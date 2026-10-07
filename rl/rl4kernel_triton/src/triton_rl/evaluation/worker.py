# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate one request through fresh, timed pytest subprocesses.

This independent implementation reads the archived external reference format.
It does not import or copy the GEAK evaluator. The container is the isolation
boundary. The subprocesses only separate execution state and enforce timeouts.
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from pathlib import Path

from triton_rl.contracts import EvaluationResult
from triton_rl.evaluation.protocol import (
    REFERENCE_SEPARATOR,
    resolve_reference,
    subprocess_environment,
    validate_request,
)
from triton_rl.evaluation.verification import (
    CandidateOutputError,
    CandidateTimingError,
    ReferenceError,
    aggregate_timings,
    compare_outputs,
)

MAX_FEEDBACK = 32768
COMPILE_ERRORS = {
    "CompilationError",
    "CompileTimeAssertionFailure",
    "OutOfResources",
    "SyntaxError",
    "IndentationError",
    "ImportError",
    "ModuleNotFoundError",
    "NameError",
}


class CandidateFailure(Exception):
    def __init__(self, result: EvaluationResult):
        self.result = result


def candidate_source(code: str) -> str:
    """Remove generated tests before appending the external reference tests."""
    tree = ast.parse(code)

    def test_call(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id.startswith("test_")
        )

    def supplied_test(node: ast.AST) -> bool:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return node.name.startswith(("test_", "Test"))
        if isinstance(node, ast.Expr):
            return test_call(node.value)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            return test_call(node.value)
        if isinstance(node, ast.If):
            return any(
                isinstance(child, ast.Name) and child.id == "__name__"
                for child in ast.walk(node.test)
            )
        return False

    tree.body = [node for node in tree.body if not supplied_test(node)]
    return ast.unparse(tree) + "\n"


def prepare_sources(reference: Path, code: str, root: Path, *, seed: int = 42) -> tuple[Path, Path]:
    source = reference.read_text(encoding="utf-8")
    parts = source.split(REFERENCE_SEPARATOR)
    if len(parts) != 2 or not parts[1].strip():
        raise ReferenceError(
            "The reference must contain one separator of 146 hashes and a nonempty test section."
        )
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise ReferenceError("The reference source contains a Python syntax error.") from exc
    try:
        generated = candidate_source(code)
    except (SyntaxError, ValueError) as exc:
        raise CandidateFailure(
            EvaluationResult(
                compiled=False,
                correct=False,
                feedback=str(exc),
                error_type="compilation_error",
                diagnostics={"stage": "python_parse"},
            )
        ) from exc
    seed_boundary = (
        "\nfrom triton_rl.evaluation.pytest_runner import seed_process as _triton_rl_seed_process\n"
        f"_triton_rl_seed_process({seed!r})\n"
    )
    tests = seed_boundary + parts[1]
    files = []
    for name, text in (
        ("reference", parts[0] + REFERENCE_SEPARATOR + tests),
        ("candidate", generated + tests),
    ):
        folder = root / name
        folder.mkdir()
        path = folder / "test_kernel.py"
        path.write_text(text, encoding="utf-8")
        files.append(path)
    return files[0], files[1]


def _tail(path: Path) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - MAX_FEEDBACK))
        return stream.read().decode("utf-8", errors="replace")


def _run_stage(
    path: Path,
    *,
    reference_parent: Path,
    mode: str,
    reference: bool,
    seed: int,
    deadline: float,
    perf_dir: Path,
) -> dict:
    label = "reference" if reference else "candidate"
    stage = f"{label}_{mode}"
    remaining = deadline - time.monotonic()
    report_path = path.parent / f"{mode}_report.json"
    started_path = report_path.with_suffix(".started.json")
    report_path.unlink(missing_ok=True)
    started_path.unlink(missing_ok=True)
    log_path = path.parent / f"{mode}.log"
    perf_dir.mkdir(exist_ok=True)
    env = subprocess_environment()
    env["PERF_OUTPUT_DIR"] = str(perf_dir)
    env["PYTHONHASHSEED"] = str(seed)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    try:
        with log_path.open("wb") as log:
            if remaining <= 0:
                raise subprocess.TimeoutExpired("pytest", 0)
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "triton_rl.evaluation.pytest_runner",
                    "--file",
                    str(path),
                    "--report",
                    str(report_path),
                    "--reference-parent",
                    str(reference_parent),
                    "--seed",
                    str(seed),
                    "--mode",
                    mode,
                ],
                cwd=path.parent,
                env=env,
                stdout=log,
                stderr=log,
                timeout=remaining,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        if reference:
            raise ReferenceError(f"The {stage} stage exceeded the execution timeout.") from exc
        raise CandidateFailure(
            EvaluationResult(
                compiled=mode == "performance",
                correct=mode == "performance",
                feedback=f"The {stage} stage exceeded the execution timeout.",
                error_type="timeout",
                diagnostics={"stage": stage},
            )
        ) from exc
    feedback = _tail(log_path)
    if not report_path.is_file():
        if not reference and started_path.is_file():
            raise CandidateFailure(
                EvaluationResult(
                    compiled=mode == "performance",
                    correct=mode == "performance",
                    feedback=f"The {stage} process exited without a report. Exit code: {process.returncode}.\n{feedback}",
                    error_type="execution_error",
                    diagnostics={"stage": stage, "returncode": process.returncode},
                )
            )
        raise ReferenceError(f"The {stage} stage exited without a pytest report.\n{feedback}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        required = {
            "collected",
            "passed",
            "failed",
            "skipped",
            "collection_errors",
            "exceptions",
            "exit_code",
        }
        if not isinstance(report, dict) or set(report) != required:
            raise ValueError("Invalid report fields.")
        if any(
            type(report[key]) is not int or report[key] < 0 for key in required - {"exceptions"}
        ):
            raise ValueError("Invalid report counters.")
        if not isinstance(report["exceptions"], list) or any(
            not isinstance(item, str) for item in report["exceptions"]
        ):
            raise ValueError("Invalid report exceptions.")
    except (ValueError, OSError) as exc:
        raise ReferenceError(f"The {stage} stage returned an invalid pytest report.") from exc
    if report["exit_code"] != process.returncode:
        raise ReferenceError(f"The {stage} stage returned inconsistent pytest exit codes.")
    if process.returncode in (3, 4, 5) or process.returncode < 0:
        raise ReferenceError(f"The {stage} stage cannot execute its tests.\n{feedback}")
    if process.returncode != 0:
        if reference:
            raise ReferenceError(f"The {stage} tests failed.\n{feedback}")
        compiled = mode == "performance" or (
            report["collection_errors"] == 0
            and not COMPILE_ERRORS.intersection(report["exceptions"])
            and ("AssertionError" in report["exceptions"] or mode == "performance")
        )
        raise CandidateFailure(
            EvaluationResult(
                compiled=compiled,
                correct=mode == "performance" and compiled,
                feedback=feedback,
                error_type=(
                    "performance_error"
                    if mode == "performance"
                    else "execution_error"
                    if compiled
                    else "compilation_error"
                ),
                diagnostics={"stage": stage, "pytest": report},
            )
        )
    if report["collected"] == 0 or report["passed"] == 0 or report["skipped"] or report["failed"]:
        raise ReferenceError(f"The {stage} stage did not execute every selected test successfully.")
    report["stage"] = stage
    return report


def load_outputs(path: Path, *, reference: bool) -> dict:
    import torch

    output_path = Path(str(path).replace(".", "_") + ".pt")
    error = ReferenceError if reference else CandidateOutputError
    if not output_path.is_file():
        raise error("The tests did not save their required output dictionary.")
    try:
        return torch.load(output_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise error("The tests did not save a supported tensor dictionary.") from exc


def evaluate_job(job: dict, root: Path) -> EvaluationResult:
    request = validate_request(job["request"])
    reference = resolve_reference(Path(job["reference_root"]), request["filename"])
    import torch
    import triton

    if not torch.version.hip or not torch.cuda.is_available():
        raise ReferenceError(
            "The execution worker requires ROCm PyTorch and an available assigned GPU."
        )
    if triton.__version__ != "3.3.0":
        raise ReferenceError(
            "The execution worker requires Triton 3.3.0 for the documented timer protocol."
        )
    deadline = time.monotonic() + job["timeout_seconds"]
    metadata = {
        "seed": job["seed"],
        "atol": request["atol"],
        "rtol": request["rtol"],
        "equal_nan": False,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "rocm_version": torch.version.hip,
        "device": torch.cuda.get_device_name(0),
        "reference_file": request["filename"],
        "pytest": [],
    }
    try:
        reference_path, candidate_path = prepare_sources(
            reference, request["code"], root, seed=job["seed"]
        )
        reference_perf, candidate_perf = root / "reference-perf", root / "candidate-perf"
        common = {"reference_parent": reference.parent, "seed": job["seed"], "deadline": deadline}
        metadata["pytest"].append(
            _run_stage(
                reference_path,
                mode="correctness",
                reference=True,
                perf_dir=reference_perf,
                **common,
            )
        )
        reference_outputs = load_outputs(reference_path, reference=True)
        # Check the baseline independently before examining any candidate evidence.
        reference_matches, _ = compare_outputs(
            reference_outputs,
            reference_outputs,
            atol=request["atol"],
            rtol=request["rtol"],
        )
        if not reference_matches:
            raise ReferenceError(
                "The reference outputs fail their own comparison, including NaN rejection."
            )
        metadata["pytest"].append(
            _run_stage(
                candidate_path,
                mode="correctness",
                reference=False,
                perf_dir=candidate_perf,
                **common,
            )
        )
        candidate_outputs = load_outputs(candidate_path, reference=False)
        correct, feedback = compare_outputs(
            reference_outputs,
            candidate_outputs,
            atol=request["atol"],
            rtol=request["rtol"],
        )
        if not correct:
            return EvaluationResult(
                True,
                False,
                feedback=feedback[:MAX_FEEDBACK],
                error_type="correctness_error",
                diagnostics=metadata,
            )
        metadata["pytest"].append(
            _run_stage(
                reference_path,
                mode="performance",
                reference=True,
                perf_dir=reference_perf,
                **common,
            )
        )
        # Validate every reference timing before the candidate performance stage.
        aggregate_timings(reference_perf, reference_perf)
        metadata["pytest"].append(
            _run_stage(
                candidate_path,
                mode="performance",
                reference=False,
                perf_dir=candidate_perf,
                **common,
            )
        )
        timing = aggregate_timings(reference_perf, candidate_perf)
        metadata.update(timing.diagnostics)
        return EvaluationResult(
            True,
            True,
            speedup=timing.speedup,
            latency_ms=timing.latency_ms,
            baseline_latency_ms=timing.baseline_latency_ms,
            feedback=feedback,
            diagnostics=metadata,
        )
    except CandidateFailure as exc:
        value = exc.result.to_dict()
        value["diagnostics"] = {**metadata, **value["diagnostics"]}
        return EvaluationResult.from_dict(value)
    except CandidateOutputError as exc:
        return EvaluationResult(
            exc.compiled,
            False,
            feedback=str(exc),
            error_type="candidate_output_error",
            diagnostics={**metadata, "stage": exc.stage},
        )
    except CandidateTimingError as exc:
        return EvaluationResult(
            True, True, feedback=str(exc), error_type="performance_error", diagnostics=metadata
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    try:
        job = json.loads(args.request.read_text(encoding="utf-8"))
        result = evaluate_job(job, args.request.parent)
        envelope = {"result": result.to_dict()}
    except Exception as exc:
        envelope = {"error": {"type": type(exc).__name__, "message": str(exc)[:MAX_FEEDBACK]}}
    args.result.write_text(json.dumps(envelope, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
