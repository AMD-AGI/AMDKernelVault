# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""GPU subprocess entry point. Never import external tasks in the LLM client.

Run this worker on a dedicated, credential-free GPU host or inside a container.
A subprocess provides fault containment, but it is not a hostile-code sandbox.
The oracle and candidate use separate processes and separate artifact trees.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import random
import traceback
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

import torch

from .contracts import Case, VerificationSettings
from .errors import ReferenceValidationError
from .forwarding import check_forwarding_wrapper
from .output_metadata import compare_output_devices
from .source_loader import TrustedSourceImporter, load_module
from .tensor_utils import clone_tree, compare_outputs, load_tree, save_tree, tree_metadata
from .timing import bench_ms


class CandidateContractError(ValueError):
    """The candidate does not implement the release interface."""


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
    except ImportError:
        pass
    else:
        np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_environment(settings: VerificationSettings) -> tuple[dict[str, Any], torch.device]:
    import triton

    if triton.__version__ != "3.3.0":
        raise RuntimeError(f"Triton 3.3.0 is required. Found {triton.__version__}.")
    if not torch.version.hip:
        raise RuntimeError("PyTorch must use ROCm. torch.version.hip is empty.")
    if not torch.cuda.is_available():
        raise RuntimeError("An AMD GPU is required.")
    if not 0 <= settings.gpu < torch.cuda.device_count():
        raise RuntimeError(f"GPU index {settings.gpu} is unavailable.")
    torch.cuda.set_device(settings.gpu)
    device = torch.device("cuda", settings.gpu)
    properties = torch.cuda.get_device_properties(device)
    target = triton.runtime.driver.active.get_current_target()
    arch = str(target.arch).split(":", 1)[0]
    if target.backend != "hip":
        raise RuntimeError(f"Triton must use the HIP backend. Found {target.backend}.")
    if arch not in {"gfx942", "gfx950"}:
        raise RuntimeError(f"Unsupported AMD architecture {arch}. Use gfx942 or gfx950.")
    if settings.target_arch is not None and arch != settings.target_arch:
        raise RuntimeError(f"Requested {settings.target_arch}, but the GPU uses {arch}.")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "triton": triton.__version__,
        "backend": target.backend,
        "target_arch": arch,
        "gpu_index": settings.gpu,
        "gpu_name": properties.name,
        "gpu_total_memory": properties.total_memory,
        "worker_pid": os.getpid(),
    }, device


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _assert_unchanged(before: Any, after: Any, label: str) -> None:
    """Reject mutation, including writes to storage gaps and aliased views."""
    before_metadata = tree_metadata(before)
    after_metadata = tree_metadata(after)
    for metadata in (before_metadata, after_metadata):
        for tensor in metadata["tensors"]:
            # Inference clones intentionally detach parameters from autograd.
            tensor.pop("requires_grad", None)
    if before_metadata != after_metadata:
        raise ReferenceValidationError(
            f"{label} mutated input geometry, aliases, or scalar values."
        )
    if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor):
        if (
            before.dtype != after.dtype
            or before.shape != after.shape
            or before.stride() != after.stride()
        ):
            raise ReferenceValidationError(f"{label} changed tensor geometry.")
        if before.storage_offset() != after.storage_offset():
            raise ReferenceValidationError(f"{label} changed a tensor storage offset.")
        before_bytes = torch.empty(0, dtype=torch.uint8, device=before.device).set_(
            before.untyped_storage()
        )
        after_bytes = torch.empty(0, dtype=torch.uint8, device=after.device).set_(
            after.untyped_storage()
        )
        if not torch.equal(before_bytes.cpu(), after_bytes.cpu()):
            raise ReferenceValidationError(
                f"{label} mutated tensor storage. Mutable tasks are unsupported."
            )
        return
    if type(before) is not type(after):
        raise ReferenceValidationError(f"{label} changed an input type.")
    if isinstance(before, dict):
        if before.keys() != after.keys():
            raise ReferenceValidationError(f"{label} changed input keys.")
        for key in before:
            _assert_unchanged(before[key], after[key], label)
    elif isinstance(before, (tuple, list)):
        if len(before) != len(after):
            raise ReferenceValidationError(f"{label} changed an input length.")
        for left, right in zip(before, after):
            _assert_unchanged(left, right, label)
    elif before != after and not (
        isinstance(before, float) and math.isnan(before) and math.isnan(after)
    ):
        raise ReferenceValidationError(f"{label} changed an input value.")


def _assert_match(
    expected: Any, actual: Any, atol: float, rtol: float, label: str
) -> dict[str, Any]:
    result = compare_outputs(expected, actual, atol=atol, rtol=rtol)
    if not result["success"]:
        raise ReferenceValidationError(f"{label}: {result['message']}")
    return result


def _assert_native_output(expected: dict[str, Any], actual: Any, label: str) -> dict[str, Any]:
    metadata = tree_metadata(actual)
    result = compare_output_devices(expected, metadata)
    if not result["success"]:
        raise ReferenceValidationError(f"{label}: {result['message']}")
    return metadata


def _case_payload(case: Case) -> dict[str, Any]:
    return {
        "args": case.args,
        "kwargs": case.kwargs,
        "init_args": case.init_args,
        "init_kwargs": case.init_kwargs,
    }


def _validate_case(case: Case, settings: VerificationSettings) -> tuple[float, float]:
    if not isinstance(case, Case):
        raise ReferenceValidationError("The case provider must return Case objects.")
    if not isinstance(case.case_id, str) or not case.case_id.strip():
        raise ReferenceValidationError("Every case needs a nonempty string case_id.")
    if not isinstance(case.args, tuple) or not isinstance(case.init_args, tuple):
        raise ReferenceValidationError("Case args and init_args must be tuples.")
    if not isinstance(case.kwargs, dict) or not isinstance(case.init_kwargs, dict):
        raise ReferenceValidationError("Case kwargs and init_kwargs must be dictionaries.")
    if any(not isinstance(key, str) for key in (*case.kwargs, *case.init_kwargs)):
        raise ReferenceValidationError("Argument keyword names must be strings.")
    if "fn" in case.kwargs:
        raise ReferenceValidationError("The fn keyword belongs to the verifier.")
    if not isinstance(case.metadata, dict):
        raise ReferenceValidationError("Case metadata must be a dictionary.")
    json.dumps(case.metadata, allow_nan=False)
    atol = settings.atol if case.atol is None else case.atol
    rtol = settings.rtol if case.rtol is None else case.rtol
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in (atol, rtol)
    ):
        raise ReferenceValidationError("Case tolerances must be finite, nonnegative numbers.")
    return float(atol), float(rtol)


def _check_wrapper(functional: ModuleType) -> None:
    if not callable(getattr(functional, "module_fn", None)):
        raise ReferenceValidationError("The functional reference must export callable module_fn.")
    model_type = getattr(functional, "Model", None)
    if not isinstance(model_type, type) or not issubclass(model_type, torch.nn.Module):
        raise ReferenceValidationError(
            "The functional reference Model must be a torch.nn.Module class."
        )
    signature = inspect.signature(model_type.forward)
    parameter = signature.parameters.get("fn")
    if (
        parameter is None
        or parameter.kind
        not in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
        or parameter.default is inspect.Parameter.empty
    ):
        raise ReferenceValidationError(
            "Model.forward needs an explicit fn keyword with a default reference function."
        )
    check_forwarding_wrapper(model_type, functional.module_fn)


def _model_state(model: torch.nn.Module) -> dict[str, Any]:
    """Capture tensor state and ordinary module attributes, including private counters."""
    bookkeeping = set(vars(torch.nn.Module())) - {"training"}

    def attribute(value: Any) -> Any:
        if isinstance(value, torch.Tensor) or type(value) in {
            str,
            int,
            float,
            complex,
            bool,
            type(None),
        }:
            return value
        if isinstance(value, (torch.dtype, torch.device)):
            return {"attribute_type": type(value).__name__, "value": str(value)}
        if isinstance(value, torch.Size):
            return {"attribute_type": "torch.Size", "items": tuple(value)}
        if isinstance(value, list):
            return [attribute(child) for child in value]
        if isinstance(value, tuple):
            return tuple(attribute(child) for child in value)
        if isinstance(value, dict):
            return {key: attribute(child) for key, child in value.items()}
        if isinstance(value, set) and all(isinstance(child, str) for child in value):
            return {"attribute_type": "set", "items": tuple(sorted(value))}
        if callable(value):
            return {"callable_identity": id(value)}
        raise ReferenceValidationError(f"Unsupported model attribute type: {type(value).__name__}.")

    return {
        "parameters": dict(model.named_parameters(remove_duplicate=False)),
        "buffers": dict(model.named_buffers(remove_duplicate=False)),
        "attributes": {
            name: {
                key: attribute(value)
                for key, value in vars(module).items()
                if key not in bookkeeping
            }
            for name, module in model.named_modules(remove_duplicate=False)
        },
    }


def _model_call(
    module: ModuleType,
    payload: dict[str, Any],
    seed: int,
    device: torch.device,
    injected: Callable[..., Any] | None = None,
    *,
    forwarding_only: bool = False,
) -> Any:
    materialized = clone_tree(payload, device=device)
    untouched = clone_tree(materialized)
    set_seed(seed)
    model = (
        module.Model(*materialized["init_args"], **materialized["init_kwargs"]).to(device).eval()
    )
    if not isinstance(model, torch.nn.Module):
        raise ReferenceValidationError("Model must be a torch.nn.Module instance.")
    if forwarding_only:
        from .forwarding import check_forwarding_instance

        check_forwarding_instance(model)
    state = clone_tree(_model_state(model))
    set_seed(seed)
    kwargs = dict(materialized["kwargs"])
    if injected is not None:
        kwargs["fn"] = injected
    with torch.no_grad():
        output = model(*materialized["args"], **kwargs)
    _synchronize(device)
    _assert_unchanged(untouched, materialized, "The reference")
    _assert_unchanged(state, _model_state(model), "The reference model")
    return output


def validate_reference_case(
    original: ModuleType,
    functional: ModuleType,
    case: Case,
    settings: VerificationSettings,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    """Validate a transparent wrapper and capture the function call boundary."""
    _check_wrapper(functional)
    atol, rtol = _validate_case(case, settings)
    payload = clone_tree(_case_payload(case), device="cpu")
    native_expected = _model_call(original, payload, seed, device)
    native_outputs = {"original": tree_metadata(native_expected)}
    expected = clone_tree(native_expected, device="cpu")
    _assert_match(expected, expected, 0.0, 0.0, "The original output is invalid")
    repeated = _model_call(original, payload, seed, device)
    native_outputs["repeat"] = _assert_native_output(
        native_outputs["original"], repeated, "The repeated original device differs"
    )
    _assert_match(
        expected, repeated, 0.0, 0.0, "The original is not deterministic under its fixed seed"
    )
    default_output = _model_call(functional, payload, seed, device, forwarding_only=True)
    native_outputs["default"] = _assert_native_output(
        native_outputs["original"], default_output, "The default reference device differs"
    )
    default_check = _assert_match(
        expected, default_output, atol, rtol, "The default functional reference differs"
    )
    captures: list[tuple[dict[str, Any], Any]] = []

    def capture(*args: Any, **kwargs: Any) -> Any:
        call = {"args": args, "kwargs": kwargs}
        before = clone_tree(call)
        frozen = clone_tree(call, device="cpu")
        output = functional.module_fn(*args, **kwargs)
        _synchronize(device)
        _assert_unchanged(before, call, "module_fn")
        captures.append((frozen, output))
        return output

    injected_output = _model_call(functional, payload, seed, device, capture, forwarding_only=True)
    native_outputs["injected"] = _assert_native_output(
        native_outputs["original"], injected_output, "The injected reference device differs"
    )
    if len(captures) != 1:
        raise ReferenceValidationError("Model.forward must call fn exactly once.")
    call, returned = captures[0]
    if injected_output is not returned:
        raise ReferenceValidationError(
            "Model.forward must return the fn output object directly. Wrapper transforms are unsupported."
        )
    injected_check = _assert_match(
        expected, injected_output, atol, rtol, "The injected functional reference differs"
    )
    replay = clone_tree(call, device=device)
    before = clone_tree(replay)
    set_seed(seed)
    with torch.no_grad():
        direct_output = functional.module_fn(*replay["args"], **replay["kwargs"])
    _synchronize(device)
    _assert_unchanged(before, replay, "The direct reference")
    native_outputs["replay"] = _assert_native_output(
        native_outputs["original"], direct_output, "The replayed reference device differs"
    )
    _assert_match(
        expected,
        direct_output,
        atol,
        rtol,
        "The captured function call cannot replay the reference",
    )
    return (
        call,
        expected,
        {"default": default_check, "injected": injected_check, "native_outputs": native_outputs},
    )


def prepare_reference(request: dict[str, Any], root: Path) -> dict[str, Any]:
    dependencies = tuple(Path(path) for path in request["task"].get("dependency_paths", []))
    with TrustedSourceImporter(dependencies) as importer:
        return _prepare_reference(request, root, importer)


def _prepare_reference(
    request: dict[str, Any],
    root: Path,
    importer: TrustedSourceImporter,
) -> dict[str, Any]:
    settings = VerificationSettings(**request["settings"])
    environment, device = runtime_environment(settings)
    task = request["task"]
    set_seed(settings.seed)
    original = importer.load_module(Path(task["module_path"]), "_kernelgen_original")
    set_seed(settings.seed)
    functional = importer.load_module(Path(task["functional_path"]), "_kernelgen_functional")
    set_seed(settings.seed)
    provider = importer.load_module(Path(task["case_provider_path"]), "_kernelgen_cases")
    if not callable(getattr(provider, "get_cases", None)):
        raise ReferenceValidationError("The case provider must export callable get_cases.")
    if not isinstance(getattr(original, "Model", None), type) or not issubclass(
        original.Model, torch.nn.Module
    ):
        raise ReferenceValidationError("The original Model must be a torch.nn.Module class.")
    _check_wrapper(functional)
    frozen: list[tuple[str, int, Case, float, float]] = []
    for kind in ("correctness", "performance"):
        set_seed(settings.seed)
        ids: set[str] = set()
        for index, case in enumerate(provider.get_cases(task["task_id"], original, kind=kind)):
            atol, rtol = _validate_case(case, settings)
            if case.case_id in ids:
                raise ReferenceValidationError(f"Duplicate {kind} case_id: {case.case_id}.")
            ids.add(case.case_id)
            materialized = clone_tree(_case_payload(case), device="cpu")
            copied = Case(
                case.case_id,
                **materialized,
                atol=atol,
                rtol=rtol,
                metadata=json.loads(json.dumps(case.metadata)),
            )
            frozen.append((kind, index, copied, atol, rtol))
        if not ids:
            raise ReferenceValidationError(f"The provider returned no {kind} cases.")
    records: list[dict[str, Any]] = []
    for ordinal, (kind, index, case, atol, rtol) in enumerate(frozen):
        seed = settings.seed + ordinal
        key = f"{kind}_{index}"
        call, expected, checks = validate_reference_case(
            original, functional, case, settings, seed, device
        )
        save_tree(root / "inputs" / key, call)
        save_tree(root / "expected" / key, expected)
        save_tree(root / "cases" / key, _case_payload(case))
        # Enforce codec size and view limits before the first model request.
        load_tree(root / "inputs" / key)
        load_tree(root / "expected" / key)
        record = {
            "case_id": case.case_id,
            "key": key,
            "kind": kind,
            "seed": seed,
            "atol": atol,
            "rtol": rtol,
            "metadata": case.metadata,
            "inputs": tree_metadata(_case_payload(case)),
            "function_inputs": tree_metadata(call),
            "output": tree_metadata(expected),
            "reference_checks": checks,
            "native_output": checks["native_outputs"]["original"],
        }
        if kind == "performance":
            inputs = clone_tree(call, device=device)
            before = clone_tree(inputs)
            set_seed(seed)
            with torch.no_grad():
                latency = bench_ms(
                    lambda: functional.module_fn(*inputs["args"], **inputs["kwargs"]), settings
                )
                set_seed(seed)
                output = functional.module_fn(*inputs["args"], **inputs["kwargs"])
            _synchronize(device)
            _assert_unchanged(before, inputs, "The timed reference")
            record["reference_native_timed_output"] = _assert_native_output(
                record["native_output"],
                output,
                "The timed reference device differs",
            )
            record["reference_timed_check"] = _assert_match(
                expected,
                output,
                atol,
                rtol,
                "The timed reference output differs",
            )
            record.update(
                reference_ms=latency, baseline_ms=latency, baseline_kind="functional_reference"
            )
        records.append(record)
    return {
        "success": True,
        "stage1_success": True,
        "task_id": task["task_id"],
        "environment": environment,
        "settings": asdict(settings),
        "cases": records,
        "correctness_count": sum(record["kind"] == "correctness" for record in records),
        "performance_count": sum(record["kind"] == "performance" for record in records),
        "timing_protocol": {
            "method": "triton.testing.do_bench",
            "warmup_ms": settings.warmup_ms,
            "measure_ms": settings.measure_ms,
            "quantiles": [0.5],
            "jit_and_autotuning": "excluded by an explicit warmup invocation",
            "input_cloning": "outside timed calls",
            "wrapper_allocations": "inside timed calls",
            "launch_instrumentation": "Launch probes apply to validation calls only. All benchmark callbacks use the ordinary module_fn wrapper without probes.",
            "raw_samples": "not exposed by do_bench with quantiles=[0.5]",
        },
    }


class TritonLaunchProbe:
    """Observe real JIT launches without changing autotuning or launch options."""

    def __init__(self) -> None:
        from triton.runtime.jit import JITFunction

        self.jit_type = JITFunction
        self.original_run = JITFunction.run
        self.launches = 0

    def __enter__(self) -> "TritonLaunchProbe":
        probe = self

        def observed(kernel: Any, *args: Any, **kwargs: Any) -> Any:
            result = probe.original_run(kernel, *args, **kwargs)
            if not kwargs.get("warmup", False):
                probe.launches += 1
            return result

        self.jit_type.run = observed
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.jit_type.run = self.original_run

    @contextmanager
    def suspended(self):
        """Remove the probe while the timer measures the ordinary wrapper."""
        observed_run = self.jit_type.run
        self.jit_type.run = self.original_run
        try:
            yield
        finally:
            self.jit_type.run = observed_run

    def require_kernel(self, module: ModuleType) -> None:
        for value in vars(module).values():
            for _depth in range(8):
                if isinstance(value, self.jit_type):
                    if getattr(value.fn, "__module__", None) == module.__name__:
                        return
                    break
                # Triton Autotuner and Heuristics wrap another object in fn.
                if not type(value).__module__.startswith("triton."):
                    break
                value = getattr(value, "fn", None)
        raise CandidateContractError("The candidate must define an actual Triton JIT kernel.")


def _failure_stage(exc: BaseException, phase: str) -> str:
    if phase == "environment":
        return "infrastructure"
    if phase == "timing":
        return "timing"
    if isinstance(exc, CandidateContractError):
        return "runtime" if phase == "runtime" else "compile"
    if isinstance(exc, (SyntaxError, ImportError)):
        return "compile"
    module = type(exc).__module__
    name = type(exc).__name__
    if module.startswith("triton.compiler") or name in {
        "CompilationError",
        "CompileTimeAssertionFailure",
        "PTXASError",
    }:
        return "compile"
    return "runtime"


def run_candidate(
    request: dict[str, Any], root: Path, *, benchmark: bool = False
) -> dict[str, Any]:
    settings = VerificationSettings(**request["settings"])
    report: dict[str, Any] = {
        "success": False,
        "compile_success": False,
        "runtime_success": False,
        "timing_success": False,
        "failure_stage": None,
        "phase": "environment",
        "cases": [],
        "environment": {},
    }
    report_path = root / "report.json"
    _write_json(report_path, report)
    try:
        environment, device = runtime_environment(settings)
        report["environment"] = environment
        report["phase"] = "import"
        _write_json(report_path, report)
        set_seed(settings.seed)
        with TritonLaunchProbe() as probe:
            candidate = load_module(Path(request["candidate_path"]), "_kernelgen_candidate")
            function = getattr(candidate, "module_fn", None)
            if not callable(function):
                raise CandidateContractError("The candidate must export callable module_fn.")
            probe.require_kernel(candidate)
            cases = request["cases"]
            if not cases:
                raise CandidateContractError("The candidate worker received no cases.")
            report["phase"] = "jit_warmup"
            _write_json(report_path, report)
            for case in cases:
                inputs = clone_tree(load_tree(root / "inputs" / case["key"]), device=device)
                before = clone_tree(inputs)
                launches = probe.launches
                set_seed(case["seed"])
                with torch.no_grad():
                    function(*inputs["args"], **inputs["kwargs"])
                _synchronize(device)
                if probe.launches == launches:
                    raise CandidateContractError(
                        f"Case {case['case_id']} did not invoke a Triton kernel."
                    )
                _assert_unchanged(before, inputs, "The candidate")
                report["cases"].append(
                    {
                        "key": case["key"],
                        "case_id": case["case_id"],
                        "jit_launches": probe.launches - launches,
                    }
                )
                _write_json(report_path, report)
            report["compile_success"] = True
            report["phase"] = "runtime"
            _write_json(report_path, report)
            for case, record in zip(cases, report["cases"]):
                inputs = clone_tree(load_tree(root / "inputs" / case["key"]), device=device)
                before = clone_tree(inputs)
                launches = probe.launches
                set_seed(case["seed"])
                with torch.no_grad():
                    output = function(*inputs["args"], **inputs["kwargs"])
                _synchronize(device)
                if probe.launches == launches:
                    raise CandidateContractError(
                        f"Case {case['case_id']} skipped Triton during execution."
                    )
                _assert_unchanged(before, inputs, "The candidate")
                record["native_output"] = tree_metadata(output)
                save_tree(root / "outputs" / case["key"], clone_tree(output, device="cpu"))
                record["runtime_launches"] = probe.launches - launches
                _write_json(report_path, report)
            report["runtime_success"] = True
            _write_json(report_path, report)
            if benchmark:
                report["phase"] = "timing"
                _write_json(report_path, report)
                for case, record in zip(cases, report["cases"]):
                    inputs = clone_tree(load_tree(root / "inputs" / case["key"]), device=device)
                    before = clone_tree(inputs)
                    set_seed(case["seed"])

                    with torch.no_grad():
                        with probe.suspended():
                            latency = bench_ms(
                                lambda: function(*inputs["args"], **inputs["kwargs"]), settings
                            )
                        # This is a new correctness boundary, outside the timed calls.
                        set_seed(case["seed"])
                        launches = probe.launches
                        output = function(*inputs["args"], **inputs["kwargs"])
                        if probe.launches == launches:
                            raise CandidateContractError(
                                f"Case {case['case_id']} skipped Triton after timing."
                            )
                    _synchronize(device)
                    _assert_unchanged(before, inputs, "The timed candidate")
                    record["native_timed_output"] = tree_metadata(output)
                    save_tree(
                        root / "timed_outputs" / case["key"], clone_tree(output, device="cpu")
                    )
                    record["candidate_ms"] = latency
                    record["post_timing_validation_launches"] = probe.launches - launches
                    _write_json(report_path, report)
                report["timing_success"] = True
            report.update(
                success=True, phase="complete", message="The worker completed all requested cases."
            )
    except Exception as exc:
        report.update(
            failure_stage=_failure_stage(exc, report["phase"]),
            message=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
    _write_json(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Execute an isolated TritonKernelGen GPU worker.")
    parser.add_argument("mode", choices=("reference", "execute", "benchmark"))
    parser.add_argument("request", type=Path)
    arguments = parser.parse_args(argv)
    request = json.loads(arguments.request.read_text(encoding="utf-8"))
    root = arguments.request.resolve().parent
    if arguments.mode != "reference":
        return (
            0
            if run_candidate(request, root, benchmark=arguments.mode == "benchmark")["success"]
            else 1
        )
    try:
        report = prepare_reference(request, root)
    except Exception as exc:
        report = {
            "success": False,
            "message": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    _write_json(root / "report.json", report)
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
