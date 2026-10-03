# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Freeze trusted tests, execute separate GPU workers, and compare their outputs.

The worker host must contain no client credentials or valuable writable files.
Process isolation and hashes do not replace an OS sandbox for hostile code.
The client never imports a task, a provider, or a candidate Python module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import signal
import stat
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .contracts import PreparedTask, TaskSpec, VerificationResult, VerificationSettings
from .errors import ReferenceValidationError
from .output_metadata import compare_output_devices

_MAX_REPORT_BYTES = 8 * 1024 * 1024
_WORKER_ENVIRONMENT = {
    "PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "ROCM_PATH",
    "HIP_PATH",
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "HSA_ENABLE_SDMA",
    "HSA_XNACK",
}


def _sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}.")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Reference artifacts cannot contain a symbolic link: {path}.")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _sha256(path)
    return result


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_report(path: Path) -> dict[str, Any]:
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        information = os.fstat(handle.fileno())
        if not stat.S_ISREG(information.st_mode) or information.st_size > _MAX_REPORT_BYTES:
            raise ValueError("The worker report has an invalid type or size.")
        payload = handle.read(_MAX_REPORT_BYTES + 1)
    if len(payload) > _MAX_REPORT_BYTES:
        raise ValueError("The worker report exceeds the size limit.")

    def reject_constant(value: str) -> None:
        raise ValueError(f"The worker report contains {value}.")

    report = json.loads(payload, parse_constant=reject_constant)
    if not isinstance(report, dict):
        raise ValueError("The worker report must be a JSON object.")
    return report


class Verifier:
    """Validate external tasks before any model request can incur cost."""

    def __init__(self, settings: VerificationSettings) -> None:
        self.settings = settings
        self._prepared: dict[int, dict[str, Any]] = {}
        for name in ("atol", "rtol", "warmup_ms", "measure_ms", "timeout_seconds"):
            value = getattr(settings, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number.")
            minimum_is_valid = name in {"atol", "rtol", "warmup_ms"}
            if value < 0 or (value == 0 and not minimum_is_valid):
                raise ValueError(f"{name} is outside its permitted range.")
        if type(settings.gpu) is not int or settings.gpu < 0:
            raise ValueError("gpu must be a nonnegative integer.")
        if type(settings.seed) is not int or not 0 <= settings.seed < 2**63:
            raise ValueError("seed must be an integer from 0 through 2**63 - 1.")
        if settings.target_arch not in {None, "gfx942", "gfx950"}:
            raise ValueError("target_arch must be gfx942, gfx950, or None.")

    def _worker_command(self, mode: str, request_path: Path) -> list[str]:
        return [sys.executable, "-m", "triton_kernel_gen.worker", mode, str(request_path)]

    def _run_worker(self, mode: str, root: Path, request: dict[str, Any]) -> dict[str, Any]:
        root.mkdir(parents=True, exist_ok=True)
        request_path = root / "request.json"
        _write_json(request_path, request)
        home = root / "home"
        home.mkdir(exist_ok=True)
        temporary = root / "tmp"
        temporary.mkdir(exist_ok=True)
        environment = {
            key: value for key, value in os.environ.items() if key in _WORKER_ENVIRONMENT
        }
        for key in self.settings.excluded_environment:
            environment.pop(key, None)
        environment.update(
            HOME=str(home),
            TMPDIR=str(temporary),
            PYTHONNOUSERSITE="1",
            PYTHONPATH=str(Path(__file__).resolve().parents[1]),
            TRITON_CACHE_DIR=str(root / "triton_cache"),
            LANG="C.UTF-8",
            LC_ALL="C.UTF-8",
        )
        timed_out = False
        with (root / "stdout.log").open("wb") as stdout, (root / "stderr.log").open("wb") as stderr:
            process = subprocess.Popen(
                self._worker_command(mode, request_path),
                cwd=root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            try:
                process.wait(timeout=self.settings.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Also remove descendants after a normal exit. They must not outlive a trial.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        try:
            report = _read_report(root / "report.json")
        except (OSError, ValueError, TypeError) as exc:
            report = {
                "success": False,
                "failure_stage": "infrastructure",
                "message": f"The worker report is unavailable: {exc}",
            }
        report["worker_exit_code"] = process.returncode
        report["worker_artifacts"] = str(root)
        if timed_out:
            phase = report.get("phase", "startup")
            report.update(
                success=False,
                failure_stage="timing"
                if mode == "benchmark"
                else ("runtime" if phase in {"runtime", "jit_warmup"} else "infrastructure"),
                message=f"The {mode} worker exceeded {self.settings.timeout_seconds:g} seconds during {phase}.",
                timed_out=True,
            )
        elif process.returncode != 0:
            report["success"] = False
            report.setdefault("message", f"The worker exited with code {process.returncode}.")
        return report

    @staticmethod
    def _source_hashes(task: TaskSpec) -> dict[str, str]:
        paths = [task.module_path, task.functional_path, task.case_provider_path]
        paths.extend(task.dependency_paths)
        if task.seed_kernel_path is not None:
            paths.append(task.seed_kernel_path)
        if task.baseline_kernel_path is not None:
            paths.append(task.baseline_kernel_path)
        return {str(path.resolve()): _sha256(path.resolve()) for path in paths}

    @staticmethod
    def _guard(state: dict[str, Any]) -> None:
        try:
            if _fingerprint(state["root"]) != state["fingerprint"]:
                raise ReferenceValidationError(
                    "Frozen reference artifacts changed.", failure_stage="input_modified"
                )
            for filename, expected in state["source_hashes"].items():
                if _sha256(Path(filename)) != expected:
                    raise ReferenceValidationError(
                        f"Trusted input changed: {filename}.", failure_stage="input_modified"
                    )
        except (OSError, ValueError) as exc:
            if isinstance(exc, ReferenceValidationError):
                raise
            raise ReferenceValidationError(
                f"Trusted inputs changed: {exc}", failure_stage="input_modified"
            ) from exc

    def prepare(self, task: TaskSpec, work_dir: Path) -> PreparedTask:
        work_dir = Path(work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            source_hashes = self._source_hashes(task)
            root = Path(tempfile.mkdtemp(prefix="oracle-", dir=work_dir))
            request = {
                "task": {
                    "task_id": task.task_id,
                    "module_path": str(task.module_path.resolve()),
                    "functional_path": str(task.functional_path.resolve()),
                    "case_provider_path": str(task.case_provider_path.resolve()),
                    "dependency_paths": [str(path.resolve()) for path in task.dependency_paths],
                },
                "settings": asdict(self.settings),
            }
            record = self._run_worker("reference", root, request)
            if record.get("success") is not True:
                raise ReferenceValidationError(
                    str(record.get("message", "The reference worker failed.")),
                    failure_stage=record.get("failure_stage", "reference"),
                )
            if self._source_hashes(task) != source_hashes:
                raise ReferenceValidationError(
                    "Trusted inputs changed during preparation.", failure_stage="input_modified"
                )
            cases = record.get("cases", [])
            for kind in ("correctness", "performance"):
                selected = [case for case in cases if case["kind"] == kind]
                if not selected or len({case["case_id"] for case in selected}) != len(selected):
                    raise ReferenceValidationError(
                        f"The oracle did not establish unique {kind} cases."
                    )
            record["source_hashes"] = source_hashes
            record["provenance"] = copy.deepcopy(task.provenance)
            prepared = PreparedTask(
                task, work_dir, copy.deepcopy(record), {"reference_artifacts": str(root)}
            )
            state = {
                "prepared": prepared,
                "root": root,
                "record": copy.deepcopy(record),
                "source_hashes": source_hashes,
                "fingerprint": _fingerprint(root),
            }
            self._prepared[id(prepared)] = state
            if task.baseline_kernel_path is not None:
                result = self.verify(prepared, task.baseline_kernel_path)
                if not result.success:
                    self._prepared.pop(id(prepared), None)
                    raise ReferenceValidationError(
                        f"The baseline kernel failed validation: {result.message}"
                    )
                by_id = {case["case_id"]: case for case in result.performance_cases}
                for case in record["cases"]:
                    if case["kind"] == "performance":
                        case.update(
                            baseline_kernel_ms=by_id[case["case_id"]]["candidate_ms"],
                            baseline_ms=by_id[case["case_id"]]["candidate_ms"],
                            baseline_kind="baseline_kernel",
                        )
                record["baseline_validation"] = asdict(result)
                prepared.reference_record = copy.deepcopy(record)
                state["record"] = copy.deepcopy(record)
            _write_json(work_dir / "prepared_reference.json", record)
            return prepared
        except ReferenceValidationError:
            raise
        except Exception as exc:
            raise ReferenceValidationError(
                f"Reference preparation failed: {type(exc).__name__}: {exc}",
                failure_stage="infrastructure",
            ) from exc

    def _candidate_trial(
        self,
        prepared: PreparedTask,
        state: dict[str, Any],
        candidate: bytes,
        mode: str,
    ) -> tuple[dict[str, Any], Path]:
        root = Path(tempfile.mkdtemp(prefix=f"candidate-{mode}-", dir=prepared.work_dir))
        candidate_path = root / "candidate.py"
        candidate_path.write_bytes(candidate)
        candidate_path.chmod(0o444)
        # Copy only frozen function inputs. Do not expose expected outputs or source paths.
        from .tensor_utils import load_tree, save_tree

        cases = []
        for case in state["record"]["cases"]:
            if mode == "benchmark" and case["kind"] != "performance":
                continue
            save_tree(
                root / "inputs" / case["key"], load_tree(state["root"] / "inputs" / case["key"])
            )
            cases.append({key: case[key] for key in ("key", "case_id", "kind", "seed")})
        report = self._run_worker(
            mode,
            root,
            {
                "candidate_path": str(candidate_path),
                "settings": asdict(self.settings),
                "cases": cases,
            },
        )
        if _sha256(candidate_path) != hashlib.sha256(candidate).hexdigest():
            report.update(
                success=False,
                failure_stage="input_modified",
                message="The candidate changed its source file.",
            )
        return report, root

    @staticmethod
    def _worker_message(report: dict[str, Any]) -> str:
        message = str(report.get("message", "The worker failed."))
        if report.get("traceback"):
            message += "\n" + str(report["traceback"])[-12000:]
        return message

    @staticmethod
    def _same_environment(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
        for key in ("torch", "hip", "triton", "backend", "target_arch", "gpu_index", "gpu_name"):
            if reference.get(key) != candidate.get(key):
                raise ValueError(
                    f"The worker environment changed at {key}. Baseline timings are not comparable."
                )

    def verify(self, prepared: PreparedTask, candidate_path: Path) -> VerificationResult:
        result = VerificationResult(
            False, False, False, "The candidate did not complete verification."
        )
        state = self._prepared.get(id(prepared))
        if state is None or state["prepared"] is not prepared:
            result.failure_stage = "infrastructure"
            result.message = "Call prepare with this Verifier before verify. Prepared tasks cannot transfer between verifiers."
            return result
        try:
            self._guard(state)
            from .tensor_utils import compare_outputs, load_tree

            # Keep the authority in trusted memory before the candidate process starts.
            expected = {
                case["key"]: load_tree(state["root"] / "expected" / case["key"])
                for case in state["record"]["cases"]
            }
            records = [copy.deepcopy(case) for case in state["record"]["cases"]]
            for case in records:
                case.update(
                    compile_success=False,
                    runtime_success=False,
                    correctness_success=False,
                    timing_success=False,
                )
            result.correctness_cases = [case for case in records if case["kind"] == "correctness"]
            result.performance_cases = [case for case in records if case["kind"] == "performance"]
            candidate = Path(candidate_path).read_bytes()
            report, root = self._candidate_trial(prepared, state, candidate, "execute")
            self._guard(state)
            result.environment = copy.deepcopy(report.get("environment", {}))
            result.environment["worker_artifacts"] = {"execute": str(root)}
            result.compile_success = report.get("compile_success") is True
            result.runtime_success = report.get("runtime_success") is True
            progress = {case["key"]: case for case in report.get("cases", [])}
            for case in records:
                current = progress.get(case["key"], {})
                case["compile_success"] = current.get("jit_launches", 0) > 0
                case["runtime_success"] = current.get("runtime_launches", 0) > 0
                case["jit_launches"] = current.get("jit_launches", 0)
                case["runtime_launches"] = current.get("runtime_launches", 0)
            if (
                not report.get("success")
                or not result.compile_success
                or not result.runtime_success
            ):
                result.failure_stage = report.get("failure_stage") or "runtime"
                result.message = self._worker_message(report)
                return result
            self._same_environment(state["record"]["environment"], report.get("environment", {}))
            if set(progress) != {case["key"] for case in records} or not all(
                case["runtime_success"] for case in records
            ):
                raise ValueError("The worker skipped a requested configuration.")
            failures = []
            for case in records:
                native_output = progress[case["key"]].get("native_output")
                if not isinstance(native_output, dict):
                    raise ValueError("The candidate omitted native output metadata.")
                device_comparison = compare_output_devices(case["native_output"], native_output)
                comparison = compare_outputs(
                    expected[case["key"]],
                    load_tree(root / "outputs" / case["key"]),
                    atol=case["atol"],
                    rtol=case["rtol"],
                )
                case["comparison"] = comparison
                case["candidate_native_output"] = native_output
                case["device_comparison"] = device_comparison
                case["correctness_success"] = comparison["success"] and device_comparison["success"]
                if not case["correctness_success"]:
                    detail = (
                        comparison["message"]
                        if not comparison["success"]
                        else device_comparison["message"]
                    )
                    failures.append(f"{case['kind']}/{case['case_id']}: {detail}")
            if failures:
                result.failure_stage = "correctness"
                result.message = "\n".join(failures)
                return result
            result.correctness_success = True
            timing, timing_root = self._candidate_trial(prepared, state, candidate, "benchmark")
            self._guard(state)
            result.environment["worker_artifacts"]["benchmark"] = str(timing_root)
            result.environment["timing_environment"] = copy.deepcopy(timing.get("environment", {}))
            if not timing.get("success") or timing.get("timing_success") is not True:
                result.failure_stage = (
                    "input_modified"
                    if timing.get("failure_stage") == "input_modified"
                    else "timing"
                )
                result.message = self._worker_message(timing)
                return result
            self._same_environment(report.get("environment", {}), timing.get("environment", {}))
            timed = {case["key"]: case for case in timing.get("cases", [])}
            if set(timed) != {case["key"] for case in result.performance_cases}:
                raise ValueError("The timing worker skipped a performance configuration.")
            for case in result.performance_cases:
                native_output = timed[case["key"]].get("native_timed_output")
                if not isinstance(native_output, dict):
                    raise ValueError("The candidate omitted native post-timing output metadata.")
                device_comparison = compare_output_devices(case["native_output"], native_output)
                comparison = compare_outputs(
                    expected[case["key"]],
                    load_tree(timing_root / "timed_outputs" / case["key"]),
                    atol=case["atol"],
                    rtol=case["rtol"],
                )
                case["timed_comparison"] = comparison
                case["candidate_native_timed_output"] = native_output
                case["timed_device_comparison"] = device_comparison
                if not comparison["success"] or not device_comparison["success"]:
                    result.correctness_success = False
                    case["correctness_success"] = False
                    result.failure_stage = "correctness"
                    detail = (
                        comparison["message"]
                        if not comparison["success"]
                        else device_comparison["message"]
                    )
                    result.message = f"The timed output differs for {case['case_id']}: {detail}"
                    return result
                latency = timed[case["key"]].get("candidate_ms")
                baseline = case["baseline_ms"]
                if any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                    for value in (latency, baseline)
                ):
                    raise ValueError("Timing values must be finite, positive numbers.")
                case.update(candidate_ms=latency, speedup=baseline / latency, timing_success=True)
            result.timing_success = True
            result.message = (
                "The candidate passed all frozen cases and completed every performance measurement."
            )
            return result
        except ReferenceValidationError as exc:
            result.failure_stage = exc.failure_stage
            result.message = str(exc)
            return result
        except Exception as exc:
            result.failure_stage = "infrastructure"
            result.message = f"Verification failed: {type(exc).__name__}: {exc}"
            return result


__all__ = ["ReferenceValidationError", "Verifier"]
