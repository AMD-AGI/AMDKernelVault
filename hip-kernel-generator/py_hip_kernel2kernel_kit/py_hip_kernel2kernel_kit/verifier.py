# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import copy
import math
import random
import signal
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable

import torch

from .hip_runtime import cleanup_dir, load_hip_forward, load_python_module

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

REQUIRED_ORIGINAL_EXPORTS = ("Model", "get_inputs", "get_init_inputs")
REQUIRED_FUNCTIONAL_EXPORTS = ("module_fn", "Model", "get_inputs", "get_init_inputs")


def _path_text(path: Path) -> str:
    return path.as_posix()


@dataclass(slots=True)
class BaselineVerificationResult:
    success: bool
    message: str
    compile_success: bool = False
    correctness_success: bool = False
    module_latency_ms: float | None = None
    baseline_latency_ms: float | None = None


@dataclass(slots=True)
class CandidateVerificationResult:
    success: bool
    message: str
    compile_success: bool = False
    correctness_success: bool = False
    speedup_vs_baseline: float | None = None
    speedup_vs_module: float | None = None
    module_latency_ms: float | None = None
    baseline_latency_ms: float | None = None
    candidate_latency_ms: float | None = None


class VerificationTimeoutError(TimeoutError):
    def __init__(self, phase: str, timeout_seconds: float):
        self.phase = phase
        self.timeout_seconds = timeout_seconds
        super().__init__(f"{phase} timed out after {timeout_seconds:.2f}s.")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        # Tensor.clone() can remove storage gaps and expanded dimensions.
        # Detach first because deepcopy rejects nonleaf tensors with gradients.
        return copy.deepcopy(value.detach()).requires_grad_(value.requires_grad)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _ensure_required_exports(module: ModuleType, module_path: Path, required_exports: tuple[str, ...]) -> None:
    for export_name in required_exports:
        if not hasattr(module, export_name):
            raise AttributeError(f"{_path_text(module_path)} is missing required export `{export_name}`.")


def _normalize_call_args(value: Any) -> tuple[list[Any], dict[str, Any]]:
    if value is None:
        return [], {}
    if isinstance(value, dict):
        return [], _clone_value(value)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[0], list) and isinstance(value[1], dict):
        return _clone_value(value[0]), _clone_value(value[1])
    if isinstance(value, tuple):
        return list(_clone_value(value)), {}
    if isinstance(value, list):
        return _clone_value(value), {}
    return [_clone_value(value)], {}


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        if value.layout == torch.strided and not value.is_quantized:
            # Transfer the storage so sliced and expanded inputs retain their layout.
            storage_size = value.untyped_storage().nbytes() // value.element_size()
            storage = value.detach().as_strided((storage_size,), (1,), 0)
            moved = storage.to(device).as_strided(value.shape, value.stride(), value.storage_offset())
            return moved.requires_grad_(value.requires_grad)
        return value.to(device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _compare_scalars(expected: Any, actual: Any, rtol: float, atol: float) -> tuple[bool, str]:
    if isinstance(expected, bool) and isinstance(actual, bool):
        return expected == actual, f"Boolean mismatch: expected {expected}, got {actual}."
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isclose(float(expected), float(actual), rel_tol=rtol, abs_tol=atol):
            return True, ""
        return False, f"Scalar mismatch: expected {expected}, got {actual}."
    if expected == actual:
        return True, ""
    return False, f"Value mismatch: expected {expected!r}, got {actual!r}."


def _compare_outputs(expected: Any, actual: Any, rtol: float, atol: float, path: str = "output") -> tuple[bool, str]:
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if expected.shape != actual.shape:
            return False, f"{path} tensor shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}."
        expected_cpu = expected.detach().cpu()
        actual_cpu = actual.detach().cpu()
        matches = torch.allclose(expected_cpu, actual_cpu, rtol=rtol, atol=atol, equal_nan=True)
        if matches:
            return True, ""
        max_abs = torch.max(torch.abs(expected_cpu - actual_cpu)).item()
        return False, f"{path} tensor mismatch: max abs diff={max_abs:.6g}."

    if type(expected) is not type(actual):
        return False, f"{path} type mismatch: expected {type(expected).__name__}, got {type(actual).__name__}."

    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"{path} length mismatch: expected {len(expected)}, got {len(actual)}."
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            matches, detail = _compare_outputs(expected_item, actual_item, rtol, atol, f"{path}[{index}]")
            if not matches:
                return False, detail
        return True, ""

    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return False, f"{path} dict keys mismatch: expected {sorted(expected.keys())}, got {sorted(actual.keys())}."
        for key in expected:
            matches, detail = _compare_outputs(expected[key], actual[key], rtol, atol, f"{path}[{key!r}]")
            if not matches:
                return False, detail
        return True, ""

    matches, detail = _compare_scalars(expected, actual, rtol, atol)
    if matches:
        return True, ""
    return False, f"{path} {detail}"


class _TrackedForward:
    def __init__(self, forward: Callable[..., Any]):
        self.forward = forward
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return self.forward(*args, **kwargs)


def _normalize_timeout_seconds(timeout_seconds: float | None) -> float | None:
    if timeout_seconds is None:
        return None
    resolved = float(timeout_seconds)
    if resolved <= 0:
        return None
    return resolved


def _run_with_signal_timeout(
    fn: Callable[[], Any],
    *,
    phase: str,
    timeout_seconds: float,
) -> Any:
    def _handle_timeout(_signum, _frame) -> None:
        raise VerificationTimeoutError(phase, timeout_seconds)

    previous_handler = signal.getsignal(signal.SIGALRM)
    if hasattr(signal, "setitimer") and hasattr(signal, "ITIMER_REAL"):
        previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
        try:
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            if previous_timer[0] > 0 or previous_timer[1] > 0:
                signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])

    previous_alarm = signal.alarm(0)
    try:
        signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(max(1, math.ceil(timeout_seconds)))
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_alarm > 0:
            signal.alarm(previous_alarm)


def _run_with_thread_timeout(
    fn: Callable[[], Any],
    *,
    phase: str,
    timeout_seconds: float,
) -> Any:
    outcome: dict[str, Any] = {}

    def _runner() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # pragma: no cover - passthrough container
            outcome["error"] = exc

    worker = threading.Thread(target=_runner, daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise VerificationTimeoutError(phase, timeout_seconds)
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _run_with_timeout(
    fn: Callable[[], Any],
    *,
    phase: str,
    timeout_seconds: float | None,
) -> Any:
    resolved_timeout = _normalize_timeout_seconds(timeout_seconds)
    if resolved_timeout is None:
        return fn()

    if hasattr(signal, "SIGALRM") and threading.current_thread() is threading.main_thread():
        return _run_with_signal_timeout(fn, phase=phase, timeout_seconds=resolved_timeout)
    return _run_with_thread_timeout(fn, phase=phase, timeout_seconds=resolved_timeout)


def _format_phase_exception(phase: str, exc: Exception) -> str:
    return f"{phase} failed with {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"


def _best_effort_cuda_cleanup() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def _measure_latency_ms(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    timeout_seconds: float | None = None,
    phase: str = "benchmark execution",
) -> float:
    if iterations < 1:
        raise ValueError("iterations must be at least 1.")

    def _runner() -> float:
        with torch.no_grad():
            for _ in range(max(warmup, 0)):
                fn()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        with torch.no_grad():
            for _ in range(iterations):
                fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / iterations

    return _run_with_timeout(_runner, phase=phase, timeout_seconds=timeout_seconds)


class VerificationContext:
    def __init__(
        self,
        *,
        functional_model: Any,
        original_model: Any,
        expected_output: Any,
        forward_args: list[Any],
        forward_kwargs: dict[str, Any],
        seed: int,
        rtol: float,
        atol: float,
        perf_warmup: int,
        perf_iterations: int,
        module_latency_ms: float,
        baseline_latency_ms: float,
        offload_arch: str | None = None,
        hip_compile_timeout_seconds: float | None = None,
        execution_timeout_seconds: float | None = None,
        benchmark_timeout_seconds: float | None = None,
        extra_include_paths: Iterable[str] | None = None,
    ):
        self.functional_model = functional_model
        self.original_model = original_model
        self.expected_output = expected_output
        self.forward_args = forward_args
        self.forward_kwargs = forward_kwargs
        self.seed = seed
        self.rtol = rtol
        self.atol = atol
        self.perf_warmup = perf_warmup
        self.perf_iterations = perf_iterations
        self.module_latency_ms = module_latency_ms
        self.baseline_latency_ms = baseline_latency_ms
        self.offload_arch = offload_arch
        self.hip_compile_timeout_seconds = hip_compile_timeout_seconds
        self.execution_timeout_seconds = execution_timeout_seconds
        self.benchmark_timeout_seconds = benchmark_timeout_seconds
        self.extra_include_paths = tuple(extra_include_paths or ())

    def verify_candidate(
        self,
        candidate_hip_path: Path,
        *,
        build_dir: Path,
        keep_build_dir: bool = False,
    ) -> CandidateVerificationResult:
        compile_success = False
        correctness_success = False
        current_phase = "compile candidate HIP extension"
        try:
            candidate_fn = _run_with_timeout(
                lambda: load_hip_forward(
                    candidate_hip_path,
                    build_dir,
                    verbose=False,
                    offload_arch=self.offload_arch,
                    extra_include_paths=self.extra_include_paths,
                ),
                phase=current_phase,
                timeout_seconds=self.hip_compile_timeout_seconds,
            )
            compile_success = True

            current_phase = "execute candidate HIP forward"
            _set_seed(self.seed)
            tracked_candidate_fn = _TrackedForward(candidate_fn)
            actual = _run_with_timeout(
                lambda: self.functional_model(
                    *_clone_value(self.forward_args),
                    **_clone_value(self.forward_kwargs),
                    fn=tracked_candidate_fn,
                ),
                phase=current_phase,
                timeout_seconds=self.execution_timeout_seconds,
            )
            if tracked_candidate_fn.calls == 0:
                return CandidateVerificationResult(
                    False,
                    "The functional model did not call the candidate HIP forward function.",
                    compile_success=compile_success,
                    correctness_success=False,
                    module_latency_ms=self.module_latency_ms,
                    baseline_latency_ms=self.baseline_latency_ms,
                )
            matches, detail = _compare_outputs(self.expected_output, actual, self.rtol, self.atol)
            if not matches:
                return CandidateVerificationResult(
                    False,
                    detail,
                    compile_success=compile_success,
                    correctness_success=False,
                    module_latency_ms=self.module_latency_ms,
                    baseline_latency_ms=self.baseline_latency_ms,
                )
            correctness_success = True

            current_phase = "benchmark candidate HIP function"
            candidate_latency_ms = _measure_latency_ms(
                lambda: self.functional_model(
                    *_clone_value(self.forward_args),
                    **_clone_value(self.forward_kwargs),
                    fn=candidate_fn,
                ),
                warmup=self.perf_warmup,
                iterations=self.perf_iterations,
                timeout_seconds=self.benchmark_timeout_seconds,
                phase=current_phase,
            )

            speedup_vs_baseline = None
            if candidate_latency_ms > 0:
                speedup_vs_baseline = self.baseline_latency_ms / candidate_latency_ms

            speedup_vs_module = None
            if candidate_latency_ms > 0:
                speedup_vs_module = self.module_latency_ms / candidate_latency_ms

            summary = "Correctness passed."
            if speedup_vs_baseline is not None:
                summary += (
                    f" Speedup vs baseline HIP={speedup_vs_baseline:.4f}x "
                    f"(baseline={self.baseline_latency_ms:.4f}ms, candidate={candidate_latency_ms:.4f}ms)."
                )
            if speedup_vs_module is not None:
                summary += (
                    f" Speedup vs PyTorch={speedup_vs_module:.4f}x "
                    f"(PyTorch={self.module_latency_ms:.4f}ms)."
                )

            return CandidateVerificationResult(
                True,
                summary,
                compile_success=compile_success,
                correctness_success=correctness_success,
                speedup_vs_baseline=speedup_vs_baseline,
                speedup_vs_module=speedup_vs_module,
                module_latency_ms=self.module_latency_ms,
                baseline_latency_ms=self.baseline_latency_ms,
                candidate_latency_ms=candidate_latency_ms,
            )
        except VerificationTimeoutError as exc:
            return CandidateVerificationResult(
                False,
                str(exc),
                compile_success=compile_success,
                correctness_success=correctness_success,
                module_latency_ms=self.module_latency_ms,
                baseline_latency_ms=self.baseline_latency_ms,
            )
        except Exception as exc:
            return CandidateVerificationResult(
                False,
                _format_phase_exception(current_phase, exc),
                compile_success=compile_success,
                correctness_success=correctness_success,
                module_latency_ms=self.module_latency_ms,
                baseline_latency_ms=self.baseline_latency_ms,
            )
        finally:
            _best_effort_cuda_cleanup()
            if not keep_build_dir:
                cleanup_dir(build_dir)


def prepare_verification_context(
    *,
    original_module_path: Path,
    functional_module_path: Path,
    baseline_hip_path: Path,
    baseline_build_dir: Path,
    seed: int,
    rtol: float,
    atol: float,
    perf_warmup: int,
    perf_iterations: int,
    keep_build_dir: bool = False,
    offload_arch: str | None = None,
    python_load_timeout_seconds: float | None = None,
    hip_compile_timeout_seconds: float | None = None,
    execution_timeout_seconds: float | None = None,
    benchmark_timeout_seconds: float | None = None,
) -> tuple[BaselineVerificationResult, VerificationContext | None]:
    if not torch.cuda.is_available():
        return BaselineVerificationResult(False, "CUDA/HIP device is not available."), None

    compile_success = False
    correctness_success = False
    current_phase = "load original PyTorch module"
    try:
        original_module = _run_with_timeout(
            lambda: load_python_module(
                original_module_path,
                f"hip2hip_original_{original_module_path.stem}_{seed}",
            ),
            phase=current_phase,
            timeout_seconds=python_load_timeout_seconds,
        )

        current_phase = "load functional PyTorch module"
        functional_module = _run_with_timeout(
            lambda: load_python_module(
                functional_module_path,
                f"hip2hip_functional_{functional_module_path.stem}_{seed}",
            ),
            phase=current_phase,
            timeout_seconds=python_load_timeout_seconds,
        )
        _ensure_required_exports(original_module, original_module_path, REQUIRED_ORIGINAL_EXPORTS)
        _ensure_required_exports(functional_module, functional_module_path, REQUIRED_FUNCTIONAL_EXPORTS)

        current_phase = "read module init inputs"
        _set_seed(seed)
        init_args = _run_with_timeout(
            lambda: getattr(original_module, "get_init_inputs")(),
            phase=current_phase,
            timeout_seconds=execution_timeout_seconds,
        )
        init_call_args, init_call_kwargs = _normalize_call_args(init_args)

        current_phase = "construct original PyTorch model"
        _set_seed(seed)
        original_model = _run_with_timeout(
            lambda: getattr(original_module, "Model")(*_clone_value(init_call_args), **_clone_value(init_call_kwargs)),
            phase=current_phase,
            timeout_seconds=execution_timeout_seconds,
        )

        current_phase = "construct functional PyTorch model"
        _set_seed(seed)
        functional_model = _run_with_timeout(
            lambda: getattr(functional_module, "Model")(*_clone_value(init_call_args), **_clone_value(init_call_kwargs)),
            phase=current_phase,
            timeout_seconds=execution_timeout_seconds,
        )

        current_phase = "move models to CUDA/HIP device"
        device = torch.device("cuda")
        original_model = original_model.to(device).eval()
        functional_model = functional_model.to(device).eval()

        current_phase = "read module forward inputs"
        _set_seed(seed)
        forward_inputs = _run_with_timeout(
            lambda: getattr(original_module, "get_inputs")(),
            phase=current_phase,
            timeout_seconds=execution_timeout_seconds,
        )
        forward_args, forward_kwargs = _normalize_call_args(forward_inputs)
        current_phase = "move forward inputs to CUDA/HIP device"
        forward_args = _move_to_device(_clone_value(forward_args), device)
        forward_kwargs = _move_to_device(_clone_value(forward_kwargs), device)

        current_phase = "compile baseline HIP extension"
        baseline_fn = _run_with_timeout(
            lambda: load_hip_forward(
                baseline_hip_path,
                baseline_build_dir,
                verbose=False,
                offload_arch=offload_arch,
            ),
            phase=current_phase,
            timeout_seconds=hip_compile_timeout_seconds,
        )
        compile_success = True

        current_phase = "execute original PyTorch forward"
        _set_seed(seed)
        expected_output = _run_with_timeout(
            lambda: original_model(
                *_clone_value(forward_args),
                **_clone_value(forward_kwargs),
            ),
            phase=current_phase,
            timeout_seconds=execution_timeout_seconds,
        )

        current_phase = "execute baseline HIP forward"
        _set_seed(seed)
        tracked_baseline_fn = _TrackedForward(baseline_fn)
        baseline_output = _run_with_timeout(
            lambda: functional_model(
                *_clone_value(forward_args),
                **_clone_value(forward_kwargs),
                fn=tracked_baseline_fn,
            ),
            phase=current_phase,
            timeout_seconds=execution_timeout_seconds,
        )

        if tracked_baseline_fn.calls == 0:
            return (
                BaselineVerificationResult(
                    False,
                    "The functional model did not call the baseline HIP forward function.",
                    compile_success=compile_success,
                    correctness_success=False,
                ),
                None,
            )

        matches, detail = _compare_outputs(expected_output, baseline_output, rtol, atol)
        if not matches:
            return (
                BaselineVerificationResult(
                    False,
                    f"Baseline HIP failed correctness validation: {detail}",
                    compile_success=compile_success,
                    correctness_success=False,
                ),
                None,
            )
        correctness_success = True

        current_phase = "benchmark original PyTorch module"
        module_latency_ms = _measure_latency_ms(
            lambda: original_model(
                *_clone_value(forward_args),
                **_clone_value(forward_kwargs),
            ),
            warmup=perf_warmup,
            iterations=perf_iterations,
            timeout_seconds=benchmark_timeout_seconds,
            phase=current_phase,
        )

        current_phase = "benchmark baseline HIP function"
        baseline_latency_ms = _measure_latency_ms(
            lambda: functional_model(
                *_clone_value(forward_args),
                **_clone_value(forward_kwargs),
                fn=baseline_fn,
            ),
            warmup=perf_warmup,
            iterations=perf_iterations,
            timeout_seconds=benchmark_timeout_seconds,
            phase=current_phase,
        )

        result = BaselineVerificationResult(
            True,
            (
                "Baseline HIP correctness passed. "
                f"PyTorch={module_latency_ms:.4f}ms, baseline HIP={baseline_latency_ms:.4f}ms."
            ),
            compile_success=compile_success,
            correctness_success=correctness_success,
            module_latency_ms=module_latency_ms,
            baseline_latency_ms=baseline_latency_ms,
        )
        context = VerificationContext(
            functional_model=functional_model,
            original_model=original_model,
            expected_output=expected_output,
            forward_args=forward_args,
            forward_kwargs=forward_kwargs,
            seed=seed,
            rtol=rtol,
            atol=atol,
            perf_warmup=perf_warmup,
            perf_iterations=perf_iterations,
            module_latency_ms=module_latency_ms,
            baseline_latency_ms=baseline_latency_ms,
            offload_arch=offload_arch,
            hip_compile_timeout_seconds=hip_compile_timeout_seconds,
            execution_timeout_seconds=execution_timeout_seconds,
            benchmark_timeout_seconds=benchmark_timeout_seconds,
            extra_include_paths=(
                str(baseline_hip_path.parent.resolve()),
                str((baseline_hip_path.parent / "include").resolve()),
            ),
        )
        return result, context
    except VerificationTimeoutError as exc:
        return (
            BaselineVerificationResult(
                False,
                str(exc),
                compile_success=compile_success,
                correctness_success=correctness_success,
            ),
            None,
        )
    except Exception as exc:
        return (
            BaselineVerificationResult(
                False,
                _format_phase_exception(current_phase, exc),
                compile_success=compile_success,
                correctness_success=correctness_success,
            ),
            None,
        )
    finally:
        _best_effort_cuda_cleanup()
        if not keep_build_dir:
            cleanup_dir(baseline_build_dir)
