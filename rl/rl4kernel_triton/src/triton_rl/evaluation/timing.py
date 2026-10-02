# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Time caller operations with the documented benchmark protocol.

Independent implementation from protocol descriptions. This module contains no
copied GEAK implementation source. Triton imports occur only during measurement.

``warm_up`` and ``repetition`` specify millisecond budgets, not call counts.
The archived fields use four decimal places and preserve requested order.
With default quantiles, ``ms`` is p50, ``min_ms`` is p80, and ``max_ms`` is p20.
Neither label denotes an extremum. Explicit percentile fields retain unrounded
measurements when the caller requests the corresponding percentile.

The process-local collector supports one writer per output directory. A save
replaces each operator file. A failed save raises and retains collected records.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any, Callable

_PROTOCOL_QUANTILES = (0.5, 0.8, 0.2)
_RETURN_MODES = frozenset({"min", "max", "mean", "median", "all"})
_OPERATOR_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*\Z")
_RESULTS_LOCK = threading.RLock()
PYTEST_BENCHMARK_RESULTS: dict[str, list[dict[str, Any]]] = {}


def _finite_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        condition = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {condition}.")
    return number


def _validate_operator_name(op_name: str) -> None:
    if not isinstance(op_name, str) or _OPERATOR_NAME.fullmatch(op_name) is None:
        raise ValueError("The operator name must be a filename without path separators.")


@dataclass
class do_bench_config:
    """Set millisecond budgets and three requested percentiles.

    The default percentiles are p50, p80, and p20, in that order. Percentiles
    take precedence over ``return_mode`` in Triton. Explicit ``None`` selects
    the default percentiles.
    """

    warm_up: float = 25
    repetition: float = 100
    quantiles: list[float] | None = field(default_factory=lambda: list(_PROTOCOL_QUANTILES))
    return_mode: str = "median"

    def __post_init__(self) -> None:
        self.warm_up = _finite_number(self.warm_up, "warm_up", allow_zero=True)
        self.repetition = _finite_number(self.repetition, "repetition")
        try:
            quantiles = list(_PROTOCOL_QUANTILES if self.quantiles is None else self.quantiles)
        except TypeError as exc:
            raise ValueError("quantiles must contain three finite numbers in [0, 1].") from exc
        if (
            len(quantiles) != 3
            or any(isinstance(value, bool) or not isinstance(value, Real) for value in quantiles)
            or any(not math.isfinite(value) or not 0 <= value <= 1 for value in quantiles)
        ):
            raise ValueError("quantiles must contain three finite numbers in [0, 1].")
        self.quantiles = [float(value) for value in quantiles]
        if not isinstance(self.return_mode, str) or self.return_mode not in _RETURN_MODES:
            raise ValueError("return_mode must be min, max, mean, median, or all.")

    def as_dict(self) -> dict[str, Any]:
        """Return a separate configuration record with explicit timing units."""
        return {
            "warm_up": self.warm_up,
            "repetition": self.repetition,
            "quantiles": list(self.quantiles),
            "return_mode": self.return_mode,
            "budget_unit": "ms",
            "timer": "triton.testing.do_bench",
        }


def add_benchmark_result(op_name: str, result_dict: dict[str, Any]) -> None:
    """Store a separate result copy under a valid operator name."""
    _validate_operator_name(op_name)
    if not isinstance(result_dict, dict):
        raise TypeError("The benchmark result must be a dictionary.")
    with _RESULTS_LOCK:
        PYTEST_BENCHMARK_RESULTS.setdefault(op_name, []).append(copy.deepcopy(result_dict))


class BenchmarkJSONEncoder(json.JSONEncoder):
    """Encode PyTorch dtype and device values without importing PyTorch."""

    def default(self, value: Any) -> Any:
        torch = sys.modules.get("torch")
        if torch is not None:
            for type_name in ("dtype", "device"):
                value_type = getattr(torch, type_name, None)
                if isinstance(value_type, type) and isinstance(value, value_type):
                    return str(value)
        return super().default(value)


# External tests can import the encoder under its historical name.
TorchEncoder = BenchmarkJSONEncoder


def _output_directory(output_dir: str | os.PathLike[str]) -> Path:
    raw_path = os.fspath(output_dir)
    if not isinstance(raw_path, str) or not raw_path.strip() or "\x00" in raw_path:
        raise ValueError("The output directory must contain a valid path.")
    directory = Path(raw_path)
    if ".." in directory.parts:
        raise ValueError("The output directory must not contain parent traversal.")
    if directory.is_symlink():
        raise ValueError("The output directory must not be a symbolic link.")
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise ValueError("The output path must be a directory.")
    return directory


def save_all_benchmark_results(output_dir: str | os.PathLike[str]) -> None:
    """Save operator JSON files and clear the collector only after success.

    Serialize every record before writing files. Replace each file atomically.
    A write failure can leave earlier files updated. The collector retains all
    records after a failure, so the caller can retry the same save.
    """
    with _RESULTS_LOCK:
        directory = _output_directory(output_dir)
        serialized: dict[str, str] = {}
        for op_name, records in PYTEST_BENCHMARK_RESULTS.items():
            _validate_operator_name(op_name)
            serialized[op_name] = (
                json.dumps(records, cls=BenchmarkJSONEncoder, indent=2, allow_nan=False) + "\n"
            )
        for op_name, payload in serialized.items():
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=directory,
                    prefix=f".{op_name}.",
                    suffix=".tmp",
                    delete=False,
                ) as stream:
                    temporary_path = Path(stream.name)
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_path, directory / f"{op_name}.json")
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
        PYTEST_BENCHMARK_RESULTS.clear()


def _throughput(
    calculator: Callable[[dict[str, Any], float], Any] | None,
    primary_ms: float,
    params: dict[str, Any],
) -> float | str:
    if calculator is None:
        return "N/A"
    try:
        value = _finite_number(calculator(params, primary_ms), "throughput", allow_zero=True)
        return round(value, 2)
    except Exception:
        return "N/A"


class PytestBenchmarker:
    """Measure one callable and collect records for its operator."""

    def __init__(
        self,
        op_callable: Callable[[], Any],
        op_name: str,
        config: do_bench_config | None = None,
    ) -> None:
        if not callable(op_callable):
            raise TypeError("The operation must be callable.")
        _validate_operator_name(op_name)
        if config is not None and not isinstance(config, do_bench_config):
            raise TypeError("config must be a do_bench_config instance.")
        self.op_callable = op_callable
        self.op_name = op_name
        self.config = copy.deepcopy(config) if config is not None else do_bench_config()

    def run_benchmark(
        self,
        current_params_dict: dict[str, Any],
        gbps_calculator: Callable[[dict[str, Any], float], Any] | None = None,
        tflops_calculator: Callable[[dict[str, Any], float], Any] | None = None,
    ) -> dict[str, Any]:
        """Collect a latency record, or an explicit error record if timing fails.

        Calculators receive the caller's parameters, then the unrounded first latency.
        The first latency is the median with default quantiles.
        Calculator errors produce ``N/A`` fields. Timing errors omit latency
        fields, so an evaluator can reject the failed case.
        """
        if not isinstance(current_params_dict, dict):
            raise TypeError("The benchmark parameters must be a dictionary.")
        params = copy.deepcopy(current_params_dict)
        try:
            # Revalidate the public configuration if the caller changes it.
            config = do_bench_config(
                self.config.warm_up,
                self.config.repetition,
                self.config.quantiles,
                self.config.return_mode,
            )
            from triton.testing import do_bench

            samples = list(
                do_bench(
                    self.op_callable,
                    warmup=config.warm_up,
                    rep=config.repetition,
                    quantiles=list(config.quantiles),
                    return_mode=config.return_mode,
                )
            )
            if len(samples) != len(config.quantiles):
                raise ValueError("The timer must return one latency for each requested quantile.")
            samples = [_finite_number(value, "latency") for value in samples]
            ranked_samples = sorted(zip(config.quantiles, samples))
            if any(left[1] > right[1] for left, right in zip(ranked_samples, ranked_samples[1:])):
                raise ValueError(
                    "The timer must return nondecreasing latency as the quantile increases."
                )
            if any(round(value, 4) <= 0 for value in samples):
                raise ValueError("The archived latency fields must remain positive after rounding.")
            result = {
                "params": params,
                "ms": round(samples[0], 4),
                "min_ms": round(samples[1], 4),
                "max_ms": round(samples[2], 4),
                "GB/s": _throughput(gbps_calculator, samples[0], current_params_dict),
                "TFLOPS": _throughput(tflops_calculator, samples[0], current_params_dict),
                "timing_config": config.as_dict(),
            }
            latencies = dict(zip(config.quantiles, samples))
            for field_name, quantile in (("median_ms", 0.5), ("p80_ms", 0.8), ("p20_ms", 0.2)):
                if quantile in latencies:
                    result[field_name] = latencies[quantile]
        except Exception as exc:
            result = {"params": params, "error": f"{type(exc).__name__}: {exc}"}
        add_benchmark_result(self.op_name, result)
        return result
