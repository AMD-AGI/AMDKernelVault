# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare saved outputs and aggregate complete per-case timing records.

This module uses only the standard library until an output comparison needs
PyTorch. Timing inputs contain one JSON list per operator. Each row supplies a
parameter dictionary and a positive, finite latency in milliseconds under ``ms``.
The module never interprets an aggregate ``ms`` value as a speedup.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from statistics import mean
from typing import Any


class ReferenceError(RuntimeError):
    """Report an invalid reference or an unavailable comparison dependency."""


class CandidateOutputError(ValueError):
    """Report invalid candidate outputs and the available call-success evidence."""

    def __init__(self, message: str, *, compiled: bool = False, stage: str = "output_flag"):
        super().__init__(message)
        self.compiled = compiled
        self.stage = stage


class CandidateTimingError(ValueError):
    """Report invalid or incomplete candidate timing records."""


@dataclass(frozen=True)
class TimingSummary:
    """Store the mean operator speedup and separate sums of case latencies."""

    speedup: float
    latency_ms: float
    baseline_latency_ms: float
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _TimingCase:
    params: dict[str, Any]
    latency_ms: float
    metadata: dict[str, Any]


def _output_error(message: str, *, reference: bool, compiled: bool = False, stage: str):
    if reference:
        return ReferenceError(message)
    return CandidateOutputError(message, compiled=compiled, stage=stage)


def _call_success(outputs: Any, *, reference: bool, torch: Any) -> bool:
    label = "Reference" if reference else "Candidate"

    def error(message):
        return _output_error(message, reference=reference, stage="output_flag")

    if not isinstance(outputs, dict):
        raise error(f"{label} outputs must be a dictionary.")
    flag = outputs.get("_CALL_SUCCESS_")
    if not isinstance(flag, torch.Tensor) or flag.numel() != 1:
        raise error(f"{label} _CALL_SUCCESS_ must be a one-value torch tensor.")
    try:
        value = flag.item()
    except Exception as exc:
        raise error(f"{label} _CALL_SUCCESS_ cannot supply a scalar value.") from exc
    if not isinstance(value, Real) or value not in (0, 1):
        raise error(f"{label} _CALL_SUCCESS_ must contain Boolean true, Boolean false, 0, or 1.")
    if not value:
        raise error(f"{label} _CALL_SUCCESS_ reports a failed call.")
    return True


def _count_leaves(
    value: Any, *, reference: bool, torch: Any, path: str, ancestors: set[int]
) -> int:
    """Reject unsupported values and cycles before comparing either tree."""
    label = "Reference" if reference else "Candidate"

    def error(message):
        return _output_error(message, reference=reference, compiled=True, stage="output_tree")

    if isinstance(value, torch.Tensor):
        return int(value.numel() > 0)
    if type(value) in (bool, int, float, complex, str, type(None)):
        return 1
    if type(value) not in (dict, list, tuple):
        raise error(
            f"{label} output {path} contains an unsupported value type: {type(value).__name__}."
        )
    if id(value) in ancestors:
        raise error(f"{label} output {path} contains a cycle.")
    ancestors.add(id(value))
    try:
        children = value.items() if isinstance(value, dict) else enumerate(value)
        return sum(
            _count_leaves(
                child,
                reference=reference,
                torch=torch,
                path=f"{path}[{key!r}]",
                ancestors=ancestors,
            )
            for key, child in children
        )
    finally:
        ancestors.remove(id(value))


def _compare_tree(
    reference: Any, candidate: Any, *, torch: Any, atol: float, rtol: float, path: str
):
    if isinstance(reference, torch.Tensor):
        if not isinstance(candidate, torch.Tensor):
            return False, f"Candidate output {path} must be a torch tensor."
    elif type(candidate) is not type(reference):
        return False, f"Candidate output {path} has a different type from the reference."
    if isinstance(reference, dict):
        if reference.keys() != candidate.keys():
            return (
                False,
                f"Candidate output {path} has different dictionary keys from the reference.",
            )
        children = ((key, reference[key], candidate[key]) for key in reference)
    elif isinstance(reference, (list, tuple)):
        if len(reference) != len(candidate):
            return False, f"Candidate output {path} has a different length from the reference."
        children = (
            (index, expected, actual)
            for index, (expected, actual) in enumerate(zip(reference, candidate))
        )
    else:
        if not isinstance(reference, torch.Tensor):
            if candidate != reference:
                return False, f"Candidate output {path} differs from the reference."
        else:
            try:
                torch.testing.assert_close(
                    actual=candidate, expected=reference, atol=atol, rtol=rtol, equal_nan=False
                )
            except AssertionError as exc:
                return False, f"Candidate output {path} differs from the reference.\n{exc}"
        return True, ""
    for key, expected, actual in children:
        matches, feedback = _compare_tree(
            expected, actual, torch=torch, atol=atol, rtol=rtol, path=f"{path}[{key!r}]"
        )
        if not matches:
            return matches, feedback
    return True, ""


def compare_outputs(
    reference: dict, candidate: dict, *, atol: float, rtol: float
) -> tuple[bool, str]:
    """Compare two saved output dictionaries with the caller's tolerances.

    Both dictionaries require a one-value tensor named ``_CALL_SUCCESS_``. Boolean
    values and real numeric values exactly equal to 0 or 1 are valid flags.
    Reference failures raise ``ReferenceError``. Invalid candidate outputs raise
    ``CandidateOutputError``. A false or invalid candidate flag sets ``compiled``
    to false. A true flag establishes call success, not independent compiler proof.

    Each output tree must contain at least one leaf after removing its flag.
    A scalar or a nonempty tensor counts as one leaf. Empty tensors supply no
    leaves. This structural rule does not establish benchmark coverage. Container
    types, scalar types, keys, and lengths must match. Tensors use
    ``torch.testing.assert_close`` with the reference as ``expected``. All
    non-tensor scalars use exact equality. The function does not mutate inputs.
    """
    for name, value in (("atol", atol), ("rtol", rtol)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{name} must be a finite, nonnegative number.")
    try:
        import torch
    except ImportError as exc:
        raise ReferenceError("Output comparison requires PyTorch.") from exc

    _call_success(reference, reference=True, torch=torch)
    expected = {key: value for key, value in reference.items() if key != "_CALL_SUCCESS_"}
    leaf_count = _count_leaves(
        expected, reference=True, torch=torch, path="outputs", ancestors=set()
    )
    if not leaf_count:
        raise ReferenceError(
            "Reference outputs must contain at least one leaf after removing _CALL_SUCCESS_."
        )
    _call_success(candidate, reference=False, torch=torch)
    actual = {key: value for key, value in candidate.items() if key != "_CALL_SUCCESS_"}
    if not _count_leaves(actual, reference=False, torch=torch, path="outputs", ancestors=set()):
        raise CandidateOutputError(
            "Candidate outputs must contain at least one leaf after removing _CALL_SUCCESS_.",
            compiled=True,
            stage="output_tree",
        )
    matches, feedback = _compare_tree(
        expected, actual, torch=torch, atol=atol, rtol=rtol, path="outputs"
    )
    return matches, f"Compared {leaf_count} output leaves." if matches else feedback


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains a duplicate object key: {key!r}.")
        result[key] = value
    return result


def _parameter_key(value: Any) -> tuple:
    """Preserve JSON dictionary equality, including numeric equality, in keys."""
    if isinstance(value, dict):
        return ("dict", tuple((key, _parameter_key(child)) for key, child in sorted(value.items())))
    if isinstance(value, list):
        return ("list", tuple(_parameter_key(child) for child in value))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Timing parameters must contain only finite numbers.")
    return ("scalar", value)


def _load_timings(folder: Path, *, reference: bool) -> dict[str, dict[tuple, _TimingCase]]:
    error = ReferenceError if reference else CandidateTimingError
    label = "Reference" if reference else "Candidate"
    folder = Path(folder)
    try:
        files = sorted(path for path in folder.iterdir() if path.suffix == ".json")
    except OSError as exc:
        raise error(f"{label} timing folder cannot be read: {folder}.") from exc
    if not files:
        raise error(f"{label} timing folder contains no operator JSON files: {folder}.")
    operators: dict[str, dict[tuple, _TimingCase]] = {}
    for path in files:
        if path.name == "all_perf_results.json":
            raise error(
                f"{label} timings must contain per-case records, not all_perf_results.json."
            )
        try:
            with path.open(encoding="utf-8") as stream:
                records = json.load(stream, object_pairs_hook=_unique_object)
        except (OSError, UnicodeError, ValueError) as exc:
            raise error(f"{label} timing file cannot supply valid JSON: {path.name}.") from exc
        if not isinstance(records, list) or not records:
            raise error(
                f"{label} timing file must contain a nonempty list of case records: {path.name}."
            )
        cases: dict[tuple, _TimingCase] = {}
        for index, record in enumerate(records):
            context = f"{label} timing {path.name}, case {index}"
            if not isinstance(record, dict):
                raise error(f"{context} must be a dictionary.")
            if "error" in record:
                raise error(f"{context} contains an error record.")
            if not isinstance(record.get("params"), dict):
                raise error(f"{context} must contain a params dictionary.")
            try:
                key = _parameter_key(record["params"])
            except ValueError as exc:
                raise error(f"{context} contains invalid parameters.") from exc
            if key in cases:
                raise error(f"{context} duplicates an earlier params dictionary.")
            value = record.get("ms")
            if type(value) not in (int, float):
                raise error(f"{context} must contain a numeric ms latency.")
            try:
                latency = float(value)
            except (OverflowError, ValueError) as exc:
                raise error(f"{context} must contain a finite ms latency.") from exc
            if not math.isfinite(latency) or latency <= 0:
                raise error(f"{context} must contain a positive, finite ms latency.")
            metadata = {
                name: record[name]
                for name in ("timing_config", "median_ms", "p80_ms", "p20_ms")
                if name in record
            }
            try:
                json.dumps(metadata, allow_nan=False)
            except ValueError as exc:
                raise error(f"{context} contains invalid timing metadata.") from exc
            cases[key] = _TimingCase(params=record["params"], latency_ms=latency, metadata=metadata)
        operators[path.name] = cases
    return operators


def _latency_sum(values: Any, *, reference: bool) -> float:
    error = ReferenceError if reference else CandidateTimingError
    label = "Reference" if reference else "Candidate"
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise error(f"{label} summed latency exceeds the finite numeric range.") from exc
    if not math.isfinite(total) or total <= 0:
        raise error(f"{label} summed latency must be positive and finite.")
    return total


def _speedup(baseline: float, candidate: float) -> float:
    speedup = baseline / candidate
    if not math.isfinite(speedup) or speedup <= 0:
        raise CandidateTimingError("The speedup ratio exceeds the positive, finite numeric range.")
    return speedup


def aggregate_timings(reference_folder: Path, candidate_folder: Path) -> TimingSummary:
    """Require complete case coverage and return the mean operator speedup.

    Each operator file must contain a nonempty JSON list with ``params`` and
    ``ms`` in every record. Parameter dictionaries identify cases by exact JSON
    value equality, independent of dictionary key order. Duplicate parameters,
    error records, invalid latencies, missing cases, and extra cases fail the
    complete aggregate. Both folders must contain the same operator filenames.
    Reference errors raise ``ReferenceError``. Candidate errors raise
    ``CandidateTimingError``. Small speedups remain valid and receive no cutoff.

    Each operator speedup divides its summed reference latencies by its summed
    candidate latencies. ``speedup`` is the arithmetic mean of these unrounded
    operator speedups. Each operator receives equal weight. Diagnostics include
    four-decimal values for source-style display. Rounding does not affect the
    returned speedup or remove small ratios.

    ``latency_ms`` and ``baseline_latency_ms`` separately sum all case latencies.
    They do not measure complete program latency. Their ratio generally differs
    from the mean operator speedup. Diagnostics retain every matched case.
    """
    reference = _load_timings(reference_folder, reference=True)
    baseline_total = _latency_sum(
        (case.latency_ms for cases in reference.values() for case in cases.values()), reference=True
    )
    candidate = _load_timings(candidate_folder, reference=False)
    missing = sorted(reference.keys() - candidate.keys())
    extra = sorted(candidate.keys() - reference.keys())
    if missing or extra:
        raise CandidateTimingError(
            f"Candidate operator coverage differs. Missing files: {missing}. Extra files: {extra}."
        )
    operators: dict[str, Any] = {}
    case_count = 0
    for name, expected in reference.items():
        actual = candidate[name]
        missing_count = len(expected.keys() - actual.keys())
        extra_count = len(actual.keys() - expected.keys())
        if missing_count or extra_count:
            raise CandidateTimingError(
                f"Candidate parameters differ in {name}. Missing cases: {missing_count}. Extra cases: {extra_count}."
            )
        if any(
            case.metadata.get("timing_config") != actual[key].metadata.get("timing_config")
            for key, case in expected.items()
        ):
            raise CandidateTimingError(
                f"Candidate timing settings differ from the reference in {name}."
            )
        baseline = _latency_sum((case.latency_ms for case in expected.values()), reference=True)
        latency = _latency_sum((case.latency_ms for case in actual.values()), reference=False)
        speedup = _speedup(baseline, latency)
        operators[name] = {
            "case_count": len(expected),
            "speedup": speedup,
            "rounded_speedup": round(speedup, 4),
            "latency_ms": latency,
            "baseline_latency_ms": baseline,
            "cases": [
                {
                    "params": case.params,
                    "reference_ms": case.latency_ms,
                    "candidate_ms": actual[key].latency_ms,
                    **({"reference_timing": case.metadata} if case.metadata else {}),
                    **({"candidate_timing": actual[key].metadata} if actual[key].metadata else {}),
                }
                for key, case in expected.items()
            ],
        }
        case_count += len(expected)
    candidate_total = _latency_sum(
        (case.latency_ms for cases in candidate.values() for case in cases.values()),
        reference=False,
    )
    return TimingSummary(
        speedup=mean(operator["speedup"] for operator in operators.values()),
        latency_ms=candidate_total,
        baseline_latency_ms=baseline_total,
        diagnostics={
            "aggregation": "arithmetic_mean(operator_speedups)",
            "operator_aggregation": "sum(reference_ms) / sum(candidate_ms)",
            "display_rounding_digits": 4,
            "latency_aggregation": "sum_of_case_latencies",
            "operator_count": len(operators),
            "case_count": case_count,
            "operators": operators,
        },
    )
