# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Format execution feedback for the next turn of the shared policy."""

from __future__ import annotations

import json
import math
from numbers import Real

from .contracts import EvaluationResult


def _finite_number(value: float | None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_feedback(result: EvaluationResult, turn: int) -> str:
    """Keep execution facts separate from instructions for reflection and revision."""
    facts = {
        "turn": turn,
        "compiled": result.compiled,
        "correct": result.correct,
        "speedup": _finite_number(result.speedup),
        "latency_ms": _finite_number(result.latency_ms),
        "baseline_latency_ms": _finite_number(result.baseline_latency_ms),
        "feedback": result.feedback,
        "error_type": result.error_type,
    }
    action = (
        "Improve the kernel while preserving correctness."
        if result.correct
        else "Correct the kernel."
    )
    return (
        "\n\n<execution_result>\n"
        + json.dumps(facts, ensure_ascii=False, allow_nan=False)
        + "\n</execution_result>\n"
        + "Reflect on the execution result.\n"
        + action
        + "\nReturn your reflection and the complete revised kernel.\n"
        + "Put the kernel in one Python fence or one <answer> block.\n"
    )
