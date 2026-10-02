# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Calculate the paper's turn rewards and unnormalized trajectory return."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

from .config import load_config
from .contracts import EvaluationResult


def turn_reward(result: EvaluationResult, algorithm: Mapping[str, Any] | None = None) -> float:
    """Give the speed bonus only to correct candidates with finite speedup >= 1."""
    settings = load_config()["algorithm"] if algorithm is None else algorithm
    if not result.compiled:
        return 0.0
    reward = float(settings["compile_reward"])
    if not result.correct:
        return reward
    reward += float(settings["correct_reward"])
    speedup = result.speedup
    if isinstance(speedup, bool) or not isinstance(speedup, Real):
        return reward
    try:
        speedup = float(speedup)
    except (OverflowError, ValueError):
        return reward
    if not math.isfinite(speedup) or speedup < 1:
        return reward
    bonus = float(settings["performance_scale"]) * math.log2(speedup) ** 2
    return reward + min(bonus, float(settings["performance_cap"]))


def discounted_return(rewards: Iterable[float], discount: float = 0.6) -> float:
    """Calculate sum(discount**turn * reward), starting with turn zero."""
    if isinstance(discount, bool) or not isinstance(discount, Real):
        raise ValueError("The discount must be a finite number between zero and one.")
    if not math.isfinite(discount) or not 0 <= discount <= 1:
        raise ValueError("The discount must be a finite number between zero and one.")
    values = list(rewards)
    if any(
        isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
        for value in values
    ):
        raise ValueError("Each turn reward must be a finite number.")
    return math.fsum(discount**turn * reward for turn, reward in enumerate(values))


async def reward_func(args: Any, sample: Any, **kwargs: Any) -> float:
    """Return the stored trajectory return when Slime calls its reward hook."""
    try:
        value = sample.metadata["triton_rl"]["trajectory_return"]
    except (AttributeError, KeyError, TypeError) as error:
        raise ValueError("The sample contains no Triton trajectory return.") from error
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError("The stored trajectory return must be a finite number.")
    return float(value)
