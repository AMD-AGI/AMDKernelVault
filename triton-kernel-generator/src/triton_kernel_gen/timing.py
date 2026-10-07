# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Measure GPU execution time after the first call completes compilation."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from .contracts import VerificationSettings


def _duration_ms(value: Any, name: str, *, allow_zero: bool = False) -> float:
    requirement = "finite and nonnegative" if allow_zero else "finite and positive"
    message = f"{name} must be {requirement}."
    if isinstance(value, (bool, str, bytes, bytearray)):
        raise ValueError(message)
    try:
        duration = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(message) from exc
    if not math.isfinite(duration) or duration < 0 or (duration == 0 and not allow_zero):
        raise ValueError(message)
    return duration


def bench_ms(fn: Callable[[], Any], settings: VerificationSettings) -> float:
    """Return the median execution time in milliseconds on the selected GPU.

    The first call and synchronization exclude compilation from the benchmark.
    Import the GPU packages only when this function starts a valid benchmark.
    """
    warmup_ms = _duration_ms(settings.warmup_ms, "warmup_ms", allow_zero=True)
    measure_ms = _duration_ms(settings.measure_ms, "measure_ms")

    import torch
    import triton.testing

    with torch.cuda.device(settings.gpu):
        fn()
        torch.cuda.synchronize()
        median = triton.testing.do_bench(fn, warmup=warmup_ms, rep=measure_ms, quantiles=[0.5])

    if isinstance(median, Sequence) and not isinstance(median, (str, bytes, bytearray)):
        if len(median) != 1:
            raise ValueError("The benchmark must return one median time.")
        median = median[0]
    return _duration_ms(median, "The median time")
