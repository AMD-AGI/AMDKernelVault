# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transport-independent results from the ROCm execution service."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    """Keep compilation, correctness, and timing outcomes separate."""

    compiled: bool
    correct: bool
    speedup: float | None = None
    latency_ms: float | None = None
    baseline_latency_ms: float | None = None
    feedback: str = ""
    error_type: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.compiled) is not bool or type(self.correct) is not bool:
            raise ValueError("Compilation and correctness results must be booleans.")
        if self.correct and not self.compiled:
            raise ValueError("A correct candidate must compile successfully.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvaluationResult":
        return cls(**value)
