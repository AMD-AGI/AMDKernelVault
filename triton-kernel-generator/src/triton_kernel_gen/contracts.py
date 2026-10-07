# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Public interfaces for external tasks and their verified test cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Case:
    """One externally supplied input configuration, not one random repetition."""

    case_id: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    init_args: tuple[Any, ...] = ()
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    atol: float | None = None
    rtol: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    module_path: Path
    functional_path: Path
    case_provider_path: Path
    seed_kernel_path: Path | None = None
    provenance: dict[str, Any] = field(default_factory=dict)
    dependency_paths: tuple[Path, ...] = ()
    baseline_kernel_path: Path | None = None


@dataclass(frozen=True)
class VerificationSettings:
    gpu: int = 0
    target_arch: str | None = None
    seed: int = 1234
    atol: float = 1e-3
    rtol: float = 1e-4
    warmup_ms: float = 25.0
    measure_ms: float = 100.0
    timeout_seconds: float = 900.0
    excluded_environment: tuple[str, ...] = ()


@dataclass
class PreparedTask:
    task: TaskSpec
    work_dir: Path
    reference_record: dict[str, Any]
    state: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    compile_success: bool
    correctness_success: bool
    timing_success: bool
    message: str
    correctness_cases: list[dict[str, Any]] = field(default_factory=list)
    performance_cases: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    runtime_success: bool = False
    failure_stage: str | None = None

    @property
    def success(self) -> bool:
        return (
            self.compile_success
            and self.runtime_success
            and self.correctness_success
            and self.timing_success
        )


@dataclass(frozen=True)
class ModelResponse:
    text: str
    model: str
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    reasoning: str | None = None


@dataclass(frozen=True)
class GenerationSettings:
    output_dir: Path
    artifacts_dir: Path
    max_attempts: int
    num_variants: int = 1
    temperature: float = 0.7
    max_tokens: int = 16384
    history_char_limit: int = 16000
