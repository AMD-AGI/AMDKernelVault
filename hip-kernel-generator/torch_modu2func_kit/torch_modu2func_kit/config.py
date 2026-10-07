# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    artifacts_dir: Path
    max_attempts: int = 5
    rtol: float = 1e-4
    atol: float = 1e-4
    seed: int = 1234
    overwrite: bool = False
    temperature: float = 0.0
    max_tokens: int = 5000
    history_code_char_limit: int = 6000
    history_feedback_char_limit: int = 4000
    records_file: Path | None = None
    success_file: Path | None = None
    failure_file: Path | None = None

    def with_defaults(self) -> "PipelineConfig":
        if self.records_file is None:
            self.records_file = self.artifacts_dir / "conversion_records.json"
        if self.success_file is None:
            self.success_file = self.artifacts_dir / "successful_conversions.json"
        if self.failure_file is None:
            self.failure_file = self.artifacts_dir / "failed_conversions.json"
        return self


@dataclass(slots=True)
class AttemptRecord:
    attempt: int
    prompt_path: str
    candidate_path: str
    status: str
    feedback: str | None = None
    error: str | None = None
    mismatch: str | None = None
    source_sha256: str | None = None
    candidate_sha256: str | None = None
    verification_seed: int | None = None
    verification_rtol: float | None = None
    verification_atol: float | None = None


@dataclass(slots=True)
class ConversionRecord:
    source_path: str
    relative_path: str
    output_path: str
    status: str
    attempts_used: int
    attempts: list[AttemptRecord] = field(default_factory=list)
    final_error: str | None = None
    source_sha256: str | None = None
    output_sha256: str | None = None
    verification_seed: int | None = None
    verification_rtol: float | None = None
    verification_atol: float | None = None
    skip_reason: str | None = None
