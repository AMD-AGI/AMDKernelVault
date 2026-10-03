# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .prompt_assets import resolve_prompt_assets


@dataclass(slots=True)
class PipelineConfig:
    module_dir: Path
    functional_dir: Path
    output_dir: Path
    artifacts_dir: Path
    max_attempts: int = 5
    rtol: float = 1e-4
    atol: float = 1e-4
    seed: int = 1234
    overwrite: bool = False
    temperature: float = 0.0
    max_tokens: int = 12000
    perf_warmup: int = 25
    perf_iterations: int = 200
    keep_build_dirs: bool = False
    history_code_char_limit: int = 8000
    history_feedback_char_limit: int = 4000
    instruction_file: Path | None = None
    few_shot_file: Path | None = None
    system_instruction: str | None = None
    few_shot_examples: str | None = None
    build_root: Path | None = None
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
        if self.build_root is None:
            self.build_root = self.artifacts_dir / "build"
        if self.system_instruction is None or self.few_shot_examples is None:
            instruction, few_shot = resolve_prompt_assets(self.instruction_file, self.few_shot_file)
            if self.system_instruction is None:
                self.system_instruction = instruction
            if self.few_shot_examples is None:
                self.few_shot_examples = few_shot
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
    compile_success: bool = False
    correctness_success: bool = False
    speedup: float | None = None
    module_latency_ms: float | None = None
    hip_latency_ms: float | None = None


@dataclass(slots=True)
class ConversionRecord:
    module_source_path: str
    functional_source_path: str
    relative_path: str
    output_path: str
    status: str
    attempts_used: int
    attempts: list[AttemptRecord] = field(default_factory=list)
    best_attempt: int | None = None
    best_speedup: float | None = None
    final_error: str | None = None
    input_fingerprint: str | None = None
    output_sha256: str | None = None
