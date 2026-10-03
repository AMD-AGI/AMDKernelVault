# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .prompt_assets import resolve_prompt_assets


@dataclass(slots=True)
class PipelineConfig:
    baseline_hip_dir: Path
    module_dir: Path
    functional_dir: Path
    output_dir: Path
    artifacts_dir: Path
    resume: bool = False
    num_workers: int = 1
    target_function_mode: str = "auto"
    max_attempts: int = 5
    rtol: float = 1e-4
    atol: float = 1e-4
    seed: int = 1234
    overwrite: bool = False
    temperature: float = 0.0
    max_tokens: int = 12000
    provider: str | None = None
    model_id: str | None = None
    python_load_timeout_seconds: float | None = 60.0
    hip_compile_timeout_seconds: float | None = 900.0
    execution_timeout_seconds: float | None = 300.0
    benchmark_timeout_seconds: float | None = 900.0
    perf_warmup: int = 25
    perf_iterations: int = 200
    keep_build_dirs: bool = False
    history_code_char_limit: int = 8000
    history_feedback_char_limit: int = 4000
    instruction_file: Path | None = None
    few_shot_file: Path | None = None
    system_instruction: str | None = None
    few_shot_examples: str | None = None
    offload_arch: str | None = None
    build_root: Path | None = None
    records_file: Path | None = None
    success_file: Path | None = None
    failure_file: Path | None = None
    success_cases_dir: Path | None = None

    def with_defaults(self) -> "PipelineConfig":
        if self.records_file is None:
            self.records_file = self.artifacts_dir / "optimization_records.json"
        if self.success_file is None:
            self.success_file = self.artifacts_dir / "successful_optimizations.json"
        if self.failure_file is None:
            self.failure_file = self.artifacts_dir / "failed_optimizations.json"
        if self.success_cases_dir is None:
            self.success_cases_dir = self.artifacts_dir / "successful_optimizations"
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
    function_candidate_path: str
    candidate_path: str
    status: str
    optimized_gpu_function: str | None = None
    feedback: str | None = None
    error: str | None = None
    mismatch: str | None = None
    compile_success: bool = False
    correctness_success: bool = False
    speedup_vs_baseline: float | None = None
    speedup_vs_module: float | None = None
    module_latency_ms: float | None = None
    baseline_latency_ms: float | None = None
    candidate_latency_ms: float | None = None


@dataclass(slots=True)
class ConversionRecord:
    baseline_hip_source_path: str
    module_source_path: str
    functional_source_path: str
    relative_path: str
    output_path: str
    status: str
    attempts_used: int
    target_function_mode: str = "auto"
    target_function_name: str | None = None
    target_function_kind: str | None = None
    target_function_length: int | None = None
    baseline_gpu_function: str | None = None
    optimized_gpu_function: str | None = None
    attempts: list[AttemptRecord] = field(default_factory=list)
    best_attempt: int | None = None
    best_speedup_vs_baseline: float | None = None
    best_speedup_vs_module: float | None = None
    baseline_compile_success: bool = False
    baseline_correctness_success: bool = False
    module_latency_ms: float | None = None
    baseline_latency_ms: float | None = None
    final_error: str | None = None
    skip_reason: str | None = None
    input_fingerprint: str | None = None
    output_sha256: str | None = None
