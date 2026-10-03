# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .model_factory import create_model_client
from .pipeline import run_optimization_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize baseline HIP kernels with iterative LLM-guided kernel-to-kernel rewriting."
    )
    parser.add_argument("--baseline-hip-dir", type=Path, required=True, help="Input root for baseline HIP files.")
    parser.add_argument("--module-dir", type=Path, required=True, help="Input root for paired PyTorch module files.")
    parser.add_argument("--functional-dir", type=Path, required=True, help="Input root for paired PyTorch functional files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for selected optimized HIP files.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=(
            "Number of worker threads used for dataset production. "
            "Default is 1. Each worker optimizes one baseline HIP sample at a time."
        ),
    )
    parser.add_argument(
        "--target-function-mode",
        default="auto",
        choices=["auto", "global", "device"],
        help=(
            "Target function selection policy: `auto` prefers the longest `__global__` "
            "function and falls back to `__device__`; `device` restricts optimization "
            "to `__device__`/`__host__ __device__` functions."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".artifacts"),
        help="Directory used to store prompts, function candidates, patched HIP attempts, build files, and JSON records.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing JSON records in the artifacts directory.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "standard-openai", "claude", "standard-claude", "gemini"],
        help="LLM provider adapter.",
    )
    parser.add_argument("--model-id", default="gpt-4o", help="Provider-specific model identifier. Default is gpt-4o.")
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "LLM API key. The package checks PY_HIP_KERNEL2KERNEL_API_KEY, HIP2HIP_API_KEY, "
            "TORCH2HIP_API_KEY, and TORCH_MODU2FUNC_API_KEY first. "
            "It then checks OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY/GOOGLE_API_KEY for the selected provider."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum optimized kernel candidates per sample.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for output comparison.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for output comparison.")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic seed for verification.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--max-tokens", type=int, default=12000, help="Maximum tokens per generation call.")
    parser.add_argument(
        "--python-load-timeout-seconds",
        type=float,
        default=60.0,
        help="Timeout for importing Python samples and constructing their callable objects. Use 0 to disable.",
    )
    parser.add_argument(
        "--hip-compile-timeout-seconds",
        type=float,
        default=900.0,
        help="Timeout for compiling a HIP extension with torch cpp_extension. Use 0 to disable.",
    )
    parser.add_argument(
        "--execution-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for a single correctness execution of PyTorch or HIP paths. Use 0 to disable.",
    )
    parser.add_argument(
        "--benchmark-timeout-seconds",
        type=float,
        default=900.0,
        help="Timeout for a benchmark phase, including warmup and timed iterations. Use 0 to disable.",
    )
    parser.add_argument("--perf-warmup", type=int, default=25, help="Warmup iterations before timing.")
    parser.add_argument("--perf-iterations", type=int, default=200, help="Measured iterations for latency timing.")
    parser.add_argument("--instruction-file", type=Path, default=None, help="Optional Python file that defines `hip_kernel_opt_req`.")
    parser.add_argument("--few-shot-file", type=Path, default=None, help="Optional Python file that defines `few_shot_code_instructions`.")
    parser.add_argument("--offload-arch", type=str, default=None, help="Optional ROCm offload arch, for example `gfx90a`.")
    parser.add_argument("--keep-build-dirs", action="store_true", help="Preserve per-attempt build directories for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing optimized HIP outputs.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1.")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1.")
    if args.perf_iterations < 1:
        parser.error("--perf-iterations must be at least 1.")
    if args.perf_warmup < 0:
        parser.error("--perf-warmup must be non-negative.")
    if args.python_load_timeout_seconds < 0:
        parser.error("--python-load-timeout-seconds must be non-negative.")
    if args.hip_compile_timeout_seconds < 0:
        parser.error("--hip-compile-timeout-seconds must be non-negative.")
    if args.execution_timeout_seconds < 0:
        parser.error("--execution-timeout-seconds must be non-negative.")
    if args.benchmark_timeout_seconds < 0:
        parser.error("--benchmark-timeout-seconds must be non-negative.")

    if args.num_workers == 1:
        client = create_model_client(args.provider, args.model_id, args.api_key)
    else:
        def client():
            return create_model_client(args.provider, args.model_id, args.api_key)

    config = PipelineConfig(
        baseline_hip_dir=args.baseline_hip_dir.resolve(),
        module_dir=args.module_dir.resolve(),
        functional_dir=args.functional_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        artifacts_dir=args.artifacts_dir.resolve(),
        resume=args.resume,
        num_workers=args.num_workers,
        target_function_mode=args.target_function_mode,
        max_attempts=args.max_attempts,
        rtol=args.rtol,
        atol=args.atol,
        seed=args.seed,
        overwrite=args.overwrite,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        provider=args.provider,
        model_id=args.model_id,
        python_load_timeout_seconds=args.python_load_timeout_seconds,
        hip_compile_timeout_seconds=args.hip_compile_timeout_seconds,
        execution_timeout_seconds=args.execution_timeout_seconds,
        benchmark_timeout_seconds=args.benchmark_timeout_seconds,
        perf_warmup=args.perf_warmup,
        perf_iterations=args.perf_iterations,
        instruction_file=args.instruction_file.resolve() if args.instruction_file else None,
        few_shot_file=args.few_shot_file.resolve() if args.few_shot_file else None,
        keep_build_dirs=args.keep_build_dirs,
        offload_arch=args.offload_arch,
    ).with_defaults()

    summary = run_optimization_pipeline(client, config)
    print(
        "Optimization finished: "
        f"total={summary['total']} success={summary['success']} "
        f"failed={summary['failed']} skipped={summary['skipped']}"
    )


if __name__ == "__main__":
    main()
