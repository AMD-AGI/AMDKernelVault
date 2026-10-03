# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .model_factory import create_model_client
from .pipeline import run_conversion_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert PyTorch module files into verified HIP kernels."
    )
    parser.add_argument("--module-dir", type=Path, required=True, help="Input root for PyTorch module files.")
    parser.add_argument("--functional-dir", type=Path, required=True, help="Input root for paired PyTorch functional files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root for selected HIP files.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".artifacts"),
        help="Directory used to store prompts, candidate attempts, build files, and JSON records.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "standard-openai", "claude", "standard-claude", "gemini"],
        help="Public LLM provider. The standard-openai and standard-claude names are aliases.",
    )
    parser.add_argument("--model-id", default="gpt-4o", help="Provider-specific model identifier.")
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "LLM API key. The client checks TORCH2HIP_API_KEY, then TORCH_MODU2FUNC_API_KEY, "
            "then OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY/GOOGLE_API_KEY for the selected provider."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum HIP candidates per sample.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for output comparison.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for output comparison.")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic seed for verification.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--max-tokens", type=int, default=12000, help="Maximum tokens per generation call.")
    parser.add_argument("--perf-warmup", type=int, default=25, help="Warmup iterations before timing.")
    parser.add_argument("--perf-iterations", type=int, default=200, help="Measured iterations for latency timing.")
    parser.add_argument("--instruction-file", type=Path, default=None, help="Optional Python file that defines `hip_generation_req`.")
    parser.add_argument("--few-shot-file", type=Path, default=None, help="Optional Python file that defines `few_shot_code_instructions`.")
    parser.add_argument("--keep-build-dirs", action="store_true", help="Preserve per-attempt build directories for debugging.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing selected HIP files.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1.")
    if args.perf_iterations < 1:
        parser.error("--perf-iterations must be at least 1.")
    if args.perf_warmup < 0:
        parser.error("--perf-warmup must be non-negative.")

    client = create_model_client(args.provider, args.model_id, args.api_key)
    config = PipelineConfig(
        module_dir=args.module_dir.resolve(),
        functional_dir=args.functional_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        artifacts_dir=args.artifacts_dir.resolve(),
        max_attempts=args.max_attempts,
        rtol=args.rtol,
        atol=args.atol,
        seed=args.seed,
        overwrite=args.overwrite,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        perf_warmup=args.perf_warmup,
        perf_iterations=args.perf_iterations,
        instruction_file=args.instruction_file.resolve() if args.instruction_file else None,
        few_shot_file=args.few_shot_file.resolve() if args.few_shot_file else None,
        keep_build_dirs=args.keep_build_dirs,
    ).with_defaults()

    summary = run_conversion_pipeline(client, config)
    print(
        "Conversion finished: "
        f"total={summary['total']} success={summary['success']} "
        f"failed={summary['failed']} skipped={summary['skipped']}"
    )


if __name__ == "__main__":
    main()
