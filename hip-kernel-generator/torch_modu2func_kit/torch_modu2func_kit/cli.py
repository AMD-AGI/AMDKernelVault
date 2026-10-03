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
        description="Convert PyTorch module files into verified functional equivalents."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Input root such as kernelbench_torch_modu.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output root such as kernelbench_torch_func.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path(".artifacts"),
        help="Directory used to store prompts, candidate attempts, and JSON records.",
    )
    parser.add_argument(
        "--provider",
        default="openai",
        choices=["openai", "standard-openai", "claude", "standard-claude", "gemini"],
        help="LLM provider adapter.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model identifier. Defaults: gpt-4o, claude-sonnet-4-20250514, or gemini-2.5-pro.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=("API key. Checks TORCH_MODU2FUNC_API_KEY, then OPENAI_API_KEY, "
              "ANTHROPIC_API_KEY, or GEMINI_API_KEY for the selected provider."),
    )
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum attempts per sample.")
    parser.add_argument("--rtol", type=float, default=1e-4, help="Relative tolerance for output comparison.")
    parser.add_argument("--atol", type=float, default=1e-4, help="Absolute tolerance for output comparison.")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic seed for verification.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--max-tokens", type=int, default=5000, help="Maximum tokens per generation call.")
    parser.add_argument("--overwrite", action="store_true", help="Generate and verify replacements for existing output files.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1.")

    client = create_model_client(args.provider, args.model_id, args.api_key)
    config = PipelineConfig(
        input_dir=args.input_dir.resolve(),
        output_dir=args.output_dir.resolve(),
        artifacts_dir=args.artifacts_dir.resolve(),
        max_attempts=args.max_attempts,
        rtol=args.rtol,
        atol=args.atol,
        seed=args.seed,
        overwrite=args.overwrite,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    ).with_defaults()

    summary = run_conversion_pipeline(client, config)
    print(
        "Conversion finished: "
        f"total={summary['total']} success={summary['success']} "
        f"failed={summary['failed']} skipped={summary['skipped']}"
    )


if __name__ == "__main__":
    main()
