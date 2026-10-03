# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Run Triton kernel construction with external references and fixed cases."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .client import ChatClient
from .contracts import GenerationSettings, TaskSpec, VerificationSettings
from .tasks import load_tasks, validate_task

MODEL_FAMILIES = ("gpt-oss-120b", "deepseek-r1", "qwen2.5-32b")


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("The value must be a positive integer.")
    return value


def nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("The value must be a nonnegative integer.")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path, help="External JSON task manifest.")
    source.add_argument("--module", type=Path, help="External original PyTorch module.")
    result.add_argument(
        "--functional", type=Path, help="External standardized functional reference."
    )
    result.add_argument("--case-provider", type=Path, help="External fixed-case provider.")
    result.add_argument("--task-id", help="Stable task identifier for a single task.")
    result.add_argument(
        "--seed-kernel",
        type=Path,
        help="Optional existing Triton source for conversion or refinement.",
    )
    result.add_argument(
        "--baseline-kernel",
        type=Path,
        help="Optional Triton baseline that must pass AMD validation before optimization.",
    )
    result.add_argument(
        "--dependency",
        type=Path,
        action="append",
        default=[],
        help="Declared Python helper for the reference.",
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--artifacts-dir", type=Path, required=True)
    result.add_argument(
        "--endpoint",
        default=os.environ.get("TRITON_GEN_ENDPOINT"),
        help="Complete chat-completion endpoint URL.",
    )
    result.add_argument(
        "--model-id",
        default=os.environ.get("TRITON_GEN_MODEL"),
        help="Model identifier served by the endpoint.",
    )
    result.add_argument(
        "--model-family",
        choices=MODEL_FAMILIES,
        required=True,
        help="Generation model family reported by the paper.",
    )
    result.add_argument("--reflector-model-id", help="Defaults to the generation model.")
    result.add_argument(
        "--api-key-env",
        default="TRITON_GEN_API_KEY",
        help="Environment variable containing optional endpoint authentication.",
    )
    result.add_argument(
        "--request-options",
        type=Path,
        help="External JSON object with additional sampling options.",
    )
    result.add_argument("--request-timeout", type=float, default=300)
    result.add_argument(
        "--max-attempts",
        type=positive_int,
        required=True,
        help="Total code-generation budget per task; the paper gives no fixed count.",
    )
    result.add_argument("--num-variants", type=positive_int, default=1)
    result.add_argument("--temperature", type=float, default=0.7)
    result.add_argument("--max-tokens", type=positive_int, default=16384)
    result.add_argument("--history-char-limit", type=positive_int, default=16000)
    result.add_argument("--gpu", type=nonnegative_int, default=0)
    result.add_argument("--target-arch", choices=("gfx942", "gfx950"))
    result.add_argument("--seed", type=nonnegative_int, default=1234)
    result.add_argument("--atol", type=float, default=1e-3)
    result.add_argument("--rtol", type=float, default=1e-4)
    result.add_argument("--verification-timeout", type=float, default=900)
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Check paths and print the plan without importing tasks or calling a model.",
    )
    return result


def _settings(args: argparse.Namespace) -> tuple[GenerationSettings, VerificationSettings]:
    for name in ("temperature", "atol", "rtol"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"The {name} value must be finite and nonnegative.")
    for name in ("request_timeout", "verification_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"The {name} value must be positive and finite.")
    if args.seed >= 2**32:
        raise ValueError("The seed must be less than 4294967296.")
    if args.num_variants > args.max_attempts:
        raise ValueError("The requested variant count must not exceed the generation budget.")
    generation = GenerationSettings(
        args.output_dir.resolve(),
        args.artifacts_dir.resolve(),
        args.max_attempts,
        args.num_variants,
        args.temperature,
        args.max_tokens,
        args.history_char_limit,
    )
    verification = VerificationSettings(
        gpu=args.gpu,
        target_arch=args.target_arch,
        seed=args.seed,
        atol=args.atol,
        rtol=args.rtol,
        timeout_seconds=args.verification_timeout,
        excluded_environment=(args.api_key_env,),
    )
    return generation, verification


def _tasks(args: argparse.Namespace) -> list[TaskSpec]:
    if args.manifest:
        if (
            args.functional
            or args.case_provider
            or args.task_id
            or args.seed_kernel
            or args.baseline_kernel
            or args.dependency
        ):
            raise ValueError("A manifest supplies its own task paths and identifiers.")
        return load_tasks(args.manifest)
    if args.functional is None or args.case_provider is None:
        raise ValueError("A single task requires --functional and --case-provider.")
    return [
        validate_task(
            TaskSpec(
                args.task_id or args.module.stem,
                args.module.resolve(),
                args.functional.resolve(),
                args.case_provider.resolve(),
                args.seed_kernel.resolve() if args.seed_kernel else None,
                dependency_paths=tuple(path.resolve() for path in args.dependency),
                baseline_kernel_path=args.baseline_kernel.resolve()
                if args.baseline_kernel
                else None,
            )
        )
    ]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    client = reflector = None
    try:
        generation, verification = _settings(args)
        tasks = _tasks(args)
        if not args.endpoint or not args.model_id:
            raise ValueError(
                "Supply --endpoint and --model-id, or their TRITON_GEN environment variables."
            )
        options = json.loads(args.request_options.read_text()) if args.request_options else {}
        if not isinstance(options, dict):
            raise ValueError("Additional request options must be a JSON object.")
        client = ChatClient(
            args.endpoint,
            args.model_id,
            api_key_env=args.api_key_env,
            timeout_seconds=args.request_timeout,
            request_options=options,
        )
        reflector = (
            client
            if not args.reflector_model_id
            else ChatClient(
                args.endpoint,
                args.reflector_model_id,
                api_key_env=args.api_key_env,
                timeout_seconds=args.request_timeout,
                request_options=options,
            )
        )
        plan = {
            "schema_version": 1,
            "container": {
                "reference": os.environ.get("TRITON_GEN_CONTAINER_REFERENCE"),
                "image_id": os.environ.get("TRITON_GEN_CONTAINER_IMAGE_ID"),
            },
            "model_family": args.model_family,
            "model_id": args.model_id,
            "reflector_model_id": args.reflector_model_id or args.model_id,
            "generation": asdict(generation),
            "verification": asdict(verification),
            "request_options": options,
            "tasks": [asdict(task) for task in tasks],
            "maximum_model_requests": len(tasks) * (2 * args.max_attempts - 1),
        }
        if args.dry_run:
            print(json.dumps(_jsonable(plan), indent=2, allow_nan=False))
            return 0

        from .pipeline import run_task
        from .verifier import Verifier

        verifier = Verifier(verification)
        records = []
        generation.artifacts_dir.mkdir(parents=True, exist_ok=True)
        summary_path = generation.artifacts_dir / f"summary-{uuid.uuid4().hex}.json"
        for task in tasks:
            record = run_task(task, client, reflector, verifier, generation)
            records.append(record)
            summary = {"schema_version": 1, "plan": _jsonable(plan), "records": records}
            temporary = summary_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
            temporary.replace(summary_path)
            print(f"{task.task_id}: {record['status']}")
        print(f"Run summary: {summary_path}")
        return 0 if all(record["status"] == "success" for record in records) else 1
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Cannot complete kernel generation: {exc}", file=sys.stderr)
        return 2
    finally:
        if reflector is not None and reflector is not client:
            reflector.close()
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
