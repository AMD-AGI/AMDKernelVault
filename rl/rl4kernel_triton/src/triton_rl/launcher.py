# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a synchronous Slime job from the published Triton configuration."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from triton_rl.config import load_config

SLIME_REVISION = "5f781608ba28738fc73f44fa12efef1cdb408ee2"

# Qwen3-8B architecture from the pinned Slime model configuration.
MODEL_ARGS = [
    "--swiglu",
    "--num-layers",
    "36",
    "--hidden-size",
    "4096",
    "--ffn-hidden-size",
    "12288",
    "--num-attention-heads",
    "32",
    "--group-query-attention",
    "--num-query-groups",
    "8",
    "--use-rotary-position-embeddings",
    "--disable-bias-linear",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-6",
    "--rotary-base",
    "1000000",
    "--vocab-size",
    "151936",
    "--kv-channels",
    "128",
    "--qk-layernorm",
    "--untie-embeddings-and-output-weights",
]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("The value must be a positive integer.")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--hf-checkpoint", required=True, help="External Qwen3-8B SFT checkpoint.")
    result.add_argument(
        "--reference-checkpoint", required=True, help="Converted Megatron SFT checkpoint."
    )
    result.add_argument("--train-data", required=True, help="External prepared prompt file.")
    result.add_argument(
        "--output", required=True, help="Checkpoint directory for this training run."
    )
    result.add_argument("--resume", help="Existing training checkpoint with optimizer state.")
    result.add_argument(
        "--new-curriculum-phase",
        action="store_true",
        help="Keep the resumed optimizer but start the supplied prompt phase at its beginning.",
    )
    result.add_argument("--config", help="Optional configuration file shared across all nodes.")
    result.add_argument("--slime-root", default=os.environ.get("SLIME_ROOT", "/opt/slime"))
    result.add_argument(
        "--ray-address", default=os.environ.get("RAY_JOB_ADDRESS", "http://127.0.0.1:8265")
    )
    budget = result.add_mutually_exclusive_group(required=True)
    budget.add_argument(
        "--rollout-steps", type=positive_int, help="Exact budget for this curriculum phase."
    )
    budget.add_argument(
        "--effective-prompts",
        type=positive_int,
        help="Retained prompt count for a single fixed phase, used with the epoch target.",
    )
    result.add_argument("--actor-nodes", type=positive_int, default=4)
    result.add_argument("--gpus-per-node", type=positive_int, default=8)
    result.add_argument("--tensor-parallel-size", type=positive_int, default=2)
    result.add_argument("--rollout-gpus-per-engine", type=positive_int, default=2)
    result.add_argument("--max-tokens-per-gpu", type=positive_int, default=16384)
    result.add_argument("--save-interval", type=positive_int, default=20)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--temperature", type=float, default=1.0)
    result.add_argument("--sglang-memory-fraction", type=float, default=0.4)
    result.add_argument(
        "--wandb-project", help="Enable optional W&B logging with environment authentication."
    )
    result.add_argument(
        "--dry-run", action="store_true", help="Print the plan without starting a job."
    )
    return result


def _validate_url(value: str) -> None:
    if any(character.isspace() for character in value):
        raise ValueError("Execution URLs must not contain whitespace.")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Each execution endpoint must be an HTTP or HTTPS URL.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("The endpoint port must be an integer from 1 through 65535.") from exc
    if port is not None and port < 1:
        raise ValueError("The endpoint port must be an integer from 1 through 65535.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "Execution URLs must not contain credentials, query strings, or fragments."
        )


def build_plan(
    args: argparse.Namespace, environment: dict[str, str] | None = None
) -> dict[str, Any]:
    env = os.environ if environment is None else environment
    selected_config = args.config or env.get("TRITON_RL_CONFIG")
    config = load_config(selected_config)
    training = config["training"]
    total_gpus = args.actor_nodes * args.gpus_per_node
    if total_gpus != training["total_gpus"]:
        raise ValueError(
            f"The configuration requires {training['total_gpus']} training GPUs across the actor nodes."
        )
    if total_gpus % args.tensor_parallel_size or total_gpus % args.rollout_gpus_per_engine:
        raise ValueError("The GPU count must divide evenly into the parallel groups.")
    if args.tensor_parallel_size > args.gpus_per_node:
        raise ValueError("Tensor parallelism must fit inside one node for this launcher.")
    if not math.isfinite(args.temperature) or args.temperature <= 0:
        raise ValueError("The sampling temperature must be positive and finite.")
    if not 0 < args.sglang_memory_fraction < 1:
        raise ValueError("The SGLang memory fraction must be between zero and one.")
    endpoints = env.get("TRITON_RL_SANDBOX_URLS", "")
    if not endpoints.strip():
        raise ValueError("Set TRITON_RL_SANDBOX_URLS to the execution service endpoints.")
    for endpoint in endpoints.split(","):
        _validate_url(endpoint.strip())
    _validate_url(args.ray_address)

    prompt_batch = training["prompts_per_rollout"]
    group_size = training["samples_per_prompt"]
    rollout_steps = args.rollout_steps
    budget = {"rollout_steps": rollout_steps, "prompt_batch": prompt_batch}
    if rollout_steps is None:
        rollout_steps = math.ceil(args.effective_prompts * training["epochs"] / prompt_batch)
        budget.update(
            rollout_steps=rollout_steps,
            retained_prompts=args.effective_prompts,
            target_epochs=training["epochs"],
            effective_epochs=rollout_steps * prompt_batch / args.effective_prompts,
            rounding="Round up to a complete 64-prompt rollout.",
        )
    budget["prompt_exposures"] = rollout_steps * prompt_batch
    budget["sampled_trajectories"] = rollout_steps * prompt_batch * group_size
    start_rollout = 0
    if args.resume:
        tracker = Path(args.resume) / "latest_checkpointed_iteration.txt"
        try:
            previous = int(tracker.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(
                "Resume requires a training checkpoint with a numeric iteration tracker."
            ) from exc
        if previous < 0:
            raise ValueError("The saved training iteration must be nonnegative.")
        cursor = Path(args.resume) / "rollout" / f"global_dataset_state_dict_{previous}.pt"
        if not args.new_curriculum_phase and not cursor.is_file():
            raise ValueError(
                "Resume requires the saved prompt cursor, or an explicit new curriculum phase."
            )
        start_rollout = previous + 1
    elif args.new_curriculum_phase:
        raise ValueError("A new curriculum phase requires a resumed training checkpoint.")
    stop_rollout = start_rollout + rollout_steps
    budget["start_rollout_id"] = start_rollout
    budget["stop_rollout_id_exclusive"] = stop_rollout

    runtime_env = {
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
        "PYTHONUNBUFFERED": "1",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "TRITON_RL_SANDBOX_URLS": endpoints,
    }
    for key in ("TRITON_RL_SANDBOX_TIMEOUT_SECONDS", "HF_HOME", "PYTHONPATH"):
        if env.get(key):
            runtime_env[key] = env[key]
    if selected_config:
        runtime_env["TRITON_RL_CONFIG"] = str(Path(selected_config).resolve())

    command = [sys.executable, "-m", "triton_rl.driver"]
    command += MODEL_ARGS
    values = [
        ("--hf-checkpoint", args.hf_checkpoint),
        ("--ref-load", args.reference_checkpoint),
        ("--save", args.output),
        ("--save-interval", args.save_interval),
        ("--prompt-data", args.train_data),
        ("--input-key", "prompt"),
        ("--label-key", "reward_model"),
        ("--num-rollout", stop_rollout),
        ("--rollout-batch-size", prompt_batch),
        ("--n-samples-per-prompt", group_size),
        ("--global-batch-size", prompt_batch * group_size),
        ("--rollout-max-context-len", training["sequence_length"]),
        ("--rollout-max-response-len", training["sequence_length"]),
        ("--rollout-temperature", args.temperature),
        ("--rollout-seed", args.seed),
        ("--seed", args.seed),
        ("--actor-num-nodes", args.actor_nodes),
        ("--actor-num-gpus-per-node", args.gpus_per_node),
        ("--tensor-model-parallel-size", args.tensor_parallel_size),
        ("--pipeline-model-parallel-size", 1),
        ("--context-parallel-size", 1),
        ("--expert-model-parallel-size", 1),
        ("--expert-tensor-parallel-size", 1),
        ("--rollout-num-gpus-per-engine", args.rollout_gpus_per_engine),
        ("--sglang-mem-fraction-static", args.sglang_memory_fraction),
        ("--max-tokens-per-gpu", args.max_tokens_per_gpu),
        ("--advantage-estimator", "grpo"),
        ("--kl-loss-coef", training["kl_coefficient"]),
        ("--kl-loss-type", "low_var_kl"),
        ("--eps-clip", training["clip_low"]),
        ("--eps-clip-high", training["clip_high"]),
        ("--entropy-coef", 0.0),
        ("--optimizer", "adam"),
        ("--lr", training["learning_rate"]),
        ("--lr-decay-style", "constant"),
        ("--weight-decay", 0.1),
        ("--adam-beta1", 0.9),
        ("--adam-beta2", 0.98),
        ("--attention-dropout", 0.0),
        ("--hidden-dropout", 0.0),
        ("--attention-backend", "flash"),
        ("--distributed-backend", "nccl"),
        ("--recompute-granularity", "full"),
        ("--recompute-method", "uniform"),
        ("--recompute-num-layers", 1),
        ("--custom-generate-function-path", "triton_rl.rollout.generate"),
        ("--custom-rm-path", "triton_rl.rollout.reward_func"),
    ]
    for key, value in values:
        command += [key, str(value)]
    command += [
        "--colocate",
        "--apply-chat-template",
        "--rollout-shuffle",
        "--balance-data",
        "--sequence-parallel",
        "--use-dynamic-batch-size",
        "--use-kl-loss",
        "--disable-grpo-std-normalization",
        "--no-gradient-accumulation-fusion",
        "--accumulate-allreduce-grads-in-fp32",
        "--attention-softmax-in-fp32",
    ]
    if args.resume:
        # Extend the constant scheduler horizon while restoring its step count.
        command += ["--load", args.resume, "--override-opt_param-scheduler"]
    if args.new_curriculum_phase:
        command += ["--new-curriculum-phase"]
    if args.wandb_project:
        command += ["--use-wandb", "--wandb-project", args.wandb_project]
    return {
        "slime_revision": SLIME_REVISION,
        "budget": budget,
        "runtime_env": {"env_vars": runtime_env},
        "command": command,
    }


def validate_inputs(args: argparse.Namespace) -> None:
    for label, value in (
        ("SFT checkpoint", args.hf_checkpoint),
        ("reference checkpoint", args.reference_checkpoint),
        ("training prompts", args.train_data),
    ):
        if not Path(value).exists():
            raise FileNotFoundError(f"The external {label} path does not exist: {value}")
    if not (Path(args.slime_root) / "train.py").is_file():
        raise FileNotFoundError("SLIME_ROOT must contain the pinned Slime training entry point.")
    revision = subprocess.run(
        ["git", "-C", args.slime_root, "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if revision != SLIME_REVISION:
        raise ValueError(f"Slime must use revision {SLIME_REVISION}.")
    if not (Path(args.reference_checkpoint) / "latest_checkpointed_iteration.txt").is_file():
        raise FileNotFoundError("The reference checkpoint requires a Megatron iteration tracker.")
    from triton_rl.preflight import run_preflight

    report = run_preflight("trainer")
    if not report.passed:
        failures = " ".join(check.message for check in report.checks if not check.passed)
        raise ValueError(f"The trainer runtime check failed. {failures}")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        plan = build_plan(args)
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        validate_inputs(args)
        print(json.dumps({"budget": plan["budget"], "slime_revision": SLIME_REVISION}, indent=2))
        submission = [
            "ray",
            "job",
            "submit",
            "--address",
            args.ray_address,
            "--runtime-env-json",
            json.dumps(plan["runtime_env"]),
            "--",
            *plan["command"],
        ]
        return subprocess.run(submission, check=False).returncode
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Cannot start training: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
