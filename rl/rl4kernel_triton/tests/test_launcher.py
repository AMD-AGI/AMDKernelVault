# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the paper configuration and the Slime job boundary."""

import json
from pathlib import Path

import pytest

from triton_rl.config import load_config
from triton_rl.launcher import build_plan, parser


def arguments(*extra):
    return parser().parse_args(
        [
            "--hf-checkpoint",
            "/models/SFT checkpoint",
            "--reference-checkpoint",
            "/models/reference",
            "--train-data",
            "/data/prepared prompts.parquet",
            "--output",
            "/output/run",
            *extra,
        ]
    )


def environment(**extra):
    return {"TRITON_RL_SANDBOX_URLS": "http://executor-a:8080,http://executor-b:8080", **extra}


def flag(plan, name):
    command = plan["command"]
    return command[command.index(name) + 1]


def test_paper_batch_and_mean_only_advantages():
    plan = build_plan(arguments("--rollout-steps", "7"), environment())
    assert flag(plan, "--rollout-batch-size") == "64"
    assert flag(plan, "--n-samples-per-prompt") == "8"
    assert flag(plan, "--global-batch-size") == "512"
    assert flag(plan, "--kl-loss-coef") == "0.01"
    assert flag(plan, "--lr") == "1e-06"
    assert flag(plan, "--eps-clip") == "0.2"
    assert flag(plan, "--eps-clip-high") == "0.28"
    assert flag(plan, "--rollout-max-context-len") == "16384"
    assert "--use-kl-loss" in plan["command"]
    assert "--disable-grpo-std-normalization" in plan["command"]
    assert "--disable-rewards-normalization" not in plan["command"]
    assert "--normalize-advantages" not in plan["command"]
    assert "--keep-old-actor" not in plan["command"]
    assert "--use-tis" not in plan["command"]
    assert plan["command"][1:3] == ["-m", "triton_rl.driver"]
    assert flag(plan, "--prompt-data") == "/data/prepared prompts.parquet"


def test_fractional_epoch_budget_reports_rounding():
    plan = build_plan(arguments("--effective-prompts", "100"), environment())
    assert flag(plan, "--num-rollout") == "5"
    assert plan["budget"]["effective_epochs"] == 3.2
    assert plan["budget"]["target_epochs"] == 2.7
    assert plan["budget"]["prompt_exposures"] == 320


def test_resumed_budget_adds_new_rollouts(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("39")
    (tmp_path / "rollout").mkdir()
    (tmp_path / "rollout/global_dataset_state_dict_39.pt").touch()
    plan = build_plan(arguments("--rollout-steps", "7", "--resume", str(tmp_path)), environment())
    assert flag(plan, "--num-rollout") == "47"
    assert plan["budget"]["start_rollout_id"] == 40
    assert plan["budget"]["rollout_steps"] == 7
    assert "--override-opt_param-scheduler" in plan["command"]
    assert "--use-checkpoint-opt_param-scheduler" not in plan["command"]


def test_resume_requires_saved_prompt_cursor(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("39")
    with pytest.raises(ValueError, match="saved prompt cursor"):
        build_plan(arguments("--rollout-steps", "7", "--resume", str(tmp_path)), environment())


def test_new_phase_keeps_checkpoint_but_resets_prompt_phase(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("39")
    plan = build_plan(
        arguments("--rollout-steps", "7", "--resume", str(tmp_path), "--new-curriculum-phase"),
        environment(),
    )
    assert "--new-curriculum-phase" in plan["command"]
    assert flag(plan, "--load") == str(tmp_path)


def test_new_phase_requires_resume():
    with pytest.raises(ValueError, match="requires a resumed"):
        build_plan(arguments("--rollout-steps", "7", "--new-curriculum-phase"), environment())


def test_release_checkpoint_is_not_training_resume(tmp_path):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release")
    with pytest.raises(ValueError, match="numeric iteration"):
        build_plan(arguments("--rollout-steps", "7", "--resume", str(tmp_path)), environment())


def test_environment_selected_configuration_reaches_workers(tmp_path):
    path = tmp_path / "selected config.json"
    config = load_config()
    config["training"]["learning_rate"] = 2e-6
    path.write_text(json.dumps(config))
    plan = build_plan(arguments("--rollout-steps", "7"), environment(TRITON_RL_CONFIG=str(path)))
    assert flag(plan, "--lr") == "2e-06"
    assert plan["runtime_env"]["env_vars"]["TRITON_RL_CONFIG"] == str(path.resolve())


def test_credentials_are_not_inserted_into_submission_plan():
    plan = build_plan(
        arguments("--rollout-steps", "7"),
        environment(
            HF_TOKEN="test-secret", WANDB_API_KEY="test-secret", AWS_SECRET_ACCESS_KEY="test-secret"
        ),
    )
    assert "test-secret" not in json.dumps(plan)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "file:///tmp/socket",
        "http://user:pass@host:8080",
        "https://host/?token=secret",
        "http://host/#fragment",
        "http://host:0",
        "http://host:99999",
        "http://host:invalid",
        "http://host with spaces",
    ],
)
def test_rejects_unsupported_execution_endpoint(url):
    with pytest.raises(ValueError):
        build_plan(arguments("--rollout-steps", "7"), {"TRITON_RL_SANDBOX_URLS": url})


@pytest.mark.parametrize(
    "group,key,value",
    [
        ("algorithm", "max_turns", True),
        ("algorithm", "max_turns", 0),
        ("algorithm", "discount", True),
        ("algorithm", "discount", float("nan")),
        ("algorithm", "performance_scale", -1),
        ("algorithm", "performance_cap", float("inf")),
        ("training", "samples_per_prompt", False),
        ("evaluation", "atol", -0.1),
        ("evaluation", "equal_nan", True),
    ],
)
def test_rejects_invalid_algorithm_before_generation(tmp_path, group, key, value):
    config = load_config()
    config[group][key] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_config(path)


def test_default_reward_and_discount_match_paper():
    config = load_config()
    assert config["algorithm"] == {
        "max_turns": 3,
        "discount": 0.6,
        "compile_reward": 0.4,
        "correct_reward": 1.0,
        "performance_scale": 0.375,
        "performance_cap": 1.5,
    }
    assert config["evaluation"] == {"atol": 0.001, "rtol": 0.0001, "equal_nan": False}


def test_public_package_has_no_dataset_files():
    source = Path(__file__).parents[1] / "src"
    forbidden = {".parquet", ".jsonl", ".safetensors", ".pt"}
    assert not [path for path in source.rglob("*") if path.suffix in forbidden]
