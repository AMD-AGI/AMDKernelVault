# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the configuration for the published Triton training procedure."""

from __future__ import annotations

import json
import math
import os
from importlib.resources import files
from pathlib import Path
from typing import Any


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Read the packaged paper configuration or an explicitly selected file."""
    selected = path or os.environ.get("TRITON_RL_CONFIG")
    if selected:
        contents = Path(selected).read_text(encoding="utf-8")
    else:
        contents = files("triton_rl").joinpath("paper.json").read_text(encoding="utf-8")
    config = json.loads(contents)
    if not isinstance(config, dict):
        raise ValueError("The configuration must contain a JSON object.")
    for group in ("algorithm", "training", "evaluation"):
        if not isinstance(config.get(group), dict):
            raise ValueError(f"The configuration requires the {group} object.")
    required = {
        "algorithm": {
            "max_turns",
            "discount",
            "compile_reward",
            "correct_reward",
            "performance_scale",
            "performance_cap",
        },
        "training": {
            "prompts_per_rollout",
            "samples_per_prompt",
            "sequence_length",
            "total_gpus",
            "epochs",
            "learning_rate",
            "kl_coefficient",
            "clip_low",
            "clip_high",
        },
        "evaluation": {"atol", "rtol", "equal_nan"},
    }
    for group, keys in required.items():
        missing = sorted(keys - config[group].keys())
        if missing:
            raise ValueError(f"The {group} configuration is missing: {', '.join(missing)}.")
    algorithm = config["algorithm"]
    if type(algorithm["max_turns"]) is not int or algorithm["max_turns"] < 1:
        raise ValueError("The turn limit must be a positive integer.")

    def number(value: Any, name: str, *, positive: bool = False) -> None:
        try:
            valid = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            valid = False
        if not valid or (value <= 0 if positive else value < 0):
            adjective = "positive" if positive else "nonnegative"
            raise ValueError(f"The {name} value must be finite and {adjective}.")

    number(algorithm["discount"], "discount")
    if algorithm["discount"] > 1:
        raise ValueError("The discount must be between zero and one.")
    for name in ("compile_reward", "correct_reward", "performance_scale", "performance_cap"):
        number(algorithm[name], name)
    training = config["training"]
    for name in ("prompts_per_rollout", "samples_per_prompt", "sequence_length", "total_gpus"):
        if type(training[name]) is not int or training[name] < 1:
            raise ValueError(f"The {name} value must be a positive integer.")
    for name in ("epochs", "learning_rate", "kl_coefficient", "clip_low", "clip_high"):
        number(training[name], name, positive=True)
    for name in ("atol", "rtol"):
        number(config["evaluation"][name], name)
    if config["evaluation"].get("equal_nan") is not False:
        raise ValueError("The execution protocol requires equal_nan=false.")
    return config
