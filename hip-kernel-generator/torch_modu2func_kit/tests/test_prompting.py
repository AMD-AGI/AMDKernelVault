# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import torch

from torch_modu2func_kit.config import AttemptRecord
from torch_modu2func_kit.prompt_assets import FEW_SHOT_EXAMPLES
from torch_modu2func_kit.prompting import build_prompt


def test_build_prompt_includes_failed_attempt_history(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("def module_fn(x):\n    return x - 1\n", encoding="utf-8")

    attempt_records = [
        AttemptRecord(
            attempt=1,
            prompt_path=(tmp_path / "prompt_1.txt").as_posix(),
            candidate_path=candidate_path.as_posix(),
            status="failed",
            feedback="AssertionError: output tensor mismatch",
            mismatch="AssertionError: output tensor mismatch",
        )
    ]

    prompt = build_prompt(
        "class Model:\n    pass\n",
        Path("level_1/sample.py"),
        attempt_records,
        code_char_limit=1000,
        feedback_char_limit=1000,
    )

    assert "Previous failed attempts are included below" in prompt
    assert "Attempt 1:" in prompt
    assert "return x - 1" in prompt
    assert "AssertionError: output tensor mismatch" in prompt


def test_build_prompt_truncates_long_history_blocks(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_text("x" * 200, encoding="utf-8")

    attempt_records = [
        AttemptRecord(
            attempt=1,
            prompt_path=(tmp_path / "prompt_1.txt").as_posix(),
            candidate_path=candidate_path.as_posix(),
            status="failed",
            feedback="y" * 200,
        )
    ]

    prompt = build_prompt(
        "class Model:\n    pass\n",
        Path("level_1/sample.py"),
        attempt_records,
        code_char_limit=50,
        feedback_char_limit=50,
    )

    assert "... [truncated]" in prompt


def test_first_example_preserves_original_forward_inputs() -> None:
    example = FEW_SHOT_EXAMPLES.split("Example 1:", 1)[1].split("Example 2:", 1)[0]
    original_source, functional_source = example.split("pytorch functional:", 1)
    original_source = original_source.split("pytorch module:", 1)[1]
    original_namespace: dict = {}
    functional_namespace: dict = {}
    exec(original_source, original_namespace)
    exec(functional_source, functional_namespace)
    init_args = (2, 3, 3, 2, 1)
    torch.manual_seed(1234)
    original = original_namespace["Model"](*init_args).eval()
    torch.manual_seed(1234)
    functional = functional_namespace["Model"](*init_args).eval()
    inputs = torch.randn(1, 2, 4, 4, 4)
    calls = []

    def tracked_fn(*args, **kwargs):
        calls.append(True)
        return functional_namespace["module_fn"](*args, **kwargs)

    with torch.no_grad():
        expected = original(inputs)
        actual = functional(inputs)
        injected = functional(inputs, fn=tracked_fn)

    torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-4, equal_nan=True)
    torch.testing.assert_close(expected, injected, rtol=1e-4, atol=1e-4, equal_nan=True)
    assert calls == [True]
