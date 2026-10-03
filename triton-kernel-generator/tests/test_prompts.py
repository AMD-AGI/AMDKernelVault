# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import json

import pytest

from triton_kernel_gen.prompts import build_generation_messages, build_reflection_messages


def arguments():
    return {
        "module_code": "class Model: pass\n",
        "functional_code": "def module_fn(x, *, scale=2): return x * scale\n",
        "module_name": "/private/source/original.py",
        "functional_name": "/private/source/functional.py",
    }


def test_generation_includes_both_sources_and_only_supplied_basenames():
    messages = build_generation_messages(**arguments())
    text = messages[1]["content"]
    assert "Original PyTorch module" in text
    assert "Verified functional reference" in text
    assert "original.py" in text and "functional.py" in text
    assert "/private/source" not in text
    assert "def module_fn(x, *, scale=2)" in text
    assert [message["role"] for message in messages] == ["system", "user"]


def test_generation_uses_explicit_seed_and_dependency_context():
    messages = build_generation_messages(
        **arguments(),
        seed_code="seed marker",
        seed_name="/external/seed.py",
        dependencies=[{"filename": "/source/helper.py", "source": "helper marker"}],
        environment={"device_name": "test device", "triton_version": "3.3.0"},
    )
    text = messages[1]["content"]
    assert all(
        marker in text
        for marker in ("seed marker", "helper marker", "seed.py", "helper.py", "test device")
    )
    assert "/external" not in text and "/source" not in text


@pytest.mark.parametrize("builder", [build_generation_messages, build_reflection_messages])
def test_prompts_distinguish_raw_seed_from_verified_baseline(builder):
    extra = (
        {"candidate_text": "candidate", "feedback": {}}
        if builder is build_reflection_messages
        else {}
    )
    messages = builder(
        **arguments(),
        **extra,
        seed_code="CUDA_SEED_MARKER = 1",
        seed_name="/external/cuda_seed.py",
        baseline_code="AMD_BASELINE_MARKER = 1",
        baseline_name="/external/amd_baseline.py",
    )
    context = messages[1]["content"]
    assert "Raw Triton seed (unverified source context)" in context
    assert "Verified Triton baseline" in context
    assert "CUDA_SEED_MARKER" in context and "AMD_BASELINE_MARKER" in context
    assert "cuda_seed.py" in context and "amd_baseline.py" in context
    assert "/external" not in context
    assert "unverified source context" in messages[0]["content"]


def test_generation_does_not_require_an_amd_runnable_seed():
    system = build_generation_messages(**arguments())[0]["content"]
    assert "Do not assume that the raw seed compiles or runs on AMD" in system
    assert "Port CUDA-specific operations" in system


def test_system_preserves_fixed_boundary_and_generic_real_triton():
    system = build_generation_messages(**arguments())[0]["content"]
    for requirement in (
        "Triton 3.3.0",
        "AMD ROCm",
        "@triton.jit",
        "module_fn",
        "shapes, strides, dtypes",
        "Do not hardcode",
        "numerical tolerances",
        "case provider",
        "file writes",
        "PyTorch",
    ):
        assert requirement in system
    assert "reward" not in system and "gamma" not in system


def test_history_limit_keeps_full_reference_and_recent_feedback():
    messages = build_generation_messages(
        **arguments(),
        history=[{"old": "x" * 1000}, {"reflection": "recent plan"}],
        history_char_limit=50,
    )
    text = messages[1]["content"]
    assert "recent plan" in text
    assert "leading context omitted" in text
    assert "x" * 1000 not in text
    assert "def module_fn(x, *, scale=2)" in text


def test_long_reflection_does_not_hide_feedback_or_plan_prefix():
    messages = build_generation_messages(
        **arguments(),
        history=[
            {
                "candidate": "large candidate " * 5000,
                "feedback": {"message": "missing stride argument"},
                "reflection": "First repair the stride argument. " + "later context " * 5000,
            }
        ],
        history_char_limit=16000,
    )
    text = messages[1]["content"]
    assert "missing stride argument" in text
    assert "First repair the stride argument" in text


@pytest.mark.parametrize("limit", [0, -1])
def test_history_requires_positive_limit(limit):
    with pytest.raises(ValueError):
        build_generation_messages(**arguments(), history_char_limit=limit)


def test_reflection_uses_actual_feedback_and_preserves_candidate():
    candidate = 'def module_fn(x):\n    return "```example```"\n'
    feedback = {
        "compile_success": True,
        "correctness_success": False,
        "message": "wrong output shape",
    }
    messages = build_reflection_messages(**arguments(), candidate_text=candidate, feedback=feedback)
    assert json.dumps(candidate) in messages[1]["content"]
    assert "wrong output shape" in messages[1]["content"]
    assert "Explain the observed failure" in messages[1]["content"]
    assert "not a replacement implementation" in messages[0]["content"]


def test_variant_reflection_uses_latency_without_speedup_requirement():
    messages = build_reflection_messages(
        **arguments(),
        candidate_text="candidate",
        seek_variant=True,
        feedback={"performance_cases": [{"baseline_ms": 1, "candidate_ms": 2, "speedup": 0.5}]},
    )
    text = messages[1]["content"]
    assert '"speedup": 0.5' in text
    assert "different correct implementation" in text
    assert "A correct slower candidate remains valid" in text
    assert "No universal speedup threshold" in text
