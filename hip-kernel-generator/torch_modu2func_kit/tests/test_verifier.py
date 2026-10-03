# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import os
from pathlib import Path

import pytest
import torch

from torch_modu2func_kit.verifier import _clone_value, _compare_outputs, _load_module, verify_candidate


ORIGINAL_CODE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return x + self.bias


bias_value = 3.0


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [bias_value]
"""


GENERATED_CODE = """
import torch
import torch.nn as nn


def module_fn(x, bias):
    return x + bias


class Model(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x, fn=module_fn):
        return fn(x, self.bias)


bias_value = 3.0


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [bias_value]
"""


def test_verify_candidate_accepts_equivalent_code(tmp_path: Path) -> None:
    original_path = tmp_path / "original.py"
    generated_path = tmp_path / "generated.py"
    original_path.write_text(ORIGINAL_CODE, encoding="utf-8")
    generated_path.write_text(GENERATED_CODE, encoding="utf-8")

    result = verify_candidate(
        original_path,
        generated_path,
        seed=1234,
        rtol=1e-4,
        atol=1e-4,
    )

    assert result.success is True


@pytest.mark.parametrize(
    ("signature", "body", "message"),
    [
        ("self, x", "return module_fn(x, self.bias)", "must accept the keyword parameter fn"),
        ("self, x, **kwargs", "return module_fn(x, self.bias)", "must accept the keyword parameter fn"),
        ("self, x, fn=module_fn", "return module_fn(x, self.bias)", "did not call the injected fn"),
        (
            "self, x, fn=module_fn",
            "return fn(x, self.bias) + (0 if fn is module_fn else 1)",
            "Injected fn: output tensor mismatch",
        ),
        (
            "self, x, fn=module_fn",
            "return fn(x, self.bias) + (1 if fn is module_fn else 0)",
            "output tensor mismatch",
        ),
    ],
)
def test_verify_candidate_checks_default_and_injected_fn(
    tmp_path: Path, signature: str, body: str, message: str
) -> None:
    original_path = tmp_path / "original.py"
    generated_path = tmp_path / "generated.py"
    original_path.write_text(ORIGINAL_CODE, encoding="utf-8")
    candidate = GENERATED_CODE.replace("self, x, fn=module_fn", signature)
    candidate = candidate.replace("return fn(x, self.bias)", body)
    generated_path.write_text(candidate, encoding="utf-8")

    result = verify_candidate(original_path, generated_path, seed=1234, rtol=1e-4, atol=1e-4)

    assert result.success is False
    assert message in result.message


def test_verify_candidate_uses_one_original_input_draw(tmp_path: Path) -> None:
    original_path = tmp_path / "original.py"
    generated_path = tmp_path / "generated.py"
    original = ORIGINAL_CODE.replace(
        "def get_inputs():",
        "input_draws = 0\ndef get_inputs():\n"
        "    global input_draws\n    input_draws += 1\n    assert input_draws == 1",
    )
    generated = GENERATED_CODE.replace(
        "def get_inputs():", "def get_inputs():\n    raise AssertionError('Candidate inputs must not run')"
    )
    original_path.write_text(original, encoding="utf-8")
    generated_path.write_text(generated, encoding="utf-8")

    result = verify_candidate(original_path, generated_path, seed=1234, rtol=1e-4, atol=1e-4)

    assert result.success is True, result.message


@pytest.mark.parametrize("actual", [torch.ones(3), torch.ones(1, 3), torch.ones(2, 1)])
def test_compare_outputs_rejects_broadcast_shapes(actual: torch.Tensor) -> None:
    matches, message = _compare_outputs({"value": torch.ones(2, 3)}, {"value": actual}, 1e-4, 1e-4)

    assert matches is False
    assert "output['value'] shape mismatch" in message


def test_compare_outputs_retains_nan_tolerance() -> None:
    expected = torch.tensor([float("nan"), 1.0])
    actual = torch.tensor([float("nan"), 1.0001])

    assert _compare_outputs(expected, actual, 1e-4, 1e-4)[0] is True


@pytest.mark.parametrize("layout", ["slice", "expand", "transpose", "nonleaf"])
def test_clone_value_preserves_tensor_layout(layout: str) -> None:
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    if layout == "slice":
        value = base[1:, 1::2]
    elif layout == "expand":
        value = base[1:2].expand(4, 6)
    elif layout == "transpose":
        value = base.T
    else:
        value = (base.requires_grad_() * 2)[1:, 1::2]

    cloned = _clone_value({"values": [value]})["values"][0]

    assert torch.equal(cloned, value)
    assert cloned.shape == value.shape
    assert cloned.stride() == value.stride()
    assert cloned.storage_offset() == value.storage_offset()
    assert cloned.requires_grad == value.requires_grad
    assert cloned.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
    assert cloned.is_leaf


def test_load_module_ignores_stale_bytecode(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    initial_stat = source.stat()
    spec = importlib.util.spec_from_file_location("cached_source", source)
    cached = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cached)
    assert cached.VALUE == 1
    source.write_text("VALUE = 2\n", encoding="utf-8")
    os.utime(source, ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns))

    current = _load_module(source, "current_source")

    assert current.VALUE == 2


def test_load_module_preserves_source_annotation_semantics(tmp_path: Path) -> None:
    source = tmp_path / "annotations.py"
    source.write_text("def f(x: int):\n    return x\n", encoding="utf-8")

    current = _load_module(source, "source_annotations")

    assert current.f.__annotations__["x"] is int
