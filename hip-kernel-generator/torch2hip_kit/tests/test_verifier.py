# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import torch2hip_kit.verifier as verifier


def test_comparison_rejects_broadcast_output_shapes() -> None:
    matches, detail = verifier._compare_outputs(torch.ones(2, 3), torch.ones(3), 1e-4, 1e-4)
    assert not matches
    assert "shape mismatch" in detail


def test_comparison_keeps_hip_nan_policy() -> None:
    values = torch.tensor([float("nan"), 1.0])
    assert verifier._compare_outputs(values, values.clone(), 1e-4, 1e-4) == (True, "")


@pytest.mark.parametrize(
    "value",
    [
        torch.arange(12).reshape(3, 4)[:, ::2],
        torch.arange(8)[2::2],
        torch.arange(4).expand(3, 4),
        torch.randn(4, requires_grad=True) * 2,
    ],
)
def test_cloned_inputs_keep_layout_and_independent_storage(value: torch.Tensor) -> None:
    clone = verifier._clone_value(value)
    assert clone.stride() == value.stride()
    assert clone.storage_offset() == value.storage_offset()
    assert clone.requires_grad == value.requires_grad
    assert torch.equal(clone, value)
    assert clone.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()


def test_device_transfer_keeps_slice_layout() -> None:
    value = torch.arange(24).reshape(4, 6)[1:, 1::2]
    moved = verifier._move_to_device(value, torch.device("meta"))
    assert moved.device.type == "meta"
    assert moved.shape == value.shape
    assert moved.stride() == value.stride()
    assert moved.storage_offset() == value.storage_offset()


class OriginalModel(torch.nn.Module):
    def forward(self, x):
        return x + 1

    def to(self, device):
        # The tests exercise Python orchestration with CPU tensors.
        return self


class FunctionalModel(OriginalModel):
    def forward(self, x, fn=lambda x: x + 1):
        return fn(x)


class IgnoredFunctionModel(OriginalModel):
    def forward(self, x, fn=lambda x: x + 1):
        return x + 1


@pytest.fixture
def check_candidate(monkeypatch, tmp_path: Path):
    get_inputs = Mock(return_value=[torch.tensor([1.0, 2.0])])
    original = SimpleNamespace(Model=OriginalModel, get_inputs=get_inputs, get_init_inputs=lambda: [])
    monkeypatch.setattr(verifier.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(verifier, "_set_seed", lambda seed: None)
    monkeypatch.setattr(verifier, "_move_to_device", lambda value, device: value)

    def run(functional_model=FunctionalModel, hip_fn=None, timing_error=None):
        functional = SimpleNamespace(
            Model=functional_model, module_fn=lambda x: x + 1,
            get_inputs=lambda: pytest.fail("The verifier must use the original inputs."),
            get_init_inputs=lambda: [],
        )
        modules = iter([original, functional])
        monkeypatch.setattr(verifier, "load_python_module", lambda *args: next(modules))
        compiled_fn = hip_fn if hip_fn is not None else Mock(side_effect=lambda x: x + 1)
        compiler = Mock(return_value=compiled_fn)
        monkeypatch.setattr(verifier, "load_hip_forward", compiler)
        if timing_error is not None:
            latency = Mock(side_effect=timing_error)
        else:
            latency = Mock(side_effect=[4.0, 2.0])
        monkeypatch.setattr(verifier, "_measure_latency_ms", latency)
        result = verifier.verify_candidate(
            tmp_path / "original.py", tmp_path / "functional.py", tmp_path / "candidate.hip",
            build_dir=tmp_path / "build", seed=1234, rtol=1e-4, atol=1e-4,
            perf_warmup=0, perf_iterations=1,
        )
        return result, compiler, compiled_fn, get_inputs, latency

    return run


def test_correct_kernel_runs_on_one_original_input_set(check_candidate) -> None:
    result, compiler, hip_fn, get_inputs, latency = check_candidate()
    assert result.success
    assert result.compile_success and result.correctness_success
    assert result.speedup == 2.0
    compiler.assert_called_once()
    hip_fn.assert_called_once()
    get_inputs.assert_called_once()
    assert latency.call_count == 2


def test_verifier_rejects_wrapper_that_ignores_hip_function(check_candidate) -> None:
    result, _, hip_fn, _, latency = check_candidate(functional_model=IgnoredFunctionModel)
    assert not result.success
    assert result.compile_success
    assert not result.correctness_success
    assert "did not call" in result.message
    hip_fn.assert_not_called()
    latency.assert_not_called()


def test_verifier_rejects_missing_fn_interface_before_compilation(check_candidate) -> None:
    result, compiler, _, _, _ = check_candidate(functional_model=OriginalModel)
    assert not result.success
    assert not result.compile_success
    assert "explicit `fn`" in result.message
    compiler.assert_not_called()


def test_runtime_error_preserves_compile_result(check_candidate) -> None:
    result, _, _, _, _ = check_candidate(hip_fn=Mock(side_effect=RuntimeError("kernel failed")))
    assert not result.success
    assert result.compile_success
    assert not result.correctness_success
    assert "kernel failed" in result.message


def test_timing_error_preserves_correctness_result(check_candidate) -> None:
    result, _, _, _, _ = check_candidate(timing_error=RuntimeError("timer failed"))
    assert not result.success
    assert result.compile_success and result.correctness_success
    assert "timer failed" in result.message


def test_wrong_kernel_does_not_reach_timing(check_candidate) -> None:
    result, _, _, _, latency = check_candidate(hip_fn=lambda x: x + 2)
    assert not result.success
    assert result.compile_success
    assert not result.correctness_success
    latency.assert_not_called()
