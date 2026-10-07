# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import py_hip_kernel2kernel_kit.verifier as verifier_module
from py_hip_kernel2kernel_kit.verifier import VerificationContext, VerificationTimeoutError


@pytest.fixture(autouse=True)
def cpu_only_verification(monkeypatch):
    monkeypatch.setattr(verifier_module, "_set_seed", lambda seed: None)
    monkeypatch.setattr(verifier_module, "_best_effort_cuda_cleanup", lambda: None)


def _make_context() -> VerificationContext:
    def functional_model(*args, **kwargs):
        return kwargs["fn"]()

    return VerificationContext(
        functional_model=functional_model,
        original_model=None,
        expected_output=1,
        forward_args=[],
        forward_kwargs={},
        seed=1234,
        rtol=1e-4,
        atol=1e-4,
        perf_warmup=1,
        perf_iterations=2,
        module_latency_ms=3.0,
        baseline_latency_ms=2.0,
        offload_arch=None,
        hip_compile_timeout_seconds=1.0,
        execution_timeout_seconds=1.0,
        benchmark_timeout_seconds=1.0,
    )


def test_verify_candidate_reports_compile_timeout(monkeypatch, tmp_path: Path) -> None:
    context = _make_context()

    monkeypatch.setattr(
        verifier_module,
        "load_hip_forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            VerificationTimeoutError("compile candidate HIP extension", 1.0)
        ),
    )

    result = context.verify_candidate(
        tmp_path / "candidate.hip",
        build_dir=tmp_path / "build",
    )

    assert result.success is False
    assert result.compile_success is False
    assert result.correctness_success is False
    assert "compile candidate HIP extension timed out" in result.message


def test_verify_candidate_reports_benchmark_timeout_after_correctness(monkeypatch, tmp_path: Path) -> None:
    context = _make_context()

    monkeypatch.setattr(verifier_module, "load_hip_forward", lambda *args, **kwargs: lambda *a, **k: 1)
    monkeypatch.setattr(
        verifier_module,
        "_measure_latency_ms",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            VerificationTimeoutError("benchmark candidate HIP function", 1.0)
        ),
    )

    result = context.verify_candidate(
        tmp_path / "candidate.hip",
        build_dir=tmp_path / "build",
    )

    assert result.success is False
    assert result.compile_success is True
    assert result.correctness_success is True
    assert "benchmark candidate HIP function timed out" in result.message


def test_verify_candidate_rejects_wrapper_that_ignores_hip(monkeypatch, tmp_path: Path) -> None:
    context = _make_context()
    context.functional_model = lambda *args, **kwargs: 1
    monkeypatch.setattr(verifier_module, "load_hip_forward", lambda *args, **kwargs: lambda: -1)

    def unexpected_benchmark(*args, **kwargs):
        pytest.fail("The verifier must reject the wrapper before timing it.")

    monkeypatch.setattr(verifier_module, "_measure_latency_ms", unexpected_benchmark)
    result = context.verify_candidate(tmp_path / "candidate.hip", build_dir=tmp_path / "build")

    assert not result.success
    assert result.compile_success
    assert not result.correctness_success
    assert "did not call the candidate HIP forward function" in result.message


def test_verify_candidate_uses_hip_result_and_baseline_include_paths(monkeypatch, tmp_path: Path) -> None:
    context = _make_context()
    context.extra_include_paths = (str(tmp_path / "baseline"), str(tmp_path / "baseline" / "include"))
    captured = {}
    calls = []

    def load_candidate(*args, **kwargs):
        captured.update(kwargs)

        def candidate():
            calls.append(True)
            return 1

        return candidate

    monkeypatch.setattr(verifier_module, "load_hip_forward", load_candidate)
    monkeypatch.setattr(verifier_module, "_measure_latency_ms", lambda *args, **kwargs: 1.0)
    result = context.verify_candidate(tmp_path / "candidate.hip", build_dir=tmp_path / "build")

    assert result.success
    assert result.correctness_success
    assert len(calls) == 1
    assert captured["extra_include_paths"] == context.extra_include_paths
    assert result.speedup_vs_baseline == 2.0


@pytest.mark.parametrize(
    ("expected", "actual"),
    [
        (torch.ones(2, 3), torch.ones(3)),
        (torch.ones(2, 3), torch.ones(1, 3)),
        (torch.ones(()), torch.ones(1)),
    ],
)
def test_compare_outputs_rejects_broadcastable_shapes(expected, actual) -> None:
    matches, detail = verifier_module._compare_outputs({"result": expected}, {"result": actual}, 1e-4, 1e-4)
    assert not matches
    assert "output['result'] tensor shape mismatch" in detail


def test_compare_outputs_preserves_tolerance_and_nan_policy() -> None:
    expected = torch.tensor([float("nan"), 1.0])
    actual = torch.tensor([float("nan"), 1.0001])
    assert verifier_module._compare_outputs(expected, actual, 1e-4, 1e-4) == (True, "")


@pytest.mark.parametrize("layout", ["sliced", "expanded", "nonleaf"])
def test_clone_inputs_preserves_layout_and_independent_storage(layout) -> None:
    if layout == "sliced":
        source = torch.arange(12.0).reshape(3, 4)[1:, ::2]
    elif layout == "expanded":
        source = torch.arange(4.0).expand(3, 4)
    else:
        source = torch.arange(4.0, requires_grad=True) * 2
    original_values = source.detach().clone()

    result = verifier_module._clone_value({"values": [source]})["values"][0]
    assert torch.equal(result, source)
    assert result.shape == source.shape
    assert result.stride() == source.stride()
    assert result.storage_offset() == source.storage_offset()
    assert result.requires_grad == source.requires_grad
    assert result.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    with torch.no_grad():
        result[(0,) * result.ndim] = -100
    assert torch.equal(source, original_values)


@pytest.mark.parametrize("expanded", [False, True])
def test_device_transfer_preserves_layout_without_gpu(monkeypatch, expanded) -> None:
    source = torch.arange(12.0).reshape(3, 4)[1:, ::2]
    if expanded:
        source = torch.arange(4.0).expand(3, 4)
    requested = []
    tensor_to = torch.Tensor.to

    def cpu_transfer(tensor, device):
        requested.append((device, tensor.shape, tensor.stride()))
        return tensor_to(tensor, device="cpu", copy=True)

    monkeypatch.setattr(torch.Tensor, "to", cpu_transfer)
    result = verifier_module._move_to_device(source, torch.device("cuda"))

    assert requested == [(torch.device("cuda"), (source.untyped_storage().nbytes() // source.element_size(),), (1,))]
    assert torch.equal(result, source)
    assert result.stride() == source.stride()
    assert result.storage_offset() == source.storage_offset()
    assert result.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


@pytest.mark.parametrize("uses_hip", [False, True])
def test_prepare_context_requires_baseline_hip_call(monkeypatch, tmp_path: Path, uses_hip) -> None:
    class OriginalModel:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, value):
            return value + 1

    class FunctionalModel(OriginalModel):
        def __call__(self, value, *, fn):
            return fn(value) if uses_hip else value + 1

    original = SimpleNamespace(Model=OriginalModel, get_inputs=lambda: [torch.ones(3)], get_init_inputs=lambda: [])
    functional = SimpleNamespace(
        Model=FunctionalModel, module_fn=lambda value: value + 1,
        get_inputs=original.get_inputs, get_init_inputs=original.get_init_inputs,
    )
    monkeypatch.setattr(verifier_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(verifier_module, "_move_to_device", lambda value, device: value)
    monkeypatch.setattr(
        verifier_module, "load_python_module",
        lambda path, name: original if path.name == "original.py" else functional,
    )
    monkeypatch.setattr(verifier_module, "load_hip_forward", lambda *args, **kwargs: lambda value: value + 1)
    monkeypatch.setattr(verifier_module, "_measure_latency_ms", lambda *args, **kwargs: 1.0)
    baseline_path = tmp_path / "baseline" / "kernel.hip"
    result, context = verifier_module.prepare_verification_context(
        original_module_path=tmp_path / "original.py",
        functional_module_path=tmp_path / "functional.py",
        baseline_hip_path=baseline_path,
        baseline_build_dir=tmp_path / "build",
        seed=1234, rtol=1e-4, atol=1e-4, perf_warmup=0, perf_iterations=1,
    )

    assert result.success is uses_hip
    assert result.compile_success
    assert result.correctness_success is uses_hip
    if uses_hip:
        assert context is not None
        assert context.extra_include_paths == (
            str(baseline_path.parent), str(baseline_path.parent / "include"),
        )
    else:
        assert context is None
        assert "did not call the baseline HIP forward function" in result.message
