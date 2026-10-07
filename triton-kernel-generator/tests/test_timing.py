# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Check timing behavior without importing or executing a GPU runtime."""

from __future__ import annotations

import builtins
import importlib
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from triton_kernel_gen import timing
from triton_kernel_gen.contracts import VerificationSettings


@pytest.fixture
def gpu_runtime(monkeypatch):
    events = []

    @contextmanager
    def device_context(index):
        events.append(("device_enter", index))
        try:
            yield
        finally:
            events.append(("device_exit", index))

    device = Mock(side_effect=device_context)
    synchronize = Mock(side_effect=lambda: events.append("synchronize"))
    do_bench = Mock(return_value=[1.25])

    def benchmark(*args, **kwargs):
        events.append("do_bench")
        return do_bench.return_value

    do_bench.side_effect = benchmark
    torch_module = ModuleType("torch")
    torch_module.cuda = SimpleNamespace(device=device, synchronize=synchronize)
    triton_module = ModuleType("triton")
    testing_module = ModuleType("triton.testing")
    testing_module.do_bench = do_bench
    triton_module.testing = testing_module
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "triton", triton_module)
    monkeypatch.setitem(sys.modules, "triton.testing", testing_module)
    return SimpleNamespace(events=events, device=device, synchronize=synchronize, do_bench=do_bench)


def _block_gpu_imports(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "triton"}:
            raise AssertionError(f"The test must not import {name}.")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_import_requires_no_gpu_packages(monkeypatch):
    _block_gpu_imports(monkeypatch)
    importlib.reload(timing)


def test_first_call_and_synchronization_precede_benchmark(gpu_runtime):
    fn = Mock(side_effect=lambda: gpu_runtime.events.append("fn"))
    settings = VerificationSettings(gpu=3, warmup_ms=7.5, measure_ms=40.0)

    assert timing.bench_ms(fn, settings) == 1.25

    fn.assert_called_once_with()
    gpu_runtime.device.assert_called_once_with(3)
    gpu_runtime.synchronize.assert_called_once_with()
    gpu_runtime.do_bench.assert_called_once_with(fn, warmup=7.5, rep=40.0, quantiles=[0.5])
    assert gpu_runtime.events == [
        ("device_enter", 3),
        "fn",
        "synchronize",
        "do_bench",
        ("device_exit", 3),
    ]


def test_default_durations_reach_benchmark(gpu_runtime):
    fn = Mock()

    timing.bench_ms(fn, VerificationSettings())

    gpu_runtime.do_bench.assert_called_once_with(fn, warmup=25.0, rep=100.0, quantiles=[0.5])


def test_zero_warmup_is_valid(gpu_runtime):
    fn = Mock()

    assert timing.bench_ms(fn, VerificationSettings(warmup_ms=0.0)) == 1.25

    gpu_runtime.do_bench.assert_called_once_with(fn, warmup=0.0, rep=100.0, quantiles=[0.5])


@pytest.mark.parametrize("median", [2.5, 2, [2.5], (2.5,)])
def test_accepts_scalar_or_one_median(gpu_runtime, median):
    gpu_runtime.do_bench.return_value = median

    actual = timing.bench_ms(Mock(), VerificationSettings())

    assert isinstance(actual, float)
    assert actual == (2.0 if median == 2 else 2.5)


@pytest.mark.parametrize("field", ["warmup_ms", "measure_ms"])
@pytest.mark.parametrize(
    "value", [-1.0, float("nan"), float("inf"), -float("inf"), None, "2", True]
)
def test_rejects_invalid_durations_before_imports_or_callback(monkeypatch, field, value):
    _block_gpu_imports(monkeypatch)
    fn = Mock()

    with pytest.raises(ValueError, match=field):
        timing.bench_ms(fn, VerificationSettings(**{field: value}))

    fn.assert_not_called()


def test_rejects_zero_measurement_before_imports_or_callback(monkeypatch):
    _block_gpu_imports(monkeypatch)
    fn = Mock()

    with pytest.raises(ValueError, match="measure_ms"):
        timing.bench_ms(fn, VerificationSettings(measure_ms=0.0))

    fn.assert_not_called()


@pytest.mark.parametrize(
    "median",
    [
        0.0,
        -1.0,
        float("nan"),
        float("inf"),
        -float("inf"),
        None,
        "2",
        True,
        [0.0],
        [float("nan")],
        [float("inf")],
        [],
        [1.0, 2.0],
        [[1.0]],
    ],
)
def test_rejects_invalid_benchmark_results(gpu_runtime, median):
    gpu_runtime.do_bench.return_value = median

    with pytest.raises(ValueError, match="median"):
        timing.bench_ms(Mock(), VerificationSettings())


@pytest.mark.parametrize("stage", ["fn", "synchronize", "do_bench"])
def test_runtime_errors_propagate_and_restore_device(gpu_runtime, stage):
    failure = RuntimeError(f"{stage} failed")
    fn = Mock()
    operation = fn if stage == "fn" else getattr(gpu_runtime, stage)
    operation.side_effect = failure

    with pytest.raises(RuntimeError) as raised:
        timing.bench_ms(fn, VerificationSettings(gpu=2))

    assert raised.value is failure
    assert gpu_runtime.events[-1] == ("device_exit", 2)
    if stage == "fn":
        gpu_runtime.synchronize.assert_not_called()
    if stage != "do_bench":
        gpu_runtime.do_bench.assert_not_called()
