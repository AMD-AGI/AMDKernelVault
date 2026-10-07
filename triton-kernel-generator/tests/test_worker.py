# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Exercise worker contracts with real CPU tensors and a fake GPU runtime.

The fake Triton runtime tests launch accounting. It does not compile GPU code.
"""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import asdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from triton_kernel_gen import worker
from triton_kernel_gen.contracts import Case, VerificationSettings
from triton_kernel_gen.errors import ReferenceValidationError
from triton_kernel_gen.tensor_utils import compare_outputs, load_tree, save_tree

_HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.\n"
)
_CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def no_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        Mock(side_effect=AssertionError("A CPU test attempted GPU synchronization.")),
    )


def _source(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / f"{name}.py"
    path.write_text(_HEADER + textwrap.dedent(source), encoding="utf-8")
    return path


def _module(tmp_path: Path, name: str, source: str) -> ModuleType:
    return worker.load_module(_source(tmp_path, name, source), f"_worker_test_{name}")


def _simple_pair(
    tmp_path: Path,
    *,
    original_body: str = "return x * 2",
    function_body: str = "return x * 2",
    forward_body: str = "return fn(x)",
    signature: str = "self, x, fn=module_fn",
):
    original = _module(
        tmp_path,
        "original",
        "import torch\nclass Model(torch.nn.Module):\n"
        "    def forward(self, x):\n" + textwrap.indent(original_body, "        ") + "\n",
    )
    functional = _module(
        tmp_path,
        "functional",
        "import torch\ndef module_fn(x):\n"
        + textwrap.indent(function_body, "    ")
        + f"\nclass Model(torch.nn.Module):\n    def forward({signature}):\n"
        + textwrap.indent(forward_body, "        ")
        + "\n",
    )
    return original, functional


def _validate(pair, case=None, *, seed=17, settings=None):
    return worker.validate_reference_case(
        *pair,
        case or Case("vector", args=(torch.arange(4, dtype=torch.float32),)),
        settings or VerificationSettings(),
        seed,
        _CPU,
    )


def test_original_default_and_injected_outputs_agree(tmp_path):
    call, expected, checks = _validate(_simple_pair(tmp_path))

    torch.testing.assert_close(expected, torch.tensor([0.0, 2.0, 4.0, 6.0]), atol=0, rtol=0)
    torch.testing.assert_close(
        call["args"][0], torch.arange(4, dtype=torch.float32), atol=0, rtol=0
    )
    assert call["kwargs"] == {}
    assert set(checks) == {"default", "injected", "native_outputs"}
    for result in (checks["default"], checks["injected"]):
        assert result["success"]
        assert result["max_abs_error"] == 0.0
        assert result["tensor_count"] == 1
    assert set(checks["native_outputs"]) == {"original", "repeat", "default", "injected", "replay"}
    for metadata in checks["native_outputs"].values():
        assert metadata["tensors"][0]["device"] == "cpu"


@pytest.mark.parametrize("field", ["atol", "rtol"])
@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("source", ["case", "settings"])
def test_reference_rejects_boolean_tolerances(tmp_path, field, value, source):
    case_options = {field: value} if source == "case" else {}
    setting_options = {field: value} if source == "settings" else {}
    case = Case("invalid_tolerance", args=(torch.ones(4),), **case_options)
    settings = VerificationSettings(**setting_options)

    with pytest.raises(
        ReferenceValidationError, match="Case tolerances must be finite, nonnegative numbers"
    ):
        _validate(_simple_pair(tmp_path), case, settings=settings)


@pytest.mark.parametrize("metadata", [[], [["source", "fixture"]]])
def test_reference_rejects_list_metadata(tmp_path, metadata):
    case = Case("invalid_metadata", args=(torch.ones(4),), metadata=metadata)

    with pytest.raises(ReferenceValidationError, match="Case metadata must be a dictionary"):
        _validate(_simple_pair(tmp_path), case)


@pytest.mark.parametrize(
    ("forward_body", "message"),
    [
        ("return module_fn(x)", "call fn exactly once"),
        ("fn(x)\nreturn fn(x)", "forwarding-only"),
        ("fn(x)\nreturn module_fn(x)", "forwarding-only"),
        ("return fn(x).clone()", "return the fn output object directly"),
    ],
    ids=["bypass", "two_calls", "ignored_result", "copied_result"],
)
def test_wrapper_must_call_and_return_injected_function(tmp_path, forward_body, message):
    with pytest.raises(ReferenceValidationError, match=message):
        _validate(_simple_pair(tmp_path, forward_body=forward_body))


@pytest.mark.parametrize(
    "signature",
    ["self, x", "self, x, **kwargs", "self, x, fn", "self, x, fn=module_fn, /"],
    ids=["missing", "implicit_kwargs", "missing_default", "positional_only"],
)
def test_wrapper_requires_explicit_fn_keyword_and_default(tmp_path, signature):
    with pytest.raises(ReferenceValidationError, match="explicit fn keyword"):
        _validate(_simple_pair(tmp_path, signature=signature, forward_body="return module_fn(x)"))


@pytest.mark.parametrize(
    ("function_body", "message"),
    [
        ("return (x * 2).reshape(2, 2)", "tensor shapes differ"),
        ("return (x * 2).to(torch.float64)", "tensor dtypes differ"),
        ("return x * 2 + 1", "tensor values differ"),
    ],
)
def test_default_reference_rejects_shape_dtype_and_value_changes(tmp_path, function_body, message):
    with pytest.raises(
        ReferenceValidationError, match=f"default functional reference differs.*{message}"
    ):
        _validate(_simple_pair(tmp_path, function_body=function_body))


def test_injected_reference_must_agree_even_when_default_matches(tmp_path):
    pair = _simple_pair(tmp_path, forward_body="return fn(x if fn is module_fn else x + 1)")

    with pytest.raises(ReferenceValidationError, match="forwarding-only"):
        _validate(pair)


@pytest.mark.parametrize("output", ["[]", "()", "{}", "None", "''", "torch.empty(0)"])
def test_empty_original_output_cannot_establish_correctness(tmp_path, output):
    pair = _simple_pair(
        tmp_path, original_body=f"return {output}", function_body=f"return {output}"
    )

    with pytest.raises(ReferenceValidationError, match="original output is invalid.*empty"):
        _validate(pair)


def test_model_initialization_and_forward_randomness_use_the_case_seed(tmp_path):
    original = _module(
        tmp_path,
        "seeded_original",
        """
        import torch
        weights = []
        class Model(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(width))
                weights.append(self.weight.detach().clone())
            def forward(self, x):
                return x * self.weight + torch.rand_like(x)
    """,
    )
    functional = _module(
        tmp_path,
        "seeded_functional",
        """
        import torch
        weights = []
        def module_fn(x, weight):
            return x * weight + torch.rand_like(x)
        class Model(torch.nn.Module):
            def __init__(self, width):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.rand(width))
                weights.append(self.weight.detach().clone())
            def forward(self, x, *, fn=module_fn):
                return fn(x, self.weight)
    """,
    )
    case = Case("seeded", args=(torch.ones(4),), init_args=(4,), atol=0, rtol=0)

    first_call, first_output, _ = _validate((original, functional), case, seed=23)
    second_call, second_output, _ = _validate((original, functional), case, seed=23)
    _, other_output, _ = _validate((original, functional), case, seed=24)

    assert torch.equal(first_output, second_output)
    assert torch.equal(first_call["args"][1], second_call["args"][1])
    assert not torch.equal(first_output, other_output)
    for weight in original.weights[:4] + functional.weights[:4]:
        assert torch.equal(weight, first_call["args"][1])
    assert len(original.weights) == len(functional.weights) == 6
    assert not first_call["args"][1].requires_grad


def test_original_repeat_must_match_exactly_despite_loose_tolerances(tmp_path):
    original = _module(
        tmp_path,
        "nonrepeatable",
        """
        import torch
        calls = 0
        class Model(torch.nn.Module):
            def forward(self, x):
                global calls
                calls += 1
                return x * 2 + calls * 0.000001
    """,
    )
    _, functional = _simple_pair(tmp_path)

    with pytest.raises(ReferenceValidationError, match="not deterministic under its fixed seed"):
        _validate((original, functional), settings=VerificationSettings(atol=1.0, rtol=1.0))


def test_captured_call_must_replay_reference_without_wrapper_rng_effects(tmp_path):
    pair = _simple_pair(
        tmp_path,
        original_body="torch.rand_like(x)\nreturn x + torch.rand_like(x)",
        function_body="return x + torch.rand_like(x)",
        forward_body="torch.rand_like(x)\nreturn fn(x)",
    )

    with pytest.raises(ReferenceValidationError, match="forwarding-only"):
        _validate(pair)


@pytest.mark.parametrize(
    ("original_body", "function_body"),
    [
        ("x.add_(1)\nreturn x * 2", "return x * 2"),
        ("return x * 2", "x.add_(1)\nreturn x * 2"),
        (
            "torch.empty(0, dtype=x.dtype).set_(x.untyped_storage())[1] = 99\nreturn x * 2",
            "return x * 2",
        ),
    ],
    ids=["original_input", "functional_input", "storage_gap"],
)
def test_references_reject_input_storage_mutation(tmp_path, original_body, function_body):
    base = torch.arange(8, dtype=torch.float32)
    case = Case("strided", args=(base[::2],))

    with pytest.raises(ReferenceValidationError, match="mutated tensor storage"):
        _validate(
            _simple_pair(tmp_path, original_body=original_body, function_body=function_body), case
        )
    assert torch.equal(base, torch.arange(8, dtype=torch.float32))


@pytest.mark.parametrize("persistent", [True, False])
def test_reference_rejects_model_buffer_mutation(tmp_path, persistent):
    original = _module(
        tmp_path,
        "stateful",
        f"""
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("calls", torch.tensor(0), persistent={persistent!r})
            def forward(self, x):
                self.calls.add_(1)
                return x * 2
    """,
    )
    _, functional = _simple_pair(tmp_path)

    with pytest.raises(ReferenceValidationError, match="reference model.*mutated tensor storage"):
        _validate((original, functional))


@pytest.mark.parametrize("attribute", ["calls", "_calls"])
def test_reference_rejects_python_counter_mutation(tmp_path, attribute):
    original = _module(
        tmp_path,
        "python_state",
        f"""
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.{attribute} = 0
            def forward(self, x):
                self.{attribute} += 1
                return x * 2
    """,
    )
    _, functional = _simple_pair(tmp_path)

    with pytest.raises(ReferenceValidationError, match="reference model.*mutated"):
        _validate((original, functional))


@pytest.mark.parametrize("mutate", [False, True], ids=["unchanged_list", "list_to_tuple"])
def test_reference_preserves_model_attribute_container_types(tmp_path, mutate):
    original = _module(
        tmp_path,
        "container_state",
        f"""
        import torch
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.axes = [2, 3]
            def forward(self, x):
                if {mutate!r}:
                    self.axes = tuple(self.axes)
                return x * 2
    """,
    )
    _, functional = _simple_pair(tmp_path)

    if mutate:
        with pytest.raises(ReferenceValidationError, match="reference model.*mutated"):
            _validate((original, functional))
    else:
        _, expected, checks = _validate((original, functional))
        assert torch.equal(expected, torch.arange(4, dtype=torch.float32) * 2)
        assert checks["default"]["success"] and checks["injected"]["success"]


def test_reference_rejects_unsupported_mutable_model_attributes(tmp_path):
    original = _module(
        tmp_path,
        "unsupported_state",
        """
        import torch
        class CustomState:
            pass
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.state = CustomState()
            def forward(self, x):
                return x * 2
    """,
    )
    _, functional = _simple_pair(tmp_path)

    with pytest.raises(
        ReferenceValidationError, match="Unsupported model attribute type: CustomState"
    ):
        _validate((original, functional))


def test_reference_rejects_changed_input_alias_graph(tmp_path):
    original = _module(
        tmp_path,
        "alias_mutation",
        """
        import torch
        class Model(torch.nn.Module):
            def forward(self, values):
                values["second"] = values["second"].clone()
                return values["first"] * 2
    """,
    )
    functional = _module(
        tmp_path,
        "alias_functional",
        """
        import torch
        def module_fn(values):
            return values["first"] * 2
        class Model(torch.nn.Module):
            def forward(self, values, fn=module_fn):
                return fn(values)
    """,
    )
    tensor = torch.arange(4, dtype=torch.float32)
    case = Case("alias", args=({"first": tensor, "second": tensor},))

    with pytest.raises(ReferenceValidationError, match="alias|mutat"):
        _validate((original, functional), case)


def test_captured_call_preserves_strides_offsets_aliases_and_keywords(tmp_path):
    original = _module(
        tmp_path,
        "view_original",
        """
        import torch
        class Model(torch.nn.Module):
            def forward(self, x, other, *, again, scale):
                assert x is again
                assert x.untyped_storage()._cdata == other.untyped_storage()._cdata
                return (x + other) * scale
    """,
    )
    functional = _module(
        tmp_path,
        "view_functional",
        """
        import torch
        def module_fn(x, other, *, again, scale):
            assert x is again
            assert x.untyped_storage()._cdata == other.untyped_storage()._cdata
            return (x + other) * scale
        class Model(torch.nn.Module):
            def forward(self, x, other, *, again, scale, fn=module_fn):
                return fn(x, other, again=again, scale=scale)
    """,
    )
    base = torch.arange(30, dtype=torch.float32)
    view = base.as_strided((2, 3), (7, 2), 2)
    other = base.as_strided((2, 3), (7, 2), 3)
    case = Case("views", args=(view, other), kwargs={"again": view, "scale": 3})

    call, expected, _ = _validate((original, functional), case)

    left, right = call["args"]
    assert left is call["kwargs"]["again"]
    assert left.stride() == right.stride() == (7, 2)
    assert left.storage_offset() == 2
    assert right.storage_offset() == 3
    assert left.untyped_storage()._cdata == right.untyped_storage()._cdata
    assert left.untyped_storage()._cdata != base.untyped_storage()._cdata
    assert call["kwargs"]["scale"] == 3
    assert torch.equal(expected, (view + other) * 3)


@pytest.fixture
def cpu_environment(monkeypatch):
    environment = {"backend": "test_cpu", "target_arch": "test_cpu"}
    runtime = Mock(return_value=(environment, _CPU))
    monkeypatch.setattr(worker, "runtime_environment", runtime)
    return runtime


def _reference_request(tmp_path, provider_source):
    original = _source(
        tmp_path,
        "reference_original",
        """
        import torch
        class Model(torch.nn.Module):
            def __init__(self, bias=3):
                super().__init__()
                self.register_buffer("bias", torch.tensor(float(bias)))
            def forward(self, x):
                return x + self.bias
    """,
    )
    functional = _source(
        tmp_path,
        "reference_functional",
        """
        import torch
        def module_fn(x, bias):
            return x + bias
        class Model(torch.nn.Module):
            def __init__(self, bias=3):
                super().__init__()
                self.register_buffer("bias", torch.tensor(float(bias)))
            def forward(self, x, fn=module_fn):
                return fn(x, self.bias)
    """,
    )
    provider = _source(tmp_path, "case_provider", provider_source)
    return {
        "settings": asdict(VerificationSettings(seed=71, warmup_ms=5, measure_ms=15)),
        "task": {
            "task_id": "external_task",
            "module_path": str(original),
            "functional_path": str(functional),
            "case_provider_path": str(provider),
        },
    }


def test_prepare_reference_records_actual_coverage_and_separate_timing(
    tmp_path, cpu_environment, monkeypatch
):
    request = _reference_request(
        tmp_path,
        """
        import torch
        from triton_kernel_gen.contracts import Case
        def get_cases(task_id, original, *, kind):
            assert task_id == "external_task"
            assert issubclass(original.Model, torch.nn.Module)
            if kind == "correctness":
                yield Case("matrix", args=(torch.arange(12, dtype=torch.float32).reshape(3, 4).t(),),
                           init_kwargs={"bias": 5}, atol=0, rtol=0,
                           metadata={"claimed_shape": [999], "source": "fixture"})
                yield Case("vector", args=(torch.arange(2, dtype=torch.float64),))
            else:
                yield Case("matrix", args=(torch.arange(7, dtype=torch.float32),),
                           metadata={"purpose": "performance"})
    """,
    )
    timed_outputs = []

    def benchmark(fn, settings):
        timed_outputs.append(fn())
        assert settings.warmup_ms == 5
        assert settings.measure_ms == 15
        return 1.75

    bench = Mock(side_effect=benchmark)
    monkeypatch.setattr(worker, "bench_ms", bench)

    report = worker.prepare_reference(request, tmp_path)

    assert report["success"] and report["stage1_success"]
    assert report["correctness_count"] == 2
    assert report["performance_count"] == 1
    assert report["environment"] == cpu_environment.return_value[0]
    assert [record["seed"] for record in report["cases"]] == [71, 72, 73]
    assert [record["key"] for record in report["cases"]] == [
        "correctness_0",
        "correctness_1",
        "performance_0",
    ]
    matrix, vector, performance = report["cases"]
    assert matrix["metadata"] == {"claimed_shape": [999], "source": "fixture"}
    assert matrix["inputs"]["tensors"][0]["shape"] == [4, 3]
    assert matrix["inputs"]["tensors"][0]["stride"] == [1, 4]
    assert matrix["inputs"]["tensors"][0]["dtype"] == "float32"
    assert len(matrix["inputs"]["tensors"]) == 1
    assert len(matrix["function_inputs"]["tensors"]) == 2
    assert matrix["output"]["tensors"][0]["shape"] == [4, 3]
    assert vector["inputs"]["tensors"][0]["dtype"] == "float64"
    assert performance["inputs"]["tensors"][0]["shape"] == [7]
    assert matrix["atol"] == matrix["rtol"] == 0
    assert "reference_ms" not in matrix and "reference_ms" not in vector
    assert performance["reference_ms"] == performance["baseline_ms"] == 1.75
    assert performance["baseline_kind"] == "functional_reference"
    assert bench.call_count == 1
    assert torch.equal(timed_outputs[0], torch.arange(7, dtype=torch.float32) + 3)
    for record in report["cases"]:
        call = load_tree(tmp_path / "inputs" / record["key"])
        expected = load_tree(tmp_path / "expected" / record["key"])
        payload = load_tree(tmp_path / "cases" / record["key"])
        assert torch.equal(expected, call["args"][0] + call["args"][1])
        assert torch.equal(payload["args"][0], call["args"][0])
        assert record["reference_checks"]["default"]["success"]
        assert record["reference_checks"]["injected"]["success"]
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("kind", ["correctness", "performance"])
def test_prepare_reference_rejects_duplicate_case_ids(tmp_path, cpu_environment, kind):
    request = _reference_request(
        tmp_path,
        f"""
        import torch
        from triton_kernel_gen.contracts import Case
        def get_cases(task_id, original, *, kind):
            yield Case("duplicate", args=(torch.ones(3),))
            if kind == {kind!r}:
                yield Case("duplicate", args=(torch.zeros(3),))
    """,
    )

    with pytest.raises(ReferenceValidationError, match=f"Duplicate {kind} case_id"):
        worker.prepare_reference(request, tmp_path)
    assert not (tmp_path / "inputs").exists()


def test_prepare_freezes_each_yield_before_provider_reuses_storage(
    tmp_path, cpu_environment, monkeypatch
):
    request = _reference_request(
        tmp_path,
        """
        import torch
        from triton_kernel_gen.contracts import Case
        def get_cases(task_id, original, *, kind):
            value = torch.tensor([1.0])
            case = Case("first", args=(value,), metadata={"value": 1})
            yield case
            value.add_(1)
            case.case_id = "second"
            case.metadata["value"] = 2
            yield case
    """,
    )
    monkeypatch.setattr(worker, "bench_ms", lambda fn, settings: (fn(), 1.0)[1])
    report = worker.prepare_reference(request, tmp_path)
    for kind in ("correctness", "performance"):
        records = [case for case in report["cases"] if case["kind"] == kind]
        assert [case["case_id"] for case in records] == ["first", "second"]
        assert [case["metadata"]["value"] for case in records] == [1, 2]
        for index, expected_input in enumerate((1.0, 2.0)):
            case = records[index]
            assert load_tree(tmp_path / "inputs" / case["key"])["args"][0].item() == expected_input
            assert load_tree(tmp_path / "expected" / case["key"]).item() == expected_input + 3


@pytest.mark.parametrize("case_id", ["", " \t", None, 5])
def test_prepare_reference_rejects_invalid_case_ids(tmp_path, cpu_environment, case_id):
    request = _reference_request(
        tmp_path,
        f"""
        import torch
        from triton_kernel_gen.contracts import Case
        def get_cases(task_id, original, *, kind):
            yield Case({case_id!r}, args=(torch.ones(3),))
    """,
    )

    with pytest.raises(ReferenceValidationError, match="nonempty string case_id"):
        worker.prepare_reference(request, tmp_path)


@pytest.mark.parametrize("empty_kind", ["correctness", "performance"])
def test_prepare_reference_requires_both_case_groups(tmp_path, cpu_environment, empty_kind):
    request = _reference_request(
        tmp_path,
        f"""
        import torch
        from triton_kernel_gen.contracts import Case
        def get_cases(task_id, original, *, kind):
            if kind != {empty_kind!r}:
                yield Case("one", args=(torch.ones(3),))
    """,
    )

    with pytest.raises(ReferenceValidationError, match=f"no {empty_kind} cases"):
        worker.prepare_reference(request, tmp_path)


@pytest.fixture
def gpu_runtime(monkeypatch):
    triton = ModuleType("triton")
    triton.__version__ = "3.3.0"
    target = SimpleNamespace(backend="hip", arch="gfx942:sramecc+:xnack-")
    triton.runtime = SimpleNamespace(
        driver=SimpleNamespace(active=SimpleNamespace(get_current_target=Mock(return_value=target)))
    )
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setattr(torch.version, "hip", "6.3.0")
    monkeypatch.setattr(torch.cuda, "is_available", Mock(return_value=True))
    monkeypatch.setattr(torch.cuda, "device_count", Mock(return_value=2))
    monkeypatch.setattr(torch.cuda, "set_device", Mock())
    properties = SimpleNamespace(name="AMD Instinct test device", total_memory=128 * 1024**3)
    monkeypatch.setattr(torch.cuda, "get_device_properties", Mock(return_value=properties))
    return SimpleNamespace(triton=triton, target=target, properties=properties)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_runtime_selects_requested_amd_gpu_and_reports_actual_target(gpu_runtime, arch):
    gpu_runtime.target.arch = f"{arch}:sramecc+:xnack-"

    environment, device = worker.runtime_environment(VerificationSettings(gpu=1, target_arch=arch))

    assert device == torch.device("cuda:1")
    torch.cuda.set_device.assert_called_once_with(1)
    torch.cuda.get_device_properties.assert_called_once_with(device)
    assert environment["triton"] == "3.3.0"
    assert environment["hip"] == "6.3.0"
    assert environment["target_arch"] == arch
    assert environment["backend"] == "hip"
    assert environment["gpu_index"] == 1
    assert environment["gpu_total_memory"] == gpu_runtime.properties.total_memory
    assert environment["worker_pid"] > 0


@pytest.mark.parametrize(
    ("change", "settings", "message"),
    [
        ("version", {}, "Triton 3.3.0 is required"),
        ("hip", {}, "torch.version.hip is empty"),
        ("available", {}, "An AMD GPU is required"),
        ("none", {"gpu": -1}, "GPU index -1 is unavailable"),
        ("none", {"gpu": 2}, "GPU index 2 is unavailable"),
        ("backend", {}, "HIP backend"),
        ("arch", {}, "Unsupported AMD architecture gfx90a"),
        ("none", {"target_arch": "gfx950"}, "Requested gfx950, but the GPU uses gfx942"),
    ],
)
def test_runtime_rejects_unsupported_environment(
    gpu_runtime, monkeypatch, change, settings, message
):
    if change == "version":
        gpu_runtime.triton.__version__ = "3.2.0"
    elif change == "hip":
        monkeypatch.setattr(torch.version, "hip", None)
    elif change == "available":
        torch.cuda.is_available.return_value = False
    elif change == "backend":
        gpu_runtime.target.backend = "cuda"
    elif change == "arch":
        gpu_runtime.target.arch = "gfx90a"

    with pytest.raises(RuntimeError, match=message):
        worker.runtime_environment(VerificationSettings(**settings))


@pytest.fixture
def fake_triton(monkeypatch):
    triton = ModuleType("triton")
    runtime = ModuleType("triton.runtime")
    jit = ModuleType("triton.runtime.jit")

    class JITFunction:
        def __init__(self, fn):
            self.fn = fn

        def run(self, *args, **kwargs):
            if kwargs.pop("warmup", False):
                return None
            return self.fn(*args, **kwargs)

        def __getitem__(self, grid):
            return self.run

    JITFunction.__module__ = "triton.runtime.jit"
    jit.JITFunction = JITFunction
    triton.jit = JITFunction
    runtime.jit = jit
    triton.runtime = runtime
    triton.CompilationError = type(
        "CompilationError", (RuntimeError,), {"__module__": "triton.compiler.errors"}
    )
    monkeypatch.setitem(sys.modules, "triton", triton)
    monkeypatch.setitem(sys.modules, "triton.runtime", runtime)
    monkeypatch.setitem(sys.modules, "triton.runtime.jit", jit)
    return SimpleNamespace(module=triton, jit_type=JITFunction, original_run=JITFunction.run)


_KERNEL = """
import triton
@triton.jit
def kernel(x):
    return x * 2
"""


def _candidate_request(tmp_path, body, *, values=(1.0,)):
    candidate = _source(tmp_path, "candidate", body)
    cases = []
    for index, value in enumerate(values):
        key = f"correctness_{index}"
        cases.append({"key": key, "case_id": f"case_{index}", "seed": 31 + index})
        save_tree(
            tmp_path / "inputs" / key, {"args": (torch.tensor([value, value + 1]),), "kwargs": {}}
        )
    return {
        "settings": asdict(VerificationSettings()),
        "candidate_path": str(candidate),
        "cases": cases,
    }


def _run_candidate(tmp_path, request, fake_triton, *, benchmark=False):
    report = worker.run_candidate(request, tmp_path, benchmark=benchmark)
    assert json.loads((tmp_path / "report.json").read_text()) == report
    assert fake_triton.jit_type.run is fake_triton.original_run
    return report


def test_candidate_executes_and_saves_cpu_outputs_with_observed_launches(
    tmp_path, cpu_environment, fake_triton
):
    request = _candidate_request(
        tmp_path, _KERNEL + "\ndef module_fn(x):\n    return kernel[(1,)](x)\n", values=(1.0, 3.0)
    )

    report = _run_candidate(tmp_path, request, fake_triton)

    assert report["success"] and report["compile_success"] and report["runtime_success"]
    assert not report["timing_success"]
    assert report["failure_stage"] is None
    assert report["phase"] == "complete"
    assert len(report["cases"]) == 2
    for case in report["cases"]:
        assert case["jit_launches"] == case["runtime_launches"] == 1
        inputs = load_tree(tmp_path / "inputs" / case["key"])
        output = load_tree(tmp_path / "outputs" / case["key"])
        assert torch.equal(output, inputs["args"][0] * 2)


@pytest.mark.parametrize(
    ("source", "phase", "stage", "message"),
    [
        ("def module_fn(:\n", "import", "compile", "SyntaxError"),
        (
            "import kernelgen_nonexistent_fixture_module\n",
            "import",
            "compile",
            "ModuleNotFoundError",
        ),
        (_KERNEL, "import", "compile", "export callable module_fn"),
        ("def module_fn(x):\n    return x * 2\n", "import", "compile", "actual Triton JIT kernel"),
        (
            _KERNEL + "\ndef module_fn(x):\n    return x * 2\n",
            "jit_warmup",
            "compile",
            "did not invoke a Triton kernel",
        ),
        (
            _KERNEL + "\ndef module_fn(x):\n    kernel.run(x, warmup=True)\n    return x * 2\n",
            "jit_warmup",
            "compile",
            "did not invoke a Triton kernel",
        ),
        (
            _KERNEL + "\ndef module_fn(x):\n    raise triton.CompilationError('invalid kernel')\n",
            "jit_warmup",
            "compile",
            "invalid kernel",
        ),
        (
            _KERNEL + "\ndef module_fn(x):\n    raise RuntimeError('launch failed')\n",
            "jit_warmup",
            "runtime",
            "launch failed",
        ),
        (
            _KERNEL + "\ndef module_fn(x):\n    x.add_(1)\n    return kernel[(1,)](x)\n",
            "jit_warmup",
            "runtime",
            "mutated tensor storage",
        ),
    ],
    ids=[
        "syntax",
        "import",
        "missing_export",
        "no_kernel",
        "unused_kernel",
        "compile_only",
        "compiler_error",
        "launch_error",
        "input_mutation",
    ],
)
def test_candidate_reports_import_compile_and_launch_failures(
    tmp_path,
    cpu_environment,
    fake_triton,
    source,
    phase,
    stage,
    message,
):
    report = _run_candidate(tmp_path, _candidate_request(tmp_path, source), fake_triton)

    assert not report["success"]
    assert not report["compile_success"]
    assert not report["runtime_success"]
    assert not report["timing_success"]
    assert report["phase"] == phase
    assert report["failure_stage"] == stage
    assert message in report["message"]


def test_every_candidate_case_requires_a_launch(tmp_path, cpu_environment, fake_triton):
    source = _KERNEL + "\ndef module_fn(x):\n    return kernel[(1,)](x) if x[0] > 0 else x * 2\n"
    request = _candidate_request(tmp_path, source, values=(1.0, -3.0))

    report = _run_candidate(tmp_path, request, fake_triton)

    assert not report["success"]
    assert report["failure_stage"] == "compile"
    assert "case_1 did not invoke" in report["message"]
    assert len(report["cases"]) == 1


def test_candidate_cannot_satisfy_kernel_requirement_with_an_imported_kernel(
    tmp_path,
    cpu_environment,
    fake_triton,
):
    _source(tmp_path, "worker_external_kernel", _KERNEL)
    request = _candidate_request(
        tmp_path,
        """
        from worker_external_kernel import kernel
        def module_fn(x):
            return kernel[(1,)](x)
    """,
    )

    report = _run_candidate(tmp_path, request, fake_triton)

    assert not report["success"]
    assert report["phase"] == "import"
    assert report["failure_stage"] == "compile"
    assert "dependency_paths" in report["message"]


def test_probe_accepts_local_kernel_inside_triton_wrappers(tmp_path, fake_triton):
    module = _module(tmp_path, "wrapped_kernel", _KERNEL)

    class Autotuner:
        __module__ = "triton.runtime.autotuner"

        def __init__(self, fn):
            self.fn = fn

    module.kernel = Autotuner(Autotuner(module.kernel))

    with worker.TritonLaunchProbe() as probe:
        probe.require_kernel(module)
        assert probe.launches == 0
    assert fake_triton.jit_type.run is fake_triton.original_run


def test_candidate_environment_error_reports_failure_before_import(
    tmp_path,
    cpu_environment,
    fake_triton,
):
    cpu_environment.side_effect = RuntimeError("An AMD GPU is required.")
    request = _candidate_request(tmp_path, "raise AssertionError('The candidate must not load.')\n")

    report = _run_candidate(tmp_path, request, fake_triton)

    assert not report["success"]
    assert not report["compile_success"] and not report["runtime_success"]
    assert report["phase"] == "environment"
    assert report["failure_stage"] == "infrastructure"
    assert "An AMD GPU is required" in report["message"]


@pytest.mark.parametrize("failure", ["raise RuntimeError('execution failed')", "return x * 2"])
def test_candidate_runtime_failure_preserves_compile_success(
    tmp_path, cpu_environment, fake_triton, failure
):
    source = (
        _KERNEL
        + "\ncalls = 0\ndef module_fn(x):\n    global calls\n    calls += 1\n    if calls == 2:\n"
    )
    source += textwrap.indent(failure, "        ") + "\n    return kernel[(1,)](x)\n"

    report = _run_candidate(tmp_path, _candidate_request(tmp_path, source), fake_triton)

    assert not report["success"]
    assert report["compile_success"]
    assert not report["runtime_success"]
    assert not report["timing_success"]
    assert report["phase"] == "runtime"
    assert report["failure_stage"] == "runtime"


def test_candidate_timing_failure_preserves_compile_and_runtime_success(
    tmp_path, cpu_environment, fake_triton, monkeypatch
):
    request = _candidate_request(
        tmp_path, _KERNEL + "\ndef module_fn(x):\n    return kernel[(1,)](x)\n"
    )
    monkeypatch.setattr(worker, "bench_ms", Mock(side_effect=RuntimeError("timer failed")))

    report = _run_candidate(tmp_path, request, fake_triton, benchmark=True)

    assert not report["success"]
    assert report["compile_success"] and report["runtime_success"]
    assert not report["timing_success"]
    assert report["phase"] == report["failure_stage"] == "timing"
    assert "timer failed" in report["message"]


def test_candidate_benchmark_records_latency_and_output(
    tmp_path, cpu_environment, fake_triton, monkeypatch
):
    request = _candidate_request(
        tmp_path, _KERNEL + "\ndef module_fn(x):\n    return kernel[(1,)](x)\n"
    )
    benchmark = Mock(side_effect=lambda fn, settings: (fn(), 0.625)[1])
    monkeypatch.setattr(worker, "bench_ms", benchmark)

    report = _run_candidate(tmp_path, request, fake_triton, benchmark=True)

    assert report["success"]
    assert report["compile_success"] and report["runtime_success"] and report["timing_success"]
    assert report["cases"][0]["candidate_ms"] == 0.625
    benchmark.assert_called_once()
    expected = load_tree(tmp_path / "outputs" / "correctness_0")
    timed = load_tree(tmp_path / "timed_outputs" / "correctness_0")
    assert compare_outputs(expected, timed, atol=0, rtol=0)["success"]


def test_candidate_must_execute_triton_in_post_timing_validation(
    tmp_path, cpu_environment, fake_triton, monkeypatch
):
    source = (
        _KERNEL
        + """
calls = 0
def module_fn(x):
    global calls
    calls += 1
    if calls <= 2:
        return kernel[(1,)](x)
    return x * 2
"""
    )
    request = _candidate_request(tmp_path, source)
    monkeypatch.setattr(worker, "bench_ms", Mock(side_effect=lambda fn, settings: (fn(), 0.01)[1]))

    report = _run_candidate(tmp_path, request, fake_triton, benchmark=True)

    assert not report["success"]
    assert report["compile_success"] and report["runtime_success"]
    assert not report["timing_success"]
    assert report["failure_stage"] == "timing"
    assert "Triton" in report["message"]


def test_candidate_benchmark_uses_no_launch_probe(
    tmp_path, cpu_environment, fake_triton, monkeypatch
):
    request = _candidate_request(
        tmp_path, _KERNEL + "\ndef module_fn(x):\n    return kernel[(1,)](x)\n"
    )

    def benchmark(fn, settings):
        assert fake_triton.jit_type.run is fake_triton.original_run
        fn()
        assert fake_triton.jit_type.run is fake_triton.original_run
        return 0.5

    monkeypatch.setattr(worker, "bench_ms", benchmark)
    report = _run_candidate(tmp_path, request, fake_triton, benchmark=True)
    assert report["success"]
    record = report["cases"][0]
    assert (
        record["jit_launches"]
        == record["runtime_launches"]
        == record["post_timing_validation_launches"]
        == 1
    )
