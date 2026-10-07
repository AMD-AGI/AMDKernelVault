# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks for tensor views, strict comparisons, and the safe tree format."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import triton_kernel_gen.tensor_utils as tensor_utils
from triton_kernel_gen.tensor_utils import (
    clone_tree,
    compare_outputs,
    load_tree,
    save_tree,
    tree_metadata,
)


def _alias_tree():
    base = torch.arange(30, dtype=torch.float32, requires_grad=True)
    view = base.as_strided((3, 4), (7, 2), 2)
    return {
        "view": view,
        "other": (base[3::2], [view, base[2:3].expand(5)]),
        "scalar": (3, True, None, "result", 0.125, 2 + 3j),
    }


@pytest.mark.parametrize("through_disk", [False, True])
def test_clone_and_codec_preserve_views_aliases_and_independence(tmp_path, through_disk):
    original = _alias_tree()
    if through_disk:
        save_tree(tmp_path / "tree", original)
        result = load_tree(tmp_path / "tree")
    else:
        result = clone_tree(original, device="cpu")
    assert compare_outputs(original, result, atol=0, rtol=0)["success"]
    view = result["view"]
    assert view.stride() == (7, 2)
    assert view.storage_offset() == 2
    assert view is result["other"][1][0]
    assert view.untyped_storage().nbytes() == original["view"].untyped_storage().nbytes()
    assert view.untyped_storage()._cdata == result["other"][0].untyped_storage()._cdata
    assert view.untyped_storage()._cdata != original["view"].untyped_storage()._cdata
    assert not view.requires_grad
    assert result["other"][1][1].stride() == (0,)
    view[0, 0] = 999
    assert result["other"][1][1].tolist() == [999] * 5
    assert original["view"][0, 0].item() == 2


@pytest.mark.parametrize("through_disk", [False, True])
def test_mixed_dtype_storage_aliases(tmp_path, through_disk):
    base = torch.arange(8, dtype=torch.float32)
    original = [base[2:], base.view(torch.uint8)[3:]]
    if through_disk:
        save_tree(tmp_path / "tree", original)
        result = load_tree(tmp_path / "tree")
    else:
        result = clone_tree(original)
    assert result[0].untyped_storage()._cdata == result[1].untyped_storage()._cdata
    assert result[1].storage_offset() == 3
    assert result[1].dtype == torch.uint8
    assert compare_outputs(original, result, 0, 0)["success"]


@pytest.mark.parametrize("through_disk", [False, True])
def test_empty_views_and_scalar_tensors(tmp_path, through_disk):
    original = (
        torch.tensor(3.0),
        torch.empty(0),
        torch.as_strided(torch.empty(0), (0, 2), (9, 2), 99),
    )
    if through_disk:
        save_tree(tmp_path / "tree", original)
        result = load_tree(tmp_path / "tree")
    else:
        result = clone_tree(original)
    assert result[0].shape == torch.Size([])
    assert result[2].storage_offset() == 99
    assert result[2].stride() == (9, 2)
    assert compare_outputs(original, result, 0, 0)["success"]


def test_metadata_is_json_and_identifies_storage_aliases():
    original = _alias_tree()
    metadata = tree_metadata(original)
    json.dumps(metadata, allow_nan=False)
    assert len(metadata["storages"]) == 1
    assert len(metadata["tensors"]) == 3
    descriptor = metadata["tensors"][0]
    assert descriptor["dtype"] == "float32"
    assert descriptor["device"] == "cpu"
    assert descriptor["shape"] == [3, 4]
    assert descriptor["stride"] == [7, 2]
    assert descriptor["storage_offset"] == 2
    assert all(item["storage"] == 0 for item in metadata["tensors"])


@pytest.mark.parametrize(
    "unsupported",
    [
        lambda: torch.sparse_coo_tensor([[0]], [1.0], (2,)),
        lambda: torch.ones(2, dtype=torch.complex64).conj(),
        lambda: torch.ones(2, device="meta"),
        lambda: torch.quantize_per_tensor(
            torch.ones(2), scale=0.1, zero_point=0, dtype=torch.qint8
        ),
        lambda: {"bad": object()},
        lambda: {float("nan"): 2},
        lambda: {torch.tensor(1): 2},
    ],
)
def test_unsupported_types_fail_explicitly(unsupported, tmp_path):
    tree = unsupported()
    with pytest.raises((ValueError, TypeError)):
        clone_tree(tree)
    with pytest.raises((ValueError, TypeError)):
        save_tree(tmp_path / "tree", tree)


@pytest.mark.parametrize("through_disk", [False, True])
def test_parameter_normalizes_to_detached_tensor_and_preserves_aliases(tmp_path, through_disk):
    parameter = torch.nn.Parameter(torch.arange(12, dtype=torch.float32))
    value = {"weight": parameter, "view": parameter[2::3], "repeat": parameter}
    if through_disk:
        save_tree(tmp_path / "tree", value)
        result = load_tree(tmp_path / "tree")
    else:
        result = clone_tree(value)
    assert type(result["weight"]) is torch.Tensor
    assert not result["weight"].requires_grad
    assert not result["view"].requires_grad
    assert result["weight"] is result["repeat"]
    assert result["weight"].untyped_storage()._cdata == result["view"].untyped_storage()._cdata
    assert result["view"].stride() == (3,)
    assert result["view"].storage_offset() == 2
    assert compare_outputs(value, result, 0, 0)["success"]
    metadata = tree_metadata(value)
    assert len(metadata["storages"]) == 1
    assert metadata["tensors"][0]["requires_grad"]


def test_arbitrary_tensor_subclass_fails_explicitly():
    class CustomTensor(torch.Tensor):
        pass

    with pytest.raises(TypeError, match="subclasses"):
        clone_tree(torch.ones(2).as_subclass(CustomTensor))


@pytest.mark.filterwarnings(
    "ignore:Named tensors and all their associated APIs are an experimental feature"
)
@pytest.mark.parametrize("names", [("batch", "feature"), (None, "feature")])
@pytest.mark.parametrize("parameter", [False, True])
def test_named_tensors_fail_before_cloning_or_serialization(tmp_path, names, parameter):
    tensor = torch.ones(2, 3, names=names)
    if parameter:
        tensor = torch.nn.Parameter(tensor)
    tree = {"args": (tensor,)}
    with pytest.raises(TypeError, match="Named tensors are not supported"):
        clone_tree(tree)
    with pytest.raises(TypeError, match="Named tensors are not supported"):
        tree_metadata(tree)
    directory = tmp_path / "tree"
    with pytest.raises(TypeError, match="Named tensors are not supported"):
        save_tree(directory, tree)
    assert not directory.exists()
    result = compare_outputs(tensor, tensor.rename(None), 0, 0)
    assert not result["success"]
    assert "Named tensors are not supported" in result["message"]


def test_all_none_tensor_names_remain_supported():
    tensor = torch.ones(2, 3, names=(None, None))
    result = clone_tree(tensor)
    assert result.names == (None, None)
    assert torch.equal(result, tensor)


def test_cyclic_tree_fails_explicitly():
    tree = []
    tree.append(tree)
    with pytest.raises(ValueError, match="Cyclic"):
        clone_tree(tree)
    assert not compare_outputs(tree, tree, 0, 0)["success"]


def test_overlapping_distinct_storages_fail_explicitly():
    buffer = bytearray(range(20))
    first = torch.frombuffer(buffer, dtype=torch.uint8)
    second = torch.frombuffer(buffer, dtype=torch.uint8, offset=4)
    with pytest.raises(TypeError, match="Overlapping distinct"):
        clone_tree([first, second])


def test_codec_preserves_scalar_types_without_pickle(tmp_path, monkeypatch):
    def reject_pickle(*args, **kwargs):
        raise AssertionError("The codec must not call torch.load.")

    monkeypatch.setattr(torch, "load", reject_pickle)
    value = {
        None: (2**100, True, -0.0, "text", 3 + 4j),
        2: [float("inf"), float("nan")],
        3.5: False,
    }
    save_tree(tmp_path / "tree", value)
    result = load_tree(tmp_path / "tree")
    assert result[None] == value[None]
    assert math.copysign(1, result[None][2]) == -1
    assert math.isinf(result[2][0])
    assert math.isnan(result[2][1])
    assert result[3.5] is False


@pytest.mark.parametrize(
    "dtype",
    [
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.complex64,
        torch.complex128,
    ],
)
def test_codec_preserves_dtype_values(tmp_path, dtype):
    value = torch.tensor([0, 1, 1], dtype=dtype)
    save_tree(tmp_path / "tree", value)
    result = load_tree(tmp_path / "tree")
    assert result.dtype == dtype
    assert torch.equal(result, value)


def _change_manifest(directory: Path, mutate):
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    mutate(manifest)
    path.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "malformed",
    [
        lambda data: data.update(format="pickle"),
        lambda data: data["storages"][0].update(file="../outside.bin"),
        lambda data: data["storages"][0].update(file="/etc/passwd"),
        lambda data: data["storages"][0].update(nbytes=100000),
        lambda data: data["storages"][0].update(nbytes=-1),
        lambda data: data["storages"][0].update(nbytes=True),
        lambda data: data["tensors"][0].update(dtype="__import__('os')"),
        lambda data: data["tensors"][0].update(shape=[999999]),
        lambda data: data["tensors"][0].update(stride=[-1]),
        lambda data: data["tensors"][0].update(storage_offset=500),
        lambda data: data["tensors"][0].update(storage=9),
        lambda data: data["tensors"][0].update(shape=[True]),
        lambda data: data["tree"].update(id=99),
        lambda data: data["tree"].update(kind="pickle"),
    ],
)
def test_codec_rejects_malformed_manifest(tmp_path, malformed):
    directory = tmp_path / "tree"
    save_tree(directory, torch.arange(3))
    _change_manifest(directory, malformed)
    with pytest.raises((ValueError, TypeError)):
        load_tree(directory)


def test_codec_rejects_truncated_storage(tmp_path):
    directory = tmp_path / "tree"
    save_tree(directory, torch.arange(3))
    (directory / "storage_000000.bin").write_bytes(b"abc")
    with pytest.raises(ValueError, match="byte count"):
        load_tree(directory)


@pytest.mark.parametrize("filename", ["manifest.json", "storage_000000.bin"])
def test_codec_rejects_symbolic_files(tmp_path, filename):
    directory = tmp_path / "tree"
    save_tree(directory, torch.arange(3))
    target = tmp_path / "target"
    (directory / filename).rename(target)
    (directory / filename).symlink_to(target)
    with pytest.raises(ValueError):
        load_tree(directory)


def test_codec_rejects_symbolic_directory(tmp_path):
    directory = tmp_path / "tree"
    save_tree(directory, torch.arange(3))
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(ValueError):
        load_tree(alias)
    with pytest.raises(ValueError):
        save_tree(alias, 2)


def test_codec_retains_original_directory_during_path_replacement(tmp_path, monkeypatch):
    directory = tmp_path / "tree"
    outside = tmp_path / "outside"
    save_tree(directory, torch.tensor([3]))
    save_tree(outside, torch.tensor([99]))
    read_original = tensor_utils._read_regular_file

    def read_then_replace(filename, max_bytes, directory_fd):
        data = read_original(filename, max_bytes, directory_fd)
        if filename == "manifest.json":
            directory.rename(tmp_path / "retained")
            directory.symlink_to(outside, target_is_directory=True)
        return data

    monkeypatch.setattr(tensor_utils, "_read_regular_file", read_then_replace)
    assert load_tree(directory).tolist() == [3]


def test_codec_enforces_aggregate_byte_limit(tmp_path):
    directory = tmp_path / "tree"
    save_tree(directory, [torch.ones(3), torch.zeros(3)])
    with pytest.raises(ValueError, match="byte limit"):
        load_tree(directory, max_bytes=16)


def test_codec_rejects_excessive_broadcast_shape(tmp_path):
    directory = tmp_path / "tree"
    save_tree(directory, torch.ones(1))
    _change_manifest(directory, lambda data: data["tensors"][0].update(shape=[2**50], stride=[0]))
    with pytest.raises(ValueError, match="logical byte limit"):
        load_tree(directory)


def test_codec_does_not_overwrite_existing_tree(tmp_path):
    directory = tmp_path / "tree"
    save_tree(directory, torch.arange(3))
    with pytest.raises(ValueError, match="empty"):
        save_tree(directory, torch.arange(5))
    assert load_tree(directory).tolist() == [0, 1, 2]


@pytest.mark.parametrize("empty", [None, [], {}, (), "", torch.empty(0), [torch.empty(2, 0), None]])
def test_comparison_rejects_empty_outputs(empty):
    result = compare_outputs(empty, clone_tree(empty), 0, 0)
    assert not result["success"]
    assert "empty" in result["message"]


@pytest.mark.parametrize(
    "expected,actual,reason",
    [
        (torch.ones(2, 1), torch.ones(2, 2), "shapes"),
        (torch.ones(2), torch.ones(2, dtype=torch.float64), "dtypes"),
        ([torch.ones(2)], (torch.ones(2),), "types"),
        ([1], [1, 2], "lengths"),
        ({"a": 1}, {"b": 1}, "keys"),
        ({1: 1}, {True: 1}, "keys"),
        (1, True, "types"),
        ("one", "two", "scalar"),
        (torch.tensor([True]), torch.tensor([False]), "Boolean"),
    ],
)
def test_comparison_rejects_structure_and_dtype_mismatches(expected, actual, reason):
    result = compare_outputs(expected, actual, 100, 100)
    assert not result["success"]
    assert reason in result["message"]


def test_comparison_applies_tolerance_and_reports_max_errors():
    expected = {"result": torch.tensor([1.0, 2.0], dtype=torch.float64)}
    actual = {"result": torch.tensor([1.125, 2.25], dtype=torch.float64)}
    result = compare_outputs(expected, actual, atol=0, rtol=0.125)
    assert result["success"]
    assert result["max_abs_error"] == 0.25
    assert result["max_rel_error"] == 0.125
    assert result["tensor_count"] == 1
    assert not compare_outputs(expected, actual, atol=0, rtol=0.12)["success"]


@pytest.mark.parametrize(
    "expected,actual",
    [
        (torch.tensor([float("nan")]), torch.tensor([float("nan")])),
        (float("nan"), float("nan")),
        (torch.tensor([float("inf")]), torch.tensor([-float("inf")])),
        (float("inf"), -float("inf")),
        (complex(float("nan"), 0), complex(float("nan"), 0)),
    ],
)
def test_nonfinite_mismatches_never_pass_and_metrics_remain_json(expected, actual):
    result = compare_outputs(expected, actual, 1, 1)
    assert not result["success"]
    json.dumps(result, allow_nan=False)
    assert result["max_abs_error"] is None


@pytest.mark.parametrize(
    "value", [float("inf"), -float("inf"), torch.tensor([float("inf"), -float("inf")])]
)
def test_equal_infinities_match_with_zero_error(value):
    result = compare_outputs(value, clone_tree(value), 0, 0)
    assert result["success"]
    assert result["max_abs_error"] == 0
    assert result["max_rel_error"] == 0


def test_int64_comparison_does_not_lose_low_bits():
    result = compare_outputs(torch.tensor([2**63 - 1]), torch.tensor([2**63 - 2]), atol=10, rtol=10)
    assert not result["success"]
    assert result["max_abs_error"] == 1


def test_comparison_supports_complex_values():
    left = (torch.tensor([1 + 2j], dtype=torch.complex128), 2 + 3j)
    right = (torch.tensor([1.125 + 2j], dtype=torch.complex128), 2.125 + 3j)
    result = compare_outputs(left, right, atol=0.125, rtol=0)
    assert result["success"]
    assert result["max_abs_error"] == 0.125


@pytest.mark.parametrize(
    "expected,actual",
    [
        (1.7e308, -1.7e308),
        (complex(1.7e308, 1.7e308), complex(-1.7e308, -1.7e308)),
    ],
)
def test_scalar_overflow_cannot_create_false_match(expected, actual):
    result = compare_outputs(expected, actual, atol=0, rtol=1.1)
    assert not result["success"]
    assert result["max_abs_error"] is None
    assert result["max_rel_error"] == 2
    assert compare_outputs(expected, actual, atol=0, rtol=2.1)["success"]


@pytest.mark.parametrize("atol", [0, 1])
@pytest.mark.parametrize("swap_components", [False, True])
def test_adjacent_huge_complex_scalars_cannot_lose_finite_difference(atol, swap_components):
    expected = complex(1.7e308, 8.500000000000001e307)
    actual = complex(1.7e308, 8.500000000000002e307)
    if swap_components:
        expected = complex(expected.imag, expected.real)
        actual = complex(actual.imag, actual.real)
    difference = abs(actual - expected)
    assert difference == 9.9792015476736e291
    result = compare_outputs(expected, actual, atol=atol, rtol=0)
    assert not result["success"]
    assert result["max_abs_error"] == difference
    assert result["max_rel_error"] > 0


@pytest.mark.parametrize(
    "expected,actual",
    [
        (complex(1.7e308, 8.500000000000001e307), complex(1.7e308, 8.500000000000002e307)),
        (complex(1.4e308, 1.4e308), complex(1.4e308, math.nextafter(1.4e308, 0))),
    ],
)
def test_huge_complex_scalars_preserve_adjacent_absolute_tolerance_boundaries(expected, actual):
    difference = abs(actual - expected)
    assert not compare_outputs(expected, actual, math.nextafter(difference, 0), 0)["success"]
    assert compare_outputs(expected, actual, difference, 0)["success"]
    assert compare_outputs(expected, actual, math.nextafter(difference, math.inf), 0)["success"]


@pytest.mark.parametrize(
    "rtol,matches",
    [
        (5.250393652181006e-17, False),
        (5.250393652181007e-17, True),
    ],
)
def test_huge_complex_scalars_preserve_adjacent_relative_tolerance_boundaries(rtol, matches):
    expected = complex(1.7e308, 8.500000000000001e307)
    actual = complex(1.7e308, 8.500000000000002e307)
    assert compare_outputs(expected, actual, 0, rtol)["success"] is matches


@pytest.mark.parametrize(
    "expected,actual,rejected_rtol,accepted_rtol",
    [
        (complex(-8e307, -8e307), complex(8e307, 8e307), math.nextafter(2, 0), 2),
        (complex(1.7e308, 8e307), complex(-1.7e308, 7e307), 1.81041995034317, 1.8104199503431704),
    ],
)
def test_overflowing_complex_errors_preserve_relative_boundaries(
    expected,
    actual,
    rejected_rtol,
    accepted_rtol,
):
    rejected = compare_outputs(expected, actual, 0, rejected_rtol)
    accepted = compare_outputs(expected, actual, 0, accepted_rtol)
    assert not rejected["success"]
    assert accepted["success"]
    assert accepted["max_abs_error"] is None
    assert accepted["max_rel_error"] > 0


def test_overflowing_tolerance_does_not_erase_subnormal_error_metric():
    difference = math.ulp(0.0)
    expected = complex(1e308, 0)
    actual = complex(1e308, difference)
    result = compare_outputs(expected, actual, 0, 2)
    assert result["success"]
    assert result["max_abs_error"] == difference


def test_tiny_component_difference_beside_huge_component_respects_absolute_tolerance():
    step = math.ulp(0.0)
    expected = complex(1e308, step)
    actual = complex(1e308, 3 * step)
    result = compare_outputs(expected, actual, step, 0)
    assert not result["success"]
    assert result["max_abs_error"] == 2 * step


@pytest.mark.parametrize("expected", [1.7e308, complex(1.7e308, 8.500000000000001e307)])
def test_zero_tolerance_accepts_identical_huge_scalars(expected):
    result = compare_outputs(expected, expected, 0, 0)
    assert result["success"]
    assert result["max_abs_error"] == result["max_rel_error"] == 0


@pytest.mark.parametrize(
    "atol,rtol", [(-1, 0), (0, -1), (float("nan"), 0), (0, float("inf")), (True, 0)]
)
def test_comparison_rejects_invalid_tolerances(atol, rtol):
    with pytest.raises(ValueError):
        compare_outputs(torch.ones(2), torch.ones(2), atol, rtol)
