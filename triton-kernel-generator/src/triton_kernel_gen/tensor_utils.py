# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Preserve tensor views, compare outputs, and read trees without pickle.

The disk format contains JSON and raw storage bytes. It never imports classes or
executes deserialization hooks. Trees support tensors, Parameters, lists, tuples,
dictionaries with scalar keys, and Python scalar values. Tensor storage aliases
and repeated tensor references survive a round trip. Container identity does not.
"""

from __future__ import annotations

import json
import math
import os
import stat
from fractions import Fraction
from pathlib import Path
from typing import Any

import torch

_FORMAT = "triton-kernel-gen-tree-v1"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_NODES = 100_000
_MAX_DEPTH = 100
_DEFAULT_MAX_BYTES = 1024 * 1024 * 1024
_RELATIVE_FLOOR = 1e-12
_DTYPES = {
    name: getattr(torch, name)
    for name in (
        "bool",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "complex64",
        "complex128",
        "uint16",
        "uint32",
        "uint64",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
    )
    if hasattr(torch, name)
}


def _check_tensor(value: torch.Tensor) -> None:
    if type(value) not in (torch.Tensor, torch.nn.Parameter):
        raise TypeError("Tensor subclasses are not supported.")
    if value.layout != torch.strided or value.is_quantized or value.is_nested:
        raise TypeError("Only dense, strided tensors are supported.")
    if value.has_names():
        raise TypeError("Named tensors are not supported.")
    if value.device.type not in ("cpu", "cuda"):
        raise TypeError(f"Tensor device {value.device.type!r} is not supported.")
    if value.is_conj() or value.is_neg():
        raise TypeError("Conjugate and negative tensor views are not supported.")
    if str(value.dtype).removeprefix("torch.") not in _DTYPES:
        raise TypeError(f"Tensor dtype {value.dtype} is not supported.")
    if any(stride < 0 for stride in value.stride()):
        raise TypeError("Negative tensor strides are not supported.")


def _scalar_node(value: Any) -> dict[str, Any]:
    kind = type(value)
    if value is None:
        return {"kind": "none"}
    if kind is bool:
        return {"kind": "bool", "value": value}
    if kind is int:
        return {"kind": "int", "value": str(value)}
    if kind is float:
        return {"kind": "float", "value": value.hex()}
    if kind is complex:
        return {"kind": "complex", "real": value.real.hex(), "imag": value.imag.hex()}
    if kind is str:
        return {"kind": "str", "value": value}
    raise TypeError(f"Tree value type {kind.__name__!r} is not supported.")


def _check_key(value: Any) -> None:
    _scalar_node(value)
    if type(value) in (float, complex) and not (
        math.isfinite(value.real) and math.isfinite(value.imag)
    ):
        raise TypeError("Dictionary keys must be finite scalar values.")


def _describe_tree(value: Any) -> tuple[dict[str, Any], list[torch.Tensor]]:
    tensor_ids: dict[int, int] = {}
    storage_ids: dict[tuple[str, int], int] = {}
    tensor_descriptors: list[dict[str, Any]] = []
    storage_descriptors: list[dict[str, Any]] = []
    storage_examples: list[torch.Tensor] = []
    storage_ranges: dict[str, list[tuple[int, int]]] = {}
    active: set[int] = set()
    node_count = 0

    def visit(item: Any, depth: int) -> dict[str, Any]:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_NODES or depth > _MAX_DEPTH:
            raise ValueError("The tensor tree exceeds the size or depth limit.")
        if isinstance(item, torch.Tensor):
            _check_tensor(item)
            tensor_id = tensor_ids.get(id(item))
            if tensor_id is None:
                tensor_id = len(tensor_descriptors)
                tensor_ids[id(item)] = tensor_id
                storage = item.untyped_storage()
                storage_key = (str(item.device), storage._cdata)
                storage_id = storage_ids.get(storage_key)
                if storage_id is None:
                    start, length = storage.data_ptr(), storage.nbytes()
                    ranges = storage_ranges.setdefault(str(item.device), [])
                    if length and any(
                        start < end and begin < start + length for begin, end in ranges
                    ):
                        raise TypeError("Overlapping distinct tensor storages are not supported.")
                    if length:
                        ranges.append((start, start + length))
                    storage_id = len(storage_descriptors)
                    storage_ids[storage_key] = storage_id
                    storage_descriptors.append(
                        {
                            "id": storage_id,
                            "device": str(item.device),
                            "nbytes": storage.nbytes(),
                        }
                    )
                    storage_examples.append(item)
                tensor_descriptors.append(
                    {
                        "id": tensor_id,
                        "storage": storage_id,
                        "dtype": str(item.dtype).removeprefix("torch."),
                        "shape": list(item.shape),
                        "stride": list(item.stride()),
                        "storage_offset": item.storage_offset(),
                        "device": str(item.device),
                        "requires_grad": item.requires_grad,
                    }
                )
            return {"kind": "tensor", "id": tensor_id}
        if type(item) in (list, tuple, dict):
            if id(item) in active:
                raise ValueError("Cyclic tensor trees are not supported.")
            active.add(id(item))
            try:
                if type(item) is dict:
                    entries = []
                    for key, child in item.items():
                        _check_key(key)
                        entries.append([visit(key, depth + 1), visit(child, depth + 1)])
                    return {"kind": "dict", "items": entries}
                return {
                    "kind": type(item).__name__,
                    "items": [visit(child, depth + 1) for child in item],
                }
            finally:
                active.remove(id(item))
        return _scalar_node(item)

    root = visit(value, 0)
    return {
        "format": _FORMAT,
        "tree": root,
        "tensors": tensor_descriptors,
        "storages": storage_descriptors,
    }, storage_examples


def tree_metadata(value: Any) -> dict[str, Any]:
    """Return JSON metadata, including each tensor's storage alias group."""
    return _describe_tree(value)[0]


def _storage_bytes(value: torch.Tensor) -> torch.Tensor:
    storage = value.untyped_storage()
    return torch.empty(0, dtype=torch.uint8, device=value.device).set_(
        storage, 0, (storage.nbytes(),), (1,)
    )


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise ValueError(f"{label} must be a nonnegative 64-bit integer.")
    return value


def _restore_tree(
    manifest: dict[str, Any],
    raw_storages: list[torch.Tensor],
    *,
    max_logical_bytes: int | None = None,
) -> Any:
    descriptors = manifest.get("tensors")
    if type(descriptors) is not list or len(descriptors) > _MAX_NODES:
        raise ValueError("The tensor descriptors are invalid.")
    tensors = []
    logical_bytes = 0
    for index, descriptor in enumerate(descriptors):
        if (
            type(descriptor) is not dict
            or type(descriptor.get("id")) is not int
            or descriptor["id"] != index
        ):
            raise ValueError("The tensor identifier is invalid.")
        storage_id = _nonnegative_int(descriptor.get("storage"), "Storage identifier")
        if storage_id >= len(raw_storages):
            raise ValueError("The tensor refers to a missing storage.")
        dtype_name = descriptor.get("dtype")
        if type(dtype_name) is not str or dtype_name not in _DTYPES:
            raise ValueError("The tensor dtype is invalid.")
        shape, stride = descriptor.get("shape"), descriptor.get("stride")
        if (
            type(shape) is not list
            or type(stride) is not list
            or len(shape) != len(stride)
            or len(shape) > 64
        ):
            raise ValueError("The tensor shape or stride is invalid.")
        shape = [_nonnegative_int(size, "Tensor dimension") for size in shape]
        stride = [_nonnegative_int(step, "Tensor stride") for step in stride]
        offset = _nonnegative_int(descriptor.get("storage_offset"), "Storage offset")
        raw = raw_storages[storage_id]
        dtype = _DTYPES[dtype_name]
        element_size = torch.empty(0, dtype=dtype).element_size()
        logical_bytes += math.prod(shape) * element_size
        if max_logical_bytes is not None and logical_bytes > max_logical_bytes:
            raise ValueError("The tensor tree exceeds the logical byte limit.")
        elements = raw.numel() // element_size
        required = (
            0
            if 0 in shape
            else offset + 1 + sum((size - 1) * step for size, step in zip(shape, stride))
        )
        if required > elements:
            raise ValueError("The tensor view exceeds its storage.")
        try:
            base = torch.empty(0, dtype=dtype, device=raw.device).set_(
                raw.untyped_storage(), 0, (elements,), (1,)
            )
            tensors.append(torch.as_strided(base, shape, stride, offset).detach())
        except (RuntimeError, OverflowError) as error:
            raise ValueError("The tensor view is invalid.") from error

    node_count = 0

    def visit(node: Any, depth: int) -> Any:
        nonlocal node_count
        node_count += 1
        if node_count > _MAX_NODES or depth > _MAX_DEPTH or type(node) is not dict:
            raise ValueError("The tree node exceeds the limits or has an invalid type.")
        kind = node.get("kind")
        if kind == "tensor":
            index = _nonnegative_int(node.get("id"), "Tensor identifier")
            if index >= len(tensors):
                raise ValueError("The tree refers to a missing tensor.")
            return tensors[index]
        if kind in ("list", "tuple", "dict"):
            items = node.get("items")
            if type(items) is not list:
                raise ValueError("The container items are invalid.")
            if kind == "dict":
                result = {}
                for pair in items:
                    if type(pair) is not list or len(pair) != 2:
                        raise ValueError("The dictionary entry is invalid.")
                    key = visit(pair[0], depth + 1)
                    _check_key(key)
                    if key in result:
                        raise ValueError("The dictionary contains a duplicate key.")
                    result[key] = visit(pair[1], depth + 1)
                return result
            result = [visit(child, depth + 1) for child in items]
            return tuple(result) if kind == "tuple" else result
        if kind == "none":
            return None
        item = node.get("value")
        if kind == "bool" and type(item) is bool:
            return item
        if kind == "str" and type(item) is str:
            return item
        try:
            if kind == "int" and type(item) is str:
                return int(item)
            if kind == "float" and type(item) is str:
                return float.fromhex(item)
            if (
                kind == "complex"
                and type(node.get("real")) is str
                and type(node.get("imag")) is str
            ):
                return complex(float.fromhex(node["real"]), float.fromhex(node["imag"]))
        except (ValueError, OverflowError) as error:
            raise ValueError("The scalar value is invalid.") from error
        raise ValueError("The tree node has an unsupported scalar type.")

    return visit(manifest.get("tree"), 0)


def clone_tree(value: Any, device: Any = None) -> Any:
    """Clone full backing storage once and preserve strides, offsets, and aliases.

    The clone detaches tensors from autograd and converts Parameters to tensors.
    Unsupported tensor layouts,
    subclasses, and cyclic containers raise explicit errors.
    """
    manifest, examples = _describe_tree(value)
    target = torch.device(device) if device is not None else None
    if target is not None and target.type not in ("cpu", "cuda"):
        raise TypeError(f"Target device {target.type!r} is not supported.")
    raw_storages = []
    for tensor in examples:
        raw = _storage_bytes(tensor)
        raw_storages.append(raw.to(device=target or tensor.device, copy=True))
    return _restore_tree(manifest, raw_storages)


def save_tree(path: str | Path, value: Any) -> None:
    """Write JSON and raw storage files to a new or empty directory."""
    manifest, examples = _describe_tree(value)
    directory = Path(path)
    if directory.is_symlink():
        raise ValueError("The tree directory must not be a symbolic link.")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("The tree directory must be empty.")
    for index, tensor in enumerate(examples):
        filename = f"storage_{index:06d}.bin"
        manifest["storages"][index]["file"] = filename
        raw = _storage_bytes(tensor).cpu()
        with (directory / filename).open("xb") as output:
            output.write(memoryview(raw.numpy()).cast("B"))
    with (directory / "manifest.json").open("x", encoding="utf-8") as output:
        json.dump(manifest, output, allow_nan=False, separators=(",", ":"))


def _read_regular_file(filename: str, max_bytes: int, directory_fd: int) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(filename, flags, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"Cannot open tree file {filename!r}.") from error
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError("The tree file is not regular or exceeds the byte limit.")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            data = source.read(max_bytes + 1)
        if len(data) != info.st_size:
            raise ValueError("The tree file changed size while the reader read it.")
        return data
    finally:
        os.close(descriptor)


def load_tree(path: str | Path, *, max_bytes: int = _DEFAULT_MAX_BYTES) -> Any:
    """Read an untrusted tree without pickle and restore all tensors on the CPU.

    The default limit is one GiB for storage and for all unique tensor elements.
    Callers can set an explicit limit. The reader rejects symbolic links and
    arbitrary file paths.
    """
    max_bytes = _nonnegative_int(max_bytes, "Maximum byte count")
    directory = Path(path)
    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise ValueError("The safe tree reader requires O_NOFOLLOW and O_DIRECTORY.")
    try:
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError("The tree path must name a regular directory.") from error
    try:
        return _load_tree_directory(directory_fd, max_bytes)
    finally:
        os.close(directory_fd)


def _load_tree_directory(directory_fd: int, max_bytes: int) -> Any:
    try:
        manifest = json.loads(
            _read_regular_file("manifest.json", _MAX_MANIFEST_BYTES, directory_fd)
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("The tree manifest is not valid JSON.") from error
    if type(manifest) is not dict or manifest.get("format") != _FORMAT:
        raise ValueError("The tree format is invalid.")
    descriptors = manifest.get("storages")
    if type(descriptors) is not list or len(descriptors) > _MAX_NODES:
        raise ValueError("The storage descriptors are invalid.")
    raw_storages = []
    total_bytes = 0
    for index, descriptor in enumerate(descriptors):
        if (
            type(descriptor) is not dict
            or type(descriptor.get("id")) is not int
            or descriptor["id"] != index
        ):
            raise ValueError("The storage identifier is invalid.")
        filename = f"storage_{index:06d}.bin"
        if descriptor.get("file") != filename:
            raise ValueError("The storage filename is invalid.")
        size = _nonnegative_int(descriptor.get("nbytes"), "Storage byte count")
        total_bytes += size
        if total_bytes > max_bytes:
            raise ValueError("The tensor tree exceeds the storage byte limit.")
        data = _read_regular_file(filename, size, directory_fd)
        if len(data) != size:
            raise ValueError("The storage file has an incorrect byte count.")
        raw_storages.append(
            torch.frombuffer(bytearray(data), dtype=torch.uint8)
            if data
            else torch.empty(0, dtype=torch.uint8)
        )
    return _restore_tree(manifest, raw_storages, max_logical_bytes=max_bytes)


def _exact_scalar_close(
    expected: float | complex,
    actual: float | complex,
    atol: float,
    rtol: float,
) -> bool:
    """Compare finite scalars exactly when floating intermediates overflow."""
    real_difference = Fraction(actual.real) - Fraction(expected.real)
    imag_difference = Fraction(actual.imag) - Fraction(expected.imag)
    difference_squared = real_difference**2 + imag_difference**2
    magnitude_squared = Fraction(expected.real) ** 2 + Fraction(expected.imag) ** 2
    absolute_squared = Fraction(atol) ** 2
    relative_squared = Fraction(rtol) ** 2 * magnitude_squared
    remainder = difference_squared - absolute_squared - relative_squared
    # sqrt(D) <= A + R*sqrt(E). If D-A^2-R^2*E is positive, square
    # the remaining nonnegative sides to compare without a rounded square root.
    return remainder <= 0 or remainder**2 <= 4 * absolute_squared * relative_squared


def _compare_scalar_numbers(
    expected: float | complex,
    actual: float | complex,
    atol: float,
    rtol: float,
) -> tuple[bool, float, float]:
    components = (expected.real, expected.imag, actual.real, actual.imag)
    if any(math.isnan(component) for component in components):
        return False, math.inf, math.inf
    if expected == actual:
        return True, 0.0, 0.0
    if not all(math.isfinite(component) for component in components):
        return False, math.inf, math.inf

    # Keep finite component differences even when the expected magnitude overflows.
    # Scaling operands before subtraction can erase adjacent representable values.
    real_difference = actual.real - expected.real
    imag_difference = actual.imag - expected.imag
    difference = math.hypot(real_difference, imag_difference)
    magnitude = math.hypot(expected.real, expected.imag)
    scale = max(atol, *(abs(component) for component in components))
    scaled_real_difference = (
        real_difference / scale
        if math.isfinite(real_difference)
        else actual.real / scale - expected.real / scale
    )
    scaled_imag_difference = (
        imag_difference / scale
        if math.isfinite(imag_difference)
        else actual.imag / scale - expected.imag / scale
    )
    scaled_difference = math.hypot(scaled_real_difference, scaled_imag_difference)
    scaled_magnitude = math.hypot(expected.real / scale, expected.imag / scale)
    if math.isfinite(difference) and math.isfinite(magnitude):
        relative = difference / max(magnitude, _RELATIVE_FLOOR)
    else:
        denominator = max(scaled_magnitude, _RELATIVE_FLOOR / scale)
        relative = scaled_difference / denominator if denominator else math.inf

    if atol == 0 and rtol == 0:
        # The original scalars differ. Rounded intermediates cannot establish equality.
        matches = False
    else:
        bound = atol if rtol == 0 else atol + rtol * magnitude
        if all(math.isfinite(value) for value in (difference, magnitude, bound)):
            matches = difference <= bound
        else:
            # Normalized differences preserve error metrics. Exact rational norms
            # also preserve tolerance boundaries when normalized bounds round together.
            matches = _exact_scalar_close(expected, actual, atol, rtol)
    return matches, difference, relative


def compare_outputs(expected: Any, actual: Any, atol: float, rtol: float) -> dict[str, Any]:
    """Compare exact tree types, tensor shapes, and dtypes before numeric values.

    Floating values use abs(actual - expected) <= atol + rtol * abs(expected).
    Integer and Boolean values must match exactly. NaN values never match.
    Relative error uses max(abs(expected), 1e-12) as its denominator.
    Non-finite error metrics use JSON null. Empty outputs do not pass.
    """
    for label, tolerance in (("atol", atol), ("rtol", rtol)):
        if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"{label} must be finite and nonnegative.")
    messages: list[str] = []
    tensor_count = scalar_count = value_count = 0
    max_abs = max_rel = 0.0

    def record_metrics(difference: float, magnitude: float) -> None:
        nonlocal max_abs, max_rel
        if math.isnan(difference):
            max_abs = max_rel = math.inf
            return
        max_abs = max(max_abs, difference)
        relative = difference / max(magnitude, _RELATIVE_FLOOR)
        max_rel = max(max_rel, relative if not math.isnan(relative) else math.inf)

    def compare(left: Any, right: Any, location: str) -> None:
        nonlocal tensor_count, scalar_count, value_count, max_abs, max_rel
        if type(left) is not type(right) and not (
            isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        ):
            messages.append(f"{location}: output types differ.")
            return
        if isinstance(left, torch.Tensor):
            tensor_count += 1
            if left.shape != right.shape:
                messages.append(
                    f"{location}: tensor shapes differ ({list(left.shape)} != {list(right.shape)})."
                )
                return
            if left.dtype != right.dtype:
                messages.append(
                    f"{location}: tensor dtypes differ ({left.dtype} != {right.dtype})."
                )
                return
            value_count += left.numel()
            if not left.numel():
                return
            left, right = left.detach().cpu(), right.detach().cpu()
            if not (left.is_floating_point() or left.is_complex()):
                if not torch.equal(left, right):
                    messages.append(f"{location}: integer or Boolean values differ.")
                    for a, b in zip(left.reshape(-1).tolist(), right.reshape(-1).tolist()):
                        if a != b:
                            record_metrics(float(abs(int(a) - int(b))), float(abs(int(a))))
                return
            numeric_dtype = torch.complex128 if left.is_complex() else torch.float64
            left, right = left.to(numeric_dtype), right.to(numeric_dtype)
            equal = torch.isclose(right, left, atol=atol, rtol=rtol, equal_nan=False)
            difference = (right - left).abs()
            difference = torch.where(left == right, torch.zeros_like(difference), difference)
            relative = difference / left.abs().clamp_min(_RELATIVE_FLOOR)
            relative = torch.where(difference == 0, torch.zeros_like(relative), relative)
            abs_value, rel_value = difference.max().item(), relative.max().item()
            max_abs = max(max_abs, abs_value if math.isfinite(abs_value) else math.inf)
            max_rel = max(max_rel, rel_value if math.isfinite(rel_value) else math.inf)
            if not bool(equal.all()):
                messages.append(f"{location}: tensor values differ or contain NaN.")
            return
        if type(left) in (list, tuple):
            if len(left) != len(right):
                messages.append(f"{location}: container lengths differ.")
                return
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, f"{location}[{index}]")
            return
        if type(left) is dict:
            left_keys = {(type(key), key) for key in left}
            right_keys = {(type(key), key) for key in right}
            if left_keys != right_keys:
                messages.append(f"{location}: dictionary keys differ.")
                return
            for key in left:
                compare(left[key], right[key], f"{location}[{key!r}]")
            return
        scalar_count += 1
        if left is not None and (type(left) is not str or left):
            value_count += 1
        if type(left) in (float, complex):
            matches, difference, relative = _compare_scalar_numbers(left, right, atol, rtol)
            max_abs = max(max_abs, difference)
            max_rel = max(max_rel, relative)
            if not matches:
                messages.append(f"{location}: scalar values differ or contain NaN.")
        elif left != right:
            messages.append(f"{location}: scalar values differ.")
            if type(left) in (int, bool):
                record_metrics(float(abs(int(right) - int(left))), float(abs(int(left))))

    try:
        _describe_tree(expected)
        _describe_tree(actual)
        compare(expected, actual, "output")
    except (TypeError, ValueError, RuntimeError, OverflowError) as error:
        messages.append(f"The output tree is unsupported: {error}")
    if value_count == 0 and not messages:
        messages.append("The output is empty and cannot establish correctness.")
    return {
        "success": not messages,
        "message": messages[0] if messages else "All output values match.",
        "mismatches": messages,
        "max_abs_error": max_abs if math.isfinite(max_abs) else None,
        "max_rel_error": max_rel if math.isfinite(max_rel) else None,
        "tensor_count": tensor_count,
        "scalar_count": scalar_count,
    }
