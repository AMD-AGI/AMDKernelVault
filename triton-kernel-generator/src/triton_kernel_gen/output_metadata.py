# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Compare native output devices before serialization removes placement evidence."""

from __future__ import annotations

from typing import Any


def _key_value(node: dict[str, Any]) -> Any:
    kind = node["kind"]
    if kind in {"bool", "str"}:
        return node["value"]
    if kind == "int":
        return int(node["value"])
    if kind == "float":
        return float.fromhex(node["value"])
    if kind == "complex":
        return complex(float.fromhex(node["real"]), float.fromhex(node["imag"]))
    if kind == "none":
        return None
    raise ValueError("Native output metadata contains an invalid dictionary key.")


def _output_devices(metadata: dict[str, Any]) -> dict[tuple[Any, ...], tuple[str, int | None]]:
    tensors = metadata["tensors"]
    devices: dict[tuple[Any, ...], tuple[str, int | None]] = {}

    def visit(node: dict[str, Any], path: tuple[Any, ...]) -> None:
        kind = node["kind"]
        if kind == "tensor":
            descriptor = tensors[node["id"]]
            device = descriptor["device"]
            device_type, separator, index = device.partition(":")
            if device_type not in {"cpu", "cuda"}:
                raise ValueError("Native output metadata contains an unsupported device type.")
            device_index = int(index) if separator else None
            if device_index is not None and device_index < 0:
                raise ValueError("Native output metadata contains an invalid device index.")
            devices[path] = (device_type, device_index)
        elif kind in {"list", "tuple"}:
            for index, child in enumerate(node["items"]):
                visit(child, (*path, ("index", index)))
        elif kind == "dict":
            for key, child in node["items"]:
                visit(child, (*path, ("key", _key_value(key))))

    visit(metadata["tree"], ())
    return devices


def compare_output_devices(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    """Compare device type and index for every tensor position in an output tree."""
    expected_devices = _output_devices(expected)
    actual_devices = _output_devices(actual)
    if expected_devices.keys() != actual_devices.keys():
        return {"success": False, "message": "The native output tensor positions differ."}
    for path, expected_device in expected_devices.items():
        actual_device = actual_devices[path]
        if actual_device != expected_device:
            label = "output" + "".join(f"[{part[1]!r}]" for part in path)
            return {
                "success": False,
                "message": f"{label} native device mismatch: expected {expected_device}, got {actual_device}.",
            }
    return {"success": True, "message": "Native output devices match."}
