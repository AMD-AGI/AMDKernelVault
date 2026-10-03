# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Test device metadata with CPU tensors and explicit synthetic GPU descriptors."""

from __future__ import annotations

import copy

import pytest
import torch

from triton_kernel_gen.errors import ReferenceValidationError
from triton_kernel_gen.output_metadata import compare_output_devices
from triton_kernel_gen.tensor_utils import tree_metadata
from triton_kernel_gen.worker import _assert_native_output


@pytest.mark.parametrize(
    "expected,actual", [("cuda:0", "cpu"), ("cpu", "cuda:0"), ("cuda:0", "cuda:1")]
)
def test_native_device_type_and_index_must_match(expected, actual):
    reference = tree_metadata({"nested": [torch.ones(2)]})
    candidate = copy.deepcopy(reference)
    reference["tensors"][0]["device"] = expected
    candidate["tensors"][0]["device"] = actual
    result = compare_output_devices(reference, candidate)
    assert not result["success"]
    assert "native device mismatch" in result["message"]
    assert "nested" in result["message"]


def test_cpu_reference_remains_a_valid_device_contract():
    reference = tree_metadata((torch.ones(2), {"value": torch.zeros(3)}))
    assert compare_output_devices(reference, reference)["success"]


def test_device_check_uses_tensor_positions_instead_of_storage_aliases():
    tensor = torch.ones(2)
    reference = tree_metadata({"a": tensor, "b": tensor})
    candidate = tree_metadata({"b": tensor.clone(), "a": tensor.clone()})
    assert compare_output_devices(reference, candidate)["success"]


def test_device_check_preserves_python_dictionary_key_equality():
    reference = tree_metadata({-0.0: torch.ones(1)})
    candidate = tree_metadata({0: torch.ones(1)})
    assert compare_output_devices(reference, candidate)["success"]


def test_stage1_device_check_precedes_cpu_value_comparison():
    expected = tree_metadata(torch.ones(2))
    expected["tensors"][0]["device"] = "cuda:0"
    with pytest.raises(ReferenceValidationError, match="native device mismatch"):
        _assert_native_output(expected, torch.ones(2), "The functional device differs")
