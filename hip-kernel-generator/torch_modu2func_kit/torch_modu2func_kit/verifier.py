# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import copy
import importlib.util
import inspect
import math
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional.
    np = None


REQUIRED_ORIGINAL_EXPORTS = ("Model", "get_inputs", "get_init_inputs")
REQUIRED_CANDIDATE_EXPORTS = ("module_fn", "Model", "get_inputs", "get_init_inputs")


def _path_text(path: Path) -> str:
    return path.as_posix()


@dataclass(slots=True)
class VerificationResult:
    success: bool
    message: str


def _set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {_path_text(module_path)}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    # Execute current bytes instead of a timestamp-based bytecode cache.
    exec(compile(module_path.read_bytes(), str(module_path), "exec", dont_inherit=True), module.__dict__)
    return module


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        # Verification needs independent values with the original storage layout.
        return copy.deepcopy(value.detach()).requires_grad_(value.requires_grad)
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _ensure_required_exports(module: ModuleType, module_path: Path, required_exports: tuple[str, ...]) -> None:
    for export_name in required_exports:
        if not hasattr(module, export_name):
            raise AttributeError(f"{_path_text(module_path)} is missing required export `{export_name}`.")


def _normalize_call_args(value: Any) -> tuple[list[Any], dict[str, Any]]:
    if value is None:
        return [], {}
    if isinstance(value, dict):
        return [], _clone_value(value)
    if isinstance(value, tuple):
        return list(_clone_value(value)), {}
    if isinstance(value, list):
        return _clone_value(value), {}
    return [_clone_value(value)], {}


def _compare_scalars(expected: Any, actual: Any, rtol: float, atol: float) -> tuple[bool, str]:
    if isinstance(expected, bool) and isinstance(actual, bool):
        return expected == actual, f"Boolean mismatch: expected {expected}, got {actual}."
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if math.isclose(float(expected), float(actual), rel_tol=rtol, abs_tol=atol):
            return True, ""
        return False, f"Scalar mismatch: expected {expected}, got {actual}."
    if expected == actual:
        return True, ""
    return False, f"Value mismatch: expected {expected!r}, got {actual!r}."


def _compare_outputs(expected: Any, actual: Any, rtol: float, atol: float, path: str = "output") -> tuple[bool, str]:
    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        if expected.shape != actual.shape:
            return False, f"{path} shape mismatch: expected {tuple(expected.shape)}, got {tuple(actual.shape)}."
        matches = torch.allclose(
            expected.detach().cpu(),
            actual.detach().cpu(),
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
        if matches:
            return True, ""
        max_abs = torch.max(torch.abs(expected.detach().cpu() - actual.detach().cpu())).item()
        return False, f"{path} tensor mismatch: max abs diff={max_abs:.6g}."

    if type(expected) is not type(actual):
        return False, f"{path} type mismatch: expected {type(expected).__name__}, got {type(actual).__name__}."

    if isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            return False, f"{path} length mismatch: expected {len(expected)}, got {len(actual)}."
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            matches, detail = _compare_outputs(expected_item, actual_item, rtol, atol, f"{path}[{index}]")
            if not matches:
                return False, detail
        return True, ""

    if isinstance(expected, dict):
        if expected.keys() != actual.keys():
            return False, f"{path} dict keys mismatch: expected {sorted(expected.keys())}, got {sorted(actual.keys())}."
        for key in expected:
            matches, detail = _compare_outputs(expected[key], actual[key], rtol, atol, f"{path}[{key!r}]")
            if not matches:
                return False, detail
        return True, ""

    matches, detail = _compare_scalars(expected, actual, rtol, atol)
    if matches:
        return True, ""
    return False, f"{path} {detail}"


def verify_candidate(
    original_module_path: Path,
    candidate_module_path: Path,
    *,
    seed: int,
    rtol: float,
    atol: float,
) -> VerificationResult:
    try:
        original_module = _load_module(
            original_module_path, f"torch_modu2func_original_{original_module_path.stem}_{seed}"
        )
        candidate_module = _load_module(
            candidate_module_path, f"torch_modu2func_candidate_{candidate_module_path.stem}_{seed}"
        )
        _ensure_required_exports(candidate_module, candidate_module_path, REQUIRED_CANDIDATE_EXPORTS)
        _ensure_required_exports(original_module, original_module_path, REQUIRED_ORIGINAL_EXPORTS)
        module_fn = getattr(candidate_module, "module_fn")
        if not callable(module_fn):
            raise TypeError("Candidate module_fn must be callable.")

        _set_seed(seed)
        init_args = getattr(original_module, "get_init_inputs")()
        original_init_args, original_init_kwargs = _normalize_call_args(init_args)

        _set_seed(seed)
        original_model = getattr(original_module, "Model")(*_clone_value(original_init_args), **_clone_value(original_init_kwargs))

        _set_seed(seed)
        candidate_model = getattr(candidate_module, "Model")(*_clone_value(original_init_args), **_clone_value(original_init_kwargs))

        original_model.eval()
        candidate_model.eval()

        _set_seed(seed)
        forward_inputs = getattr(original_module, "get_inputs")()
        original_forward_args, original_forward_kwargs = _normalize_call_args(forward_inputs)
        candidate_forward_args = _clone_value(original_forward_args)
        candidate_forward_kwargs = _clone_value(original_forward_kwargs)

        fn_calls = 0

        def tracked_fn(*args, **kwargs):
            nonlocal fn_calls
            fn_calls += 1
            return module_fn(*args, **kwargs)

        forward_signature = inspect.signature(candidate_model.forward)
        fn_parameter = forward_signature.parameters.get("fn")
        if fn_parameter is None or fn_parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            raise TypeError("Candidate Model.forward must accept the keyword parameter fn.")
        forward_signature.bind(*candidate_forward_args, **candidate_forward_kwargs, fn=tracked_fn)

        with torch.no_grad():
            _set_seed(seed)
            expected = original_model(*_clone_value(original_forward_args), **_clone_value(original_forward_kwargs))
            _set_seed(seed)
            actual = candidate_model(*candidate_forward_args, **candidate_forward_kwargs)

        matches, detail = _compare_outputs(expected, actual, rtol, atol)
        if not matches:
            return VerificationResult(False, detail)

        # A fresh model keeps the injected call independent of forward mutations.
        _set_seed(seed)
        injected_model = getattr(candidate_module, "Model")(
            *_clone_value(original_init_args), **_clone_value(original_init_kwargs)
        )
        injected_model.eval()
        with torch.no_grad():
            _set_seed(seed)
            injected = injected_model(
                *_clone_value(original_forward_args),
                **_clone_value(original_forward_kwargs),
                fn=tracked_fn,
            )
        if fn_calls == 0:
            return VerificationResult(False, "Candidate Model.forward did not call the injected fn.")
        matches, detail = _compare_outputs(expected, injected, rtol, atol)
        if not matches:
            return VerificationResult(False, f"Injected fn: {detail}")
        return VerificationResult(True, "Default and injected outputs matched within tolerance.")
    except Exception as exc:
        return VerificationResult(False, f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}")
