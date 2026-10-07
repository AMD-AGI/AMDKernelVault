# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the JSON protocol without importing a GPU runtime."""

from __future__ import annotations

import math
import os
from pathlib import Path, PurePosixPath
from typing import Any

from triton_rl.contracts import EvaluationResult

PROTOCOL_VERSION = 1
MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
REFERENCE_SEPARATOR = "#" * 146


class ProtocolError(ValueError):
    """The request or response does not satisfy the execution protocol."""


def subprocess_environment() -> dict[str, str]:
    """Keep package imports valid after a subprocess changes its directory."""
    env = os.environ.copy()
    paths = [str(Path(__file__).resolve().parents[2])]
    paths.extend(
        str(Path(path).resolve()) for path in env.get("PYTHONPATH", "").split(os.pathsep) if path
    )
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def tolerance(value: Any, name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ProtocolError(f"{name} must be a finite, nonnegative number.")
    return float(value)


def validate_filename(filename: Any) -> str:
    if not isinstance(filename, str) or not filename or "\\" in filename or "\x00" in filename:
        raise ProtocolError("The reference filename must be a relative POSIX path.")
    path = PurePosixPath(filename)
    if path.is_absolute() or any(part in ("", ".", "..") for part in filename.split("/")):
        raise ProtocolError("The reference filename must stay inside the reference directory.")
    if path.suffix != ".py":
        raise ProtocolError("The reference filename must end with .py.")
    return filename


def resolve_reference(root: Path, filename: str) -> Path:
    validate_filename(filename)
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ProtocolError("The reference root must be a directory.")
    try:
        reference = (resolved_root / filename).resolve(strict=True)
    except OSError as exc:
        raise ProtocolError("The requested reference file does not exist.") from exc
    if not reference.is_relative_to(resolved_root) or not reference.is_file():
        raise ProtocolError("The reference file must stay inside the reference directory.")
    return reference


def validate_request(value: Any) -> dict[str, Any]:
    expected = {"protocol_version", "code", "filename", "atol", "rtol"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ProtocolError("The request contains missing or unsupported fields.")
    if type(value["protocol_version"]) is not int or value["protocol_version"] != PROTOCOL_VERSION:
        raise ProtocolError("The execution protocol version is not supported.")
    if not isinstance(value["code"], str) or not value["code"].strip():
        raise ProtocolError("The candidate source must be a nonempty string.")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "code": value["code"],
        "filename": validate_filename(value["filename"]),
        "atol": tolerance(value["atol"], "atol"),
        "rtol": tolerance(value["rtol"], "rtol"),
    }


def validate_result(value: Any) -> EvaluationResult:
    if not isinstance(value, dict):
        raise ProtocolError("The execution result must be an object.")
    try:
        result = EvaluationResult.from_dict(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError("The execution result has an invalid schema.") from exc
    if not isinstance(result.feedback, str) or not isinstance(result.diagnostics, dict):
        raise ProtocolError("The execution feedback has an invalid type.")
    if result.error_type is not None and not isinstance(result.error_type, str):
        raise ProtocolError("The execution error type must be a string or null.")
    for name in ("speedup", "latency_ms", "baseline_latency_ms"):
        number = getattr(result, name)
        if number is not None:
            if type(number) not in (int, float) or not math.isfinite(number) or number <= 0:
                raise ProtocolError(f"{name} must be a finite, positive number or null.")
            if not result.correct:
                raise ProtocolError(
                    "An incorrect candidate cannot receive a performance measurement."
                )
    return result
