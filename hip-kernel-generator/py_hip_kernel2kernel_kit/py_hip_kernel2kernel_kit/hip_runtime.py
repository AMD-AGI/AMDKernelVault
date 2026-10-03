# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import hashlib
import importlib.util
import re
import shutil
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterable


def _path_text(path: Path) -> str:
    return path.as_posix()


def load_python_module(module_path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to create import spec for {_path_text(module_path)}.")
    module = importlib.util.module_from_spec(spec)
    missing = object()
    previous_module = sys.modules.get(module_name, missing)
    sys.modules[module_name] = module
    try:
        # Generated files can change without a timestamp or size change.
        # Compile the current bytes instead of accepting cached bytecode.
        exec(compile(module_path.read_bytes(), str(module_path), "exec", dont_inherit=True), module.__dict__)
    except BaseException:
        if sys.modules.get(module_name) is module:
            if previous_module is missing:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module
        raise
    return module


def cleanup_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def unique_extension_name(hip_path: Path, build_dir: Path) -> str:
    safe_stem = re.sub(r"\W+", "_", hip_path.stem)
    payload = f"{hip_path.resolve()}::{build_dir.resolve()}::{hip_path.read_text(encoding='utf-8')}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"hip2hip_{safe_stem}_{digest}"


def load_hip_forward(
    hip_path: Path,
    build_dir: Path,
    *,
    extension_name: str | None = None,
    verbose: bool = False,
    offload_arch: str | None = None,
    extra_include_paths: Iterable[str] | None = None,
) -> Callable[..., Any]:
    from torch.utils.cpp_extension import load

    build_dir.mkdir(parents=True, exist_ok=True)
    include_dir = hip_path.parent / "include"
    resolved_include_paths = list(extra_include_paths or [])
    if include_dir.exists():
        resolved_include_paths.append(str(include_dir))

    extra_cuda_cflags: list[str] = []
    if offload_arch:
        extra_cuda_cflags.append(f"--offload-arch={offload_arch}")

    extension = load(
        name=extension_name or unique_extension_name(hip_path, build_dir),
        sources=[str(hip_path)],
        build_directory=str(build_dir),
        extra_include_paths=resolved_include_paths,
        extra_cuda_cflags=extra_cuda_cflags,
        with_cuda=True,
        verbose=verbose,
    )
    if not hasattr(extension, "forward"):
        raise AttributeError(f"Compiled HIP extension {_path_text(hip_path)} does not expose `forward`.")
    return extension.forward
