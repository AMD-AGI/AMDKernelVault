# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import os
import py_compile
from pathlib import Path

from torch2hip_kit.hip_runtime import load_python_module


def test_python_loader_ignores_stale_timestamp_bytecode(tmp_path: Path) -> None:
    source = tmp_path / "reference.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    original_stat = source.stat()
    py_compile.compile(str(source), cfile=importlib.util.cache_from_source(str(source)), doraise=True)
    source.write_text("VALUE = 2\n", encoding="utf-8")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))

    loaded = load_python_module(source, "torch2hip_audit_fresh_source")

    assert loaded.VALUE == 2


def test_python_loader_preserves_source_annotation_semantics(tmp_path: Path) -> None:
    source = tmp_path / "annotations.py"
    source.write_text("VALUE: int = 1\n", encoding="utf-8")

    loaded = load_python_module(source, "torch2hip_audit_annotations")

    assert loaded.__annotations__["VALUE"] is int
