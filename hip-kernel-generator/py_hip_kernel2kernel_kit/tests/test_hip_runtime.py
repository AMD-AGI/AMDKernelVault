# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os
import py_compile
import sys
from types import SimpleNamespace

import pytest
import torch.utils.cpp_extension

from py_hip_kernel2kernel_kit.hip_runtime import load_hip_forward, load_python_module


def test_load_forward_preserves_candidate_and_baseline_include_paths(monkeypatch, tmp_path):
    source = tmp_path / "candidate" / "kernel.hip"
    source.parent.mkdir()
    source.write_text("// CPU test source\n", encoding="utf-8")
    candidate_include = source.parent / "include"
    candidate_include.mkdir()
    baseline_include = tmp_path / "baseline" / "include"
    captured = {}

    def fake_load(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(forward=lambda value: value + 1)

    monkeypatch.setattr(torch.utils.cpp_extension, "load", fake_load)
    forward = load_hip_forward(
        source, tmp_path / "build", offload_arch="gfx90a",
        extra_include_paths=[str(baseline_include)],
    )

    assert forward(2) == 3
    assert captured["extra_include_paths"] == [str(baseline_include), str(candidate_include)]
    assert captured["extra_cuda_cflags"] == ["--offload-arch=gfx90a"]
    assert captured["with_cuda"] is True


def test_python_loader_ignores_stale_bytecode_after_same_size_rewrite(tmp_path):
    source = tmp_path / "sample.py"
    source.write_text("VALUE = 'first'\n", encoding="utf-8")
    original_stat = source.stat()
    py_compile.compile(str(source), doraise=True)
    source.write_text("VALUE = 'other'\n", encoding="utf-8")
    os.utime(source, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    module_name = "hip2hip_test_same_size_rewrite"
    try:
        module = load_python_module(source, module_name)
        assert module.VALUE == "other"
        assert module.__file__ == str(source)
        assert sys.modules[module_name] is module
    finally:
        sys.modules.pop(module_name, None)


def test_python_loader_does_not_inherit_future_annotations(tmp_path):
    source = tmp_path / "annotations.py"
    source.write_text("VALUE: int = 1\n", encoding="utf-8")
    module_name = "hip2hip_test_annotation_semantics"
    try:
        module = load_python_module(source, module_name)
        assert module.__annotations__["VALUE"] is int
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize("existing", [False, True])
def test_python_loader_cleans_failed_imports(monkeypatch, tmp_path, existing):
    source = tmp_path / "failure.py"
    source.write_text("RETAINED = bytearray(1024)\nraise RuntimeError('test failure')\n", encoding="utf-8")
    module_name = "hip2hip_test_failed_import"
    prior_module = SimpleNamespace()
    if existing:
        monkeypatch.setitem(sys.modules, module_name, prior_module)
    else:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    with pytest.raises(RuntimeError, match="test failure"):
        load_python_module(source, module_name)

    if existing:
        assert sys.modules[module_name] is prior_module
    else:
        assert module_name not in sys.modules
