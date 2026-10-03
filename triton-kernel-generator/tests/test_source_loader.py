# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Check declared source imports without GPU packages or model services."""

from __future__ import annotations

import importlib.util
import inspect
import math
import os
import py_compile
import sys
from pathlib import Path
from types import ModuleType

import pytest

from triton_kernel_gen.source_loader import TrustedSourceImporter, load_module

_HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.\n"
)


def _source(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HEADER + body, encoding="utf-8")
    return path


def _stale_cache(path: Path):
    py_compile.compile(str(path), doraise=True)
    cache = Path(importlib.util.cache_from_source(str(path)))
    cached_bytes = cache.read_bytes()
    before = path.stat()
    original = path.read_text()
    replacement = original.replace("VALUE = 1", "VALUE = 2")
    assert len(replacement) == len(original) and replacement != original
    path.write_text(replacement)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    return cache, cached_bytes


def test_load_module_ignores_stale_main_bytecode(tmp_path):
    source = _source(tmp_path / "main.py", "VALUE = 1\n")
    cache, cached_bytes = _stale_cache(source)

    module = load_module(source, "_source_main")

    assert module.VALUE == 2
    assert module.__file__ == str(source.resolve())
    assert module.__cached__ is None
    assert cache.read_bytes() == cached_bytes
    assert "_source_main" not in sys.modules


def test_declared_helper_ignores_stale_bytecode_and_existing_module(tmp_path, monkeypatch):
    helper = _source(tmp_path / "source_helper.py", "VALUE = 1\n")
    cache, cached_bytes = _stale_cache(helper)
    main = _source(tmp_path / "main.py", "from source_helper import VALUE\n")
    stale = ModuleType("source_helper")
    stale.__file__ = str(helper)
    stale.VALUE = 99
    monkeypatch.setitem(sys.modules, "source_helper", stale)

    module = load_module(main, "_source_main", (helper,))

    assert module.VALUE == 2
    assert sys.modules["source_helper"] is stale
    assert cache.read_bytes() == cached_bytes


def test_source_imports_create_no_bytecode_and_preserve_metadata(tmp_path):
    helper = _source(tmp_path / "source_helper.py", "VALUE = 7\n")
    main = _source(
        tmp_path / "main.py", "from source_helper import VALUE\ndef read():\n    return VALUE\n"
    )
    original_setting = sys.dont_write_bytecode

    module = load_module(main, "_source_main", (helper,))

    assert module.read() == 7
    assert module.read.__code__.co_filename == str(main.resolve())
    assert "def read():" in inspect.getsource(module.read)
    assert "def read():" in module.__loader__.get_source(module.__name__)
    assert module.__spec__.origin == str(main.resolve())
    assert not tuple(tmp_path.rglob("*.pyc"))
    assert sys.dont_write_bytecode is original_setting


def test_compile_bytes_honors_python_source_encoding(tmp_path):
    path = tmp_path / "latin.py"
    path.write_bytes(b"# coding: latin-1\n" + _HEADER.encode() + b"TEXT = 'caf\xe9'\n")

    assert load_module(path, "_source_latin").TEXT == "caf\xe9"


def test_package_relative_imports_and_declared_initializers(tmp_path):
    package = tmp_path / "source_package"
    init = _source(package / "__init__.py", "BASE = 10\n")
    helper = _source(package / "helper.py", "VALUE = 3\n")
    inner_init = _source(package / "nested" / "__init__.py", "LABEL = 'nested'\n")
    nested = _source(
        package / "nested" / "math.py", "from ..helper import VALUE\nDOUBLE = 2 * VALUE\n"
    )
    main = _source(
        package / "main.py",
        "from . import helper\nfrom .nested.math import DOUBLE\nfrom . import BASE\nVALUE = BASE + helper.VALUE + DOUBLE\n",
    )

    module = load_module(main, "_source_package_main", (init, helper, inner_init, nested))

    assert module.VALUE == 19
    assert module.__package__ == "source_package"
    assert module.__name__ == "_source_package_main"
    assert not any(
        name == "source_package" or name.startswith("source_package.") for name in sys.modules
    )
    assert not tuple(tmp_path.rglob("*.pyc"))


def test_package_initializers_also_ignore_stale_bytecode(tmp_path):
    package = tmp_path / "source_package"
    init = _source(package / "__init__.py", "VALUE = 1\n")
    cache, cached_bytes = _stale_cache(init)
    main = _source(package / "main.py", "from . import VALUE\n")

    module = load_module(main, "_source_package_main", (init,))

    assert module.VALUE == 2
    assert cache.read_bytes() == cached_bytes


@pytest.mark.parametrize("missing", ["outer", "inner"])
def test_every_enclosing_package_initializer_requires_declaration(tmp_path, missing):
    package = tmp_path / "source_package"
    outer = _source(
        package / "__init__.py", "raise AssertionError('This initializer must not execute.')\n"
    )
    inner = _source(
        package / "nested" / "__init__.py",
        "raise AssertionError('This initializer must not execute.')\n",
    )
    helper = _source(package / "nested" / "helper.py", "VALUE = 3\n")
    main = _source(package / "nested" / "main.py", "from .helper import VALUE\n")
    dependencies = [helper, inner if missing == "outer" else outer]

    with pytest.raises(ImportError, match="Package initializer.*must appear in dependency_paths"):
        load_module(main, "_source_package_main", dependencies)
    assert "_source_package_main" not in sys.modules


def test_context_supports_delayed_imports_and_importlib(tmp_path):
    helper = _source(tmp_path / "source_helper.py", "VALUE = 4\n")
    main = _source(
        tmp_path / "main.py",
        "def read():\n    from source_helper import VALUE\n    return VALUE\ndef dynamic():\n    import importlib\n    return importlib.import_module('source_helper').VALUE\n",
    )

    with TrustedSourceImporter((helper,)) as importer:
        module = importer.load_module(main, "_source_main")
        assert "source_helper" not in sys.modules
        assert module.read() == 4
        assert module.dynamic() == 4
    assert "source_helper" not in sys.modules
    assert "_source_main" not in sys.modules


def test_separate_contexts_do_not_reuse_helper_modules(tmp_path):
    modules = []
    for index in (1, 2):
        helper = _source(tmp_path / str(index) / "source_helper.py", f"VALUE = {index}\n")
        main = _source(
            tmp_path / str(index) / "main.py",
            "import source_helper\ndef read():\n    return source_helper.VALUE\n",
        )
        modules.append(load_module(main, f"_source_main_{index}", (helper,)))

    assert modules[0].read() == 1
    assert modules[1].read() == 2
    assert modules[0].source_helper is not modules[1].source_helper
    assert "source_helper" not in sys.modules


def test_context_restores_existing_package_tree_and_import_hooks(tmp_path, monkeypatch):
    init = _source(tmp_path / "source_package" / "__init__.py", "VALUE = 3\n")
    helper = _source(tmp_path / "source_package" / "helper.py", "VALUE = 4\n")
    main = _source(
        tmp_path / "main.py",
        "from source_package.helper import VALUE\nraise RuntimeError('source failed')\n",
    )
    existing_package = ModuleType("source_package")
    existing_package.__file__ = str(init)
    existing_helper = ModuleType("source_package.helper")
    existing_helper.__file__ = str(helper)
    existing_package.helper = existing_helper
    monkeypatch.setitem(sys.modules, "source_package", existing_package)
    monkeypatch.setitem(sys.modules, "source_package.helper", existing_helper)
    original_path = list(sys.path)
    original_meta = list(sys.meta_path)

    with pytest.raises(RuntimeError, match="source failed"):
        load_module(main, "_source_main", (init, helper))

    assert sys.path == original_path
    assert sys.meta_path == original_meta
    assert sys.modules["source_package"] is existing_package
    assert sys.modules["source_package.helper"] is existing_helper
    assert existing_package.helper is existing_helper
    assert "_source_main" not in sys.modules


@pytest.mark.parametrize("as_package", [False, True])
def test_undeclared_local_helper_is_rejected(tmp_path, as_package):
    root = tmp_path / "source_package" if as_package else tmp_path
    init = _source(root / "__init__.py", "") if as_package else None
    _source(
        root / "source_hidden.py", "raise AssertionError('Undeclared source must not execute.')\n"
    )
    statement = (
        "from .source_hidden import VALUE" if as_package else "from source_hidden import VALUE"
    )
    main = _source(root / "main.py", statement + "\n")

    with pytest.raises(ModuleNotFoundError, match="not declared in dependency_paths"):
        load_module(main, "_source_main", (init,) if init else ())


def test_cached_undeclared_sibling_cannot_bypass_declaration_check(tmp_path, monkeypatch):
    _source(tmp_path / "source_hidden.py", "VALUE = 3\n")
    main = _source(tmp_path / "main.py", "from source_hidden import VALUE\n")
    stale = ModuleType("source_hidden")
    stale.__file__ = str(tmp_path / "source_hidden.py")
    stale.VALUE = 99
    monkeypatch.setitem(sys.modules, "source_hidden", stale)

    with pytest.raises(ModuleNotFoundError, match="not declared in dependency_paths"):
        load_module(main, "_source_main")
    assert sys.modules["source_hidden"] is stale


def test_context_handles_circular_declared_helpers(tmp_path):
    first = _source(
        tmp_path / "source_first.py",
        "VALUE = 2\nimport source_second\nTOTAL = VALUE + source_second.VALUE\n",
    )
    second = _source(
        tmp_path / "source_second.py", "import source_first\nVALUE = source_first.VALUE + 1\n"
    )
    main = _source(tmp_path / "main.py", "from source_first import TOTAL\n")

    assert load_module(main, "_source_main", (first, second)).TOTAL == 5


def test_decorator_introspection_sees_module_during_execution(tmp_path):
    helper = _source(
        tmp_path / "source_helper.py",
        "from __future__ import annotations\nfrom dataclasses import dataclass\n@dataclass\nclass Item:\n    value: int\n",
    )
    main = _source(tmp_path / "main.py", "from source_helper import Item\nitem = Item(3)\n")

    assert load_module(main, "_source_main", (helper,)).item.value == 3


def test_relative_and_qualified_imports_share_one_packaged_helper(tmp_path):
    init = _source(tmp_path / "source_package" / "__init__.py", "")
    helper = _source(tmp_path / "source_package" / "source_helper.py", "TOKEN = object()\n")
    main = _source(
        tmp_path / "source_package" / "main.py",
        "from . import source_helper\nimport source_package.source_helper\nSAME = source_helper is source_package.source_helper\n",
    )

    assert load_module(main, "_source_main", (init, helper)).SAME
    assert "source_helper" not in sys.modules


def test_circular_import_through_an_alias_executes_helper_once(tmp_path):
    init = _source(tmp_path / "source_package" / "__init__.py", "")
    helper = _source(
        tmp_path / "source_package" / "source_helper.py",
        "RUNS = globals().get('RUNS', 0) + 1\nimport source_package.source_helper\n",
    )
    with TrustedSourceImporter((init, helper)) as importer:
        module = importer.load_module(helper, "_source_helper")
        assert module.RUNS == 1
        assert sys.modules["source_package"].source_helper is module


def test_failed_helper_import_removes_aliases_and_package_attributes(tmp_path):
    init = _source(tmp_path / "source_package" / "__init__.py", "source_helper = 'sentinel'\n")
    helper = _source(
        tmp_path / "source_package" / "source_helper.py",
        "import source_package.source_helper\nraise RuntimeError('helper failed')\n",
    )
    with TrustedSourceImporter((init, helper)) as importer:
        with pytest.raises(RuntimeError, match="helper failed"):
            importer.load_module(helper, "_source_helper")
        assert "_source_helper" not in sys.modules
        assert "source_package.source_helper" not in sys.modules
        assert sys.modules["source_package"].source_helper == "sentinel"
        with pytest.raises(RuntimeError, match="helper failed"):
            __import__("source_package.source_helper")


def test_packaged_helpers_with_same_basename_use_qualified_imports(tmp_path):
    dependencies = []
    for value, name in enumerate(("source_first", "source_second"), start=1):
        dependencies.append(_source(tmp_path / name / "__init__.py", ""))
        dependencies.append(_source(tmp_path / name / "source_helper.py", f"VALUE = {value}\n"))
    main = _source(
        tmp_path / "main.py",
        "from source_first.source_helper import VALUE as FIRST\n"
        "from source_second.source_helper import VALUE as SECOND\nVALUE = FIRST + SECOND\n",
    )

    assert load_module(main, "_source_main", dependencies).VALUE == 3
    assert "source_helper" not in sys.modules


def test_importer_requires_an_active_context(tmp_path):
    main = _source(tmp_path / "main.py", "VALUE = 1\n")

    with pytest.raises(RuntimeError, match="Enter the source import context"):
        TrustedSourceImporter().load_module(main, "_source_main")


def test_direct_package_load_binds_parent_and_restores_previous_package(tmp_path, monkeypatch):
    package = tmp_path / "source_package"
    init = _source(package / "__init__.py", "original = 'initializer value'\n")
    original = _source(package / "original.py", "VALUE = 1\n")
    functional = _source(
        package / "functional.py",
        "import source_package.original\nVALUE = source_package.original.VALUE\n",
    )
    previous_package = ModuleType("source_package")
    previous_package.__file__ = str(init)
    previous_child = ModuleType("source_package.original")
    previous_child.__file__ = str(original)
    previous_package.original = previous_child
    monkeypatch.setitem(sys.modules, "source_package", previous_package)
    monkeypatch.setitem(sys.modules, "source_package.original", previous_child)

    with TrustedSourceImporter((init,)) as importer:
        first = importer.load_module(original, "_source_original")
        result = importer.load_module(functional, "_source_functional")
        active_package = sys.modules["source_package"]
        assert active_package.original is first
        assert result.VALUE == 1
    assert sys.modules["source_package"] is previous_package
    assert previous_package.original is previous_child
    assert sys.modules["source_package.original"] is previous_child
    assert active_package.original == "initializer value"

    cache, cached_bytes = _stale_cache(original)
    with TrustedSourceImporter((init,)) as importer:
        second = importer.load_module(original, "_source_original")
        result = importer.load_module(functional, "_source_functional")
        assert sys.modules["source_package"].original is second
        assert result.VALUE == 2
    assert second is not first
    assert first.VALUE == 1
    assert cache.read_bytes() == cached_bytes
    assert previous_package.original is previous_child


@pytest.mark.parametrize("previous", [None, "sentinel"])
def test_context_restores_retained_parent_attributes(tmp_path, previous):
    package = tmp_path / "source_package"
    init = _source(package / "__init__.py", "" if previous is None else "main = 'sentinel'\n")
    main = _source(
        package / "main.py",
        "import source_package\nVALUE = 7\ndef read():\n    return source_package.main.VALUE\n",
    )

    with TrustedSourceImporter((init,)) as importer:
        module = importer.load_module(main, "_source_main")
        retained_package = sys.modules["source_package"]
        assert module.read() == 7
        assert retained_package.main is module
    if previous is None:
        assert "main" not in vars(retained_package)
    else:
        assert retained_package.main == previous
    assert "source_package" not in sys.modules
    assert "source_package.main" not in sys.modules


@pytest.mark.parametrize("cached", [False, True])
def test_packaged_math_helper_does_not_shadow_standard_math(tmp_path, monkeypatch, cached):
    if not cached:
        monkeypatch.delitem(sys.modules, "math", raising=False)
    package = tmp_path / "source_package"
    init = _source(package / "__init__.py", "")
    helper = _source(package / "math.py", "VALUE = 9\n")
    main = _source(
        package / "main.py",
        "import math\nfrom . import math as helper\nVALUE = math.sqrt(4) + helper.VALUE\n",
    )

    with TrustedSourceImporter((init, helper)) as importer:
        module = importer.load_module(main, "_source_main")
        assert module.VALUE == 11.0
        assert module.math.sqrt(4) == 2.0
        assert module.helper is sys.modules["source_package.math"]
        assert module.helper is not module.math
        if cached:
            assert sys.modules["math"] is math
    assert sys.modules["math"].sqrt(4) == 2.0


@pytest.mark.parametrize("cached", [False, True])
def test_main_named_math_can_import_standard_math(tmp_path, monkeypatch, cached):
    if not cached:
        monkeypatch.delitem(sys.modules, "math", raising=False)
    main = _source(tmp_path / "math.py", "import math\nVALUE = math.sqrt(4)\n")

    module = load_module(main, "_source_main")

    assert module.VALUE == 2.0
    assert sys.modules["math"].sqrt(4) == 2.0
    if cached:
        assert sys.modules["math"] is math


def test_packaged_helper_has_no_unqualified_alias(tmp_path):
    package = tmp_path / "source_package"
    init = _source(package / "__init__.py", "")
    helper = _source(package / "source_helper.py", "VALUE = 3\n")
    main = _source(package / "main.py", "import source_helper\n")

    with pytest.raises(ModuleNotFoundError, match="source_helper"):
        load_module(main, "_source_main", (init, helper))


def test_standalone_helper_cannot_replace_standard_math(tmp_path):
    helper = _source(tmp_path / "math.py", "VALUE = 99\n")

    with pytest.raises(ImportError, match="conflicts with an unrelated Python module"):
        TrustedSourceImporter((helper,))
    assert sys.modules["math"] is math
    assert math.sqrt(4) == 2.0


def test_standalone_helper_cannot_replace_an_installed_module(tmp_path):
    helper = _source(tmp_path / "pytest.py", "VALUE = 99\n")

    with pytest.raises(ImportError, match="conflicts with an unrelated Python module"):
        TrustedSourceImporter((helper,))
    assert sys.modules["pytest"] is pytest
