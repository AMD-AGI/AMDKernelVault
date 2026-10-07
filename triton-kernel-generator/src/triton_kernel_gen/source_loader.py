# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Execute declared Python sources without reading or writing bytecode caches.

Keep TrustedSourceImporter active while external code executes delayed imports.
The convenience function serves self-contained modules and immediate imports.
Package initializers must appear in dependency_paths before package code loads.
This process-wide import context is intended for one dedicated worker thread.
It restores declared module names when the context exits. It is not a sandbox.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

_STANDARD_MODULES = frozenset(sys.builtin_module_names) | frozenset(
    getattr(sys, "stdlib_module_names", ())
)
_MISSING_ATTRIBUTE = object()


@dataclass(frozen=True)
class _Source:
    path: Path
    canonical_name: str
    package: str
    is_package: bool


def _python_path(path: Path) -> Path:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file() or resolved.suffix != ".py":
        raise ImportError(f"The source path must name a Python file: {resolved}.")
    return resolved


def _source_identity(path: Path, declared: set[Path]) -> _Source:
    is_package = path.name == "__init__.py"
    parent = path.parent
    parts = [] if is_package else [path.stem]
    while (parent / "__init__.py").is_file():
        initializer = (parent / "__init__.py").resolve()
        if initializer not in declared:
            raise ImportError(f"Package initializer {initializer} must appear in dependency_paths.")
        parts.insert(0, parent.name)
        parent = parent.parent
    if not parts or any(not part.isidentifier() for part in parts):
        raise ImportError(f"The source has an unsupported Python package name: {path}.")
    canonical = ".".join(parts)
    package = canonical if is_package else canonical.rpartition(".")[0]
    return _Source(path, canonical, package, is_package)


class _SourceLoader(importlib.abc.Loader):
    def __init__(self, owner: "TrustedSourceImporter", source: _Source):
        self.owner = owner
        self.source = source

    def create_module(self, spec: Any) -> ModuleType | None:
        return self.owner._modules_by_path.get(self.source.path)

    def exec_module(self, module: ModuleType) -> None:
        path = self.source.path
        if path in self.owner._executed or path in self.owner._executing:
            return
        self.owner._modules_by_path[path] = module
        self.owner._executing.add(path)
        module.__file__ = str(path)
        module.__cached__ = None
        module.__package__ = self.source.package
        if module.__spec__ is not None:
            module.__spec__.cached = None
        try:
            # compile(bytes) honors Python encoding declarations and ignores .pyc files.
            code = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
        except BaseException:
            self.owner._forget(module, path)
            raise
        finally:
            self.owner._executing.discard(path)
        self.owner._executed.add(path)
        if self.source.package:
            self.owner._bind_parent(self.source.canonical_name, module)

    def get_source(self, fullname: str) -> str:
        return importlib.util.decode_source(self.source.path.read_bytes())

    def get_filename(self, fullname: str) -> str:
        return str(self.source.path)

    def is_package(self, fullname: str) -> bool:
        return self.source.is_package


class TrustedSourceImporter(importlib.abc.MetaPathFinder):
    """Scope source-only imports and restore prior modules after execution.

    Declare every local helper and every enclosing __init__.py file.
    Qualified and relative imports use regular Python package names.
    Packaged helpers use qualified or relative imports, never basename aliases.
    Standalone helpers must not replace standard or installed library modules.
    Installed libraries continue to use the normal Python import machinery.
    """

    def __init__(self, dependency_paths: Sequence[Path] = ()) -> None:
        self._declared = {_python_path(path) for path in dependency_paths}
        self._sources: dict[str, _Source] = {}
        self._directories: set[Path] = set()
        self._package_roots: set[str] = set()
        self._roots: set[str] = set()
        self._saved: dict[str, ModuleType | None] = {}
        self._modules_by_path: dict[Path, ModuleType] = {}
        self._executed: set[Path] = set()
        self._executing: set[Path] = set()
        self._parent_bindings: dict[tuple[ModuleType, str], tuple[Any, ModuleType]] = {}
        self._active = False
        for path in sorted(self._declared):
            source = _source_identity(path, self._declared)
            self._register(source.canonical_name, source)

    @staticmethod
    def _unrelated_import(name: str, expected_path: Path) -> bool:
        if name in _STANDARD_MODULES:
            return True
        existing = sys.modules.get(name)
        if existing is not None:
            filename = vars(existing).get("__file__")
            if filename is None or Path(filename).resolve() != expected_path:
                return True
        spec = importlib.machinery.PathFinder.find_spec(name)
        if spec is not None:
            if spec.origin is None or spec.origin in {"built-in", "frozen"}:
                return True
            if Path(spec.origin).resolve() != expected_path:
                return True
        return False

    def _check_import_name(self, name: str, source: _Source) -> None:
        root = name.split(".")[0]
        if self._active and root in self._roots:
            return
        expected_path = source.path
        if source.package and root == source.package.split(".")[0]:
            directory = source.path.parent
            for _ in range(len(source.package.split(".")) - 1):
                directory = directory.parent
            expected_path = directory / "__init__.py"
        if self._unrelated_import(root, expected_path):
            raise ImportError(
                f"Declared source name {root!r} conflicts with an unrelated Python module. "
                "Use a distinct package name."
            )

    def _register(self, name: str, source: _Source) -> None:
        previous = self._sources.get(name)
        if previous is not None and previous.path != source.path:
            raise ImportError(f"Declared sources have the same import name {name!r}.")
        self._check_import_name(name, source)
        self._sources[name] = source
        new_directory = not source.package and source.path.parent not in self._directories
        if not source.package:
            self._directories.add(source.path.parent)
        if source.package:
            self._package_roots.add(source.package.split(".")[0])
        if self._active:
            self._claim(name)
            if new_directory:
                self._claim_local_modules(source.path.parent)

    def _claim_local_modules(self, directory: Path) -> None:
        # Cached siblings must not bypass the finder and its declaration checks.
        for root in {name.split(".")[0] for name in tuple(sys.modules)}:
            candidate = directory / f"{root}.py"
            if not candidate.is_file():
                candidate = directory / root / "__init__.py"
            if candidate.is_file() and not self._unrelated_import(root, candidate.resolve()):
                self._claim(root)

    def _forget(self, module: ModuleType, path: Path) -> None:
        self._modules_by_path.pop(path, None)
        self._executed.discard(path)
        self._restore_parent_bindings(module)
        for alias, value in tuple(sys.modules.items()):
            if value is not module or alias.split(".")[0] not in self._roots:
                continue
            parent_name, _, attribute_name = alias.rpartition(".")
            parent = sys.modules.get(parent_name)
            if parent is not None and vars(parent).get(attribute_name) is module:
                delattr(parent, attribute_name)
            del sys.modules[alias]

    def _claim(self, name: str) -> None:
        root = name.split(".")[0]
        if root in self._roots:
            return
        self._roots.add(root)
        for existing in tuple(sys.modules):
            if existing == root or existing.startswith(root + "."):
                self._saved[existing] = sys.modules.pop(existing)

    def __enter__(self) -> "TrustedSourceImporter":
        if self._active:
            raise RuntimeError("The source import context is already active.")
        self._active = True
        self._saved = {}
        self._roots = set()
        self._modules_by_path = {}
        self._executed = set()
        self._executing = set()
        self._parent_bindings = {}
        for name in self._sources:
            self._claim(name)
        for directory in self._directories:
            self._claim_local_modules(directory)
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self in sys.meta_path:
            sys.meta_path.remove(self)
        self._restore_parent_bindings()
        for name in tuple(sys.modules):
            if any(name == root or name.startswith(root + ".") for root in self._roots):
                del sys.modules[name]
        sys.modules.update(self._saved)
        self._active = False
        self._modules_by_path = {}
        self._executed = set()
        self._executing = set()

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        source = self._sources.get(fullname)
        if source is not None:
            return self._spec(fullname, source)
        if fullname.split(".")[0] in self._package_roots:
            raise ModuleNotFoundError(
                f"Source module {fullname!r} is not declared in dependency_paths."
            )
        if "." not in fullname:
            for directory in self._directories:
                candidate = directory / f"{fullname}.py"
                if not candidate.is_file():
                    candidate = directory / fullname / "__init__.py"
                if candidate.is_file() and not self._unrelated_import(
                    fullname, candidate.resolve()
                ):
                    raise ModuleNotFoundError(
                        f"Local source module {fullname!r} is not declared in dependency_paths."
                    )
        return None

    def _spec(self, name: str, source: _Source) -> Any:
        return importlib.util.spec_from_file_location(
            name,
            source.path,
            loader=_SourceLoader(self, source),
            submodule_search_locations=[str(source.path.parent)] if source.is_package else None,
        )

    def _bind_parent(self, name: str, module: ModuleType) -> None:
        parent_name, separator, child = name.rpartition(".")
        parent = sys.modules.get(parent_name) if separator else None
        if parent is not None:
            key = (parent, child)
            previous, _ = self._parent_bindings.get(
                key, (vars(parent).get(child, _MISSING_ATTRIBUTE), module)
            )
            self._parent_bindings[key] = (previous, module)
            setattr(parent, child, module)

    def _restore_parent_bindings(self, failed_module: ModuleType | None = None) -> None:
        for (parent, child), (previous, module) in reversed(tuple(self._parent_bindings.items())):
            if failed_module is not None and module is not failed_module:
                continue
            if previous is _MISSING_ATTRIBUTE:
                vars(parent).pop(child, None)
            else:
                setattr(parent, child, previous)
            del self._parent_bindings[(parent, child)]

    def load_module(self, path: Path, name: str) -> ModuleType:
        """Compile the source and resolve its declared imports within this context."""
        if not self._active:
            raise RuntimeError("Enter the source import context before loading a module.")
        if (
            not isinstance(name, str)
            or not name
            or any(not part.isidentifier() for part in name.split("."))
        ):
            raise ImportError("The requested module name must be a valid Python module name.")
        path = _python_path(path)
        self._declared.add(path)
        source = _source_identity(path, self._declared)
        self._register(name, source)
        if source.package:
            self._register(source.canonical_name, source)
            parent_package = (
                source.canonical_name.rpartition(".")[0] if source.is_package else source.package
            )
            if parent_package:
                importlib.import_module(parent_package)
        existing = self._modules_by_path.get(path)
        if existing is not None:
            sys.modules[name] = existing
            if source.package:
                self._bind_parent(source.canonical_name, existing)
            return existing
        spec = self._spec(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        if source.package:
            sys.modules[source.canonical_name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            self._forget(module, path)
            raise
        if source.package:
            self._bind_parent(source.canonical_name, module)
        return module


def load_module(path: Path, name: str, dependency_paths: Sequence[Path] = ()) -> ModuleType:
    """Load current source bytes and restore import state after immediate imports.

    Keep TrustedSourceImporter active for lazy imports or package-attribute access.
    """
    with TrustedSourceImporter(dependency_paths) as importer:
        return importer.load_module(path, name)
