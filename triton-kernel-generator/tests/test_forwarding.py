# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Check the forwarding contract without a GPU or an API connection."""

from __future__ import annotations

import linecache
import textwrap
from pathlib import Path
from types import ModuleType

import pytest
import torch

from triton_kernel_gen.errors import ReferenceValidationError
from triton_kernel_gen.forwarding import check_forwarding_instance, check_forwarding_wrapper

_HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.\n"
)


def _module(tmp_path: Path, body: str, *, signature="self, x, fn=module_fn", extra=""):
    path = tmp_path / "forwarding_fixture.py"
    source = (
        _HEADER + "import torch\ndef module_fn(*args, **kwargs):\n    return args[0]\n"
        "class Model(torch.nn.Module):\n"
        + textwrap.indent(textwrap.dedent(extra), "    ")
        + f"    def forward({signature}):\n"
        + textwrap.indent(body, "        ")
        + "\n"
    )
    path.write_text(source, encoding="utf-8")
    module = ModuleType("_forwarding_fixture")
    exec(compile(source, str(path), "exec", dont_inherit=True), module.__dict__)
    return module, path


def _check(module):
    check_forwarding_wrapper(module.Model, module.module_fn)


@pytest.mark.parametrize(
    ("body", "signature"),
    [
        ("return fn(x)", "self, x, fn=module_fn"),
        ('"""Pass the workload arguments to fn."""\nreturn fn(x)', "self, x, fn=module_fn"),
        ("if fn is None:\n    fn = module_fn\nreturn fn(x)", "self, x, fn=None"),
        ("fn = module_fn if fn is None else fn\nreturn fn(x)", "self, x, fn=None"),
        ("call = fn\nvalue = x\nreturn call(value)", "self, x, fn=module_fn"),
        ("owner = self\nweight = owner.weight\nreturn fn(x, weight)", "self, x, fn=module_fn"),
        ("x = self.weight\nreturn fn(x)", "self, x, fn=module_fn"),
        ("return fn(x, self.weight, scale=-1, axis=+2)", "self, x, fn=module_fn"),
        (
            "return fn(x, [None, True, 1, 2.5, 'value'], config={'axes': (1, -1)})",
            "self, x, fn=module_fn",
        ),
        ("return fn(*args, **kwargs)", "self, *args, fn=module_fn, **kwargs"),
        (
            "values = args\noptions = kwargs\nreturn fn(*values, **options)",
            "self, *args, fn=module_fn, **kwargs",
        ),
        (
            "values = (x, self.weight)\noptions = {'axis': -1}\nreturn fn(*values, **options)",
            "self, x, fn=module_fn",
        ),
        (
            "return fn(*[x, *args], **{'axis': -1, **kwargs})",
            "self, x, *args, fn=module_fn, **kwargs",
        ),
    ],
)
def test_accepts_forwarding_syntax(tmp_path, body, signature):
    module, _ = _module(tmp_path, body, signature=signature)

    _check(module)


@pytest.mark.parametrize(
    "body",
    [
        "return fn(x * 2)",
        "value = x * 2\nreturn fn(value)",
        "return fn(x.clone())",
        "return fn(torch.sin(x))",
        "return fn(x[::2])",
        "return fn(x.shape)",
        "return fn(self.layer.weight)",
        "return fn(-x)",
        "return fn(x + 0)",
        "return fn(x if fn is module_fn else x + 1)",
        "return fn(x and x)",
        "return fn([item for item in x])",
        "return fn((item for item in x))",
        "return fn(lambda: x)",
        'return fn(f"{x}")',
        "return fn((value := x))",
        "return fn({x: 1})",
        "return fn({x})",
        "return fn(*x)",
        "return fn(**x)",
        "values = x\nreturn fn(*values)",
        "return fn(x) + 1",
        "return fn(x).clone()",
        "return module_fn(x)",
        "fn(x)\nreturn fn(x)",
        "torch.rand_like(x)\nreturn fn(x)",
        "for value in x:\n    pass\nreturn fn(x)",
        "with torch.no_grad():\n    return fn(x)",
        "if x is None:\n    return fn(x)\nreturn fn(x)",
        "pass\nreturn fn(x)",
        "self.weight = x\nreturn fn(x)",
        "first, second = x\nreturn fn(first, second)",
        "x += 1\nreturn fn(x)",
        "value = result = x\nreturn fn(value)",
        "fn = module_fn\nreturn fn(x)",
        "call = module_fn\nreturn call(x)",
        "call = fn\ncall = x\nreturn call(x)",
        "fn = x\nreturn fn(x)",
        "return fn(global_tensor)",
        "return fn(self)",
        "return fn(fn)",
    ],
)
def test_rejects_work_outside_the_reference_function(tmp_path, body):
    module, _ = _module(tmp_path, body)

    with pytest.raises(ReferenceValidationError, match="forwarding-only"):
        _check(module)


def test_rejects_a_wrapper_that_moves_the_entire_workload_outside_module_fn(tmp_path):
    module, _ = _module(tmp_path, "return fn(x * 2)")
    value = torch.arange(4, dtype=torch.float32)
    # Output agreement alone accepts this identity module_fn and expensive wrapper.
    torch.testing.assert_close(module.Model()(value), value * 2)
    torch.testing.assert_close(module.module_fn(value), value)

    with pytest.raises(ReferenceValidationError, match="Move all computation into module_fn"):
        _check(module)


@pytest.mark.parametrize(
    "body",
    [
        "return fn(x)",
        "call = fn\nif fn is None:\n    fn = module_fn\nreturn call(x)",
        "if fn == None:\n    fn = module_fn\nreturn fn(x)",
        "if fn is None:\n    fn = module_fn\n    x = x\nreturn fn(x)",
        "if fn is None:\n    fn = module_fn\nelse:\n    fn = module_fn\nreturn fn(x)",
        "fn = module_fn if fn is None else module_fn\nreturn fn(x)",
        "if fn is None:\n    fn = x\nreturn fn(x)",
        "fn = x\nif fn is None:\n    fn = module_fn\nreturn fn(x)",
    ],
)
def test_none_default_requires_an_exact_safe_fallback(tmp_path, body):
    module, _ = _module(tmp_path, body, signature="self, x, fn=None")

    with pytest.raises(ReferenceValidationError, match="forwarding-only"):
        _check(module)


def test_accepts_a_closure_reference_fallback():
    def module_fn(x):
        return x * 2

    class Model(torch.nn.Module):
        def forward(self, x, fn=None):
            if fn is None:
                fn = module_fn
            return fn(x)

    check_forwarding_wrapper(Model, module_fn)
    check_forwarding_instance(Model())


def test_rejects_a_different_default_reference(tmp_path):
    module, _ = _module(tmp_path, "return fn(x)")

    with pytest.raises(ReferenceValidationError, match="fn default"):
        check_forwarding_wrapper(module.Model, lambda x: x)


def test_rejects_a_rebound_fallback_reference(tmp_path):
    module, _ = _module(
        tmp_path, "if fn is None:\n    fn = module_fn\nreturn fn(x)", signature="self, x, fn=None"
    )
    original = module.module_fn
    module.module_fn = lambda x: x

    with pytest.raises(ReferenceValidationError, match="fallback must assign"):
        check_forwarding_wrapper(module.Model, original)


@pytest.mark.parametrize("metadata", ["__wrapped__", "__signature__"])
def test_rejects_forged_signature_metadata(tmp_path, metadata):
    module, _ = _module(tmp_path, "return fn(x)")
    setattr(module.Model.forward, metadata, module.module_fn)

    with pytest.raises(ReferenceValidationError, match="signature"):
        _check(module)


def test_rejects_source_that_changes_after_import(tmp_path):
    module, path = _module(tmp_path, "return fn(x * 2)")
    path.write_text(path.read_text().replace("fn(x * 2)", "fn(x)"), encoding="utf-8")

    with pytest.raises(ReferenceValidationError, match="current source does not match"):
        _check(module)


def test_reads_current_bytes_instead_of_linecache(tmp_path):
    module, path = _module(tmp_path, "return fn(x * 2)")
    harmless = path.read_text().replace("fn(x * 2)", "fn(x)")
    linecache.cache[str(path)] = (len(harmless), None, harmless.splitlines(True), str(path))
    try:
        with pytest.raises(ReferenceValidationError, match="Move all computation"):
            _check(module)
    finally:
        linecache.cache.pop(str(path), None)


def test_missing_source_fails_explicitly(tmp_path):
    module, path = _module(tmp_path, "return fn(x)")
    path.unlink()

    with pytest.raises(ReferenceValidationError, match="Cannot read valid source"):
        _check(module)


def test_dynamic_code_without_source_fails_explicitly():
    namespace = {"torch": torch}
    exec(
        "def module_fn(x): return x\nclass Model(torch.nn.Module):\n def forward(self, x, fn=module_fn): return fn(x)",
        namespace,
    )

    with pytest.raises(ReferenceValidationError, match="Cannot read valid source"):
        check_forwarding_wrapper(namespace["Model"], namespace["module_fn"])


@pytest.mark.parametrize("kind", ["property", "descriptor", "inherited_property"])
def test_rejects_computed_model_attributes_without_accessing_them(tmp_path, kind):
    module, _ = _module(tmp_path, "return fn(x, self.weight)")

    def fail(*args):
        raise AssertionError("The checker executed a descriptor.")

    if kind == "descriptor":

        class Descriptor:
            __get__ = fail

        module.Model.weight = Descriptor()
    elif kind == "inherited_property":

        class Inherited(module.Model):
            weight = property(fail)

        module.Model = Inherited
    else:
        module.Model.weight = property(fail)

    with pytest.raises(ReferenceValidationError, match="properties and descriptors"):
        _check(module)


@pytest.mark.parametrize("hook", ["__getattr__", "__getattribute__"])
def test_rejects_custom_attribute_hooks(tmp_path, hook):
    module, _ = _module(tmp_path, "return fn(x)")
    setattr(module.Model, hook, lambda self, name: None)

    with pytest.raises(ReferenceValidationError, match="attribute access"):
        _check(module)


def test_accepts_registered_parameters_buffers_and_scalar_attributes(tmp_path):
    module, _ = _module(
        tmp_path,
        "return fn(x, self.weight, self.offset, scale=self.scale)",
        extra="""
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("offset", torch.zeros(2))
            self.scale = 2
    """,
    )

    _check(module)
    check_forwarding_instance(module.Model())


def test_rejects_a_property_installed_by_the_constructor(tmp_path):
    module, _ = _module(
        tmp_path,
        "return fn(x, self.weight)",
        extra="""
        def __init__(self):
            super().__init__()
            type(self).weight = property(lambda self: torch.ones(2) * 2)
    """,
    )
    _check(module)

    with pytest.raises(ReferenceValidationError, match="properties and descriptors"):
        check_forwarding_instance(module.Model())


@pytest.mark.parametrize("name", ["_parameters", "_buffers", "_modules"])
def test_rejects_custom_state_dictionaries_that_compute_on_access(tmp_path, name):
    module, _ = _module(tmp_path, "return fn(x, self.weight)")
    _check(module)
    model = module.Model()

    class ComputingState(dict):
        def __getitem__(self, key):
            return super().__getitem__(key) * 2

    setattr(model, name, ComputingState(getattr(model, name)))

    with pytest.raises(ReferenceValidationError, match="ordinary dictionaries"):
        check_forwarding_instance(model)


@pytest.mark.parametrize("method", ["__call__", "_call_impl", "_wrapped_call_impl"])
@pytest.mark.parametrize("target", ["class", "instance"])
def test_rejects_custom_dispatch(tmp_path, method, target):
    module, _ = _module(tmp_path, "return fn(x)")
    _check(module)
    model = module.Model()
    setattr(module.Model if target == "class" else model, method, lambda *args, **kwargs: None)

    with pytest.raises(ReferenceValidationError, match="dispatch"):
        check_forwarding_instance(model)


@pytest.mark.parametrize("target", ["class", "instance", "code"])
def test_rejects_forward_replacement_after_source_check(tmp_path, target):
    module, _ = _module(tmp_path, "return fn(x)")
    _check(module)
    model = module.Model()

    def replacement(self, x, fn=None):
        return fn(x * 2)

    if target == "code":
        module.Model.forward.__code__ = replacement.__code__
    else:
        setattr(module.Model if target == "class" else model, "forward", replacement)

    with pytest.raises(ReferenceValidationError, match="forward"):
        check_forwarding_instance(model)


def test_rejects_compiled_dispatch(tmp_path):
    module, _ = _module(tmp_path, "return fn(x)")
    _check(module)
    model = module.Model()
    model._compiled_call_impl = lambda *args, **kwargs: None

    with pytest.raises(ReferenceValidationError, match="compiled call override"):
        check_forwarding_instance(model)


@pytest.mark.parametrize("kind", ["pre", "post"])
@pytest.mark.parametrize("scope", ["instance", "global"])
def test_rejects_forward_hooks(tmp_path, kind, scope):
    module, _ = _module(tmp_path, "return fn(x)")
    _check(module)
    model = module.Model()
    owner = model if scope == "instance" else torch.nn.modules.module
    name = (
        "register_"
        + ("module_" if scope == "global" else "")
        + "forward_"
        + ("pre_hook" if kind == "pre" else "hook")
    )
    handle = getattr(owner, name)(lambda *args: None)
    try:
        with pytest.raises(ReferenceValidationError, match="hooks"):
            check_forwarding_instance(model)
    finally:
        handle.remove()
