# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Require the functional wrapper to leave all workload operations in module_fn."""

from __future__ import annotations

import ast
import inspect
from collections import OrderedDict
from pathlib import Path
from types import CodeType, FunctionType
from typing import Any, Callable
from weakref import WeakKeyDictionary

import torch

from .errors import ReferenceValidationError

_MISSING = object()
_CHECKED_FORWARDS: WeakKeyDictionary[type, tuple[FunctionType, CodeType, frozenset[str]]] = (
    WeakKeyDictionary()
)
_DISPATCH_METHODS = ("__call__", "_call_impl", "_wrapped_call_impl")
_ATTRIBUTE_METHODS = ("__getattribute__", "__getattr__")


def _reject(message: str) -> None:
    raise ReferenceValidationError(f"Model.forward must be forwarding-only. {message}")


def _nested_codes(code: CodeType):
    yield code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            yield from _nested_codes(value)


def _source_node(forward: FunctionType) -> ast.FunctionDef:
    """Read current source bytes and verify that they describe the loaded code."""
    code = forward.__code__
    try:
        source = Path(code.co_filename).read_bytes()
        tree = ast.parse(source, filename=code.co_filename)
        compiled = compile(source, code.co_filename, "exec", dont_inherit=True)
    except (OSError, SyntaxError, ValueError, TypeError) as error:
        _reject(f"Cannot read valid source for the forward function: {error}.")
    matches = [
        candidate
        for candidate in _nested_codes(compiled)
        if candidate.co_name == code.co_name and candidate.co_firstlineno == code.co_firstlineno
    ]
    if len(matches) != 1 or matches[0] != code:
        _reject("The current source does not match the loaded forward function.")
    nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == code.co_name
        and min([node.lineno, *(item.lineno for item in node.decorator_list)])
        == code.co_firstlineno
    ]
    if len(nodes) != 1 or not isinstance(nodes[0], ast.FunctionDef):
        _reject(
            "The current source must contain one ordinary forward function at its recorded location."
        )
    if nodes[0].decorator_list:
        _reject("The forward function cannot use decorators.")
    return nodes[0]


def _check_attribute_access(model_type: type) -> None:
    for name in _ATTRIBUTE_METHODS:
        if inspect.getattr_static(model_type, name, _MISSING) is not inspect.getattr_static(
            torch.nn.Module, name, _MISSING
        ):
            _reject(
                f"Model cannot override {name} because attribute access can compute workload operations."
            )


def _check_model_attribute(model_type: type, name: str) -> None:
    attribute = inspect.getattr_static(model_type, name, _MISSING)
    if (
        attribute is not _MISSING
        and inspect.getattr_static(type(attribute), "__get__", _MISSING) is not _MISSING
    ):
        _reject(
            "Model properties and descriptors cannot bind arguments because they can compute workload operations."
        )


class _ForwardingCheck:
    def __init__(
        self,
        model_type: type,
        forward: FunctionType,
        reference_fn: Callable[..., Any],
        node: ast.FunctionDef,
    ) -> None:
        self.model_type = model_type
        self.forward = forward
        self.reference_fn = reference_fn
        self.node = node
        self.attributes: set[str] = set()
        signature = inspect.signature(forward, follow_wrapped=False)
        parameters = list(signature.parameters.values())
        if not parameters or parameters[0].name == "fn":
            _reject(
                "The forward function needs an instance parameter before its workload parameters."
            )
        parameter = signature.parameters.get("fn")
        if (
            parameter is None
            or parameter.kind
            not in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
            or parameter.default is inspect.Parameter.empty
        ):
            _reject(
                "The forward function needs an explicit fn keyword with a default reference function."
            )
        if parameter.default is not None and parameter.default is not reference_fn:
            _reject("The fn default must be module_fn or None.")
        self.bindings = {parameter.name: "value" for parameter in parameters}
        self.bindings[parameters[0].name] = "self"
        self.bindings["fn"] = "optional_fn" if parameter.default is None else "fn"
        for item in parameters:
            if item.kind == inspect.Parameter.VAR_POSITIONAL:
                self.bindings[item.name] = "sequence"
            elif item.kind == inspect.Parameter.VAR_KEYWORD:
                self.bindings[item.name] = "mapping"

    def check(self) -> None:
        body = list(self.node.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                body.pop(0)
        if not body or not isinstance(body[-1], ast.Return):
            self._reject_return()
        for statement in body[:-1]:
            if isinstance(statement, ast.If) and self._none_test(statement.test):
                if statement.orelse or len(statement.body) != 1:
                    _reject("An fn fallback can contain only one assignment and no else branch.")
                self._fallback(statement.body[0])
            elif isinstance(statement, ast.Assign):
                self._assignment(statement)
            else:
                _reject("Only aliases, an fn fallback, and a direct return call are allowed.")
        call = body[-1].value
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            self._reject_return()
        if self.bindings.get(call.func.id) != "fn":
            self._reject_return()
        for argument in call.args:
            if isinstance(argument, ast.Starred):
                self._expansion(argument.value, "sequence")
            else:
                self._argument(argument)
        for keyword in call.keywords:
            if keyword.arg is None:
                self._expansion(keyword.value, "mapping")
            else:
                self._argument(keyword.value)

    @staticmethod
    def _reject_return() -> None:
        _reject("The wrapper must call fn exactly once and return the fn output object directly.")

    @staticmethod
    def _none_test(expression: ast.expr) -> bool:
        return (
            isinstance(expression, ast.Compare)
            and isinstance(expression.left, ast.Name)
            and expression.left.id == "fn"
            and len(expression.ops) == 1
            and isinstance(expression.ops[0], ast.Is)
            and len(expression.comparators) == 1
            and isinstance(expression.comparators[0], ast.Constant)
            and expression.comparators[0].value is None
        )

    def _reference(self, expression: ast.expr) -> bool:
        if not isinstance(expression, ast.Name) or expression.id in self.bindings:
            return False
        namespace = dict(self.forward.__globals__)
        for name, cell in zip(self.forward.__code__.co_freevars, self.forward.__closure__ or ()):
            try:
                namespace[name] = cell.cell_contents
            except ValueError:
                return False
        return namespace.get(expression.id, _MISSING) is self.reference_fn

    def _fallback(self, statement: ast.stmt) -> None:
        if not (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == "fn"
            and self._reference(statement.value)
            and self.bindings.get("fn") in {"fn", "optional_fn"}
        ):
            _reject("An fn fallback must assign the reference function directly to fn.")
        self.bindings["fn"] = "fn"

    def _assignment(self, statement: ast.Assign) -> None:
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            _reject(
                "An alias must bind one local name without assignment unpacking or model mutation."
            )
        name = statement.targets[0].id
        value = statement.value
        if isinstance(value, ast.IfExp):
            if (
                name == "fn"
                and self._none_test(value.test)
                and self._reference(value.body)
                and isinstance(value.orelse, ast.Name)
                and value.orelse.id == "fn"
                and self.bindings.get("fn") in {"fn", "optional_fn"}
            ):
                self.bindings["fn"] = "fn"
                return
            _reject("A conditional assignment can only select the fn fallback.")
        if isinstance(value, ast.Name) and self.bindings.get(value.id) in {
            "self",
            "fn",
            "optional_fn",
        }:
            kind = self.bindings[value.id]
        else:
            kind = self._argument(value)
        self.bindings[name] = kind

    def _expansion(self, expression: ast.expr, expected: str) -> None:
        if self._argument(expression) != expected:
            _reject("Argument expansion needs a variadic parameter or a literal container alias.")

    @staticmethod
    def _literal(expression: ast.expr) -> bool:
        try:
            ast.literal_eval(expression)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return False
        return True

    def _argument(self, expression: ast.expr) -> str:
        if isinstance(expression, ast.Name):
            kind = self.bindings.get(expression.id)
            if kind in {"value", "sequence", "mapping"}:
                return kind
        elif isinstance(expression, ast.Attribute):
            if (
                not isinstance(expression.value, ast.Name)
                or self.bindings.get(expression.value.id) != "self"
            ):
                _reject(
                    "Only direct model attributes can bind arguments. Move nested attribute access into module_fn."
                )
            _check_model_attribute(self.model_type, expression.attr)
            self.attributes.add(expression.attr)
            return "value"
        elif isinstance(expression, ast.Constant):
            return "value"
        elif isinstance(expression, ast.UnaryOp) and isinstance(
            expression.op, (ast.UAdd, ast.USub)
        ):
            if isinstance(expression.operand, ast.Constant) and type(expression.operand.value) in {
                int,
                float,
                complex,
            }:
                return "value"
        elif isinstance(expression, (ast.Tuple, ast.List)):
            for element in expression.elts:
                if isinstance(element, ast.Starred):
                    self._expansion(element.value, "sequence")
                else:
                    self._argument(element)
            return "sequence"
        elif isinstance(expression, ast.Dict):
            for key, value in zip(expression.keys, expression.values):
                if key is None:
                    self._expansion(value, "mapping")
                else:
                    if not self._literal(key):
                        _reject(
                            "Dictionary keys must be literals because dynamic keys can compute hash operations."
                        )
                    self._argument(value)
            return "mapping"
        _reject(
            "Arguments can only bind parameters, direct model attributes, literals, and containers. Move all computation into module_fn."
        )


def check_forwarding_wrapper(model_type: type, reference_fn: Callable[..., Any]) -> None:
    """Validate the source before the worker executes the functional wrapper."""
    forward = inspect.getattr_static(model_type, "forward", None)
    if not isinstance(forward, FunctionType):
        _reject("The forward method must be an ordinary Python function with readable source.")
    if "__wrapped__" in vars(forward) or "__signature__" in vars(forward):
        _reject("The forward function cannot replace its signature or wrap another function.")
    _check_attribute_access(model_type)
    checker = _ForwardingCheck(model_type, forward, reference_fn, _source_node(forward))
    checker.check()
    _CHECKED_FORWARDS[model_type] = (forward, forward.__code__, frozenset(checker.attributes))


def check_forwarding_instance(model: torch.nn.Module) -> None:
    """Reject instance changes and hooks that move work outside the checked method."""
    model_type = type(model)
    _check_attribute_access(model_type)
    validated = _CHECKED_FORWARDS.get(model_type)
    forward = inspect.getattr_static(model_type, "forward", None)
    if validated is None or forward is not validated[0] or forward.__code__ is not validated[1]:
        _reject("The model must use the forward function that passed the source check.")
    if "forward" in vars(model):
        _reject("The model cannot replace forward on an instance.")
    for name in ("_parameters", "_buffers", "_modules"):
        if type(inspect.getattr_static(model, name, None)) not in {dict, OrderedDict}:
            _reject(
                "Model state must use ordinary dictionaries because custom dictionaries can compute during attribute access."
            )
    for name in validated[2]:
        _check_model_attribute(model_type, name)
    for name in _DISPATCH_METHODS:
        if name in vars(model) or inspect.getattr_static(
            model_type, name, _MISSING
        ) is not inspect.getattr_static(torch.nn.Module, name, _MISSING):
            _reject(
                f"Model cannot override {name} because dispatch can compute workload operations."
            )
    if inspect.getattr_static(model, "_compiled_call_impl", None) is not None:
        _reject("The model cannot use a compiled call override.")
    for name in ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks"):
        if inspect.getattr_static(model, name, None):
            _reject("The model cannot register forward or backward hooks.")
    for name in (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_backward_pre_hooks",
        "_global_backward_hooks",
    ):
        if getattr(torch.nn.modules.module, name, None):
            _reject("Global module hooks must be empty before the wrapper executes.")
