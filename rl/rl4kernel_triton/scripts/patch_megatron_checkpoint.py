#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allow Slime's zero-based update IDs in Megatron checkpoint metadata.

The known Megatron read_metadata assertion rejects update zero. This release
patch permits zero while retaining rejection of negative numeric iterations.
It changes no model, optimizer, scheduler, or checkpoint payload. The patch
does not claim that the original experiment image contained this change.

Only the known assertion inside read_metadata(tracker_filename) is supported.
An unknown source layout fails before any file changes.
"""

from __future__ import annotations

import argparse
import ast
import io
import os
import re
import stat
import tempfile
import tokenize
from pathlib import Path


class CompatibilityPatchError(ValueError):
    """The installed source does not contain the supported assertion."""


def _comparison(source: str) -> ast.Compare:
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise CompatibilityPatchError(
            "Megatron checkpoint source contains invalid syntax."
        ) from error
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "read_metadata"
    ]
    if len(functions) != 1 or not isinstance(functions[0], ast.FunctionDef):
        raise CompatibilityPatchError("Expected one synchronous, top-level read_metadata function.")
    function = functions[0]
    arguments = function.args
    if (
        arguments.posonlyargs
        or [argument.arg for argument in arguments.args] != ["tracker_filename"]
        or arguments.defaults
        or arguments.vararg
        or arguments.kwonlyargs
        or arguments.kwarg
    ):
        raise CompatibilityPatchError(
            "The read_metadata signature does not match the supported source."
        )

    comparisons = []
    for statement in function.body:
        if not isinstance(statement, ast.Assert):
            continue
        test = statement.test
        if not isinstance(test, ast.BoolOp) or not isinstance(test.op, ast.Or):
            continue
        if len(test.values) != 2:
            continue
        comparison, release = test.values
        if (
            not isinstance(release, ast.Name)
            or release.id != "release"
            or not isinstance(comparison, ast.Compare)
            or not isinstance(comparison.left, ast.Name)
            or comparison.left.id != "iteration"
            or len(comparison.ops) != 1
            or not isinstance(comparison.ops[0], (ast.Gt, ast.GtE))
            or len(comparison.comparators) != 1
            or not isinstance(comparison.comparators[0], ast.Constant)
            or type(comparison.comparators[0].value) is not int
            or comparison.comparators[0].value != 0
        ):
            continue
        comparisons.append(comparison)
    if len(comparisons) != 1:
        raise CompatibilityPatchError(
            "Expected one direct assertion: iteration > 0 or release, or its patched form."
        )
    return comparisons[0]


def patched_source(source: str) -> tuple[str, bool]:
    """Return the narrowly patched source and whether the operator changed."""
    comparison = _comparison(source)
    segment = ast.get_source_segment(source, comparison)
    match = re.fullmatch(r"iteration[ \t]*(?P<operator>>=?)[ \t]*0", segment or "")
    if match is None:
        raise CompatibilityPatchError("The iteration comparison uses an unsupported source layout.")
    operator = match.group("operator")
    if isinstance(comparison.ops[0], ast.GtE):
        if operator != ">=":
            raise CompatibilityPatchError("The comparison text does not match its parsed operator.")
        return source, False
    if operator != ">":
        raise CompatibilityPatchError("The comparison text does not match its parsed operator.")

    # AST columns use UTF-8 byte offsets, even when the source uses another encoding.
    lines = source.splitlines(keepends=True)
    line_prefix = lines[comparison.lineno - 1].encode("utf-8")[: comparison.col_offset]
    start = sum(len(line) for line in lines[: comparison.lineno - 1])
    start += len(line_prefix.decode("utf-8")) + match.start("operator")
    result = source[:start] + ">=" + source[start + 1 :]
    if not isinstance(_comparison(result).ops[0], ast.GtE):
        raise CompatibilityPatchError("The checkpoint comparison did not change as expected.")
    return result, True


def patch_checkpoint_file(path: Path) -> bool:
    """Atomically replace only the supported operator in one source file."""
    if path.is_symlink():
        raise CompatibilityPatchError("The checkpoint source file must not be a symbolic link.")
    original = path.read_bytes()
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(original).readline)
        source = original.decode(encoding)
    except (SyntaxError, UnicodeError) as error:
        raise CompatibilityPatchError("The checkpoint source encoding is invalid.") from error
    result, changed = patched_source(source)
    if not changed:
        return False
    updated = result.encode(encoding)
    file_mode = stat.S_IMODE(path.stat().st_mode)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, file_mode)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--megatron-root",
        type=Path,
        default=os.environ.get("MEGATRON_LM_PATH"),
        help="Installed Megatron source root. Defaults to MEGATRON_LM_PATH.",
    )
    args = parser.parse_args(argv)
    if args.megatron_root is None:
        parser.error("Supply --megatron-root or set MEGATRON_LM_PATH.")
    path = args.megatron_root / "megatron" / "training" / "checkpointing.py"
    try:
        changed = patch_checkpoint_file(path)
    except (OSError, UnicodeError, CompatibilityPatchError) as error:
        parser.exit(2, f"Cannot patch Megatron checkpoint metadata: {error}\n")
    status = "Patched" if changed else "Already compatible"
    print(f"{status}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
