# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import re
from dataclasses import dataclass

GPU_QUALIFIERS = ("__global__", "__device__")
TARGET_FUNCTION_MODES = ("auto", "global", "device")
DECLARATION_PREFIXES = (
    "template",
    "__launch_bounds__",
    "__attribute__",
    "[[",
)


@dataclass(slots=True)
class GPUFunction:
    name: str
    qualifier: str
    signature: str
    full_text: str
    body: str
    start: int
    end: int
    body_start: int
    body_end: int

    @property
    def body_char_length(self) -> int:
        return len(self.body.strip())


def _mask_comments_and_strings(source: str) -> str:
    result: list[str] = []
    index = 0
    length = len(source)
    state = "normal"

    while index < length:
        chunk = source[index : index + 2]
        char = source[index]

        if state == "normal":
            if chunk == "//":
                result.extend([" ", " "])
                index += 2
                state = "line_comment"
                continue
            if chunk == "/*":
                result.extend([" ", " "])
                index += 2
                state = "block_comment"
                continue
            if char == '"':
                result.append(" ")
                index += 1
                state = "double_quote"
                continue
            if char == "'":
                result.append(" ")
                index += 1
                state = "single_quote"
                continue
            result.append(char)
            index += 1
            continue

        if state == "line_comment":
            if char == "\n":
                result.append("\n")
                state = "normal"
            else:
                result.append(" ")
            index += 1
            continue

        if state == "block_comment":
            if chunk == "*/":
                result.extend([" ", " "])
                index += 2
                state = "normal"
            else:
                result.append("\n" if char == "\n" else " ")
                index += 1
            continue

        if state in {"double_quote", "single_quote"}:
            quote_char = '"' if state == "double_quote" else "'"
            if char == "\\":
                result.append(" ")
                index += 1
                if index < length:
                    next_char = source[index]
                    result.append("\n" if next_char == "\n" else " ")
                    index += 1
                continue
            if char == quote_char:
                result.append(" ")
                index += 1
                state = "normal"
            else:
                result.append("\n" if char == "\n" else " ")
                index += 1

    return "".join(result)


def _looks_like_declaration_prefix_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return False
    return any(stripped.startswith(prefix) for prefix in DECLARATION_PREFIXES)


def _find_declaration_start(masked_source: str, qualifier_index: int) -> int:
    qualifier_line_start = masked_source.rfind("\n", 0, qualifier_index) + 1
    # Another declaration or a namespace can precede the qualifier on this line.
    # Starting at the line boundary would parse that earlier body repeatedly.
    line_prefix = masked_source[qualifier_line_start:qualifier_index]
    preceding_boundary = max(line_prefix.rfind(char) for char in ";{}")
    if preceding_boundary >= 0:
        start = qualifier_line_start + preceding_boundary + 1
        while start < qualifier_index and masked_source[start].isspace():
            start += 1
        return start
    declaration_start = qualifier_line_start
    current_line_start = qualifier_line_start

    while current_line_start > 0:
        previous_line_end = current_line_start - 1
        previous_line_start = masked_source.rfind("\n", 0, previous_line_end) + 1
        previous_line = masked_source[previous_line_start : previous_line_end + 1]
        if not _looks_like_declaration_prefix_line(previous_line):
            break
        declaration_start = previous_line_start
        current_line_start = previous_line_start

    return declaration_start


def _find_header_end(masked_source: str, start: int) -> int | None:
    paren_depth = 0
    index = start
    while index < len(masked_source):
        char = masked_source[index]
        if char == "(":
            paren_depth += 1
        elif char == ")":
            paren_depth = max(paren_depth - 1, 0)
        elif char == ";" and paren_depth == 0:
            return None
        elif char == "{" and paren_depth == 0:
            return index
        index += 1
    return None


def _find_matching_brace(source: str, opening_brace_index: int) -> int:
    masked_source = _mask_comments_and_strings(source)
    depth = 0
    for index in range(opening_brace_index, len(masked_source)):
        char = masked_source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("No matching closing brace found for GPU function.")


def _extract_function_name(signature: str) -> str:
    masked_signature = _mask_comments_and_strings(signature)
    ignored_names = {"__launch_bounds__", "__attribute__", "__declspec", "alignas", "decltype", "noexcept"}
    paren_depth = bracket_depth = angle_depth = 0
    for index, char in enumerate(masked_signature):
        if char == "(":
            if paren_depth == bracket_depth == angle_depth == 0:
                match = re.search(r"([A-Za-z_]\w*(?:::[A-Za-z_]\w*)*)\s*$", masked_signature[:index])
                if match and match.group(1) not in ignored_names:
                    return match.group(1)
            paren_depth += 1
        elif char == ")":
            paren_depth = max(paren_depth - 1, 0)
        elif char == "[" and paren_depth == 0:
            bracket_depth += 1
        elif char == "]" and paren_depth == 0:
            bracket_depth = max(bracket_depth - 1, 0)
        elif char == "<" and paren_depth == bracket_depth == 0:
            angle_depth += 1
        elif char == ">" and paren_depth == bracket_depth == 0:
            angle_depth = max(angle_depth - 1, 0)
    raise ValueError(f"Unable to extract function name from signature: {signature!r}")


def _extract_qualifier(signature: str) -> str:
    if "__global__" in signature:
        return "__global__"
    if "__device__" in signature:
        return "__device__"
    raise ValueError(f"Unable to determine function qualifier from signature: {signature!r}")


def extract_gpu_functions(source: str) -> list[GPUFunction]:
    masked_source = _mask_comments_and_strings(source)
    functions: list[GPUFunction] = []
    seen_ranges: set[tuple[int, int]] = set()

    for qualifier in GPU_QUALIFIERS:
        search_from = 0
        while True:
            start = masked_source.find(qualifier, search_from)
            if start == -1:
                break
            declaration_start = _find_declaration_start(masked_source, start)
            header_end = _find_header_end(masked_source, declaration_start)
            if header_end is None:
                search_from = start + len(qualifier)
                continue
            body_end = _find_matching_brace(source, header_end)
            key = (declaration_start, body_end)
            if key in seen_ranges:
                search_from = max(body_end + 1, start + len(qualifier))
                continue
            seen_ranges.add(key)
            signature = source[declaration_start:header_end].rstrip()
            body = source[header_end + 1 : body_end]
            normalized_qualifier = _extract_qualifier(signature)
            functions.append(
                GPUFunction(
                    name=_extract_function_name(signature),
                    qualifier=normalized_qualifier,
                    signature=signature,
                    full_text=source[declaration_start : body_end + 1],
                    body=body,
                    start=declaration_start,
                    end=body_end + 1,
                    body_start=header_end,
                    body_end=body_end,
                )
            )
            search_from = max(body_end + 1, start + len(qualifier))

    return sorted(functions, key=lambda item: item.start)


def _filter_functions_by_mode(functions: list[GPUFunction], mode: str) -> list[GPUFunction]:
    normalized_mode = mode.strip().lower()
    if normalized_mode not in TARGET_FUNCTION_MODES:
        raise ValueError(
            f"Unsupported target function mode `{mode}`. Expected one of {TARGET_FUNCTION_MODES}."
        )
    if normalized_mode == "global":
        return [function for function in functions if function.qualifier == "__global__"]
    if normalized_mode == "device":
        return [function for function in functions if function.qualifier == "__device__"]
    return list(functions)


def select_optimization_target(functions: list[GPUFunction], mode: str = "auto") -> GPUFunction | None:
    if not functions:
        return None

    normalized_mode = mode.strip().lower()
    if normalized_mode == "auto":
        global_functions = [function for function in functions if function.qualifier == "__global__"]
        if global_functions:
            return max(global_functions, key=lambda item: item.body_char_length)
        return max(functions, key=lambda item: item.body_char_length)

    filtered_functions = _filter_functions_by_mode(functions, normalized_mode)
    if not filtered_functions:
        return None
    return max(filtered_functions, key=lambda item: item.body_char_length)


def extract_gpu_function_body(function_code: str) -> str:
    functions = extract_gpu_functions(function_code)
    if not functions:
        raise ValueError("No __global__ or __device__ function was found in the provided code.")
    return functions[0].body.strip()


def _detect_indent(function: GPUFunction) -> str:
    lines = function.body.splitlines()
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            return line[: len(line) - len(stripped)]
    return "    "


def _indent_body(body: str, indent: str) -> str:
    normalized = body.strip("\n")
    if not normalized.strip():
        return ""
    return "\n".join(
        f"{indent}{line}" if line.strip() else ""
        for line in normalized.splitlines()
    )


def replace_function_body(source: str, function: GPUFunction, new_body: str) -> str:
    indented_body = _indent_body(new_body, _detect_indent(function))
    prefix = source[: function.body_start + 1]
    suffix = source[function.body_end :]
    replacement = "\n"
    if indented_body:
        replacement += f"{indented_body}\n"
    return prefix + replacement + suffix
