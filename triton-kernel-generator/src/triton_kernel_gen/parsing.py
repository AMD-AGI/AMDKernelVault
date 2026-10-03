# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Extract one complete Python candidate without executing or rewriting it."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import dataclass, field


class CandidateParseError(ValueError):
    """The response does not contain one complete, unambiguous Python candidate."""


_LANGUAGES = {"", "python", "python3", "py", "triton"}
_THOUGHT_TAGS = {"think", "analysis"}
_TOKEN = re.compile(
    r"(?P<tag><\s*(?P<closing>/?)\s*(?P<name>think|analysis|answer)\b(?P<attributes>[^>]*)>)"
    r"|(?P<partial><\s*/?\s*(?:think|analysis|answer)\b)"
    r"|(?P<marker>`{3,}|~{3,})(?P<label>[^\r\n]*)(?:\r\n|\n|\r|$)",
    re.IGNORECASE,
)


@dataclass
class _Fence:
    start: int
    body_start: int
    body_end: int
    end: int
    language: str


@dataclass
class _Answer:
    body_start: int
    body_end: int = 0
    fences: list[_Fence] = field(default_factory=list)


def _literal_ranges(text: str, start: int) -> list[tuple[int, int]]:
    """Protect Python strings and comments inside a raw answer region."""
    source = text[start:]
    offsets = [0] + [newline.end() for newline in re.finditer(r"\r\n|\r|\n", source)]
    normalized = re.sub(r"\r\n?", "\n", source)
    ranges = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(normalized).readline):
            if token.type in {tokenize.STRING, tokenize.COMMENT}:
                begin = start + offsets[token.start[0] - 1] + token.start[1]
                end = start + offsets[token.end[0] - 1] + token.end[1]
                ranges.append((begin, end))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Invalid raw code still reaches the final syntax check.
        pass
    return ranges


def _is_literal(position: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in ranges)


def _fence_start(text: str, position: int, region_start: int) -> int | None:
    """Require a delimiter line, including a line after an inline region tag."""
    line_start = (
        max(text.rfind("\n", 0, position), text.rfind("\r", 0, position), region_start - 1) + 1
    )
    prefix = text[line_start:position]
    if len(prefix) <= 3 and not prefix.strip(" "):
        return line_start
    return None


def _read_fence(text: str, opening: re.Match[str], start: int, region_name: str | None) -> _Fence:
    marker = opening.group("marker")
    # An inline region tag can follow the closing fence delimiter.
    closing_tag = r"|<\s*/\s*" + region_name + r"\s*>" if region_name else ""
    suffix = r"(?=\r|\n|$" + closing_tag + ")"
    closing_pattern = re.compile(
        r"(?:^|(?<=\r))[ ]{0,3}"
        + re.escape(marker[0])
        + "{"
        + str(len(marker))
        + r",}[ \t]*"
        + suffix,
        re.MULTILINE | re.IGNORECASE,
    )
    closing = closing_pattern.search(text, opening.end())
    if closing is None:
        raise CandidateParseError("The response contains an incomplete code fence.")
    return _Fence(
        start, opening.end(), closing.start(), closing.end(), opening.group("label").strip().lower()
    )


def _check_python(code: str) -> str:
    try:
        module = ast.parse(code, filename="<candidate>", mode="exec")
    except (SyntaxError, ValueError) as error:
        detail = error.msg if isinstance(error, SyntaxError) else str(error)
        raise CandidateParseError(
            f"The candidate contains invalid Python syntax: {detail}."
        ) from error
    if not module.body:
        raise CandidateParseError("The candidate contains no Python statements.")
    return code


def _check_truncated_tail(tail: str) -> None:
    """Reject a final delimiter that stops before its tag or marker is complete."""
    lines = tail.rstrip().splitlines()
    if not lines:
        return
    final_line = lines[-1].strip()
    if final_line.startswith("<"):
        prefix = final_line[1:].strip().removeprefix("/").strip().lower()
        if any(name.startswith(prefix) for name in _THOUGHT_TAGS | {"answer"}):
            raise CandidateParseError("The response contains an incomplete region tag.")
    if re.fullmatch(
        r"`{1,2}(?:[ \t]*(?:python|python3|py|triton))?|~{1,2}", final_line, re.IGNORECASE
    ):
        raise CandidateParseError("The response contains an incomplete code fence.")


def extract_candidate(text: str) -> str:
    """Return an exact source slice for one complete candidate.

    Accept case-insensitive Python, Python3, py, Triton, or unlabelled fences.
    Backtick and tilde fences follow Markdown delimiter lengths and indentation.
    Ignore complete fences with other language labels outside answer regions.
    An explicit ``<answer>`` region can contain raw Python or one eligible fence.
    Only whitespace can surround a fence inside an answer region.

    Ignore ``<think>`` and ``<analysis>`` regions, including nested regions.
    An initial lone closing thought tag marks the preceding response prefix as reasoning.
    Some model prompts supply the corresponding opening tag.
    Reject unfinished regions, unfinished fences, and multiple candidates.

    Parse Python syntax without imports or execution. This check does not establish
    Triton validity, the required entry point, runtime correctness, or safety.
    Preserve candidate whitespace, comments, and newline sequences exactly.
    """
    if not isinstance(text, str):
        raise CandidateParseError("The response must be text.")

    candidates: list[str] = []
    answers: list[_Answer] = []
    answer: _Answer | None = None
    thoughts: list[tuple[str, int]] = []
    thought_boundary_seen = False
    literals: list[tuple[int, int]] = []
    position = 0
    while match := _TOKEN.search(text, position):
        if _is_literal(match.start(), literals):
            position = match.start() + 1
            continue

        if match.group("partial") is not None:
            raise CandidateParseError("The response contains an incomplete region tag.")

        if match.group("marker") is not None:
            region_start = answer.body_start if answer else thoughts[-1][1] if thoughts else 0
            region_name = "answer" if answer else thoughts[-1][0] if thoughts else None
            start = _fence_start(text, match.start(), region_start)
            if start is None:
                position = match.start() + len(match.group("marker"))
                continue
            fence = _read_fence(text, match, start, region_name)
            position = fence.end
            if thoughts:
                continue
            if answer is not None:
                answer.fences.append(fence)
            elif fence.language in _LANGUAGES:
                candidates.append(text[fence.body_start : fence.body_end])
            continue

        position = match.end()
        name = match.group("name").lower()
        closing = bool(match.group("closing"))
        if match.group("attributes").strip():
            raise CandidateParseError("The response contains an unsupported region tag.")

        if name in _THOUGHT_TAGS:
            if not closing:
                if answer is not None:
                    raise CandidateParseError("An answer region cannot contain a reasoning region.")
                thoughts.append((name, match.end()))
                thought_boundary_seen = True
            elif thoughts:
                if thoughts.pop()[0] != name:
                    raise CandidateParseError("The reasoning region tags do not match.")
            else:
                # A model can omit the opening tag when its prompt supplies it.
                if thought_boundary_seen or answers:
                    raise CandidateParseError(
                        "The response contains an unmatched closing reasoning tag."
                    )
                thought_boundary_seen = True
                candidates.clear()
            continue

        if thoughts:
            continue
        if not closing:
            if answer is not None:
                raise CandidateParseError("Answer regions cannot contain another answer region.")
            answer = _Answer(body_start=match.end())
            answers.append(answer)
            literals = _literal_ranges(text, answer.body_start)
        else:
            if answer is None:
                raise CandidateParseError("The response contains an unmatched closing answer tag.")
            answer.body_end = match.start()
            answer = None
            literals = []

    _check_truncated_tail(text[position:])
    if answer is not None or thoughts:
        raise CandidateParseError("The response contains an incomplete answer or reasoning region.")
    if len(answers) > 1:
        raise CandidateParseError("The response contains multiple answer regions.")

    for region in answers:
        if not region.fences:
            candidates.append(text[region.body_start : region.body_end])
            continue
        if len(region.fences) != 1:
            raise CandidateParseError("The answer region contains multiple code fences.")
        fence = region.fences[0]
        if fence.language not in _LANGUAGES:
            raise CandidateParseError("The answer region uses an unsupported code language.")
        outside = text[region.body_start : fence.start] + text[fence.end : region.body_end]
        if outside.strip():
            raise CandidateParseError(
                "Only whitespace can surround a fenced candidate inside an answer region."
            )
        candidates.append(text[fence.body_start : fence.body_end])

    if not candidates:
        raise CandidateParseError("The response contains no complete candidate.")
    if len(candidates) != 1:
        raise CandidateParseError("The response contains multiple candidates.")
    return _check_python(candidates[0])
