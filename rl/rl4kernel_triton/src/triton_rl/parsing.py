# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extract one code payload without changing the generated response.

Response tags are reserved markup, including tags inside Python strings.
Ambiguous markup fails closed. The caller keeps the original response for
token accounting and training. This module only returns an original substring.
"""

from __future__ import annotations

import re


def _tags(text: str, name: str) -> list[tuple[bool, int, int]] | None:
    """Find complete tags, or reject a malformed tag with the same name."""
    partial = re.search(r"<\s*/?\s*([a-z]+)\s*\Z", text, re.IGNORECASE)
    if partial is not None and name.startswith(partial.group(1).lower()):
        return None
    prefix = re.compile(rf"<\s*/?\s*{name}\b", re.IGNORECASE)
    complete = re.compile(rf"<\s*(/?)\s*{name}\s*>", re.IGNORECASE)
    result = []
    for match in prefix.finditer(text):
        tag = complete.match(text, match.start())
        if tag is None:
            return None
        result.append((bool(tag.group(1)), tag.start(), tag.end()))
    return result


def _visible_regions(text: str) -> list[tuple[int, int]] | None:
    """Exclude complete thoughts and an initial thought from prefilling."""
    tags = _tags(text, "think")
    if tags is None:
        return None
    regions = []
    cursor = 0
    inside = False
    for index, (closing, start, end) in enumerate(tags):
        if closing:
            if not inside and index != 0:
                return None
            # An initial closing tag terminates a thought opened by the prompt.
            cursor = end
            inside = False
        else:
            if inside:
                return None
            if cursor < start:
                regions.append((cursor, start))
            inside = True
    if inside:
        return None
    if cursor < len(text):
        regions.append((cursor, len(text)))
    return regions


_FENCE_LINE = re.compile(r"[ \t]*(`{3,}|~{3,})([^\r\n]*)(?:\r\n|\r|\n)?\Z")
_PYTHON_LABELS = {"", "python", "py"}


def _fenced_spans(
    text: str,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], bool]:
    """Return Python spans, all complete spans, and whether every fence closed."""
    spans = []
    all_spans = []
    opening: tuple[str, int, str, int] | None = None
    offset = 0
    for line in text.splitlines(keepends=True):
        match = _FENCE_LINE.fullmatch(line)
        if match is not None:
            marker, label = match.groups()
            label = label.strip().lower()
            if opening is None:
                opening = (marker[0], len(marker), label, offset + len(line))
            else:
                character, length, language, body_start = opening
                if marker[0] == character and len(marker) >= length and not label:
                    all_spans.append((body_start, offset))
                    if language in _PYTHON_LABELS:
                        spans.append((body_start, offset))
                    opening = None
        offset += len(line)
    return spans, all_spans, opening is None


def _payload(text: str, *, allow_raw: bool) -> str | None:
    """Select one fence, or accept the complete raw answer payload."""
    spans, all_spans, closed = _fenced_spans(text)
    if not closed:
        return None
    if len(spans) == 1:
        start, end = spans[0]
        payload = text[start:end]
    elif allow_raw and not all_spans:
        payload = text
    else:
        return None
    return payload if payload.strip() else None


def extract_answer_code(text: str) -> str | None:
    """Return one original code substring, or return ``None``.

    A complete, unique ``<answer>`` region takes precedence over other fences.
    Its payload can contain raw code or one Python, py, or unlabeled fence.
    Without answer tags, exactly one accepted fence must occur outside thoughts.
    Fence labels are case-insensitive. Backtick and tilde fences are supported.

    The parser rejects empty answers, malformed tags, nested or repeated answer
    tags, nested thoughts, unclosed thoughts, and ambiguous accepted fences.
    An initial orphan ``</think>`` discards the entire preceding prefix.
    Any later orphan thought closing tag fails. Code cannot span a thought.
    An invalid explicit answer never selects an earlier fence as a fallback.
    Answer tags inside an enclosing complete fence are ambiguous and fail.

    The parser does not validate Python syntax, add imports, or repair code.
    It preserves every payload character, including whitespace and line endings.
    ``strip()`` only tests whether the selected payload is empty.
    """
    regions = _visible_regions(text)
    if regions is None:
        return None

    answer_tags = []
    for start, end in regions:
        tags = _tags(text[start:end], "answer")
        if tags is None:
            return None
        answer_tags.extend(
            (closing, start + tag_start, start + tag_end) for closing, tag_start, tag_end in tags
        )

    if answer_tags:
        if len(answer_tags) != 2:
            return None
        opening, closing = answer_tags
        if opening[0] or not closing[0]:
            return None
        body_start, body_end = opening[2], closing[1]
        containing_regions = [
            (start, end) for start, end in regions if start <= opening[1] and closing[2] <= end
        ]
        if not containing_regions:
            return None
        region_start, region_end = containing_regions[0]
        _, enclosing_fences, _ = _fenced_spans(text[region_start:region_end])
        if any(
            region_start + left <= opening[1] and closing[2] <= region_start + right
            for left, right in enclosing_fences
        ):
            return None
        return _payload(text[body_start:body_end], allow_raw=True)

    candidates = []
    for start, end in regions:
        spans, _, closed = _fenced_spans(text[start:end])
        if not closed:
            return None
        candidates.extend(text[start + left : start + right] for left, right in spans)
    if len(candidates) != 1 or not candidates[0].strip():
        return None
    return candidates[0]
