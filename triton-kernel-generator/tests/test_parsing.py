# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Test response boundaries without importing or executing generated code."""

from __future__ import annotations

import pytest

from triton_kernel_gen.parsing import CandidateParseError, extract_candidate

CODE = "import triton\n\n@triton.jit\ndef kernel(X):\n    pass\n"


@pytest.mark.parametrize("language", ["python", "triton", "", "Python", "PYTHON3", "py", "TRITON"])
def test_extracts_one_complete_fence(language):
    assert extract_candidate(f"Here is the code.\n```{language}\n{CODE}```\nDone.") == CODE


@pytest.mark.parametrize("marker", ["```", "````", "~~~", "~~~~~"])
def test_accepts_markdown_fence_delimiters(marker):
    assert extract_candidate(f"{marker}python\n{CODE}{marker}") == CODE


def test_preserves_all_candidate_whitespace():
    code = "\r\nimport triton  \r\n\r\ndef kernel(X):\r\n\treturn X  \r\n\r\n"
    assert extract_candidate(f"   ```python\r\n{code}   ```\r\n") == code


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("raw", [True, False])
def test_accepts_and_preserves_python_line_endings(newline, raw):
    code = newline + 'value = "</answer>"' + newline + "# <think>" + newline
    body = code if raw else "```python" + newline + code + "```"
    assert extract_candidate(f"<answer>{body}</answer>") == code


def test_shorter_fence_inside_a_python_string_is_literal():
    code = 'example = """\n```python\nvalue = 1\n```\n"""\n'
    assert extract_candidate(f"````python\n{code}`````") == code


def test_fenced_tags_and_inline_backticks_are_literal():
    code = 'labels = ("<think>", "</analysis>", "<answer>", "</answer>", "```")\n'
    assert extract_candidate(f"```python\n{code}```") == code


@pytest.mark.parametrize("code", [CODE, "value = 1", "\n\nvalue = 1  \n\n"])
def test_raw_code_requires_and_preserves_complete_answer_region(code):
    assert extract_candidate(f"<answer>{code}</answer>") == code


def test_raw_answer_preserves_literal_tags_and_comments():
    code = '\nlabels = ("</answer>", "<think>", "<answer>")\n# </answer>\n'
    assert extract_candidate(f"<answer>{code}</answer>") == code


def test_raw_answer_preserves_multiline_strings_with_tags_and_fences():
    code = 'example = """\n</answer>\n```python\nvalue = 1\n```\n<think>\n"""\n'
    assert extract_candidate(f"<answer>{code}</answer>") == code


def test_raw_answer_tracks_literal_offsets_after_other_line_separator_characters():
    code = 'value = "prefix\f\v"\nlabel = "</answer>"\n'
    assert extract_candidate(f"<answer>{code}</answer>") == code


@pytest.mark.parametrize("wrapper", ["<answer>\n{}\n</answer>", "<ANSWER>{}</ANSWER>"])
def test_fenced_answer_counts_as_one_candidate(wrapper):
    assert extract_candidate(wrapper.format(f"```python\n{CODE}```")) == CODE


@pytest.mark.parametrize("tag", ["think", "analysis", "THINK", "ANALYSIS"])
def test_thought_code_does_not_compete_with_the_final_candidate(tag):
    thought = f"<{tag}>\n```python\nwrong = 0\n```\n</{tag}>\n"
    assert extract_candidate(thought + f"```python\n{CODE}```") == CODE


@pytest.mark.parametrize("tag", ["think", "analysis"])
@pytest.mark.parametrize("opening_gap,closing_gap", [("", "\n"), ("\n", ""), ("", "")])
def test_inline_reasoning_tags_preserve_the_final_candidate(tag, opening_gap, closing_gap):
    thought = f"<{tag}>{opening_gap}```python\nwrong = 0\n```{closing_gap}</{tag}>\n"
    assert extract_candidate(thought + f"```python\n{CODE}```") == CODE


def test_nested_reasoning_and_reasoning_answer_tags_remain_reasoning():
    response = "<think><analysis><answer>wrong = 0</answer></analysis></think>\n"
    assert extract_candidate(response + f"```python\n{CODE}```") == CODE


@pytest.mark.parametrize("tag", ["think", "analysis"])
def test_prompt_can_supply_the_opening_reasoning_tag(tag):
    prefix = f"I will try this code.\n```python\nwrong = 0\n```\n</{tag}>\n"
    assert extract_candidate(prefix + f"<answer>{CODE}</answer>") == CODE


def test_ignores_complete_other_language_fences_outside_answer_regions():
    response = "```text\nUse the next Python block.\n```\n" + f"```python\n{CODE}```"
    assert extract_candidate(response) == CODE


@pytest.mark.parametrize(
    "response",
    [
        "",
        "There is no code.",
        CODE,
        "```json\n{}\n```",
        "<think>\n```python\nvalue = 1\n```\n</think>",
        "<analysis>\n```python\nvalue = 1\n```\n</analysis>",
        "<think><answer>value = 1</answer></think>",
        "```python\nvalue = 1\n```\n</think>",
    ],
)
def test_rejects_missing_or_reasoning_only_candidates(response):
    with pytest.raises(CandidateParseError, match="no complete candidate"):
        extract_candidate(response)


@pytest.mark.parametrize(
    "response",
    [
        "```python\nvalue = 1",
        "```python",
        "````python\nvalue = 1\n```",
        "<answer>value = 1",
        "<answer>\n```python\nvalue = 1\n```",
        "<answer>\n```python\nvalue = 1\n</answer>",
        "<answer>value = 1</answer",
        "<answer\n```python\nvalue = 1\n```",
        "<think>\n```python\nvalue = 1\n```",
    ],
)
def test_rejects_incomplete_regions_or_fences(response):
    with pytest.raises(CandidateParseError, match="incomplete"):
        extract_candidate(response)


@pytest.mark.parametrize(
    "tail",
    [
        "\n```python\nnew = 2",
        "\n<answer>",
        "\n<answer",
        "\n```bash\necho stop",
        "\n<answer\n```javascript\nvalue = 2\n```",
        "\n<ans",
        "\n</answe",
        "\n<thi",
        "\n``",
        "\n``python",
    ],
)
def test_incomplete_final_output_never_selects_an_earlier_fence(tail):
    with pytest.raises(CandidateParseError, match="incomplete"):
        extract_candidate(f"```python\n{CODE}```" + tail)


@pytest.mark.parametrize(
    "response",
    [
        "```python\na = 1\n```\n```triton\nb = 2\n```",
        "```python\ninvalid syntax here\n```\n```python\nb = 2\n```",
        "<answer>a = 1</answer><answer>b = 2</answer>",
        "```python\na = 1\n```\n<answer>b = 2</answer>",
        "<answer>\n```python\na = 1\n```\n```python\nb = 2\n```\n</answer>",
        "<answer>\n```python\na = 1\n```\n```text\nExplanation\n```\n</answer>",
    ],
)
def test_rejects_multiple_candidates_before_syntax_checks(response):
    with pytest.raises(CandidateParseError, match="multiple"):
        extract_candidate(response)


@pytest.mark.parametrize(
    "response",
    [
        "<answer>a = 1<answer>b = 2</answer></answer>",
        "</answer>\n```python\na = 1\n```",
        "<think><analysis></think></analysis>\n```python\na = 1\n```",
        "<answer><think>reason</think>a = 1</answer>",
        "<answer language='python'>a = 1</answer>",
    ],
)
def test_rejects_malformed_or_mixed_regions(response):
    with pytest.raises(CandidateParseError):
        extract_candidate(response)


@pytest.mark.parametrize(
    "prefix",
    [
        "<answer>a = 1</think>\n",
        "<answer>a = 1</answer></think>\n",
        "<think>reason</think>\n<answer>a = 1</answer>\n</think>\n",
        "<think>reason</think>\n```python\na = 1\n```\n</think>\n",
        "reason</think>more reason</think>\n",
    ],
)
def test_extra_reasoning_closer_cannot_erase_answers_or_previous_boundaries(prefix):
    with pytest.raises(CandidateParseError, match="unmatched closing reasoning tag"):
        extract_candidate(prefix + f"```python\n{CODE}```")


@pytest.mark.parametrize("outside", ["value = 2\n", "Here is the code.\n"])
def test_answer_fence_cannot_mix_with_other_content(outside):
    with pytest.raises(CandidateParseError, match="Only whitespace"):
        extract_candidate(f"<answer>{outside}```python\n{CODE}```\n</answer>")


def test_rejects_unsupported_fenced_answer_language():
    with pytest.raises(CandidateParseError, match="unsupported code language"):
        extract_candidate("<answer>\n```javascript\nvalue = 1\n```\n</answer>")


@pytest.mark.parametrize("code", ["", "\n \n", "# Only a comment.\n"])
@pytest.mark.parametrize("wrapper", ["```python\n{}```", "<answer>{}</answer>"])
def test_rejects_candidates_without_python_statements(code, wrapper):
    with pytest.raises(CandidateParseError, match="no Python statements"):
        extract_candidate(wrapper.format(code))


@pytest.mark.parametrize(
    "code", ["def kernel(\n", "    value = 1\n", "This is not Python.\n", "value = '\x00'\n"]
)
def test_rejects_invalid_python_without_repair(code):
    with pytest.raises(CandidateParseError, match="invalid Python syntax"):
        extract_candidate(f"```python\n{code}```")


def test_syntax_check_does_not_execute_candidate(tmp_path):
    marker = tmp_path / "executed"
    code = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    assert extract_candidate(f"```python\n{code}```") == code
    assert not marker.exists()


@pytest.mark.parametrize("response", [None, b"```python\na = 1\n```", 1])
def test_rejects_non_text_input(response):
    with pytest.raises(CandidateParseError, match="must be text"):
        extract_candidate(response)


def test_parse_error_is_a_value_error():
    assert issubclass(CandidateParseError, ValueError)
