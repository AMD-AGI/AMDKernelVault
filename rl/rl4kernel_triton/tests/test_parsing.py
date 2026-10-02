# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check response parsing with synthetic text and no model dependencies."""

import unittest

from triton_rl.parsing import extract_answer_code


class AnswerCodeParsingTests(unittest.TestCase):
    def test_python_py_and_unlabeled_fences(self):
        for label in ("python", "py", "", "Python", "PY"):
            with self.subTest(label=label):
                self.assertEqual(extract_answer_code(f"```{label}\nx = 1\n```"), "x = 1\n")

    def test_preserves_all_payload_characters(self):
        code = "\r\n\t# π\r\n  x = '  value  '  \r\n\r\n"
        response = f"Prose.\r\n  ```python\r\n{code}\t```\r\nMore prose."
        self.assertEqual(extract_answer_code(response), code)

    def test_preserves_invalid_python_and_missing_imports(self):
        code = "@triton.jit\ndef kernel(:\n    return torch.tensor([1])\n"
        self.assertEqual(extract_answer_code(f"```python\n{code}```"), code)

    def test_original_response_remains_unchanged(self):
        response = "<think>secret</think><answer>```py\nx = 1  \n```</answer>"
        original = response
        code = extract_answer_code(response)
        self.assertEqual(response, original)
        self.assertEqual(code, "x = 1  \n")
        self.assertIn(code, original)

    def test_complete_raw_answer_preserves_whitespace(self):
        code = "\n\tvalue = 3  \n\n"
        self.assertEqual(extract_answer_code(f"<answer>{code}</answer>"), code)

    def test_complete_answer_with_fenced_code(self):
        response = "<answer>Here is the code.\n```python\nx = 1\n```\n</answer>"
        self.assertEqual(extract_answer_code(response), "x = 1\n")

    def test_explicit_answer_takes_precedence(self):
        response = "```python\nearlier = 1\n```\n<answer>chosen = 2</answer>"
        self.assertEqual(extract_answer_code(response), "chosen = 2")

    def test_unmatched_earlier_fence_does_not_veto_explicit_answer(self):
        response = "```python\nearlier = 1\n<answer>chosen = 2</answer>"
        self.assertEqual(extract_answer_code(response), "chosen = 2")

    def test_bad_explicit_answer_does_not_select_earlier_fence(self):
        earlier = "```python\nearlier = 1\n```\n"
        answers = (
            "<answer></answer>",
            "<answer> \r\n\t</answer>",
            "<answer>```python\n\n```</answer>",
            "<answer>unfinished",
            "<answer",
            "<answe",
            "</answe",
            "<answer code>value = 2</answer>",
            "</answer>",
            "<answer>```python\nunclosed</answer>",
            "<answer>```javascript\nvalue = 2\n```</answer>",
        )
        for answer in answers:
            with self.subTest(answer=answer):
                self.assertIsNone(extract_answer_code(earlier + answer))

    def test_repeated_nested_and_reversed_answers_are_ambiguous(self):
        responses = (
            "<answer>x = 1</answer><answer>x = 2</answer>",
            "<answer>x = 1</answer><answer>x = 1</answer>",
            "<answer><answer>x = 1</answer></answer>",
            "</answer>x = 1<answer>",
            "<answer>x = 1</answer></answer>",
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertIsNone(extract_answer_code(response))

    def test_closed_thought_fences_never_become_candidates(self):
        response = "<think>```python\nsecret = 1\n```</think>```py\npublic = 2\n```"
        self.assertEqual(extract_answer_code(response), "public = 2\n")
        self.assertIsNone(extract_answer_code("<think>```py\nsecret = 1\n```</think>"))

    def test_multiple_complete_thoughts(self):
        response = "<think>one</think><think>two</think>```py\npublic = 2\n```"
        self.assertEqual(extract_answer_code(response), "public = 2\n")

    def test_answer_tags_inside_thoughts_are_ignored(self):
        responses = (
            "<think><answer>secret = 1</answer></think>",
            "<think><answer>unfinished</think>",
            "<think><answer code>malformed</think>",
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertEqual(
                    extract_answer_code(response + "<answer>public = 2</answer>"),
                    "public = 2",
                )

    def test_unclosed_fence_inside_complete_thought_is_ignored(self):
        response = "<think>```python\nsecret = 1</think>```py\npublic = 2\n```"
        self.assertEqual(extract_answer_code(response), "public = 2\n")

    def test_initial_orphan_thought_close_discards_entire_prefix(self):
        response = (
            "<answer>secret = 1</answer>\n```py\nsecret = 2\n```</think><answer>public = 3</answer>"
        )
        self.assertEqual(extract_answer_code(response), "public = 3")

    def test_initial_orphan_close_can_precede_complete_thought(self):
        response = "hidden</think><think>more hidden</think>```py\npublic = 1\n```"
        self.assertEqual(extract_answer_code(response), "public = 1\n")

    def test_initial_orphan_close_without_answer_returns_none(self):
        self.assertIsNone(extract_answer_code("```py\nsecret = 1\n```</think>"))

    def test_unclosed_thoughts_fail_even_after_a_visible_candidate(self):
        responses = (
            "<think>```python\nsecret = 1\n```",
            "<think><answer>secret = 1</answer>",
            "<answer>public = 2</answer><think>unfinished",
            "```py\npublic = 2\n```<think>unfinished",
            "<think\n```py\nsecret = 1\n```",
            "```py\nearlier = 2\n```\n<thin",
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertIsNone(extract_answer_code(response))

    def test_nested_or_repeated_orphan_thought_tags_are_ambiguous(self):
        responses = (
            "<think>outer<think>inner</think></think>",
            "first</think>second</think>",
            "<think>hidden</think>extra</think>",
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertIsNone(extract_answer_code(response + "<answer>public = 2</answer>"))

    def test_code_cannot_span_a_removed_thought(self):
        responses = (
            "<answer>x = <think>hidden</think>1</answer>",
            "```py\nx = <think>hidden</think>1\n```",
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertIsNone(extract_answer_code(response))

    def test_markup_inside_python_strings_remains_reserved(self):
        responses = (
            "```py\nprint('<think>')\n```",
            "```py\nprint('<answer>')\n```",
            "```py\nprint('<answer>x = 1</answer>')\n```",
            "```text\n<answer>x = 1</answer>\n```\n```py\nunfinished",
            "<answer>print('<think>x</think>')</answer>",
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertIsNone(extract_answer_code(response))

    def test_tag_case_and_spacing_are_accepted(self):
        response = "< THINK >hidden< / THINK >< ANSWER >x = 1< / ANSWER >"
        self.assertEqual(extract_answer_code(response), "x = 1")

    def test_multiple_python_fences_are_ambiguous(self):
        response = "```py\nx = 1\n```\n```python\nx = 2\n```"
        self.assertIsNone(extract_answer_code(response))
        self.assertIsNone(extract_answer_code(f"<answer>{response}</answer>"))

    def test_unsupported_fence_does_not_count_as_python(self):
        response = "```bash\necho example\n```\n```py\nx = 1\n```"
        self.assertEqual(extract_answer_code(response), "x = 1\n")
        self.assertIsNone(extract_answer_code("```javascript\nx = 1\n```"))
        self.assertIsNone(extract_answer_code("```python linenums\nx = 1\n```"))

    def test_fences_inside_unsupported_fence_do_not_count(self):
        response = "````text\n```py\nexample = 1\n```\n````"
        self.assertIsNone(extract_answer_code(response))

    def test_longer_and_tilde_fences(self):
        code = "value = '```'\n"
        for opening, closing in (("````py", "````"), ("~~~python", "~~~~")):
            with self.subTest(opening=opening):
                self.assertEqual(extract_answer_code(f"{opening}\n{code}{closing}"), code)

    def test_shorter_closing_fence_remains_payload(self):
        code = "text = '''\n```\n'''\n"
        self.assertEqual(extract_answer_code(f"````py\n{code}````"), code)

    def test_unclosed_or_mismatched_fences_fail(self):
        responses = (
            "```python\nx = 1",
            "```python\nx = 1\n~~~",
            "````python\nx = 1\n```",
            "```py\nx = 1\n```\n```bash\nunfinished",
        )
        for response in responses:
            with self.subTest(response=response):
                self.assertIsNone(extract_answer_code(response))

    def test_empty_whitespace_and_unfenced_text_fail(self):
        for response in ("", " \n\t", "Here is some code.", "x = 1", "```py\n \n```"):
            with self.subTest(response=response):
                self.assertIsNone(extract_answer_code(response))


if __name__ == "__main__":
    unittest.main()
