# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check Slime trajectory contracts without a model, GPU, or dataset."""

import copy
import json
import math
import unittest
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from unittest.mock import patch

from triton_rl import rollout
from triton_rl.config import load_config
from triton_rl.contracts import EvaluationResult
from triton_rl.feedback import format_feedback
from triton_rl.rewards import reward_func


@dataclass
class FakeSample:
    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    prompt: str = "TASK\n"
    tokens: list[int] = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    label: dict = field(default_factory=lambda: {"ground_truth": {"filename": "synthetic.py"}})
    reward: float | None = None
    loss_mask: list[int] | None = None
    rollout_log_probs: list[float] | None = None
    metadata: dict = field(default_factory=lambda: {"source": "synthetic-test"})
    status: Status = Status.PENDING


class FakeTokenizer:
    """Reject attempts to tokenize generated text again."""

    def __init__(self):
        self.calls = []
        self.feedback_length = 2
        self.prompt_text = "TASK\n"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        self.calls.append(text)
        if text == self.prompt_text:
            return {"input_ids": [11, 12]}
        if text.startswith("\n\n<execution_result>\n"):
            return {"input_ids": list(range(900, 900 + self.feedback_length))}
        raise AssertionError("The rollout tried to tokenize generated text again.")


class FakeSandbox:
    def __init__(self):
        self.results = []
        self.calls = []
        self.close_calls = 0
        self.close_error = None
        self.on_evaluate = None

    async def evaluate(self, code, filename, *, atol, rtol):
        self.calls.append((code, filename, atol, rtol))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        if self.on_evaluate is not None:
            self.on_evaluate()
        return result

    async def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


def generation(text, ids, finish="stop", probabilities=None):
    probabilities = probabilities or [-0.1 * (index + 1) for index in range(len(ids))]
    return {
        "text": text,
        "meta_info": {
            "finish_reason": {"type": finish},
            "output_token_logprobs": [
                [prob, token, None] for prob, token in zip(probabilities, ids)
            ],
            "completion_tokens": len(ids),
        },
    }


class RolloutTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = load_config()
        self.tokenizer = FakeTokenizer()
        self.state = SimpleNamespace(tokenizer=self.tokenizer, aborted=False)
        self.abort_after_post = False
        self.sandbox = FakeSandbox()
        self.outputs = []
        self.requests = []
        self.args = SimpleNamespace(
            partial_rollout=False,
            sglang_router_ip="localhost",
            sglang_router_port=30000,
            use_http2=True,
        )
        self.params = {"max_new_tokens": 20, "temperature": 0.8}

        async def post(url, payload, *, use_http2):
            self.requests.append((url, copy.deepcopy(payload), use_http2))
            if self.abort_after_post:
                self.state.aborted = True
            return self.outputs.pop(0)

        for target, replacement in [
            ("_slime_runtime", lambda args: (self.state, post)),
            ("_sandbox_client", lambda: self.sandbox),
            ("load_config", lambda: copy.deepcopy(self.config)),
        ]:
            patcher = patch.object(rollout, target, replacement)
            patcher.start()
            self.addCleanup(patcher.stop)

    async def test_exact_sampled_tokens_masks_and_three_turns_after_success(self):
        texts = [
            "  <think>Try a kernel.</think>\n```python\nx = 1\n```\n",
            "Reflection: retain correctness.\n<answer>\nx = 2\n</answer>  ",
            "Reflection: reduce latency.\n```python\nx = 3\n```\n",
        ]
        ids = [[41, 42, 43], [51, 52], [61, 62, 63, 64]]
        self.outputs = [generation(text, tokens) for text, tokens in zip(texts, ids)]
        results = [
            EvaluationResult(True, True, speedup=speedup, feedback="Correct.")
            for speedup in (1, 2, 4)
        ]
        self.sandbox.results = list(results)
        sample = FakeSample()
        original_params = copy.deepcopy(self.params)
        returned = await rollout.generate(self.args, sample, self.params)

        self.assertIs(returned, sample)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(len(self.sandbox.calls), 3)
        feedbacks = [format_feedback(result, turn + 1) for turn, result in enumerate(results[:2])]
        expected_tokens = [11, 12] + ids[0] + [900, 901] + ids[1] + [900, 901] + ids[2]
        self.assertEqual(sample.tokens, expected_tokens)
        self.assertEqual(
            sample.response, texts[0] + feedbacks[0] + texts[1] + feedbacks[1] + texts[2]
        )
        self.assertEqual(sample.loss_mask, [1] * 3 + [0] * 2 + [1] * 2 + [0] * 2 + [1] * 4)
        expected_probs = (
            [-0.1, -0.2, -0.30000000000000004]
            + [0.0, 0.0]
            + [-0.1, -0.2]
            + [0.0, 0.0]
            + [-0.1, -0.2, -0.30000000000000004, -0.4]
        )
        self.assertEqual(sample.rollout_log_probs, expected_probs)
        self.assertEqual(sample.response_length, len(expected_tokens) - 2)
        self.assertEqual(len(sample.loss_mask), sample.response_length)
        self.assertEqual(len(sample.rollout_log_probs), sample.response_length)
        self.assertEqual(self.requests[0][1]["input_ids"], [11, 12])
        self.assertEqual(self.requests[1][1]["input_ids"], [11, 12] + ids[0] + [900, 901])
        self.assertEqual(self.requests[2][1]["input_ids"], expected_tokens[:-4])
        for url, payload, use_http2 in self.requests:
            self.assertEqual(url, "http://localhost:30000/generate")
            self.assertTrue(use_http2)
            self.assertTrue(payload["return_logprob"])
            self.assertNotIn("text", payload)
            self.assertTrue(payload["sampling_params"]["no_stop_trim"])
        self.assertEqual(self.params, original_params)
        self.assertEqual(self.tokenizer.calls, [sample.prompt] + feedbacks)
        self.assertEqual(sample.status, sample.Status.COMPLETED)
        self.assertEqual(sample.metadata["source"], "synthetic-test")
        self.assertEqual(sample.metadata["triton_rl"]["stop_reason"], "turn_limit")
        self.assertAlmostEqual(await reward_func(self.args, sample), 1.4 + 0.6 * 1.775 + 0.36 * 2.9)
        self.assertIsNone(sample.reward)
        self.assertEqual(self.sandbox.calls[0][2:], (0.001, 0.0001))
        self.assertEqual(self.sandbox.close_calls, 1)

    async def test_parse_failure_receives_feedback_and_keeps_turn_discount(self):
        self.outputs = [
            generation("<think>```python\nx = 1\n```</think>Still thinking.", [21]),
            generation("```python\nx = 2\n```", [22]),
            generation("<answer>x = 3</answer>", [23]),
        ]
        self.sandbox.results = [
            EvaluationResult(True, False, feedback="Wrong result."),
            EvaluationResult(True, True),
        ]
        sample = await rollout.generate(self.args, FakeSample(), self.params)
        turns = sample.metadata["triton_rl"]["turns"]
        self.assertEqual([turn["reward"] for turn in turns], [0.0, 0.4, 1.4])
        self.assertEqual(turns[0]["evaluation"]["error_type"], "missing_code")
        self.assertEqual(len(self.sandbox.calls), 2)
        self.assertIn("no unambiguous, complete kernel", sample.response)
        self.assertAlmostEqual(await reward_func(self.args, sample), 0.6 * 0.4 + 0.36 * 1.4)

    async def test_context_limit_bounds_each_request_without_rewriting_history(self):
        self.config["training"]["sequence_length"] = 12
        self.outputs = [
            generation("<answer>x = 1</answer>", [31, 32]),
            generation("<answer>x = 2</answer>", [33, 34], "length"),
        ]
        self.sandbox.results = [EvaluationResult(True, True), EvaluationResult(True, True)]
        sample = await rollout.generate(self.args, FakeSample(), self.params)
        self.assertEqual(
            [request[1]["sampling_params"]["max_new_tokens"] for request in self.requests], [10, 6]
        )
        self.assertEqual(sample.status, sample.Status.TRUNCATED)
        self.assertEqual(sample.metadata["triton_rl"]["stop_reason"], "generation_length")
        self.assertEqual(sample.tokens, [11, 12, 31, 32, 900, 901, 33, 34])
        self.assertEqual(len(sample.metadata["triton_rl"]["turns"]), 2)

    async def test_feedback_must_fit_with_space_for_the_next_generated_token(self):
        self.config["training"]["sequence_length"] = 6
        self.outputs = [generation("<answer>x = 1</answer>", [31, 32])]
        self.sandbox.results = [EvaluationResult(True, True)]
        sample = await rollout.generate(self.args, FakeSample(), self.params)
        self.assertEqual(sample.tokens, [11, 12, 31, 32])
        self.assertEqual(sample.loss_mask, [1, 1])
        self.assertNotIn("execution_result", sample.response)
        self.assertEqual(sample.status, sample.Status.TRUNCATED)
        self.assertEqual(sample.metadata["triton_rl"]["stop_reason"], "feedback_context_limit")
        self.assertEqual(await reward_func(self.args, sample), 1.4)

    async def test_abort_preserves_returned_tokens_without_evaluating_them(self):
        self.abort_after_post = True
        self.outputs = [generation("unfinished", [77, 78], "abort")]
        sample = await rollout.generate(self.args, FakeSample(), self.params)
        self.assertEqual(sample.status, sample.Status.ABORTED)
        self.assertEqual(sample.tokens, [11, 12, 77, 78])
        self.assertEqual(sample.response, "unfinished")
        self.assertEqual(sample.loss_mask, [1, 1])
        self.assertEqual(sample.metadata["triton_rl"]["turns"], [])
        self.assertEqual(self.sandbox.calls, [])

    async def test_empty_abort_can_omit_log_probabilities(self):
        self.abort_after_post = True
        self.outputs = [{"text": "", "meta_info": {"finish_reason": {"type": "abort"}}}]
        sample = await rollout.generate(self.args, FakeSample(), self.params)
        self.assertEqual(sample.status, sample.Status.ABORTED)
        self.assertEqual(sample.tokens, [11, 12])
        self.assertEqual(sample.loss_mask, [])

    async def test_active_abort_raises_instead_of_returning_an_unrewarded_sample(self):
        self.outputs = [generation("unfinished", [77, 78], "abort")]
        sample = FakeSample()
        with self.assertRaises(rollout.GenerationAbortedError):
            await rollout.generate(self.args, sample, self.params)
        self.assertEqual(sample.status, sample.Status.ABORTED)
        self.assertEqual(sample.tokens, [11, 12, 77, 78])
        self.assertEqual(sample.response, "unfinished")
        self.assertEqual(sample.metadata["triton_rl"]["stop_reason"], "generation_abort")

    async def test_framework_abort_before_generation_makes_no_request(self):
        self.state.aborted = True
        sample = await rollout.generate(self.args, FakeSample(), self.params)
        self.assertEqual(sample.status, sample.Status.ABORTED)
        self.assertEqual(self.requests, [])

    async def test_framework_abort_during_evaluation_prevents_the_next_turn(self):
        self.outputs = [generation("<answer>x = 1</answer>", [31])]
        self.sandbox.results = [EvaluationResult(True, True)]
        self.sandbox.on_evaluate = lambda: setattr(self.state, "aborted", True)
        sample = await rollout.generate(self.args, FakeSample(), self.params)
        self.assertEqual(sample.status, sample.Status.ABORTED)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.sandbox.close_calls, 1)
        self.assertNotIn("execution_result", sample.response)
        self.assertEqual(sample.metadata["triton_rl"]["stop_reason"], "framework_abort")

    async def test_empty_non_abort_generation_is_rejected(self):
        self.outputs = [generation("", [], "length")]
        sample = FakeSample()
        with self.assertRaises(rollout.GenerationProtocolError):
            await rollout.generate(self.args, sample, self.params)
        self.assertEqual(sample.status, sample.Status.ABORTED)
        self.assertEqual(self.sandbox.calls, [])

    async def test_length_preserves_unfinished_text_and_gets_no_code_reward(self):
        self.outputs = [generation("<think>unfinished reasoning", [71, 72], "length")]
        sample = await rollout.generate(self.args, FakeSample(), self.params)
        self.assertEqual(sample.response, "<think>unfinished reasoning")
        self.assertEqual(sample.status, sample.Status.TRUNCATED)
        self.assertEqual(await reward_func(self.args, sample), 0.0)
        self.assertEqual(self.sandbox.calls, [])

    async def test_prefilled_unclosed_thought_does_not_execute_its_fence(self):
        self.tokenizer.prompt_text = "TASK\n< THINK >\n"
        text = "```python\nthought_code = 1\n```"
        self.outputs = [generation(text, [71, 72], "length")]
        sample = FakeSample(prompt=self.tokenizer.prompt_text)
        await rollout.generate(self.args, sample, self.params)
        self.assertEqual(sample.response, text)
        self.assertEqual(sample.tokens, [11, 12, 71, 72])
        self.assertEqual(await reward_func(self.args, sample), 0.0)
        self.assertEqual(self.sandbox.calls, [])

    async def test_open_thought_state_continues_until_the_policy_closes_it(self):
        self.outputs = [
            generation("<think>Unfinished thought.", [71]),
            generation("```python\nthought_code = 1\n```", [72]),
            generation("</think><answer>answer_code = 2</answer>", [73]),
        ]
        self.sandbox.results = [EvaluationResult(True, True)]
        sample = await rollout.generate(self.args, FakeSample(), self.params)
        self.assertEqual(
            [entry["reward"] for entry in sample.metadata["triton_rl"]["turns"]], [0.0, 0.0, 1.4]
        )
        self.assertEqual([entry[0] for entry in self.sandbox.calls], ["answer_code = 2"])
        self.assertAlmostEqual(await reward_func(self.args, sample), 0.36 * 1.4)

    async def test_generation_protocol_errors_do_not_produce_candidate_rewards(self):
        valid = generation("<answer>x = 1</answer>", [21])
        invalid_outputs = []
        for change in [
            lambda item: item["meta_info"].pop("output_token_logprobs"),
            lambda item: item["meta_info"].pop("completion_tokens"),
            lambda item: item["meta_info"].update(completion_tokens=2),
            lambda item: item["meta_info"].update(output_token_logprobs=[[-0.1, True]]),
            lambda item: item["meta_info"].update(output_token_logprobs=[[-0.1, -1]]),
            lambda item: item["meta_info"].update(output_token_logprobs=[[math.nan, 21]]),
            lambda item: item["meta_info"].update(output_token_logprobs=[[None, 21]]),
            lambda item: item["meta_info"].update(output_token_logprobs=[[0.1, 21]]),
            lambda item: item["meta_info"].update(finish_reason={"type": "unknown"}),
            lambda item: item.update(text=None),
            lambda item: item["meta_info"].update(output_token_logprobs=[]),
        ]:
            item = copy.deepcopy(valid)
            change(item)
            invalid_outputs.append(item)
        for output in invalid_outputs:
            with self.subTest(output=output):
                self.outputs = [output]
                sample = FakeSample()
                with self.assertRaises(rollout.GenerationProtocolError):
                    await rollout.generate(self.args, sample, self.params)
                self.assertEqual(sample.status, sample.Status.ABORTED)
                self.assertEqual(sample.response, "")
                self.assertEqual(self.sandbox.calls, [])

    async def test_server_cannot_exceed_the_request_budget(self):
        self.params["max_new_tokens"] = 1
        self.outputs = [generation("<answer>x = 1</answer>", [31, 32])]
        sample = FakeSample()
        with self.assertRaises(rollout.GenerationProtocolError):
            await rollout.generate(self.args, sample, self.params)
        self.assertEqual(sample.tokens, [11, 12])
        self.assertEqual(self.sandbox.calls, [])

    async def test_infrastructure_error_raises_and_closes_client(self):
        self.outputs = [generation("<answer>x = 1</answer>", [31])]
        self.sandbox.results = [RuntimeError("The execution service is unavailable.")]
        sample = FakeSample()
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            await rollout.generate(self.args, sample, self.params)
        self.assertEqual(sample.status, sample.Status.ABORTED)
        self.assertEqual(sample.metadata["triton_rl"]["turns"], [])
        self.assertEqual(self.sandbox.close_calls, 1)

    async def test_cleanup_failure_aborts_the_sample(self):
        self.outputs = [generation("<answer>x = 1</answer>", [31], "length")]
        self.sandbox.results = [EvaluationResult(True, True)]
        self.sandbox.close_error = RuntimeError("The client did not close.")
        sample = FakeSample()
        with self.assertRaisesRegex(RuntimeError, "did not close"):
            await rollout.generate(self.args, sample, self.params)
        self.assertEqual(sample.status, sample.Status.ABORTED)
        self.assertEqual(sample.metadata["triton_rl"]["stop_reason"], "cleanup_error")

    async def test_cleanup_failure_does_not_replace_an_earlier_error(self):
        self.outputs = [generation("<answer>x = 1</answer>", [31])]
        self.sandbox.results = [RuntimeError("The execution service is unavailable.")]
        self.sandbox.close_error = RuntimeError("The client did not close.")
        sample = FakeSample()
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            await rollout.generate(self.args, sample, self.params)
        self.assertEqual(sample.status, sample.Status.ABORTED)
        self.assertEqual(sample.metadata["triton_rl"]["cleanup_error"], "The client did not close.")

    async def test_task_tolerances_override_fallbacks(self):
        self.outputs = [generation("<answer>x = 1</answer>", [31], "length")]
        self.sandbox.results = [EvaluationResult(True, True)]
        sample = FakeSample(
            label={"ground_truth": {"filename": "synthetic.py", "atol": 0.25, "rtol": 0.5}}
        )
        await rollout.generate(self.args, sample, self.params)
        self.assertEqual(self.sandbox.calls, [("x = 1", "synthetic.py", 0.25, 0.5)])

    async def test_partial_rollouts_and_invalid_initial_context_are_rejected(self):
        self.args.partial_rollout = True
        with self.assertRaisesRegex(ValueError, "partial"):
            await rollout.generate(self.args, FakeSample(), self.params)
        self.args.partial_rollout = False
        for sample in [
            FakeSample(response="old response"),
            FakeSample(tokens=[999]),
            FakeSample(label={}),
        ]:
            with self.subTest(sample=sample):
                with self.assertRaises(ValueError):
                    await rollout.generate(self.args, sample, self.params)
        self.config["training"]["sequence_length"] = 2
        with self.assertRaisesRegex(ValueError, "leave space"):
            await rollout.generate(self.args, FakeSample(), self.params)
        self.assertEqual(self.requests, [])


class FeedbackTests(unittest.TestCase):
    def test_oversized_numbers_become_unavailable_timing(self):
        result = EvaluationResult(True, True, speedup=10**400, latency_ms=10**400)
        text = format_feedback(result, 1)
        self.assertIn('"speedup": null', text)
        self.assertIn('"latency_ms": null', text)

    def test_feedback_contains_execution_facts_and_shared_policy_instructions(self):
        result = EvaluationResult(
            True, True, speedup=math.nan, feedback="Compiler and runtime output."
        )
        text = format_feedback(result, 2)
        facts = json.loads(
            text.split("<execution_result>\n", 1)[1].split("\n</execution_result>", 1)[0]
        )
        self.assertEqual(facts["turn"], 2)
        self.assertTrue(facts["compiled"])
        self.assertTrue(facts["correct"])
        self.assertIsNone(facts["speedup"])
        self.assertEqual(facts["feedback"], result.feedback)
        self.assertIn("Reflect on the execution result.", text)
        self.assertIn("Improve the kernel while preserving correctness.", text)
        self.assertIn("complete revised kernel", text)


if __name__ == "__main__":
    unittest.main()
