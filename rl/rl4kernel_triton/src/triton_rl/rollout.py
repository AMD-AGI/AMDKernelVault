# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collect one complete, on-policy Triton trajectory for one Slime Sample."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any

from .config import load_config
from .contracts import EvaluationResult
from .feedback import format_feedback
from .parsing import extract_answer_code
from .rewards import discounted_return, reward_func, turn_reward

if TYPE_CHECKING:
    from slime.utils.types import Sample

__all__ = ["GenerationAbortedError", "GenerationProtocolError", "generate", "reward_func"]

_THOUGHT_TAG = re.compile(r"<\s*(/?)\s*think\s*>", re.IGNORECASE)


class GenerationProtocolError(ValueError):
    """The generation server did not return a complete sampled-token record."""


class GenerationAbortedError(RuntimeError):
    """The server aborted a sample that the active batch still needs."""


def _slime_runtime(args: Any) -> tuple[Any, Any]:
    # CPU tests do not import Slime, PyTorch, or Transformers.
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.http_utils import post

    return GenerateState(args), post


def _sandbox_client() -> Any:
    from .sandbox import SandboxClient

    return SandboxClient.from_env()


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not isinstance(tokens, list) or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("The tokenizer must return a flat list of nonnegative token IDs.")
    return list(tokens)


def _read_generation(output: Any) -> tuple[str, list[int], list[float], str]:
    """Validate token IDs without reconstructing them from response text."""
    if not isinstance(output, Mapping) or not isinstance(output.get("text"), str):
        raise GenerationProtocolError("The generation response requires a text string.")
    meta = output.get("meta_info")
    if not isinstance(meta, Mapping) or not isinstance(meta.get("finish_reason"), Mapping):
        raise GenerationProtocolError("The generation response requires a finish reason.")
    finish = meta["finish_reason"].get("type")
    if finish not in ("stop", "length", "abort"):
        raise GenerationProtocolError(
            "The generation response contains an unsupported finish reason."
        )
    entries = meta.get("output_token_logprobs")
    if entries is None and finish == "abort" and not output["text"]:
        entries = []
    if not isinstance(entries, list):
        raise GenerationProtocolError("The generation response requires output_token_logprobs.")
    token_ids = []
    log_probs = []
    for entry in entries:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise GenerationProtocolError(
                "Each generated token requires a log probability and token ID."
            )
        log_prob, token_id = entry[:2]
        if type(token_id) is not int or token_id < 0:
            raise GenerationProtocolError("Each sampled token ID must be a nonnegative integer.")
        if isinstance(log_prob, bool) or not isinstance(log_prob, Real):
            raise GenerationProtocolError(
                "Each sampled log probability must be finite and nonpositive."
            )
        if not math.isfinite(log_prob) or log_prob > 0:
            raise GenerationProtocolError(
                "Each sampled log probability must be finite and nonpositive."
            )
        token_ids.append(token_id)
        log_probs.append(float(log_prob))
    if not token_ids and (output["text"] or finish != "abort"):
        raise GenerationProtocolError("The generation response omitted sampled token IDs.")
    if finish != "abort" and "completion_tokens" not in meta:
        raise GenerationProtocolError("The generation response requires a completion token count.")
    if "completion_tokens" in meta:
        count = meta["completion_tokens"]
        if type(count) is not int or count != len(token_ids):
            raise GenerationProtocolError(
                "The completion token count does not match the sampled token IDs."
            )
    return output["text"], token_ids, log_probs, finish


def _tolerance(ground_truth: Mapping[str, Any], defaults: Mapping[str, Any], name: str) -> float:
    value = ground_truth.get(name, defaults[name])
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"The {name} tolerance must be a finite, nonnegative number.")
    return float(value)


async def generate(args: Any, sample: Sample, sampling_params: Mapping[str, Any]) -> Sample:
    """Keep generated tokens and environment tokens in one exact trajectory."""
    if getattr(args, "partial_rollout", False):
        raise ValueError("Triton trajectories do not support partial rollouts.")
    if not isinstance(sample.prompt, str):
        raise ValueError("The sample prompt must already contain the rendered prompt text.")
    if sample.response or sample.response_length:
        raise ValueError("Start each Triton trajectory with an empty response.")
    if sample.status not in (sample.Status.PENDING, sample.Status.ABORTED):
        raise ValueError("Start each Triton trajectory with a pending sample.")
    if not isinstance(sample.label, Mapping) or not isinstance(
        sample.label.get("ground_truth"), Mapping
    ):
        raise ValueError("The sample label requires a ground_truth object.")
    ground_truth = sample.label["ground_truth"]
    filename = ground_truth.get("filename")
    if not isinstance(filename, str) or not filename:
        raise ValueError("The ground_truth object requires a filename.")

    config = load_config()
    algorithm = config["algorithm"]
    sequence_length = config["training"]["sequence_length"]
    if type(sequence_length) is not int or sequence_length < 1:
        raise ValueError("The sequence length must be a positive integer.")
    output_limit = sampling_params.get("max_new_tokens", sequence_length)
    if type(output_limit) is not int or output_limit < 1:
        raise ValueError("The generation limit must be a positive integer.")
    atol = _tolerance(ground_truth, config["evaluation"], "atol")
    rtol = _tolerance(ground_truth, config["evaluation"], "rtol")
    state, post = _slime_runtime(args)
    tokenizer = state.tokenizer
    prompt_ids = _tokenize(tokenizer, sample.prompt)
    if not prompt_ids or len(prompt_ids) >= sequence_length:
        raise ValueError("The prompt must contain tokens and leave space for generation.")
    if sample.tokens and sample.tokens != prompt_ids:
        raise ValueError("The initial sample tokens do not match the rendered prompt.")

    sample.tokens = list(prompt_ids)
    sample.loss_mask = []
    sample.rollout_log_probs = []
    sample.reward = None
    sample.metadata = dict(sample.metadata or {})
    trajectory = {
        "turns": [],
        "trajectory_return": 0.0,
        "discount": algorithm["discount"],
        "prompt_tokens": len(prompt_ids),
        "sequence_length": sequence_length,
        "stop_reason": None,
    }
    sample.metadata["triton_rl"] = trajectory
    # A rendered prompt can prefill the opening marker before sampling starts.
    thought_depth = int(re.search(r"<\s*think\s*>\s*$", sample.prompt, re.IGNORECASE) is not None)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    client = None
    failure = None

    def append(text: str, tokens: list[int], mask: int, log_probs: list[float]) -> None:
        sample.response += text
        sample.tokens.extend(tokens)
        sample.response_length += len(tokens)
        sample.loss_mask.extend([mask] * len(tokens))
        sample.rollout_log_probs.extend(log_probs)

    try:
        for turn in range(algorithm["max_turns"]):
            if state.aborted:
                sample.status = sample.Status.ABORTED
                trajectory["stop_reason"] = "framework_abort"
                break
            remaining = sequence_length - len(sample.tokens)
            if remaining < 1:
                sample.status = sample.Status.TRUNCATED
                trajectory["stop_reason"] = "context_limit"
                break
            params = dict(sampling_params)
            params["max_new_tokens"] = min(output_limit, remaining)
            # Return all sampled stop tokens in the response text when supported.
            params["no_stop_trim"] = True
            payload = {
                "input_ids": list(sample.tokens),
                "sampling_params": params,
                "return_logprob": True,
            }
            output = await post(url, payload, use_http2=getattr(args, "use_http2", False))
            text, token_ids, log_probs, finish = _read_generation(output)
            if len(token_ids) > params["max_new_tokens"]:
                raise GenerationProtocolError("The server exceeded the requested generation limit.")
            append(text, token_ids, 1, log_probs)
            if finish == "abort":
                sample.status = sample.Status.ABORTED
                trajectory["stop_reason"] = "generation_abort"
                if not state.aborted:
                    raise GenerationAbortedError(
                        "The generation server aborted an active trajectory."
                    )
                break
            if state.aborted:
                sample.status = sample.Status.ABORTED
                trajectory["stop_reason"] = "framework_abort"
                break

            # Parse only a copy. The sampled response remains unchanged.
            code = extract_answer_code("<think>" * thought_depth + text)
            for tag in _THOUGHT_TAG.finditer(text):
                thought_depth = max(0, thought_depth + (-1 if tag.group(1) else 1))
            if code is None:
                result = EvaluationResult(
                    compiled=False,
                    correct=False,
                    feedback="The response contains no unambiguous, complete kernel code block.",
                    error_type="missing_code",
                )
            else:
                if client is None:
                    client = _sandbox_client()
                result = await client.evaluate(code, filename, atol=atol, rtol=rtol)
                if not isinstance(result, EvaluationResult):
                    raise TypeError("The sandbox must return an EvaluationResult.")
            reward = turn_reward(result, algorithm)
            trajectory["turns"].append(
                {
                    "turn": turn + 1,
                    "reward": reward,
                    "evaluation": result.to_dict(),
                    "generated_tokens": len(token_ids),
                    "finish_reason": finish,
                }
            )
            trajectory["trajectory_return"] = discounted_return(
                (entry["reward"] for entry in trajectory["turns"]), algorithm["discount"]
            )
            if state.aborted:
                sample.status = sample.Status.ABORTED
                trajectory["stop_reason"] = "framework_abort"
                break
            if finish == "length":
                sample.status = sample.Status.TRUNCATED
                trajectory["stop_reason"] = "generation_length"
                break
            if turn + 1 == algorithm["max_turns"]:
                sample.status = sample.Status.COMPLETED
                trajectory["stop_reason"] = "turn_limit"
                break

            # Correct candidates also receive another optimization turn.
            feedback = format_feedback(result, turn + 1)
            feedback_ids = _tokenize(tokenizer, feedback)
            if len(feedback_ids) + len(sample.tokens) >= sequence_length:
                sample.status = sample.Status.TRUNCATED
                trajectory["stop_reason"] = "feedback_context_limit"
                break
            append(feedback, feedback_ids, 0, [0.0] * len(feedback_ids))
    except BaseException as error:
        # Infrastructure and protocol failures must not become candidate rewards.
        sample.status = sample.Status.ABORTED
        trajectory["stop_reason"] = trajectory["stop_reason"] or "error"
        failure = error
        raise
    finally:
        if client is not None:
            try:
                await client.close()
            except Exception as error:
                sample.status = sample.Status.ABORTED
                trajectory["cleanup_error"] = str(error)
                if failure is None:
                    trajectory["stop_reason"] = "cleanup_error"
                    raise

    trajectory["response_tokens"] = sample.response_length
    trajectory["generated_tokens"] = sum(sample.loss_mask)
    trajectory["feedback_tokens"] = sample.response_length - sum(sample.loss_mask)
    return sample
