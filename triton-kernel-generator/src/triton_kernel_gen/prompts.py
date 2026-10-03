# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Build generation and reflection messages from fixed task sources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GENERATION_SYSTEM = """Generate a complete Python implementation with real Triton 3.3.0 kernels for AMD ROCm.
Use @triton.jit and triton.language for the tensor computation.
Expose module_fn with exactly the signature and return structure in the verified functional reference.
Preserve the reference semantics, tensor dtypes, device placement, and supported input layouts.
Use the original PyTorch module as context and the verified functional reference as the correctness authority.
Treat a supplied raw Triton seed as unverified source context.
Do not assume that the raw seed compiles or runs on AMD.
Port CUDA-specific operations and unsupported source constructs to AMD-compatible Triton.
Use a supplied verified Triton baseline as optimization context.
Use measured baseline timings from evaluator feedback.
Support the task's valid shapes, strides, dtypes, and boundary cases with generic indexing and launch logic.
Do not hardcode test inputs, test answers, or one input configuration.
Do not replace the tensor computation with PyTorch, an imported reference, or an external implementation.
Host-side shape checks and tensor allocation are permitted.
Do not modify the reference, seed, baseline, tests, case provider, numerical tolerances, evaluator, timings, or output records.
Do not include tests, benchmarks, file writes, command execution, or a main program.
Do not read evaluator files or inspect the call stack.
Use AMD-compatible operations and schedules.
Account for wavefront behavior, LDS usage, and occupancy on the supplied hardware.
gfx942 and gfx950 are possible targets, not an instruction to assume either target.
Return exactly one complete Python code fence, optionally inside one complete <answer> region.
Keep reasoning outside the final code fence.
"""

REFLECTION_SYSTEM = """Review a generated Triton candidate against the fixed functional reference and measured evaluator feedback.
Explain the failure cause or a concrete latency improvement that the evidence supports.
Give the generator a short repair plan or a plan for a different correct implementation.
Distinguish measured results from hypotheses.
Preserve generic shapes, strides, dtypes, and all reference semantics.
Keep real Triton 3.3.0 kernels compatible with AMD ROCm.
Treat a supplied raw Triton seed as unverified source context that can require porting.
Use a supplied verified Triton baseline and measured latency as optimization evidence.
Do not change the reference, seed, baseline, tests, case provider, tolerances, evaluator, timings, or output records.
Do not propose hardcoded test answers or PyTorch computation as the candidate implementation.
Do not claim a speedup when the measurements do not show one.
Return analysis and a plan, not a replacement implementation.
"""


def _source(label: str, filename: str, code: str) -> str:
    """Use supplied basenames instead of adding local directory names."""
    # JSON keeps source delimiters unambiguous even when the source contains fences.
    return f"{label}:\n" + json.dumps(
        {"filename": Path(filename).name, "source": code}, ensure_ascii=False
    )


def _context(
    module_code: str,
    functional_code: str,
    *,
    module_name: str,
    functional_name: str,
    seed_code: str | None,
    seed_name: str | None,
    environment: dict[str, Any] | None,
    dependencies: list[dict[str, str]] | None = None,
    baseline_code: str | None = None,
    baseline_name: str | None = None,
) -> str:
    parts = [
        _source("Original PyTorch module", module_name, module_code),
        _source("Verified functional reference", functional_name, functional_code),
    ]
    if seed_code is not None:
        parts.append(
            _source(
                "Raw Triton seed (unverified source context)", seed_name or "seed.py", seed_code
            )
        )
    if baseline_code is not None:
        parts.append(
            _source("Verified Triton baseline", baseline_name or "baseline.py", baseline_code)
        )
    for dependency in dependencies or []:
        parts.append(
            _source("Declared source dependency", dependency["filename"], dependency["source"])
        )
    if environment:
        parts.append(
            "Measured execution environment:\n"
            + json.dumps(environment, ensure_ascii=False, sort_keys=True)
        )
    return "\n\n".join(parts)


def _bounded_history(history: list[dict[str, Any]], limit: int) -> str:
    if not history:
        return ""
    serialized = json.dumps(history, ensure_ascii=False, sort_keys=True)
    if len(serialized) <= limit:
        return "Previous attempts and reflection plans:\n" + serialized
    # Do not let a long candidate or reflection remove the current failure evidence.
    latest = history[-1]
    sections = []
    if "feedback" in latest:
        sections.append(
            (
                "Latest evaluator feedback",
                json.dumps(latest["feedback"], ensure_ascii=False, sort_keys=True),
            )
        )
    sections.extend(
        (label, latest[key])
        for label, key in (
            ("Latest reflection plan", "reflection"),
            ("Previous candidate", "candidate"),
        )
        if latest.get(key)
    )
    if not sections:
        sections = [("Latest attempt", json.dumps(latest, ensure_ascii=False, sort_keys=True))]
    allowance = max(1, limit // len(sections))

    def clip(text: str) -> str:
        if len(text) <= allowance:
            return text
        marker = "\n[context omitted]\n"
        if allowance <= len(marker):
            return text[:allowance]
        retained = allowance - len(marker)
        first = (retained + 1) // 2
        last = retained // 2
        return text[:first] + marker + (text[-last:] if last else "")

    return "Previous attempts and reflection plans (leading context omitted):\n" + "\n\n".join(
        label + ":\n" + clip(text) for label, text in sections
    )


def build_generation_messages(
    module_code: str,
    functional_code: str,
    *,
    module_name: str,
    functional_name: str,
    seed_code: str | None = None,
    seed_name: str | None = None,
    environment: dict[str, Any] | None = None,
    history: list[dict[str, Any]] | None = None,
    history_char_limit: int = 16000,
    dependencies: list[dict[str, str]] | None = None,
    baseline_code: str | None = None,
    baseline_name: str | None = None,
) -> list[dict[str, str]]:
    """Include both source interfaces and the available reflection feedback."""
    if history_char_limit <= 0:
        raise ValueError("history_char_limit must exceed zero.")
    context = _context(
        module_code,
        functional_code,
        module_name=module_name,
        functional_name=functional_name,
        seed_code=seed_code,
        seed_name=seed_name,
        environment=environment,
        dependencies=dependencies,
        baseline_code=baseline_code,
        baseline_name=baseline_name,
    )
    prior = _bounded_history(history or [], history_char_limit)
    if prior:
        context += "\n\n" + prior
    context += "\n\nGenerate the next complete candidate. Follow the verified module_fn interface."
    return [{"role": "system", "content": GENERATION_SYSTEM}, {"role": "user", "content": context}]


def build_reflection_messages(
    module_code: str,
    functional_code: str,
    *,
    module_name: str,
    functional_name: str,
    candidate_text: str,
    feedback: dict[str, Any],
    seek_variant: bool = False,
    environment: dict[str, Any] | None = None,
    dependencies: list[dict[str, str]] | None = None,
    seed_code: str | None = None,
    seed_name: str | None = None,
    baseline_code: str | None = None,
    baseline_name: str | None = None,
) -> list[dict[str, str]]:
    """Ask a separate model call to interpret execution or latency evidence."""
    context = _context(
        module_code,
        functional_code,
        module_name=module_name,
        functional_name=functional_name,
        seed_code=seed_code,
        seed_name=seed_name,
        environment=environment,
        dependencies=dependencies,
        baseline_code=baseline_code,
        baseline_name=baseline_name,
    )
    context += "\n\nPrevious candidate or unparsed model response:\n" + json.dumps(
        candidate_text, ensure_ascii=False
    )
    context += "\n\nEvaluator feedback:\n" + json.dumps(
        feedback, ensure_ascii=False, sort_keys=True
    )
    if seek_variant:
        context += (
            "\n\nThis candidate passed correctness and timing. Plan a different correct implementation."
            " Use the per-case latency measurements when proposing an improvement."
            " A correct slower candidate remains valid. No universal speedup threshold applies."
        )
    else:
        context += "\n\nExplain the observed failure. Give a specific plan for the next candidate."
    return [{"role": "system", "content": REFLECTION_SYSTEM}, {"role": "user", "content": context}]
