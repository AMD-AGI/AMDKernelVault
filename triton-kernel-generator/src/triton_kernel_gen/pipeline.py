# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Generate, evaluate, and reflect without changing the trusted task."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from .contracts import GenerationSettings, ModelResponse, PreparedTask, TaskSpec, VerificationResult
from .errors import ReferenceValidationError
from .parsing import CandidateParseError, extract_candidate
from .prompts import build_generation_messages, build_reflection_messages


class GenerationClient(Protocol):
    def generate(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> ModelResponse: ...


class TaskVerifier(Protocol):
    def prepare(self, task: TaskSpec, work_dir: Path) -> PreparedTask: ...

    def verify(self, prepared: PreparedTask, candidate_path: Path) -> VerificationResult: ...


class InputModifiedError(RuntimeError):
    """A declared source, snapshot, or candidate changed during this run."""


class RecordWriteError(RuntimeError):
    """The pipeline cannot persist a complete checkpoint."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Records require JSON values, not {type(value).__name__}.")


def _atomic_json(path: Path, record: dict[str, Any]) -> None:
    """Replace a checkpoint only after its complete JSON reaches disk."""
    try:
        encoded = (
            json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default)
            + "\n"
        )
        _replace_json(path, encoded)
    except (OSError, TypeError, ValueError) as error:
        raise RecordWriteError(f"Cannot persist the record {path.name}.") from error


def _replace_json(path: Path, encoded: str) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _immutable_bytes(path: Path, content: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def _validate(task: TaskSpec, settings: GenerationSettings) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", task.task_id):
        raise ValueError("task_id must contain a safe filename identifier.")
    for name in ("max_attempts", "num_variants", "max_tokens", "history_char_limit"):
        value = getattr(settings, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if settings.num_variants > settings.max_attempts:
        raise ValueError("num_variants must not exceed max_attempts.")
    if (
        isinstance(settings.temperature, bool)
        or not math.isfinite(settings.temperature)
        or settings.temperature < 0
    ):
        raise ValueError("temperature must be finite and nonnegative.")
    for name in ("output_dir", "artifacts_dir"):
        location = Path(getattr(settings, name)).resolve()
        if location.exists() and not location.is_dir():
            raise ValueError(f"{name} must name a directory.")
        if name == "output_dir" and (location / task.task_id).is_symlink():
            raise ValueError("The task output directory must not be a symbolic link.")
    # Fail before a model call if user provenance cannot enter the result record.
    json.dumps(task.provenance, allow_nan=False, default=_json_default)


def _snapshot(
    task: TaskSpec, run_dir: Path
) -> tuple[dict[str, dict[str, str]], list[tuple[Path, str]]]:
    sources: dict[str, dict[str, str]] = {}
    guards: list[tuple[Path, str]] = []
    source_paths = [
        ("module", task.module_path),
        ("functional", task.functional_path),
        ("case_provider", task.case_provider_path),
        ("seed_kernel", task.seed_kernel_path),
        ("baseline_kernel", task.baseline_kernel_path),
    ]
    source_paths.extend(
        (f"dependency_{index}", path) for index, path in enumerate(task.dependency_paths)
    )
    for role, source_path in source_paths:
        if source_path is None:
            continue
        source_path = Path(source_path)
        try:
            content = source_path.read_bytes()
        except OSError as error:
            raise ReferenceValidationError(
                f"Cannot read the declared source {source_path.name}: {error}"
            ) from error
        sha256 = _digest(content)
        snapshot_dir = run_dir / "inputs" / role
        snapshot_dir.mkdir(parents=True, mode=0o700)
        snapshot_path = snapshot_dir / source_path.name
        _immutable_bytes(snapshot_path, content)
        sources[role] = {
            "filename": source_path.name,
            "sha256": sha256,
            "snapshot_path": str(snapshot_path),
        }
        if role == "seed_kernel":
            sources[role]["purpose"] = "unverified_source_context"
        elif role == "baseline_kernel":
            sources[role]["purpose"] = "optional_latency_baseline"
        guards.extend(((source_path, sha256), (snapshot_path, sha256)))
    return sources, guards


def _guard(guards: list[tuple[Path, str]]) -> None:
    for path, expected in guards:
        try:
            actual = _digest(path.read_bytes())
        except OSError as error:
            raise InputModifiedError(f"A guarded file is unavailable: {path.name}.") from error
        if actual != expected:
            raise InputModifiedError(f"A guarded file changed: {path.name}.")


def _error(error: Exception, phase: str) -> dict[str, str]:
    return {"type": type(error).__name__, "phase": phase, "message": str(error)}


def _result_record(result: VerificationResult) -> dict[str, Any]:
    if not isinstance(result, VerificationResult):
        raise TypeError("The verifier must return VerificationResult.")
    record = asdict(result)
    record["success"] = result.success
    record["correctness_case_count"] = len(result.correctness_cases)
    record["performance_case_count"] = len(result.performance_cases)
    return record


def _prepare_failure_status(error: ReferenceValidationError) -> str:
    if error.failure_stage in {"input_modified", "infrastructure"}:
        return (
            error.failure_stage
            if error.failure_stage == "input_modified"
            else "infrastructure_error"
        )
    return "reference_failed"


def run_task(
    task: TaskSpec,
    generator_client: GenerationClient,
    reflector_client: GenerationClient,
    verifier: TaskVerifier,
    settings: GenerationSettings,
) -> dict[str, Any]:
    """Retain verified, distinct candidates within one total generation budget.

    Each call creates a new run. It never trusts an output from another run.
    The return value matches ``record.json`` in that run's artifact directory.
    Invalid settings and failures to persist records raise exceptions.
    Provider, input, reference, and execution failures return explicit statuses.
    """
    _validate(task, settings)
    artifact_root = Path(settings.artifacts_dir).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"{task.task_id}-", dir=artifact_root))
    (run_dir / "attempts").mkdir(mode=0o700)
    (run_dir / "candidates").mkdir(mode=0o700)
    record: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "task_id": task.task_id,
        "status": "preparing",
        "started_at": _now(),
        "completed_at": None,
        "run_dir": str(run_dir),
        "record_path": str(run_dir / "record.json"),
        "provenance": task.provenance,
        "sources": {},
        "settings": asdict(settings),
        "retention_policy": "first_valid_unique_candidates",
        "reference": None,
        "attempts": [],
        "accepted_candidates": [],
        "counts": {
            "generation_calls": 0,
            "reflection_calls": 0,
            "verification_calls": 0,
            "accepted_candidates": 0,
        },
    }

    def checkpoint(attempt: dict[str, Any] | None = None) -> None:
        if attempt is not None:
            _atomic_json(run_dir / "attempts" / f"attempt-{attempt['attempt']:04d}.json", attempt)
        _atomic_json(run_dir / "record.json", record)

    def finish(status: str, error: Exception | None = None, phase: str = "") -> dict[str, Any]:
        record["status"] = status
        record["completed_at"] = _now()
        if error is not None:
            record["error"] = _error(error, phase)
        checkpoint()
        # Return the persisted JSON representation, including string paths.
        return json.loads((run_dir / "record.json").read_text(encoding="utf-8"))

    checkpoint()
    guards: list[tuple[Path, str]] = []
    try:
        sources, guards = _snapshot(task, run_dir)
        record["sources"] = sources
        _guard(guards)
        checkpoint()
        prepared = verifier.prepare(task, run_dir / "verification")
        record["reference"] = prepared.reference_record
        _guard(guards)
    except RecordWriteError:
        raise
    except InputModifiedError as error:
        return finish("input_modified", error, "prepare")
    except ReferenceValidationError as error:
        try:
            _guard(guards)
        except InputModifiedError as changed:
            return finish("input_modified", changed, "prepare")
        # Reference and case-provider failures never become candidate failures.
        record["reference_failure_stage"] = error.failure_stage
        return finish(_prepare_failure_status(error), error, "prepare")
    except (OSError, UnicodeError) as error:
        try:
            _guard(guards)
        except InputModifiedError as changed:
            return finish("input_modified", changed, "prepare")
        return finish("infrastructure_error", error, "prepare")
    except Exception as error:
        try:
            _guard(guards)
        except InputModifiedError as changed:
            return finish("input_modified", changed, "prepare")
        return finish("infrastructure_error", error, "prepare")

    try:
        prompt_sources = {
            "module_code": Path(sources["module"]["snapshot_path"]).read_text(encoding="utf-8"),
            "functional_code": Path(sources["functional"]["snapshot_path"]).read_text(
                encoding="utf-8"
            ),
            "module_name": sources["module"]["filename"],
            "functional_name": sources["functional"]["filename"],
            "environment": prepared.reference_record.get("environment", {}),
            "dependencies": [
                {
                    "filename": source["filename"],
                    "source": Path(source["snapshot_path"]).read_text(encoding="utf-8"),
                }
                for role, source in sources.items()
                if role.startswith("dependency_")
            ],
        }
        kernel_context = {}
        for context_name in ("seed", "baseline"):
            source = sources.get(f"{context_name}_kernel")
            kernel_context[f"{context_name}_code"] = (
                Path(source["snapshot_path"]).read_text(encoding="utf-8") if source else None
            )
            kernel_context[f"{context_name}_name"] = source["filename"] if source else None
    except (OSError, UnicodeError) as error:
        return finish("reference_failed", error, "prompt_sources")

    accepted_hashes: set[str] = set()
    history: list[dict[str, Any]] = []
    output_run_dir = Path(settings.output_dir).resolve() / task.task_id / run_dir.name
    record["status"] = "generating"
    checkpoint()
    for number in range(1, settings.max_attempts + 1):
        attempt: dict[str, Any] = {"attempt": number, "status": "generating", "started_at": _now()}
        record["attempts"].append(attempt)
        phase = "generation"
        candidate_text = ""
        try:
            _guard(guards)
            messages = build_generation_messages(
                **prompt_sources,
                **kernel_context,
                history=history,
                history_char_limit=settings.history_char_limit,
            )
            attempt["messages"] = messages
            record["counts"]["generation_calls"] += 1
            checkpoint(attempt)
            response = generator_client.generate(
                messages, temperature=settings.temperature, max_tokens=settings.max_tokens
            )
            if not isinstance(response, ModelResponse):
                raise TypeError("The generator must return ModelResponse.")
            attempt["generation"] = asdict(response)
            candidate_text = response.text
            checkpoint(attempt)
            _guard(guards)
            phase = "parse"
            if response.finish_reason in {"length", "max_tokens"}:
                raise CandidateParseError(
                    "The provider truncated the generation at its token limit."
                )
            if response.finish_reason in {"content_filter", "refusal"}:
                phase = "generation"
                raise RuntimeError("The provider filtered or refused the generation.")
            candidate_text = extract_candidate(response.text)
            content = candidate_text.encode("utf-8")
            sha256 = _digest(content)
            candidate_path = run_dir / "candidates" / f"attempt-{number:04d}-{sha256}.py"
            _immutable_bytes(candidate_path, content)
            guards.append((candidate_path, sha256))
            attempt.update(
                {
                    "candidate_path": str(candidate_path),
                    "candidate_sha256": sha256,
                    "status": "verifying",
                }
            )
            phase = "verification"
            record["counts"]["verification_calls"] += 1
            checkpoint(attempt)
            result = verifier.verify(prepared, candidate_path)
            attempt["verification"] = _result_record(result)
            checkpoint(attempt)
            _guard(guards)
            if result.failure_stage == "input_modified":
                raise InputModifiedError(result.message)
            if result.failure_stage not in {None, "compile", "runtime", "correctness", "timing"}:
                attempt["status"] = "infrastructure_error"
                attempt["completed_at"] = _now()
                checkpoint(attempt)
                return finish("infrastructure_error", RuntimeError(result.message), phase)
            if result.success:
                if sha256 in accepted_hashes:
                    attempt["status"] = "duplicate"
                else:
                    phase = "retention"
                    if not accepted_hashes:
                        if output_run_dir.parent.is_symlink():
                            raise ValueError("The task output directory became a symbolic link.")
                        output_run_dir.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                        # A run directory must be new even when its caller reuses an output root.
                        output_run_dir.mkdir(mode=0o700)
                    output_path = output_run_dir / f"{sha256}.py"
                    _immutable_bytes(output_path, content)
                    guards.append((output_path, sha256))
                    accepted = {
                        "attempt": number,
                        "candidate_sha256": sha256,
                        "candidate_path": str(candidate_path),
                        "output_path": str(output_path),
                        "verification": attempt["verification"],
                    }
                    record["accepted_candidates"].append(accepted)
                    accepted_hashes.add(sha256)
                    record["counts"]["accepted_candidates"] = len(accepted_hashes)
                    attempt["status"] = "accepted"
                    attempt["output_path"] = str(output_path)
            else:
                attempt["status"] = "verification_failed"
        except CandidateParseError as error:
            attempt["status"] = "parse_failed"
            attempt["error"] = _error(error, phase)
        except RecordWriteError:
            raise
        except Exception as caught:
            error = caught
            try:
                _guard(guards)
            except InputModifiedError as changed:
                error = changed
            if isinstance(error, InputModifiedError):
                status = "input_modified"
            elif phase == "generation":
                status = "provider_error"
            else:
                status = "infrastructure_error"
            attempt["status"] = status
            attempt["error"] = _error(error, phase)
            attempt["completed_at"] = _now()
            checkpoint(attempt)
            return finish(status, error, phase)

        attempt["completed_at"] = _now()
        checkpoint(attempt)
        if len(accepted_hashes) == settings.num_variants:
            return finish("success")
        if number == settings.max_attempts:
            break

        # Reflection is a distinct model call only when another generation can follow.
        feedback = attempt.get("verification", {"parse_error": attempt.get("error")})
        seek_variant = attempt["status"] in {"accepted", "duplicate"}
        reflection: dict[str, Any] = {
            "purpose": "variant" if seek_variant else "repair",
            "started_at": _now(),
        }
        attempt["reflection"] = reflection
        try:
            _guard(guards)
            reflection_messages = build_reflection_messages(
                **prompt_sources,
                **kernel_context,
                candidate_text=candidate_text,
                feedback=feedback,
                seek_variant=seek_variant,
            )
            reflection["messages"] = reflection_messages
            record["counts"]["reflection_calls"] += 1
            checkpoint(attempt)
            response = reflector_client.generate(
                reflection_messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            if not isinstance(response, ModelResponse):
                raise TypeError("The reflector must return ModelResponse.")
            reflection["response"] = asdict(response)
            reflection["completed_at"] = _now()
            checkpoint(attempt)
            _guard(guards)
            if not response.text.strip() or response.finish_reason in {"content_filter", "refusal"}:
                raise RuntimeError("The reflector returned no usable plan.")
        except RecordWriteError:
            raise
        except Exception as caught:
            error = caught
            try:
                _guard(guards)
            except InputModifiedError as changed:
                error = changed
            status = "input_modified" if isinstance(error, InputModifiedError) else "provider_error"
            reflection["error"] = _error(error, "reflection")
            reflection["completed_at"] = _now()
            checkpoint(attempt)
            return finish(status, error, "reflection")
        history.append(
            {
                "attempt": number,
                "candidate": candidate_text,
                "feedback": feedback,
                "reflection": response.text,
                "seek_variant": seek_variant,
            }
        )
        checkpoint(attempt)

    return finish("partial_success" if accepted_hashes else "exhausted")
