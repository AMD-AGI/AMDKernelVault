# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from tqdm import tqdm

from .config import AttemptRecord, ConversionRecord, PipelineConfig
from .discovery import iter_module_files
from .prompting import build_prompt, format_response
from .verifier import verify_candidate


_UNSET_RECORD = object()


def _path_text(path: Path) -> str:
    return path.as_posix()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary_path.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_records(config: PipelineConfig) -> dict[str, ConversionRecord]:
    if not config.records_file.exists():
        return {}
    try:
        payload = json.loads(config.records_file.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("The records file must contain a list.")
        records = {}
        for item in payload:
            record = ConversionRecord(
                **{key: value for key, value in item.items() if key != "attempts"},
                attempts=[AttemptRecord(**attempt) for attempt in item.get("attempts", [])],
            )
            relative_path = Path(record.relative_path)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError("A record contains an invalid relative path.")
            if record.relative_path in records:
                raise ValueError("The records file contains duplicate relative paths.")
            records[record.relative_path] = record
        return records
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError(f"Cannot resume from {config.records_file}: {exc}") from exc


def _resume_problem(
    record: ConversionRecord | None,
    source_path: Path,
    output_path: Path,
    config: PipelineConfig,
) -> str | None:
    if record is None or record.status != "success":
        return "The output lacks a verified success record. Use --overwrite to generate and verify it."
    if Path(record.source_path).resolve() != source_path.resolve():
        return "The source path changed. Use --overwrite to generate and verify the output."
    if Path(record.output_path).resolve() != output_path.resolve():
        return "The output path changed. Use --overwrite to generate and verify the output."
    if not source_path.is_file() or not output_path.is_file():
        return "The source or output file is missing. The previous success is not current."
    if record.source_sha256 is None or record.output_sha256 is None:
        return "The previous success lacks hashes. Use --overwrite to generate and verify the output."
    if record.source_sha256 != _sha256(source_path):
        return "The source bytes changed. Use --overwrite to generate and verify the output."
    if record.output_sha256 != _sha256(output_path):
        return "The output bytes changed. Use --overwrite to generate and verify the output."
    if (
        record.verification_seed != config.seed
        or record.verification_rtol != config.rtol
        or record.verification_atol != config.atol
    ):
        return "The verification settings changed or are missing. Use --overwrite to generate and verify the output."
    return None


def _validate_saved_record(record: ConversionRecord, config: PipelineConfig) -> ConversionRecord:
    if record.status == "success":
        problem = _resume_problem(
            record, config.input_dir / record.relative_path, config.output_dir / record.relative_path, config
        )
        if problem is not None:
            return replace(record, status="skipped", skip_reason=problem)
    return record


def _attempt_prompt_path(artifacts_dir: Path, relative_path: Path, attempt: int) -> Path:
    return artifacts_dir / "prompts" / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}.txt"


def _attempt_candidate_path(artifacts_dir: Path, relative_path: Path, attempt: int) -> Path:
    return artifacts_dir / "candidates" / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}.py"


def _persist_records(config: PipelineConfig, records: list[ConversionRecord]) -> None:
    serializable = [asdict(record) for record in records]
    successes = [record for record in serializable if record["status"] == "success"]
    failures = [record for record in serializable if record["status"] == "failed"]
    _write_json(config.records_file, serializable)
    _write_json(config.success_file, successes)
    _write_json(config.failure_file, failures)


def convert_single_file(
    source_path: Path,
    client,
    config: PipelineConfig,
    *,
    previous_record=_UNSET_RECORD,
) -> ConversionRecord:
    config = config.with_defaults()
    relative_path = source_path.relative_to(config.input_dir)
    output_path = config.output_dir / relative_path
    if previous_record is _UNSET_RECORD:
        previous_record = _load_records(config).get(_path_text(relative_path))

    if output_path.exists() and not config.overwrite:
        problem = _resume_problem(previous_record, source_path, output_path, config)
        if problem is None:
            return previous_record
        if previous_record is not None:
            return replace(
                previous_record,
                status="skipped",
                skip_reason=previous_record.skip_reason or problem,
            )
        return ConversionRecord(
            source_path=_path_text(source_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="skipped",
            attempts_used=0,
            skip_reason=problem,
        )

    source_sha256 = _sha256(source_path)
    module_code = source_path.read_text(encoding="utf-8")
    attempt_records = list(previous_record.attempts) if previous_record is not None else []
    attempts_used = previous_record.attempts_used if previous_record is not None else 0
    current_attempts: list[AttemptRecord] = []
    attempt = max((record.attempt for record in attempt_records), default=0)

    for _ in range(config.max_attempts):
        attempt += 1
        while (
            _attempt_prompt_path(config.artifacts_dir, relative_path, attempt).exists()
            or _attempt_candidate_path(config.artifacts_dir, relative_path, attempt).exists()
        ):
            attempt += 1
        attempts_used += 1
        prompt = build_prompt(
            module_code,
            relative_path,
            current_attempts,
            code_char_limit=config.history_code_char_limit,
            feedback_char_limit=config.history_feedback_char_limit,
        )
        prompt_path = _attempt_prompt_path(config.artifacts_dir, relative_path, attempt)
        candidate_path = _attempt_candidate_path(config.artifacts_dir, relative_path, attempt)
        _write_text(prompt_path, prompt)
        candidate_sha256 = None
        provenance = {
            "source_sha256": source_sha256,
            "verification_seed": config.seed,
            "verification_rtol": config.rtol,
            "verification_atol": config.atol,
        }

        try:
            response = client.generate(
                [{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            candidate_code = format_response(response)
            if not candidate_code.strip():
                raise ValueError("Model returned an empty candidate.")

            _write_text(candidate_path, candidate_code)
            candidate_sha256 = _sha256(candidate_path)
            verification = verify_candidate(
                source_path,
                candidate_path,
                seed=config.seed,
                rtol=config.rtol,
                atol=config.atol,
            )
            if verification.success:
                if _sha256(source_path) != source_sha256 or _sha256(candidate_path) != candidate_sha256:
                    raise ValueError("The source or candidate bytes changed during verification.")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(candidate_code, encoding="utf-8")
                attempt_records.append(
                    AttemptRecord(
                        attempt=attempt,
                        prompt_path=_path_text(prompt_path),
                        candidate_path=_path_text(candidate_path),
                        status="success",
                        candidate_sha256=candidate_sha256,
                        **provenance,
                    )
                )
                return ConversionRecord(
                    source_path=_path_text(source_path),
                    relative_path=_path_text(relative_path),
                    output_path=_path_text(output_path),
                    status="success",
                    attempts_used=attempts_used,
                    attempts=attempt_records,
                    output_sha256=_sha256(output_path),
                    **provenance,
                )

            attempt_records.append(
                AttemptRecord(
                    attempt=attempt,
                    prompt_path=_path_text(prompt_path),
                    candidate_path=_path_text(candidate_path),
                    status="failed",
                    feedback=verification.message,
                    mismatch=verification.message,
                    candidate_sha256=candidate_sha256,
                    **provenance,
                )
            )
        except Exception as exc:
            error_trace = traceback.format_exc()
            feedback = f"{type(exc).__name__}: {exc}\n\n{error_trace}"
            attempt_records.append(
                AttemptRecord(
                    attempt=attempt,
                    prompt_path=_path_text(prompt_path),
                    candidate_path=_path_text(candidate_path),
                    status="failed",
                    feedback=feedback,
                    error=feedback,
                    candidate_sha256=candidate_sha256,
                    **provenance,
                )
            )
        current_attempts.append(attempt_records[-1])

    return ConversionRecord(
        source_path=_path_text(source_path),
        relative_path=_path_text(relative_path),
        output_path=_path_text(output_path),
        status="failed",
        attempts_used=attempts_used,
        attempts=attempt_records,
        final_error=attempt_records[-1].feedback if attempt_records else None,
        source_sha256=source_sha256,
    )


def run_conversion_pipeline(client, config: PipelineConfig) -> dict[str, int]:
    config = config.with_defaults()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    module_files = iter_module_files(config.input_dir)
    saved_records = {
        key: _validate_saved_record(record, config) for key, record in _load_records(config).items()
    }
    # Rebuild the indexes before generation can fail or stop.
    _persist_records(config, list(saved_records.values()))
    records: list[ConversionRecord] = []

    for source_path in tqdm(module_files, desc="Converting module files"):
        relative_path = _path_text(source_path.relative_to(config.input_dir))
        record = convert_single_file(source_path, client, config, previous_record=saved_records.get(relative_path))
        records.append(record)
        saved_records[relative_path] = record
        _persist_records(config, list(saved_records.values()))

    summary = {
        "total": len(records),
        "success": sum(record.status == "success" for record in records),
        "failed": sum(record.status == "failed" for record in records),
        "skipped": sum(record.status == "skipped" for record in records),
    }
    return summary
