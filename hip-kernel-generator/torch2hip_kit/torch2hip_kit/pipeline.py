# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import hashlib
import json
import traceback
from dataclasses import asdict, fields
from pathlib import Path

from tqdm import tqdm

from .config import AttemptRecord, ConversionRecord, PipelineConfig
from .discovery import iter_module_files
from .pairing import functional_path_for_module, hip_output_path_for_module
from .prompting import build_prompt, format_response
from .verifier import verify_candidate


def _path_text(path: Path) -> str:
    return path.as_posix()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _attempt_prompt_path(artifacts_dir: Path, relative_path: Path, attempt: int) -> Path:
    return artifacts_dir / "prompts" / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}.txt"


def _attempt_candidate_path(artifacts_dir: Path, relative_path: Path, attempt: int) -> Path:
    return artifacts_dir / "candidates" / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}.hip"


def _attempt_build_dir(build_root: Path, relative_path: Path, attempt: int) -> Path:
    return build_root / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}"


def _persist_records(config: PipelineConfig, records: list[ConversionRecord]) -> None:
    serializable = [asdict(record) for record in records]
    successes = [record for record in serializable if record["status"] == "success"]
    failures = [record for record in serializable if record["status"] == "failed"]
    _write_json(config.records_file, serializable)
    _write_json(config.success_file, successes)
    _write_json(config.failure_file, failures)


def _load_previous_records(config: PipelineConfig) -> dict[str, ConversionRecord]:
    if config.records_file is None or not config.records_file.is_file():
        return {}
    try:
        payload = json.loads(config.records_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, list):
        return {}

    record_fields = {field.name for field in fields(ConversionRecord)}
    attempt_fields = {field.name for field in fields(AttemptRecord)}
    records = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            values = {key: value for key, value in item.items() if key in record_fields}
            values["attempts"] = [
                AttemptRecord(**{key: value for key, value in attempt.items() if key in attempt_fields})
                for attempt in values.get("attempts", [])
            ]
            record = ConversionRecord(**values)
            if isinstance(record.relative_path, str):
                records[record.relative_path] = record
        except (AttributeError, TypeError, ValueError):
            # Invalid or legacy records cannot establish that an output is current.
            continue
    return records


def _input_fingerprint(module_bytes: bytes, functional_bytes: bytes, config: PipelineConfig, client) -> str:
    model_id = getattr(client, "model_id", None)
    payload = {
        "version": 1,
        "module_sha256": hashlib.sha256(module_bytes).hexdigest(),
        "functional_sha256": hashlib.sha256(functional_bytes).hexdigest(),
        "validation": {
            "seed": config.seed,
            "rtol": config.rtol,
            "atol": config.atol,
            "perf_warmup": config.perf_warmup,
            "perf_iterations": config.perf_iterations,
        },
        "generation": {
            "client_class": f"{type(client).__module__}.{type(client).__qualname__}",
            "model_id": model_id if isinstance(model_id, str) else None,
            "max_attempts": config.max_attempts,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "history_code_char_limit": config.history_code_char_limit,
            "history_feedback_char_limit": config.history_feedback_char_limit,
            "system_instruction": config.system_instruction or "",
            "few_shot_examples": config.few_shot_examples or "",
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _can_reuse_record(
    record: ConversionRecord | None,
    source_path: Path,
    functional_path: Path,
    output_path: Path,
    input_fingerprint: str,
) -> bool:
    if (
        record is None
        or record.status != "success"
        or record.input_fingerprint != input_fingerprint
        or not record.output_sha256
        or not output_path.is_file()
    ):
        return False
    if not any(
        attempt.attempt == record.best_attempt
        and attempt.status == "success"
        and attempt.compile_success
        and attempt.correctness_success
        for attempt in record.attempts
    ):
        return False
    try:
        paths_match = all(
            Path(recorded).resolve() == current.resolve()
            for recorded, current in (
                (record.module_source_path, source_path),
                (record.functional_source_path, functional_path),
                (record.output_path, output_path),
            )
        )
        return paths_match and record.output_sha256 == hashlib.sha256(output_path.read_bytes()).hexdigest()
    except (OSError, TypeError, ValueError):
        return False


def convert_single_file(
    source_path: Path,
    client,
    config: PipelineConfig,
    *,
    previous_record: ConversionRecord | None = None,
) -> ConversionRecord:
    relative_path = source_path.relative_to(config.module_dir)
    output_path = hip_output_path_for_module(source_path, config.module_dir, config.output_dir)

    try:
        functional_path = functional_path_for_module(source_path, config.module_dir, config.functional_dir)
    except FileNotFoundError as exc:
        return ConversionRecord(
            module_source_path=_path_text(source_path),
            functional_source_path=_path_text(config.functional_dir / relative_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="failed",
            attempts_used=0,
            final_error=str(exc),
        )

    module_bytes = source_path.read_bytes()
    functional_bytes = functional_path.read_bytes()
    input_fingerprint = _input_fingerprint(module_bytes, functional_bytes, config, client)

    if (output_path.exists() or output_path.is_symlink()) and not config.overwrite:
        if _can_reuse_record(previous_record, source_path, functional_path, output_path, input_fingerprint):
            return previous_record
        return ConversionRecord(
            module_source_path=_path_text(source_path),
            functional_source_path=_path_text(functional_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="failed",
            attempts_used=0,
            final_error=(
                f"Existing output has no matching verified success record: {_path_text(output_path)}. "
                "Use --overwrite to replace this output."
            ),
            input_fingerprint=input_fingerprint,
        )

    module_code = module_bytes.decode("utf-8")
    functional_code = functional_bytes.decode("utf-8")
    attempt_records: list[AttemptRecord] = []
    best_attempt_record: AttemptRecord | None = None

    for attempt in range(1, config.max_attempts + 1):
        prompt = build_prompt(
            module_code,
            functional_code,
            relative_path,
            attempt_records,
            system_instruction=config.system_instruction or "",
            few_shot_examples=config.few_shot_examples or "",
            code_char_limit=config.history_code_char_limit,
            feedback_char_limit=config.history_feedback_char_limit,
        )
        prompt_path = _attempt_prompt_path(config.artifacts_dir, relative_path, attempt)
        candidate_path = _attempt_candidate_path(config.artifacts_dir, relative_path, attempt)
        build_dir = _attempt_build_dir(config.build_root, relative_path, attempt)
        _write_text(prompt_path, prompt)

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
            verification = verify_candidate(
                source_path,
                functional_path,
                candidate_path,
                build_dir=build_dir,
                seed=config.seed,
                rtol=config.rtol,
                atol=config.atol,
                perf_warmup=config.perf_warmup,
                perf_iterations=config.perf_iterations,
                keep_build_dir=config.keep_build_dirs,
            )
            attempt_record = AttemptRecord(
                attempt=attempt,
                prompt_path=_path_text(prompt_path),
                candidate_path=_path_text(candidate_path),
                status="success" if verification.success else "failed",
                feedback=verification.message,
                mismatch=verification.message if not verification.success else None,
                compile_success=verification.compile_success,
                correctness_success=verification.correctness_success,
                speedup=verification.speedup,
                module_latency_ms=verification.module_latency_ms,
                hip_latency_ms=verification.hip_latency_ms,
            )
            attempt_records.append(attempt_record)
            if verification.success and (
                best_attempt_record is None
                or (verification.speedup or 0.0) > (best_attempt_record.speedup or 0.0)
            ):
                best_attempt_record = attempt_record
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
                )
            )

    if best_attempt_record is not None:
        best_candidate_path = Path(best_attempt_record.candidate_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_bytes = best_candidate_path.read_bytes()
        output_path.write_bytes(output_bytes)
        return ConversionRecord(
            module_source_path=_path_text(source_path),
            functional_source_path=_path_text(functional_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="success",
            attempts_used=len(attempt_records),
            attempts=attempt_records,
            best_attempt=best_attempt_record.attempt,
            best_speedup=best_attempt_record.speedup,
            input_fingerprint=input_fingerprint,
            output_sha256=hashlib.sha256(output_bytes).hexdigest(),
        )

    return ConversionRecord(
        module_source_path=_path_text(source_path),
        functional_source_path=_path_text(functional_path),
        relative_path=_path_text(relative_path),
        output_path=_path_text(output_path),
        status="failed",
        attempts_used=len(attempt_records),
        attempts=attempt_records,
        final_error=attempt_records[-1].feedback if attempt_records else "No attempts were executed.",
        input_fingerprint=input_fingerprint,
    )


def run_conversion_pipeline(client, config: PipelineConfig) -> dict[str, int]:
    config = config.with_defaults()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    module_files = iter_module_files(config.module_dir)
    previous_records = _load_previous_records(config)
    current_paths = {_path_text(source.relative_to(config.module_dir)) for source in module_files}
    previous_records = {
        relative_path: record for relative_path, record in previous_records.items() if relative_path in current_paths
    }
    records: list[ConversionRecord] = []

    for source_path in tqdm(module_files, desc="Converting module files to HIP"):
        relative_path = _path_text(source_path.relative_to(config.module_dir))
        record = convert_single_file(
            source_path, client, config, previous_record=previous_records.pop(relative_path, None)
        )
        records.append(record)
        # Retain pending records so an interrupted rerun does not lose their evidence.
        _persist_records(config, records + list(previous_records.values()))

    if not module_files:
        _persist_records(config, records)

    return {
        "total": len(records),
        "success": sum(record.status == "success" for record in records),
        "failed": sum(record.status == "failed" for record in records),
        "skipped": sum(record.status == "skipped" for record in records),
    }
