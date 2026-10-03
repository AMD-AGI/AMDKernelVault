# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import hashlib
import json
import tempfile
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import threading
from typing import Any, Callable

from tqdm import tqdm

from .config import AttemptRecord, ConversionRecord, PipelineConfig
from .discovery import iter_hip_files
from .hip_parser import GPUFunction, extract_gpu_function_body, extract_gpu_functions, replace_function_body, select_optimization_target
from .pairing import functional_path_for_hip, module_path_for_hip, output_path_for_hip
from .prompting import build_prompt, format_response
from .verifier import prepare_verification_context


def _path_text(path: Path) -> str:
    return path.as_posix()


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(payload, indent=2, ensure_ascii=False)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_fingerprint(source_path: Path, config: PipelineConfig) -> str:
    module_path = module_path_for_hip(source_path, config.baseline_hip_dir, config.module_dir)
    functional_path = functional_path_for_hip(source_path, config.baseline_hip_dir, config.functional_dir)
    source_files = {source_path, module_path, functional_path}
    # Candidate compilation can use sibling headers and the baseline include tree.
    source_files.update(path for path in source_path.parent.iterdir() if path.is_file() and path.suffix in {".h", ".hpp", ".hxx", ".cuh", ".inc"})
    include_dir = source_path.parent / "include"
    if include_dir.is_dir():
        source_files.update(path for path in include_dir.rglob("*") if path.is_file())
    setting_names = (
        "target_function_mode", "max_attempts", "rtol", "atol", "seed", "temperature", "max_tokens",
        "provider", "model_id", "perf_warmup", "perf_iterations", "history_code_char_limit",
        "history_feedback_char_limit", "system_instruction", "few_shot_examples", "offload_arch",
    )
    payload = {
        "verification_protocol": 1,
        "sources": {path.resolve().as_posix(): _file_sha256(path) for path in sorted(source_files)},
        "settings": {name: getattr(config, name) for name in setting_names},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _success_reuse_error(record: ConversionRecord | None, source_path: Path, config: PipelineConfig) -> str | None:
    if record is None or record.status != "success":
        return "The output has no verified success record."
    output_path = output_path_for_hip(source_path, config.baseline_hip_dir, config.output_dir)
    if Path(record.output_path).resolve() != output_path.resolve():
        return "The saved record refers to a different output path."
    if not record.input_fingerprint or not record.output_sha256:
        return "The saved record has no source and output fingerprints."
    try:
        if record.input_fingerprint != _input_fingerprint(source_path, config):
            return "The inputs or configuration changed since verification."
        if record.output_sha256 != _file_sha256(output_path):
            return "The output changed since verification."
    except OSError as exc:
        return f"The pipeline cannot verify the saved files: {exc}"
    return None


def _output_conflict_record(source_path: Path, config: PipelineConfig, reason: str) -> ConversionRecord:
    relative_path = source_path.relative_to(config.baseline_hip_dir)
    return ConversionRecord(
        baseline_hip_source_path=_path_text(source_path),
        module_source_path=_path_text(config.module_dir / relative_path.with_suffix(".py")),
        functional_source_path=_path_text(config.functional_dir / relative_path.with_suffix(".py")),
        relative_path=_path_text(relative_path),
        output_path=_path_text(output_path_for_hip(source_path, config.baseline_hip_dir, config.output_dir)),
        status="failed",
        attempts_used=0,
        target_function_mode=config.target_function_mode,
        final_error=f"{reason} Use --overwrite to regenerate the existing output.",
    )


def _format_exception(context: str, exc: Exception) -> str:
    return f"{context}: {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"


def _attempt_prompt_path(artifacts_dir: Path, relative_path: Path, attempt: int) -> Path:
    return artifacts_dir / "prompts" / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}.txt"


def _attempt_function_candidate_path(artifacts_dir: Path, relative_path: Path, attempt: int) -> Path:
    return artifacts_dir / "function_candidates" / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}.cpp"


def _attempt_candidate_path(artifacts_dir: Path, relative_path: Path, attempt: int) -> Path:
    return artifacts_dir / "candidates" / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}.hip"


def _attempt_build_dir(build_root: Path, relative_path: Path, attempt: int) -> Path:
    return build_root / relative_path.parent / f"{relative_path.stem}.attempt_{attempt}"


def _baseline_build_dir(build_root: Path, relative_path: Path) -> Path:
    return build_root / "__baseline__" / relative_path.parent / f"{relative_path.stem}.baseline"


def _success_case_record_path(success_cases_dir: Path, relative_path: str) -> Path:
    return success_cases_dir / Path(relative_path).with_suffix(".json")


def _persist_success_case_records(config: PipelineConfig, successes: list[dict[str, Any]]) -> None:
    if config.success_cases_dir is None:
        return

    config.success_cases_dir.mkdir(parents=True, exist_ok=True)
    expected_files: set[Path] = set()
    for record in successes:
        relative_path = record.get("relative_path")
        if not relative_path:
            continue
        record_path = _success_case_record_path(config.success_cases_dir, relative_path)
        expected_files.add(record_path)
        _write_json(record_path, record)
    for existing_file in config.success_cases_dir.rglob("*.json"):
        if existing_file not in expected_files:
            existing_file.unlink()


def _persist_records(config: PipelineConfig, records: list[ConversionRecord]) -> None:
    serializable = [asdict(record) for record in records]
    successes = [record for record in serializable if record["status"] == "success"]
    failures = [record for record in serializable if record["status"] == "failed"]
    _write_json(config.records_file, serializable)
    _write_json(config.success_file, successes)
    _write_json(config.failure_file, failures)
    _persist_success_case_records(config, successes)


def _candidate_score(record: AttemptRecord) -> float:
    if record.speedup_vs_baseline is not None:
        return record.speedup_vs_baseline
    if record.speedup_vs_module is not None:
        return record.speedup_vs_module
    return 0.0


def _selection_mode_description(mode: str) -> str:
    normalized = mode.strip().lower()
    if normalized == "device":
        return "`__device__` / `__host__ __device__`"
    if normalized == "global":
        return "`__global__`"
    return "`__global__` or `__device__`"


def _failed_record_from_exception(
    source_path: Path,
    config: PipelineConfig,
    exc: Exception,
) -> ConversionRecord:
    try:
        relative_path = source_path.relative_to(config.baseline_hip_dir)
    except Exception:
        relative_path = Path(source_path.name)

    return ConversionRecord(
        baseline_hip_source_path=_path_text(source_path),
        module_source_path=_path_text(config.module_dir / relative_path.with_suffix(".py")),
        functional_source_path=_path_text(config.functional_dir / relative_path.with_suffix(".py")),
        relative_path=_path_text(relative_path),
        output_path=_path_text(config.output_dir / relative_path.with_suffix(".hip")),
        status="failed",
        attempts_used=0,
        target_function_mode=config.target_function_mode,
        final_error=_format_exception("Unhandled file processing error", exc),
    )


def _make_client_accessor(client_or_factory: Any) -> Callable[[], Any]:
    if hasattr(client_or_factory, "generate"):
        return lambda: client_or_factory

    if not callable(client_or_factory):
        raise TypeError(
            "client must either expose a `generate(...)` method or be a zero-argument factory "
            "that returns such an object."
        )

    thread_local = threading.local()

    def _get_client() -> Any:
        client = getattr(thread_local, "client", None)
        if client is None:
            client = client_or_factory()
            if not hasattr(client, "generate"):
                raise TypeError(
                    "client factory returned an object that does not expose `generate(...)`."
                )
            thread_local.client = client
        return client

    return _get_client


def _persist_completed_records(config: PipelineConfig, records: list[ConversionRecord | None]) -> None:
    completed_records = [record for record in records if record is not None]
    _persist_records(config, completed_records)


def _attempt_record_from_dict(payload: dict[str, Any]) -> AttemptRecord:
    return AttemptRecord(**payload)


def _conversion_record_from_dict(payload: dict[str, Any]) -> ConversionRecord:
    payload = dict(payload)
    payload["attempts"] = [
        _attempt_record_from_dict(attempt_payload) for attempt_payload in payload.get("attempts", [])
    ]
    return ConversionRecord(**payload)


def _load_existing_records(config: PipelineConfig) -> dict[str, ConversionRecord]:
    if config.records_file is None or not config.records_file.exists():
        return {}

    try:
        payload = json.loads(config.records_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: failed to load existing records from {config.records_file}: {exc}")
        return {}

    if not isinstance(payload, list):
        print(f"Warning: expected a list of records in {config.records_file}, ignoring existing file.")
        return {}

    records_by_relative_path: dict[str, ConversionRecord] = {}
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            record = _conversion_record_from_dict(item)
        except TypeError as exc:
            print(f"Warning: failed to parse a saved record from {config.records_file}: {exc}")
            continue
        if record.relative_path:
            records_by_relative_path[record.relative_path] = record
    return records_by_relative_path


def _run_single_file_task(
    index: int,
    source_path: Path,
    client_accessor: Callable[[], Any],
    config: PipelineConfig,
) -> tuple[int, ConversionRecord]:
    try:
        record = convert_single_file(source_path, client_accessor(), config)
    except Exception as exc:
        record = _failed_record_from_exception(source_path, config, exc)
    return index, record


def _select_response_function(candidate_code: str, target_function: GPUFunction) -> GPUFunction:
    functions = extract_gpu_functions(candidate_code)
    if not functions:
        raise ValueError("Model response did not contain a `__global__` or `__device__` function.")

    exact_matches = [function for function in functions if function.name == target_function.name]
    if exact_matches:
        return exact_matches[0]

    simple_target_name = target_function.name.split("::")[-1]
    simple_matches = [
        function for function in functions if function.name.split("::")[-1] == simple_target_name
    ]
    if simple_matches:
        return simple_matches[0]

    if len(functions) == 1:
        return functions[0]

    raise ValueError(
        f"Model response did not preserve the target function name `{target_function.name}`."
    )


def convert_single_file(source_path: Path, client, config: PipelineConfig) -> ConversionRecord:
    relative_path = source_path.relative_to(config.baseline_hip_dir)
    output_path = output_path_for_hip(source_path, config.baseline_hip_dir, config.output_dir)

    try:
        module_path = module_path_for_hip(source_path, config.baseline_hip_dir, config.module_dir)
        functional_path = functional_path_for_hip(source_path, config.baseline_hip_dir, config.functional_dir)
    except FileNotFoundError as exc:
        return ConversionRecord(
            baseline_hip_source_path=_path_text(source_path),
            module_source_path=_path_text(config.module_dir / relative_path.with_suffix(".py")),
            functional_source_path=_path_text(config.functional_dir / relative_path.with_suffix(".py")),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="failed",
            attempts_used=0,
            target_function_mode=config.target_function_mode,
            final_error=str(exc),
        )

    if output_path.exists() and not config.overwrite:
        return ConversionRecord(
            baseline_hip_source_path=_path_text(source_path),
            module_source_path=_path_text(module_path),
            functional_source_path=_path_text(functional_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="skipped",
            attempts_used=0,
            target_function_mode=config.target_function_mode,
            skip_reason="Output already exists and overwrite is disabled.",
        )

    hip_code = source_path.read_text(encoding="utf-8")
    module_code = module_path.read_text(encoding="utf-8")
    functional_code = functional_path.read_text(encoding="utf-8")
    input_fingerprint = _input_fingerprint(source_path, config)

    try:
        gpu_functions = extract_gpu_functions(hip_code)
        target_function = select_optimization_target(gpu_functions, mode=config.target_function_mode)
    except Exception as exc:
        return ConversionRecord(
            baseline_hip_source_path=_path_text(source_path),
            module_source_path=_path_text(module_path),
            functional_source_path=_path_text(functional_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="failed",
            attempts_used=0,
            target_function_mode=config.target_function_mode,
            final_error=_format_exception("Failed to parse baseline HIP functions", exc),
        )

    if target_function is None:
        return ConversionRecord(
            baseline_hip_source_path=_path_text(source_path),
            module_source_path=_path_text(module_path),
            functional_source_path=_path_text(functional_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="skipped",
            attempts_used=0,
            target_function_mode=config.target_function_mode,
            skip_reason=(
                "No "
                f"{_selection_mode_description(config.target_function_mode)} function "
                "was found in the baseline HIP file."
            ),
        )

    baseline_result, verification_context = prepare_verification_context(
        original_module_path=module_path,
        functional_module_path=functional_path,
        baseline_hip_path=source_path,
        baseline_build_dir=_baseline_build_dir(config.build_root, relative_path),
        seed=config.seed,
        rtol=config.rtol,
        atol=config.atol,
        perf_warmup=config.perf_warmup,
        perf_iterations=config.perf_iterations,
        keep_build_dir=config.keep_build_dirs,
        offload_arch=config.offload_arch,
        python_load_timeout_seconds=config.python_load_timeout_seconds,
        hip_compile_timeout_seconds=config.hip_compile_timeout_seconds,
        execution_timeout_seconds=config.execution_timeout_seconds,
        benchmark_timeout_seconds=config.benchmark_timeout_seconds,
    )
    if verification_context is None:
        return ConversionRecord(
            baseline_hip_source_path=_path_text(source_path),
            module_source_path=_path_text(module_path),
            functional_source_path=_path_text(functional_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="failed",
            attempts_used=0,
            target_function_mode=config.target_function_mode,
            target_function_name=target_function.name,
            target_function_kind=target_function.qualifier,
            target_function_length=target_function.body_char_length,
            baseline_gpu_function=target_function.full_text,
            baseline_compile_success=baseline_result.compile_success,
            baseline_correctness_success=baseline_result.correctness_success,
            module_latency_ms=baseline_result.module_latency_ms,
            baseline_latency_ms=baseline_result.baseline_latency_ms,
            final_error=baseline_result.message,
        )

    attempt_records: list[AttemptRecord] = []
    best_attempt_record: AttemptRecord | None = None

    for attempt in range(1, config.max_attempts + 1):
        function_candidate_code: str | None = None
        optimized_gpu_function: str | None = None
        prompt = build_prompt(
            module_code,
            functional_code,
            hip_code,
            relative_path,
            target_function,
            attempt_records,
            system_instruction=config.system_instruction or "",
            few_shot_examples=config.few_shot_examples or "",
            code_char_limit=config.history_code_char_limit,
            feedback_char_limit=config.history_feedback_char_limit,
            baseline_message=baseline_result.message,
            module_latency_ms=baseline_result.module_latency_ms,
            baseline_latency_ms=baseline_result.baseline_latency_ms,
        )
        prompt_path = _attempt_prompt_path(config.artifacts_dir, relative_path, attempt)
        function_candidate_path = _attempt_function_candidate_path(config.artifacts_dir, relative_path, attempt)
        candidate_path = _attempt_candidate_path(config.artifacts_dir, relative_path, attempt)
        build_dir = _attempt_build_dir(config.build_root, relative_path, attempt)
        _write_text(prompt_path, prompt)

        try:
            response = client.generate(
                [{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            function_candidate_code = format_response(response)
            if not function_candidate_code.strip():
                raise ValueError("Model returned an empty candidate function.")

            _write_text(function_candidate_path, function_candidate_code)
            response_function = _select_response_function(function_candidate_code, target_function)
            optimized_gpu_function = response_function.full_text
            candidate_code = replace_function_body(
                hip_code,
                target_function,
                extract_gpu_function_body(response_function.full_text),
            )
            _write_text(candidate_path, candidate_code)

            verification = verification_context.verify_candidate(
                candidate_path,
                build_dir=build_dir,
                keep_build_dir=config.keep_build_dirs,
            )
            attempt_record = AttemptRecord(
                attempt=attempt,
                prompt_path=_path_text(prompt_path),
                function_candidate_path=_path_text(function_candidate_path),
                candidate_path=_path_text(candidate_path),
                status="success" if verification.success else "failed",
                optimized_gpu_function=optimized_gpu_function,
                feedback=verification.message,
                mismatch=verification.message if not verification.success else None,
                compile_success=verification.compile_success,
                correctness_success=verification.correctness_success,
                speedup_vs_baseline=verification.speedup_vs_baseline,
                speedup_vs_module=verification.speedup_vs_module,
                module_latency_ms=verification.module_latency_ms,
                baseline_latency_ms=verification.baseline_latency_ms,
                candidate_latency_ms=verification.candidate_latency_ms,
            )
            attempt_records.append(attempt_record)
            if verification.success and (
                best_attempt_record is None or _candidate_score(attempt_record) > _candidate_score(best_attempt_record)
            ):
                best_attempt_record = attempt_record
        except Exception as exc:
            error_trace = traceback.format_exc()
            feedback = f"{type(exc).__name__}: {exc}\n\n{error_trace}"
            attempt_records.append(
                AttemptRecord(
                    attempt=attempt,
                    prompt_path=_path_text(prompt_path),
                    function_candidate_path=_path_text(function_candidate_path),
                    candidate_path=_path_text(candidate_path),
                    status="failed",
                    optimized_gpu_function=optimized_gpu_function or function_candidate_code,
                    feedback=feedback,
                    error=feedback,
                    module_latency_ms=baseline_result.module_latency_ms,
                    baseline_latency_ms=baseline_result.baseline_latency_ms,
                )
            )

    if best_attempt_record is not None:
        best_candidate_path = Path(best_attempt_record.candidate_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(best_candidate_path.read_text(encoding="utf-8"), encoding="utf-8")
        return ConversionRecord(
            baseline_hip_source_path=_path_text(source_path),
            module_source_path=_path_text(module_path),
            functional_source_path=_path_text(functional_path),
            relative_path=_path_text(relative_path),
            output_path=_path_text(output_path),
            status="success",
            attempts_used=len(attempt_records),
            target_function_mode=config.target_function_mode,
            target_function_name=target_function.name,
            target_function_kind=target_function.qualifier,
            target_function_length=target_function.body_char_length,
            baseline_gpu_function=target_function.full_text,
            optimized_gpu_function=best_attempt_record.optimized_gpu_function,
            attempts=attempt_records,
            best_attempt=best_attempt_record.attempt,
            best_speedup_vs_baseline=best_attempt_record.speedup_vs_baseline,
            best_speedup_vs_module=best_attempt_record.speedup_vs_module,
            baseline_compile_success=baseline_result.compile_success,
            baseline_correctness_success=baseline_result.correctness_success,
            module_latency_ms=baseline_result.module_latency_ms,
            baseline_latency_ms=baseline_result.baseline_latency_ms,
            input_fingerprint=input_fingerprint,
            output_sha256=_file_sha256(output_path),
        )

    return ConversionRecord(
        baseline_hip_source_path=_path_text(source_path),
        module_source_path=_path_text(module_path),
        functional_source_path=_path_text(functional_path),
        relative_path=_path_text(relative_path),
        output_path=_path_text(output_path),
        status="failed",
        attempts_used=len(attempt_records),
        target_function_mode=config.target_function_mode,
        target_function_name=target_function.name,
        target_function_kind=target_function.qualifier,
        target_function_length=target_function.body_char_length,
        baseline_gpu_function=target_function.full_text,
        optimized_gpu_function=next(
            (record.optimized_gpu_function for record in reversed(attempt_records) if record.optimized_gpu_function),
            None,
        ),
        attempts=attempt_records,
        baseline_compile_success=baseline_result.compile_success,
        baseline_correctness_success=baseline_result.correctness_success,
        module_latency_ms=baseline_result.module_latency_ms,
        baseline_latency_ms=baseline_result.baseline_latency_ms,
        final_error=attempt_records[-1].feedback if attempt_records else "No attempts were executed.",
    )


def run_optimization_pipeline(client, config: PipelineConfig) -> dict[str, int]:
    config = config.with_defaults()
    config.artifacts_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    hip_files = iter_hip_files(config.baseline_hip_dir)
    client_accessor = _make_client_accessor(client)
    existing_records = _load_existing_records(config)
    records: list[ConversionRecord | None] = [None] * len(hip_files)
    pending_tasks: list[tuple[int, Path]] = []

    for index, source_path in enumerate(hip_files):
        relative_path = _path_text(source_path.relative_to(config.baseline_hip_dir))
        existing_record = existing_records.get(relative_path)

        output_path = output_path_for_hip(source_path, config.baseline_hip_dir, config.output_dir)
        if output_path.exists() and not config.overwrite:
            reuse_error = _success_reuse_error(existing_record, source_path, config)
            records[index] = (
                existing_record if reuse_error is None else _output_conflict_record(source_path, config, reuse_error)
            )
            continue

        # Missing outputs and all prior failures must run again, including in resume mode.
        pending_tasks.append((index, source_path))
        if existing_record is not None:
            preserve_success = (
                config.overwrite and existing_record.status == "success" and output_path.exists()
            )
            if existing_record.status == "failed" or preserve_success:
                # Keep the prior result until the pending replacement returns.
                records[index] = existing_record

    _persist_completed_records(config, records)

    if config.num_workers == 1:
        for index, source_path in tqdm(pending_tasks, desc="Optimizing HIP files"):
            _, record = _run_single_file_task(index, source_path, client_accessor, config)
            records[index] = record
            _persist_completed_records(config, records)
    else:
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            futures = {
                executor.submit(_run_single_file_task, index, source_path, client_accessor, config): index
                for index, source_path in pending_tasks
            }
            with tqdm(total=len(pending_tasks), desc="Optimizing HIP files") as progress:
                for future in as_completed(futures):
                    index, record = future.result()
                    records[index] = record
                    _persist_completed_records(config, records)
                    progress.update(1)

    completed_records = [record for record in records if record is not None]

    return {
        "total": len(completed_records),
        "success": sum(record.status == "success" for record in completed_records),
        "failed": sum(record.status == "failed" for record in completed_records),
        "skipped": sum(record.status == "skipped" for record in completed_records),
    }
