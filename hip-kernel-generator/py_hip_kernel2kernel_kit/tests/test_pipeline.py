# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

import py_hip_kernel2kernel_kit.pipeline as pipeline_module
from py_hip_kernel2kernel_kit.config import AttemptRecord, ConversionRecord, PipelineConfig
from py_hip_kernel2kernel_kit.verifier import BaselineVerificationResult, CandidateVerificationResult


MODULE_SOURCE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return []
"""


FUNCTIONAL_SOURCE = """
import torch
import torch.nn as nn


def module_fn(x):
    return x + 1


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, fn=module_fn):
        return fn(x)


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return []
"""


HIP_SOURCE = """
#include <torch/extension.h>
#include "hip/hip_runtime.h"

__device__ __forceinline__ float helper(float x) {
    return x + 1.0f;
}

__global__ void sample_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = helper(x[idx]);
    }
}

torch::Tensor forward(torch::Tensor x) {
    return x + 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "forward");
}
"""


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, messages, **kwargs):
        return self._responses.pop(0)


class FakeVerificationContext:
    def __init__(self, results):
        self._results = list(results)

    def verify_candidate(self, candidate_hip_path, *, build_dir, keep_build_dir=False):
        return self._results.pop(0)


def make_conversion_record(config: PipelineConfig, source_path: Path, *, status: str, attempts_used: int = 0) -> ConversionRecord:
    relative_path = source_path.relative_to(config.baseline_hip_dir)
    input_fingerprint = output_sha256 = None
    attempts = []
    if status == "success":
        for root, content in ((config.module_dir, MODULE_SOURCE), (config.functional_dir, FUNCTIONAL_SOURCE)):
            paired_path = root / relative_path.with_suffix(".py")
            paired_path.parent.mkdir(parents=True, exist_ok=True)
            if not paired_path.exists():
                paired_path.write_text(content, encoding="utf-8")
        output_path = config.output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        input_fingerprint = pipeline_module._input_fingerprint(source_path, config)
        output_sha256 = pipeline_module._file_sha256(output_path)
        attempts = [
            AttemptRecord(
                attempt=1,
                prompt_path=(config.artifacts_dir / "prompts" / relative_path.parent / f"{relative_path.stem}.attempt_1.txt").as_posix(),
                function_candidate_path=(
                    config.artifacts_dir / "function_candidates" / relative_path.parent / f"{relative_path.stem}.attempt_1.cpp"
                ).as_posix(),
                candidate_path=(
                    config.artifacts_dir / "candidates" / relative_path.parent / f"{relative_path.stem}.attempt_1.hip"
                ).as_posix(),
                status="success",
                optimized_gpu_function="__global__ void sample_kernel() {}",
                compile_success=True,
                correctness_success=True,
                speedup_vs_baseline=1.2,
            )
        ]

    return ConversionRecord(
        baseline_hip_source_path=source_path.as_posix(),
        module_source_path=(config.module_dir / relative_path.with_suffix(".py")).as_posix(),
        functional_source_path=(config.functional_dir / relative_path.with_suffix(".py")).as_posix(),
        relative_path=relative_path.as_posix(),
        output_path=(config.output_dir / relative_path.with_suffix(".hip")).as_posix(),
        status=status,
        attempts_used=attempts_used,
        target_function_mode=config.target_function_mode,
        target_function_name="sample_kernel",
        target_function_kind="__global__",
        baseline_gpu_function="__global__ void sample_kernel() {}",
        optimized_gpu_function="__global__ void sample_kernel() {}" if status == "success" else None,
        attempts=attempts,
        best_attempt=1 if status == "success" else None,
        best_speedup_vs_baseline=1.2 if status == "success" else None,
        baseline_compile_success=True,
        baseline_correctness_success=status == "success",
        module_latency_ms=3.0,
        baseline_latency_ms=2.0,
        final_error="old failure" if status == "failed" else None,
        input_fingerprint=input_fingerprint,
        output_sha256=output_sha256,
    )


def test_convert_single_file_selects_fastest_success(monkeypatch, tmp_path: Path) -> None:
    hip_dir = tmp_path / "hip"
    module_dir = tmp_path / "module"
    functional_dir = tmp_path / "functional"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = hip_dir / "level_1" / "sample.hip"
    module_path = module_dir / "level_1" / "sample.py"
    functional_path = functional_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.parent.mkdir(parents=True, exist_ok=True)
    functional_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(HIP_SOURCE, encoding="utf-8")
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    functional_path.write_text(FUNCTIONAL_SOURCE, encoding="utf-8")

    verification_results = [
        CandidateVerificationResult(
            True,
            "attempt1",
            compile_success=True,
            correctness_success=True,
            speedup_vs_baseline=1.10,
            speedup_vs_module=1.50,
            module_latency_ms=3.0,
            baseline_latency_ms=2.0,
            candidate_latency_ms=1.82,
        ),
        CandidateVerificationResult(
            False,
            "attempt2 failed",
            compile_success=True,
            correctness_success=False,
            module_latency_ms=3.0,
            baseline_latency_ms=2.0,
        ),
        CandidateVerificationResult(
            True,
            "attempt3",
            compile_success=True,
            correctness_success=True,
            speedup_vs_baseline=1.75,
            speedup_vs_module=2.30,
            module_latency_ms=3.0,
            baseline_latency_ms=2.0,
            candidate_latency_ms=1.14,
        ),
    ]

    monkeypatch.setattr(
        pipeline_module,
        "prepare_verification_context",
        lambda **kwargs: (
            BaselineVerificationResult(
                True,
                "Baseline HIP correctness passed.",
                compile_success=True,
                correctness_success=True,
                module_latency_ms=3.0,
                baseline_latency_ms=2.0,
            ),
            FakeVerificationContext(verification_results),
        ),
    )

    config = PipelineConfig(
        baseline_hip_dir=hip_dir,
        module_dir=module_dir,
        functional_dir=functional_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        max_attempts=3,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    client = FakeClient(
        [
            "```cpp\n__global__ void sample_kernel(const float* x, float* out, int n) { out[0] = x[0]; }\n```",
            "```cpp\n__global__ void sample_kernel(const float* x, float* out, int n) { out[0] = x[0] + 1; }\n```",
            "```cpp\n__global__ void sample_kernel(const float* x, float* out, int n) { out[0] = x[0] + 2; }\n```",
        ]
    )
    record = pipeline_module.convert_single_file(source_path, client, config)

    assert record.status == "success"
    assert record.best_attempt == 3
    assert record.best_speedup_vs_baseline == 1.75
    assert record.attempts_used == 3
    assert "sample_kernel" in (record.baseline_gpu_function or "")
    assert "sample_kernel" in (record.optimized_gpu_function or "")
    assert (output_dir / "level_1" / "sample.hip").exists()

    third_prompt = (
        artifacts_dir / "prompts" / "level_1" / "sample.attempt_3.txt"
    ).read_text(encoding="utf-8")
    assert "Attempt 1:" in third_prompt
    assert "Current best validated speedup vs baseline HIP: 1.1000x." in third_prompt


def test_run_optimization_pipeline_writes_json_records(monkeypatch, tmp_path: Path) -> None:
    hip_dir = tmp_path / "hip"
    module_dir = tmp_path / "module"
    functional_dir = tmp_path / "functional"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = hip_dir / "level_1" / "sample.hip"
    module_path = module_dir / "level_1" / "sample.py"
    functional_path = functional_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.parent.mkdir(parents=True, exist_ok=True)
    functional_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(HIP_SOURCE, encoding="utf-8")
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    functional_path.write_text(FUNCTIONAL_SOURCE, encoding="utf-8")

    monkeypatch.setattr(
        pipeline_module,
        "prepare_verification_context",
        lambda **kwargs: (
            BaselineVerificationResult(
                True,
                "Baseline HIP correctness passed.",
                compile_success=True,
                correctness_success=True,
                module_latency_ms=3.0,
                baseline_latency_ms=2.0,
            ),
            FakeVerificationContext(
                [
                    CandidateVerificationResult(
                        True,
                        "ok",
                        compile_success=True,
                        correctness_success=True,
                        speedup_vs_baseline=1.5,
                        speedup_vs_module=2.0,
                        module_latency_ms=3.0,
                        baseline_latency_ms=2.0,
                        candidate_latency_ms=1.33,
                    )
                ]
            ),
        ),
    )

    config = PipelineConfig(
        baseline_hip_dir=hip_dir,
        module_dir=module_dir,
        functional_dir=functional_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    client = FakeClient(
        [
            "```cpp\n__global__ void sample_kernel(const float* x, float* out, int n) { out[0] = x[0]; }\n```"
        ]
    )
    summary = pipeline_module.run_optimization_pipeline(client, config)

    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    records = json.loads(config.records_file.read_text(encoding="utf-8"))
    successes = json.loads(config.success_file.read_text(encoding="utf-8"))
    failures = json.loads(config.failure_file.read_text(encoding="utf-8"))
    success_case_record = json.loads(
        (config.success_cases_dir / "level_1" / "sample.json").read_text(encoding="utf-8")
    )

    assert len(records) == 1
    assert len(successes) == 1
    assert failures == []
    assert success_case_record == successes[0]
    assert successes[0]["best_attempt"] == 1
    assert "baseline_hip_source_path" in successes[0]
    assert "module_source_path" in successes[0]
    assert "functional_source_path" in successes[0]
    assert "baseline_gpu_function" in successes[0]
    assert "optimized_gpu_function" in successes[0]
    assert "optimized_gpu_function" in successes[0]["attempts"][0]
    assert successes[0]["baseline_hip_source_path"] == source_path.as_posix()
    assert successes[0]["module_source_path"] == module_path.as_posix()
    assert successes[0]["functional_source_path"] == functional_path.as_posix()
    assert successes[0]["relative_path"] == Path("level_1/sample.hip").as_posix()
    assert successes[0]["output_path"] == (output_dir / "level_1" / "sample.hip").as_posix()
    assert successes[0]["attempts"][0]["prompt_path"] == (
        artifacts_dir / "prompts" / "level_1" / "sample.attempt_1.txt"
    ).as_posix()
    assert successes[0]["attempts"][0]["function_candidate_path"] == (
        artifacts_dir / "function_candidates" / "level_1" / "sample.attempt_1.cpp"
    ).as_posix()
    assert successes[0]["attempts"][0]["candidate_path"] == (
        artifacts_dir / "candidates" / "level_1" / "sample.attempt_1.hip"
    ).as_posix()
    assert len(successes[0]["input_fingerprint"]) == 64
    assert len(successes[0]["output_sha256"]) == 64
    # The client has no responses left. An unchanged rerun must preserve the verified record.
    assert pipeline_module.run_optimization_pipeline(client, config) == summary
    assert json.loads(config.success_file.read_text(encoding="utf-8")) == successes
    assert json.loads((config.success_cases_dir / "level_1" / "sample.json").read_text(encoding="utf-8")) == successes[0]


def test_run_optimization_pipeline_rewrites_success_case_record_directory(monkeypatch, tmp_path: Path) -> None:
    hip_dir = tmp_path / "hip"
    sample_paths = [hip_dir / "level_1" / name for name in ("sample_a.hip", "sample_b.hip")]
    for sample_path in sample_paths:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(HIP_SOURCE, encoding="utf-8")

    config = PipelineConfig(
        baseline_hip_dir=hip_dir,
        module_dir=tmp_path / "module",
        functional_dir=tmp_path / "functional",
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        resume=True,
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    stale_success = config.success_cases_dir / "old" / "stale.json"
    stale_success.parent.mkdir(parents=True, exist_ok=True)
    stale_success.write_text("{}", encoding="utf-8")

    saved_success = make_conversion_record(config, sample_paths[0], status="success", attempts_used=1)
    config.records_file.parent.mkdir(parents=True, exist_ok=True)
    config.records_file.write_text(json.dumps([asdict(saved_success)]), encoding="utf-8")

    monkeypatch.setattr(
        pipeline_module,
        "convert_single_file",
        lambda source_path, _client, config: make_conversion_record(config, source_path, status="success", attempts_used=1),
    )

    summary = pipeline_module.run_optimization_pipeline(FakeClient([]), config)

    shard_files = sorted(path.relative_to(config.success_cases_dir).as_posix() for path in config.success_cases_dir.rglob("*.json"))

    assert summary == {"total": 2, "success": 2, "failed": 0, "skipped": 0}
    assert shard_files == ["level_1/sample_a.json", "level_1/sample_b.json"]


def test_convert_single_file_supports_device_only_mode(monkeypatch, tmp_path: Path) -> None:
    hip_dir = tmp_path / "hip"
    module_dir = tmp_path / "module"
    functional_dir = tmp_path / "functional"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = hip_dir / "level_1" / "sample.hip"
    module_path = module_dir / "level_1" / "sample.py"
    functional_path = functional_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.parent.mkdir(parents=True, exist_ok=True)
    functional_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(HIP_SOURCE, encoding="utf-8")
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    functional_path.write_text(FUNCTIONAL_SOURCE, encoding="utf-8")

    monkeypatch.setattr(
        pipeline_module,
        "prepare_verification_context",
        lambda **kwargs: (
            BaselineVerificationResult(
                True,
                "Baseline HIP correctness passed.",
                compile_success=True,
                correctness_success=True,
                module_latency_ms=3.0,
                baseline_latency_ms=2.0,
            ),
            FakeVerificationContext(
                [
                    CandidateVerificationResult(
                        True,
                        "ok",
                        compile_success=True,
                        correctness_success=True,
                        speedup_vs_baseline=1.2,
                        speedup_vs_module=1.8,
                        module_latency_ms=3.0,
                        baseline_latency_ms=2.0,
                        candidate_latency_ms=1.67,
                    )
                ]
            ),
        ),
    )

    config = PipelineConfig(
        baseline_hip_dir=hip_dir,
        module_dir=module_dir,
        functional_dir=functional_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        target_function_mode="device",
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    client = FakeClient(
        [
            "```cpp\n__device__ __forceinline__ float helper(float x) { return x + 2.0f; }\n```"
        ]
    )
    record = pipeline_module.convert_single_file(source_path, client, config)

    assert record.status == "success"
    assert record.target_function_mode == "device"
    assert record.target_function_kind == "__device__"
    assert record.target_function_name == "helper"
    assert "__device__ __forceinline__ float helper" in (record.optimized_gpu_function or "")


def test_convert_single_file_captures_parser_failures(monkeypatch, tmp_path: Path) -> None:
    hip_dir = tmp_path / "hip"
    module_dir = tmp_path / "module"
    functional_dir = tmp_path / "functional"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = hip_dir / "level_1" / "sample.hip"
    module_path = module_dir / "level_1" / "sample.py"
    functional_path = functional_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.parent.mkdir(parents=True, exist_ok=True)
    functional_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(HIP_SOURCE, encoding="utf-8")
    module_path.write_text(MODULE_SOURCE, encoding="utf-8")
    functional_path.write_text(FUNCTIONAL_SOURCE, encoding="utf-8")

    monkeypatch.setattr(
        pipeline_module,
        "extract_gpu_functions",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("parse boom")),
    )

    config = PipelineConfig(
        baseline_hip_dir=hip_dir,
        module_dir=module_dir,
        functional_dir=functional_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    record = pipeline_module.convert_single_file(source_path, FakeClient([]), config)

    assert record.status == "failed"
    assert "Failed to parse baseline HIP functions" in (record.final_error or "")
    assert "parse boom" in (record.final_error or "")


def test_run_optimization_pipeline_continues_after_unhandled_file_error(monkeypatch, tmp_path: Path) -> None:
    hip_dir = tmp_path / "hip"
    source_path = hip_dir / "level_1" / "sample.hip"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(HIP_SOURCE, encoding="utf-8")

    monkeypatch.setattr(pipeline_module, "iter_hip_files", lambda *_args, **_kwargs: [source_path])
    monkeypatch.setattr(
        pipeline_module,
        "convert_single_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unexpected boom")),
    )

    config = PipelineConfig(
        baseline_hip_dir=hip_dir,
        module_dir=tmp_path / "module",
        functional_dir=tmp_path / "functional",
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    summary = pipeline_module.run_optimization_pipeline(FakeClient([]), config)
    records = json.loads(config.records_file.read_text(encoding="utf-8"))

    assert summary == {"total": 1, "success": 0, "failed": 1, "skipped": 0}
    assert len(records) == 1
    assert records[0]["status"] == "failed"
    assert "Unhandled file processing error" in (records[0]["final_error"] or "")


def test_run_optimization_pipeline_supports_multithreaded_workers(monkeypatch, tmp_path: Path) -> None:
    hip_dir = tmp_path / "hip"
    sample_paths = [hip_dir / "level_1" / f"sample_{index}.hip" for index in range(4)]
    for sample_path in sample_paths:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(HIP_SOURCE, encoding="utf-8")

    seen_thread_ids: set[int] = set()
    seen_client_ids: set[int] = set()
    seen_client_names: set[str] = set()
    created_client_names: list[str] = []
    lock = threading.Lock()

    class ThreadLocalClient:
        def __init__(self, name: str):
            self.name = name

        def generate(self, messages, **kwargs):
            return self.name

    def client_factory():
        with lock:
            name = f"client_{len(created_client_names)}"
            created_client_names.append(name)
        return ThreadLocalClient(name)

    def fake_convert_single_file(source_path: Path, client, config: PipelineConfig) -> ConversionRecord:
        time.sleep(0.05)
        with lock:
            seen_thread_ids.add(threading.get_ident())
            seen_client_ids.add(id(client))
            seen_client_names.add(client.generate([]))
        relative_path = source_path.relative_to(config.baseline_hip_dir)
        return ConversionRecord(
            baseline_hip_source_path=source_path.as_posix(),
            module_source_path=(config.module_dir / relative_path.with_suffix(".py")).as_posix(),
            functional_source_path=(config.functional_dir / relative_path.with_suffix(".py")).as_posix(),
            relative_path=relative_path.as_posix(),
            output_path=(config.output_dir / relative_path.with_suffix(".hip")).as_posix(),
            status="success",
            attempts_used=0,
            target_function_mode=config.target_function_mode,
        )

    monkeypatch.setattr(pipeline_module, "iter_hip_files", lambda *_args, **_kwargs: sample_paths)
    monkeypatch.setattr(pipeline_module, "convert_single_file", fake_convert_single_file)

    config = PipelineConfig(
        baseline_hip_dir=hip_dir,
        module_dir=tmp_path / "module",
        functional_dir=tmp_path / "functional",
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        num_workers=2,
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    summary = pipeline_module.run_optimization_pipeline(client_factory, config)
    records = json.loads(config.records_file.read_text(encoding="utf-8"))

    assert summary == {"total": 4, "success": 4, "failed": 0, "skipped": 0}
    assert [record["relative_path"] for record in records] == [
        path.relative_to(hip_dir).as_posix() for path in sample_paths
    ]
    assert len(seen_thread_ids) >= 2
    assert len(seen_client_ids) >= 2
    assert len(seen_client_names) >= 2


def test_run_optimization_pipeline_resume_reuses_success_records_without_rerun(monkeypatch, tmp_path: Path) -> None:
    hip_dir = tmp_path / "hip"
    source_path = hip_dir / "level_1" / "sample.hip"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(HIP_SOURCE, encoding="utf-8")

    config = PipelineConfig(
        baseline_hip_dir=hip_dir,
        module_dir=tmp_path / "module",
        functional_dir=tmp_path / "functional",
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        resume=True,
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    saved_record = make_conversion_record(config, source_path, status="success", attempts_used=1)
    config.records_file.parent.mkdir(parents=True, exist_ok=True)
    config.records_file.write_text(json.dumps([asdict(saved_record)]), encoding="utf-8")

    monkeypatch.setattr(
        pipeline_module,
        "convert_single_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume should not rerun saved success")),
    )

    summary = pipeline_module.run_optimization_pipeline(FakeClient([]), config)
    records = json.loads(config.records_file.read_text(encoding="utf-8"))

    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    assert len(records) == 1
    assert records[0]["status"] == "success"
    assert records[0]["relative_path"] == "level_1/sample.hip"


def test_run_optimization_pipeline_resume_preserves_records_and_reruns_failures(monkeypatch, tmp_path: Path) -> None:
    hip_dir = tmp_path / "hip"
    sample_paths = [hip_dir / "level_1" / name for name in ("sample_a.hip", "sample_b.hip")]
    for sample_path in sample_paths:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(HIP_SOURCE, encoding="utf-8")

    config = PipelineConfig(
        baseline_hip_dir=hip_dir,
        module_dir=tmp_path / "module",
        functional_dir=tmp_path / "functional",
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        resume=True,
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    saved_success = make_conversion_record(config, sample_paths[0], status="success", attempts_used=1)
    saved_failure = make_conversion_record(config, sample_paths[1], status="failed")
    config.records_file.parent.mkdir(parents=True, exist_ok=True)
    config.records_file.write_text(
        json.dumps([asdict(saved_success), asdict(saved_failure)]),
        encoding="utf-8",
    )

    rerun_paths: list[str] = []

    def fake_convert_single_file(source_path: Path, _client, config: PipelineConfig) -> ConversionRecord:
        persisted_records = json.loads(config.records_file.read_text(encoding="utf-8"))
        assert [record["relative_path"] for record in persisted_records] == [
            "level_1/sample_a.hip",
            "level_1/sample_b.hip",
        ]
        rerun_paths.append(source_path.relative_to(config.baseline_hip_dir).as_posix())
        return make_conversion_record(config, source_path, status="success", attempts_used=1)

    monkeypatch.setattr(pipeline_module, "convert_single_file", fake_convert_single_file)

    summary = pipeline_module.run_optimization_pipeline(FakeClient([]), config)
    records = json.loads(config.records_file.read_text(encoding="utf-8"))
    successes = json.loads(config.success_file.read_text(encoding="utf-8"))
    failures = json.loads(config.failure_file.read_text(encoding="utf-8"))

    assert summary == {"total": 2, "success": 2, "failed": 0, "skipped": 0}
    assert rerun_paths == ["level_1/sample_b.hip"]
    assert [record["relative_path"] for record in records] == [
        "level_1/sample_a.hip",
        "level_1/sample_b.hip",
    ]
    assert len(successes) == 2
    assert failures == []


def test_run_optimization_pipeline_resume_with_overwrite_reruns_success(monkeypatch, tmp_path: Path) -> None:
    hip_dir = tmp_path / "hip"
    source_path = hip_dir / "level_1" / "sample.hip"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(HIP_SOURCE, encoding="utf-8")

    config = PipelineConfig(
        baseline_hip_dir=hip_dir,
        module_dir=tmp_path / "module",
        functional_dir=tmp_path / "functional",
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        resume=True,
        overwrite=True,
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    saved_record = make_conversion_record(config, source_path, status="success", attempts_used=1)
    config.records_file.parent.mkdir(parents=True, exist_ok=True)
    config.records_file.write_text(json.dumps([asdict(saved_record)]), encoding="utf-8")

    rerun_paths: list[str] = []

    def fake_convert_single_file(source_path: Path, _client, config: PipelineConfig) -> ConversionRecord:
        rerun_paths.append(source_path.relative_to(config.baseline_hip_dir).as_posix())
        return make_conversion_record(config, source_path, status="success", attempts_used=2)

    monkeypatch.setattr(pipeline_module, "convert_single_file", fake_convert_single_file)

    summary = pipeline_module.run_optimization_pipeline(FakeClient([]), config)
    records = json.loads(config.records_file.read_text(encoding="utf-8"))

    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    assert rerun_paths == ["level_1/sample.hip"]
    assert records[0]["attempts_used"] == 2


def _saved_success_fixture(tmp_path: Path, *, resume: bool = False):
    config = PipelineConfig(
        baseline_hip_dir=tmp_path / "hip",
        module_dir=tmp_path / "module",
        functional_dir=tmp_path / "functional",
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        resume=resume,
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()
    source_path = config.baseline_hip_dir / "sample.hip"
    source_path.parent.mkdir()
    source_path.write_text(HIP_SOURCE, encoding="utf-8")
    saved_record = make_conversion_record(config, source_path, status="success", attempts_used=1)
    config.records_file.parent.mkdir(parents=True, exist_ok=True)
    config.records_file.write_text(json.dumps([asdict(saved_record)]), encoding="utf-8")
    return config, source_path, saved_record


def test_default_rerun_preserves_verified_success_records(tmp_path: Path) -> None:
    config, _, saved_record = _saved_success_fixture(tmp_path)
    summary = pipeline_module.run_optimization_pipeline(FakeClient([]), config)
    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    assert json.loads(config.success_file.read_text()) == [asdict(saved_record)]
    shard_path = config.success_cases_dir / "sample.json"
    first_stat = shard_path.stat().st_mtime_ns
    pipeline_module.run_optimization_pipeline(FakeClient([]), config)
    assert json.loads(shard_path.read_text()) == asdict(saved_record)
    assert shard_path.stat().st_mtime_ns == first_stat


@pytest.mark.parametrize("num_workers", [1, 2])
def test_interrupted_overwrite_preserves_prior_success(monkeypatch, tmp_path: Path, num_workers: int) -> None:
    config, source_path, saved_record = _saved_success_fixture(tmp_path, resume=True)
    pipeline_module._persist_records(config, [saved_record])
    config.overwrite = True
    config.num_workers = num_workers
    output_path = Path(saved_record.output_path)
    output_before = output_path.read_bytes()
    called_paths = []

    def interrupt_before_result(path, client, config):
        called_paths.append(path)
        raise KeyboardInterrupt("Interrupted before the replacement returned.")

    monkeypatch.setattr(pipeline_module, "convert_single_file", interrupt_before_result)
    with pytest.raises(KeyboardInterrupt):
        pipeline_module.run_optimization_pipeline(FakeClient([]), config)

    assert called_paths == [source_path]
    assert json.loads(config.records_file.read_text()) == [asdict(saved_record)]
    assert json.loads(config.success_file.read_text()) == [asdict(saved_record)]
    assert json.loads((config.success_cases_dir / "sample.json").read_text()) == asdict(saved_record)
    assert output_path.read_bytes() == output_before


def test_resume_regenerates_missing_output(monkeypatch, tmp_path: Path) -> None:
    config, source_path, saved_record = _saved_success_fixture(tmp_path, resume=True)
    Path(saved_record.output_path).unlink()
    rerun_paths = []

    def regenerate(path, client, config):
        rerun_paths.append(path)
        return make_conversion_record(config, path, status="success", attempts_used=1)

    monkeypatch.setattr(pipeline_module, "convert_single_file", regenerate)
    summary = pipeline_module.run_optimization_pipeline(FakeClient([]), config)
    assert summary["success"] == 1
    assert rerun_paths == [source_path]
    assert Path(saved_record.output_path).exists()


@pytest.mark.parametrize("changed", ["baseline", "module", "functional", "output", "tolerance", "model", "legacy"])
def test_stale_success_requires_overwrite(changed: str, tmp_path: Path) -> None:
    config, source_path, saved_record = _saved_success_fixture(tmp_path, resume=True)
    if changed in {"baseline", "module", "functional", "output"}:
        path = {
            "baseline": source_path,
            "module": config.module_dir / "sample.py",
            "functional": config.functional_dir / "sample.py",
            "output": Path(saved_record.output_path),
        }[changed]
        path.write_text(path.read_text() + "\n// changed\n")
    elif changed == "tolerance":
        config.rtol = 1e-5
    elif changed == "model":
        config.model_id = "different-model"
    else:
        payload = asdict(saved_record)
        payload.pop("input_fingerprint")
        payload.pop("output_sha256")
        config.records_file.write_text(json.dumps([payload]))
    output_before = Path(saved_record.output_path).read_bytes()
    summary = pipeline_module.run_optimization_pipeline(FakeClient([]), config)
    assert summary == {"total": 1, "success": 0, "failed": 1, "skipped": 0}
    assert "--overwrite" in json.loads(config.failure_file.read_text())[0]["final_error"]
    assert json.loads(config.success_file.read_text()) == []
    assert Path(saved_record.output_path).read_bytes() == output_before


def test_changed_header_invalidates_saved_success(tmp_path: Path) -> None:
    config, source_path, saved_record = _saved_success_fixture(tmp_path)
    include_dir = source_path.parent / "include"
    include_dir.mkdir()
    (include_dir / "helper.h").write_text("// added header")
    summary = pipeline_module.run_optimization_pipeline(FakeClient([]), config)
    assert summary["failed"] == 1
    assert summary["success"] == 0


def test_untracked_existing_output_is_not_a_verified_success(tmp_path: Path) -> None:
    config, _, saved_record = _saved_success_fixture(tmp_path)
    config.records_file.unlink()
    summary = pipeline_module.run_optimization_pipeline(FakeClient([]), config)
    assert summary["failed"] == 1
    assert Path(saved_record.output_path).exists()
