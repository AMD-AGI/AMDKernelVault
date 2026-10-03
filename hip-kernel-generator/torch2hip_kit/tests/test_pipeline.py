# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path

import pytest

import torch2hip_kit.pipeline as pipeline_module
from torch2hip_kit.config import PipelineConfig
from torch2hip_kit.verifier import VerificationResult


MODULE_SOURCE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return x + self.bias


bias_value = 2.5


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [bias_value]
"""


FUNCTIONAL_SOURCE = """
import torch
import torch.nn as nn


def module_fn(x, bias):
    return x + bias


class Model(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x, fn=module_fn):
        return fn(x, self.bias)


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [2.5]
"""


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, messages, **kwargs):
        return self._responses.pop(0)


@pytest.mark.parametrize("speedups", [(1.10, 1.75), (0.25, 0.75)])
def test_convert_single_file_selects_fastest_success(monkeypatch, tmp_path: Path, speedups) -> None:
    module_dir = tmp_path / "module"
    functional_dir = tmp_path / "functional"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = module_dir / "level_1" / "sample.py"
    functional_path = functional_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    functional_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(MODULE_SOURCE, encoding="utf-8")
    functional_path.write_text(FUNCTIONAL_SOURCE, encoding="utf-8")

    verification_results = [
        VerificationResult(True, "attempt1", compile_success=True, correctness_success=True, speedup=speedups[0]),
        VerificationResult(False, "attempt2 failed", compile_success=True, correctness_success=False),
        VerificationResult(True, "attempt3", compile_success=True, correctness_success=True, speedup=speedups[1]),
    ]

    def fake_verify_candidate(*args, **kwargs):
        return verification_results.pop(0)

    monkeypatch.setattr(pipeline_module, "verify_candidate", fake_verify_candidate)

    config = PipelineConfig(
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
            "```cpp\n// attempt1\n```",
            "```cpp\n// attempt2\n```",
            "```cpp\n// attempt3\n```",
        ]
    )
    record = pipeline_module.convert_single_file(source_path, client, config)

    assert record.status == "success"
    assert record.best_attempt == 3
    assert record.best_speedup == speedups[1]
    assert record.attempts_used == 3
    assert (output_dir / "level_1" / "sample.hip").read_text(encoding="utf-8") == "// attempt3"

    third_prompt = (artifacts_dir / "prompts" / "level_1" / "sample.attempt_3.txt").read_text(encoding="utf-8")
    assert "Attempt 1:" in third_prompt
    assert f"Current best validated speedup: {speedups[0]:.4f}x." in third_prompt


def test_run_conversion_pipeline_writes_json_records(monkeypatch, tmp_path: Path) -> None:
    module_dir = tmp_path / "module"
    functional_dir = tmp_path / "functional"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = module_dir / "level_1" / "sample.py"
    functional_path = functional_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    functional_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(MODULE_SOURCE, encoding="utf-8")
    functional_path.write_text(FUNCTIONAL_SOURCE, encoding="utf-8")

    monkeypatch.setattr(
        pipeline_module,
        "verify_candidate",
        lambda *args, **kwargs: VerificationResult(
            True,
            "ok",
            compile_success=True,
            correctness_success=True,
            speedup=1.5,
        ),
    )

    config = PipelineConfig(
        module_dir=module_dir,
        functional_dir=functional_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    client = FakeClient(["```cpp\n// best\n```"])
    summary = pipeline_module.run_conversion_pipeline(client, config)

    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    records = json.loads(config.records_file.read_text(encoding="utf-8"))
    successes = json.loads(config.success_file.read_text(encoding="utf-8"))
    failures = json.loads(config.failure_file.read_text(encoding="utf-8"))

    assert len(records) == 1
    assert len(successes) == 1
    assert failures == []
    assert successes[0]["best_attempt"] == 1
    assert successes[0]["module_source_path"] == source_path.as_posix()
    assert successes[0]["functional_source_path"] == functional_path.as_posix()
    assert successes[0]["relative_path"] == Path("level_1/sample.py").as_posix()
    assert successes[0]["output_path"] == (output_dir / "level_1" / "sample.hip").as_posix()
    assert successes[0]["attempts"][0]["prompt_path"] == (
        artifacts_dir / "prompts" / "level_1" / "sample.attempt_1.txt"
    ).as_posix()
    assert successes[0]["attempts"][0]["candidate_path"] == (
        artifacts_dir / "candidates" / "level_1" / "sample.attempt_1.hip"
    ).as_posix()


def test_convert_single_file_fails_when_functional_pair_missing(tmp_path: Path) -> None:
    module_dir = tmp_path / "module"
    source_path = module_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(MODULE_SOURCE, encoding="utf-8")

    config = PipelineConfig(
        module_dir=module_dir,
        functional_dir=tmp_path / "functional",
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()

    record = pipeline_module.convert_single_file(source_path, FakeClient([]), config)

    assert record.status == "failed"
    assert "Paired functional file was not found" in (record.final_error or "")
    assert "\\" not in (record.final_error or "")


def _make_config(tmp_path: Path, names=("sample",)) -> PipelineConfig:
    config = PipelineConfig(
        module_dir=tmp_path / "module",
        functional_dir=tmp_path / "functional",
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        max_attempts=1,
        system_instruction="SYSTEM",
        few_shot_examples="FEW SHOT",
    ).with_defaults()
    for name in names:
        for directory, text in (
            (config.module_dir, MODULE_SOURCE),
            (config.functional_dir, FUNCTIONAL_SOURCE),
        ):
            path = directory / "level_1" / f"{name}.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    return config


def _fake_success(*args, **kwargs) -> VerificationResult:
    return VerificationResult(
        True,
        "Correctness passed.",
        compile_success=True,
        correctness_success=True,
        speedup=0.75,
        module_latency_ms=3.0,
        hip_latency_ms=4.0,
    )


def _no_verification(*args, **kwargs):
    pytest.fail("A current output or a blocked output must not invoke the verifier.")


def test_rerun_preserves_verified_success_records(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, names=("first", "second"))
    monkeypatch.setattr(pipeline_module, "verify_candidate", _fake_success)
    pipeline_module.run_conversion_pipeline(FakeClient(["// first", "// second"]), config)
    record_files = (config.records_file, config.success_file, config.failure_file)
    original_records = [path.read_text(encoding="utf-8") for path in record_files]
    for candidate_path in (config.artifacts_dir / "candidates").rglob("*.hip"):
        candidate_path.unlink()

    monkeypatch.setattr(pipeline_module, "verify_candidate", _no_verification)
    summary = pipeline_module.run_conversion_pipeline(FakeClient([]), config)

    assert summary == {"total": 2, "success": 2, "failed": 0, "skipped": 0}
    assert [path.read_text(encoding="utf-8") for path in record_files] == original_records
    records = json.loads(config.success_file.read_text(encoding="utf-8"))
    assert all(record["best_speedup"] == 0.75 for record in records)
    assert all(record["input_fingerprint"] and record["output_sha256"] for record in records)


def test_interrupted_rerun_preserves_pending_records(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path, names=("first", "second"))
    monkeypatch.setattr(pipeline_module, "verify_candidate", _fake_success)
    pipeline_module.run_conversion_pipeline(FakeClient(["// first", "// second"]), config)
    original_records = json.loads(config.records_file.read_text(encoding="utf-8"))
    persist_records = pipeline_module._persist_records

    def persist_then_interrupt(config, records):
        persist_records(config, records)
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_module, "verify_candidate", _no_verification)
    monkeypatch.setattr(pipeline_module, "_persist_records", persist_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        pipeline_module.run_conversion_pipeline(FakeClient([]), config)

    assert json.loads(config.records_file.read_text(encoding="utf-8")) == original_records
    monkeypatch.setattr(pipeline_module, "_persist_records", persist_records)
    assert pipeline_module.run_conversion_pipeline(FakeClient([]), config) == {
        "total": 2, "success": 2, "failed": 0, "skipped": 0
    }


def test_removed_sources_do_not_remain_in_success_records(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(pipeline_module, "verify_candidate", _fake_success)
    pipeline_module.run_conversion_pipeline(FakeClient(["// original"]), config)
    (config.module_dir / "level_1" / "sample.py").unlink()

    assert pipeline_module.run_conversion_pipeline(FakeClient([]), config) == {
        "total": 0, "success": 0, "failed": 0, "skipped": 0
    }
    assert json.loads(config.records_file.read_text(encoding="utf-8")) == []
    assert json.loads(config.success_file.read_text(encoding="utf-8")) == []


def test_missing_output_runs_generation_again(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(pipeline_module, "verify_candidate", _fake_success)
    pipeline_module.run_conversion_pipeline(FakeClient(["// original"]), config)
    output_path = config.output_dir / "level_1" / "sample.hip"
    output_path.unlink()

    summary = pipeline_module.run_conversion_pipeline(FakeClient(["// regenerated"]), config)

    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    assert output_path.read_text(encoding="utf-8") == "// regenerated"


@pytest.mark.parametrize(
    "change",
    [
        "module",
        "functional",
        "output",
        "seed",
        "rtol",
        "atol",
        "perf_warmup",
        "perf_iterations",
        "max_attempts",
        "temperature",
        "max_tokens",
        "history_code_char_limit",
        "history_feedback_char_limit",
        "system_instruction",
        "few_shot_examples",
    ],
)
def test_changed_inputs_or_output_require_overwrite(monkeypatch, tmp_path: Path, change: str) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(pipeline_module, "verify_candidate", _fake_success)
    pipeline_module.run_conversion_pipeline(FakeClient(["// original"]), config)
    output_path = config.output_dir / "level_1" / "sample.hip"

    if change in ("module", "functional"):
        directory = config.module_dir if change == "module" else config.functional_dir
        source_path = directory / "level_1" / "sample.py"
        source_path.write_text(source_path.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
    elif change == "output":
        output_path.write_text("// modified externally", encoding="utf-8")
    elif change in ("system_instruction", "few_shot_examples"):
        setattr(config, change, "CHANGED")
    else:
        setattr(config, change, getattr(config, change) + 1)
    expected_output = output_path.read_bytes()
    monkeypatch.setattr(pipeline_module, "verify_candidate", _no_verification)

    summary = pipeline_module.run_conversion_pipeline(FakeClient([]), config)

    assert summary == {"total": 1, "success": 0, "failed": 1, "skipped": 0}
    assert output_path.read_bytes() == expected_output
    failures = json.loads(config.failure_file.read_text(encoding="utf-8"))
    assert failures[0]["attempts_used"] == 0
    assert "--overwrite" in failures[0]["final_error"]
    assert json.loads(config.success_file.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("change", ["model", "client_class"])
def test_changed_model_or_client_requires_overwrite(monkeypatch, tmp_path: Path, change: str) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(pipeline_module, "verify_candidate", _fake_success)
    client = FakeClient(["// original"])
    client.model_id = "test-model"
    client.api_key = "test-secret-must-not-appear-in-records"
    pipeline_module.run_conversion_pipeline(client, config)
    assert client.api_key not in config.records_file.read_text(encoding="utf-8")

    if change == "model":
        client.model_id = "different-test-model"
    else:
        class OtherFakeClient(FakeClient):
            pass

        client = OtherFakeClient([])
        client.model_id = "test-model"
    monkeypatch.setattr(pipeline_module, "verify_candidate", _no_verification)

    summary = pipeline_module.run_conversion_pipeline(client, config)

    assert summary == {"total": 1, "success": 0, "failed": 1, "skipped": 0}
    failure = json.loads(config.failure_file.read_text(encoding="utf-8"))[0]
    assert failure["attempts_used"] == 0
    assert "--overwrite" in failure["final_error"]


@pytest.mark.parametrize("record_state", ["absent", "legacy", "corrupt", "wrong_path", "unverified"])
def test_existing_untracked_output_requires_overwrite(monkeypatch, tmp_path: Path, record_state: str) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(pipeline_module, "verify_candidate", _fake_success)
    pipeline_module.run_conversion_pipeline(FakeClient(["// original"]), config)
    if record_state == "absent":
        config.records_file.unlink()
    elif record_state == "corrupt":
        config.records_file.write_text("{invalid JSON", encoding="utf-8")
    else:
        records = json.loads(config.records_file.read_text(encoding="utf-8"))
        if record_state == "legacy":
            records[0].pop("input_fingerprint")
            records[0].pop("output_sha256")
        elif record_state == "wrong_path":
            records[0]["output_path"] = str(tmp_path / "other.hip")
        else:
            records[0]["attempts"][0]["correctness_success"] = False
        config.records_file.write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(pipeline_module, "verify_candidate", _no_verification)

    summary = pipeline_module.run_conversion_pipeline(FakeClient([]), config)

    assert summary == {"total": 1, "success": 0, "failed": 1, "skipped": 0}
    failure = json.loads(config.failure_file.read_text(encoding="utf-8"))[0]
    assert failure["attempts_used"] == 0
    assert "--overwrite" in failure["final_error"]
    assert (config.output_dir / "level_1" / "sample.hip").read_text(encoding="utf-8") == "// original"


@pytest.mark.parametrize("tracked", [False, True])
def test_overwrite_replaces_existing_output(monkeypatch, tmp_path: Path, tracked: bool) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(pipeline_module, "verify_candidate", _fake_success)
    pipeline_module.run_conversion_pipeline(FakeClient(["// original"]), config)
    if not tracked:
        config.records_file.unlink()
    config.overwrite = True

    summary = pipeline_module.run_conversion_pipeline(FakeClient(["// replacement"]), config)

    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    assert (config.output_dir / "level_1" / "sample.hip").read_text(encoding="utf-8") == "// replacement"


def test_failed_overwrite_preserves_previous_output(monkeypatch, tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    monkeypatch.setattr(pipeline_module, "verify_candidate", _fake_success)
    pipeline_module.run_conversion_pipeline(FakeClient(["// original"]), config)
    config.overwrite = True
    monkeypatch.setattr(
        pipeline_module,
        "verify_candidate",
        lambda *args, **kwargs: VerificationResult(False, "incorrect", compile_success=True),
    )

    summary = pipeline_module.run_conversion_pipeline(FakeClient(["// incorrect"]), config)

    assert summary == {"total": 1, "success": 0, "failed": 1, "skipped": 0}
    assert (config.output_dir / "level_1" / "sample.hip").read_text(encoding="utf-8") == "// original"
