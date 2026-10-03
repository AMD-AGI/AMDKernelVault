# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import hashlib
import json
from pathlib import Path

import pytest

from torch_modu2func_kit import pipeline
from torch_modu2func_kit.config import PipelineConfig
from torch_modu2func_kit.pipeline import convert_single_file, run_conversion_pipeline
from torch_modu2func_kit.verifier import VerificationResult


SOURCE_CODE = """
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


BAD_GENERATION = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias, dtype=torch.float32))

    def forward(self, x):
        return x - self.bias


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [2.5]
"""


GOOD_GENERATION = """
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


bias_value = 2.5


def get_inputs():
    return [torch.tensor([1.0, 2.0])]


def get_init_inputs():
    return [bias_value]
"""


class FakeClient:
    def __init__(self, responses):
        self._responses = list(responses)

    def generate(self, messages, **kwargs):
        return self._responses.pop(0)


def test_convert_single_file_retries_until_success(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = input_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(SOURCE_CODE, encoding="utf-8")

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        max_attempts=2,
    ).with_defaults()

    client = FakeClient([BAD_GENERATION, GOOD_GENERATION])
    record = convert_single_file(source_path, client, config)

    assert record.status == "success"
    assert record.attempts_used == 2
    assert (output_dir / "level_1" / "sample.py").exists()
    assert len(record.attempts) == 2
    assert record.attempts[0].status == "failed"
    assert record.attempts[1].status == "success"
    assert record.attempts[0].feedback is not None

    second_prompt = (artifacts_dir / "prompts" / "level_1" / "sample.attempt_2.txt").read_text(encoding="utf-8")
    assert "Attempt 1:" in second_prompt
    assert "return x - self.bias" in second_prompt
    assert "Observed failure:" in second_prompt
    assert record.attempts[0].feedback in second_prompt


def test_run_conversion_pipeline_writes_json_records(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    artifacts_dir = tmp_path / "artifacts"
    source_path = input_dir / "level_1" / "sample.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(SOURCE_CODE, encoding="utf-8")

    config = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        artifacts_dir=artifacts_dir,
        max_attempts=1,
    ).with_defaults()

    client = FakeClient([GOOD_GENERATION])
    summary = run_conversion_pipeline(client, config)

    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}

    records = json.loads(config.records_file.read_text(encoding="utf-8"))
    successes = json.loads(config.success_file.read_text(encoding="utf-8"))
    failures = json.loads(config.failure_file.read_text(encoding="utf-8"))

    assert len(records) == 1
    assert len(successes) == 1
    assert failures == []
    assert successes[0]["attempts"][0]["status"] == "success"
    assert "feedback" in successes[0]["attempts"][0]
    assert successes[0]["source_path"] == source_path.as_posix()
    assert successes[0]["relative_path"] == Path("level_1/sample.py").as_posix()
    assert successes[0]["output_path"] == (output_dir / "level_1" / "sample.py").as_posix()
    assert successes[0]["attempts"][0]["prompt_path"] == (
        artifacts_dir / "prompts" / "level_1" / "sample.attempt_1.txt"
    ).as_posix()
    assert successes[0]["attempts"][0]["candidate_path"] == (
        artifacts_dir / "candidates" / "level_1" / "sample.attempt_1.py"
    ).as_posix()


@pytest.fixture
def resume_case(tmp_path, monkeypatch):
    source_path = tmp_path / "input" / "sample.py"
    source_path.parent.mkdir()
    source_path.write_text(SOURCE_CODE, encoding="utf-8")
    config = PipelineConfig(
        input_dir=source_path.parent,
        output_dir=tmp_path / "output",
        artifacts_dir=tmp_path / "artifacts",
        max_attempts=1,
    ).with_defaults()
    monkeypatch.setattr(pipeline, "verify_candidate", lambda *args, **kwargs: VerificationResult(True, "Matched."))
    return source_path, config


def _read_records(config):
    return json.loads(config.records_file.read_text(encoding="utf-8"))


def test_resume_preserves_verified_success_and_provenance(resume_case, monkeypatch):
    source_path, config = resume_case
    run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)
    original_records = _read_records(config)
    record = original_records[0]
    assert record["source_sha256"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert record["output_sha256"] == hashlib.sha256(Path(record["output_path"]).read_bytes()).hexdigest()
    assert record["verification_seed"] == config.seed
    assert record["verification_rtol"] == config.rtol
    assert record["verification_atol"] == config.atol
    assert record["attempts"][0]["candidate_sha256"] == record["output_sha256"]
    assert record["attempts"][0]["source_sha256"] == record["source_sha256"]

    def unexpected_verification(*args, **kwargs):
        pytest.fail("A matching saved success must not run verification again.")

    monkeypatch.setattr(pipeline, "verify_candidate", unexpected_verification)
    summary = run_conversion_pipeline(FakeClient([]), config)
    assert summary == {"total": 1, "success": 1, "failed": 0, "skipped": 0}
    assert _read_records(config) == original_records
    assert json.loads(config.success_file.read_text(encoding="utf-8")) == original_records
    assert convert_single_file(source_path, FakeClient([]), config).status == "success"


@pytest.mark.parametrize("change", ["source", "output", "seed", "rtol", "atol"])
def test_resume_does_not_trust_stale_success(resume_case, change):
    source_path, config = resume_case
    run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)
    original_record = _read_records(config)[0]
    if change in ("source", "output"):
        path = source_path if change == "source" else Path(original_record["output_path"])
        path.write_text(path.read_text(encoding="utf-8") + "\n# Changed bytes.\n", encoding="utf-8")
    else:
        setattr(config, change, getattr(config, change) + 1)

    for _ in range(2):
        assert run_conversion_pipeline(FakeClient([]), config)["skipped"] == 1
        saved = _read_records(config)[0]
        assert saved["status"] == "skipped"
        assert saved["skip_reason"]
        assert json.loads(config.success_file.read_text(encoding="utf-8")) == []
        for key in ("attempts", "attempts_used", "source_sha256", "output_sha256",
                    "verification_seed", "verification_rtol", "verification_atol"):
            assert saved[key] == original_record[key]


@pytest.mark.parametrize(
    "field", ["source_sha256", "output_sha256", "verification_seed", "verification_rtol", "verification_atol"]
)
def test_legacy_success_without_provenance_stays_unverified(resume_case, field):
    _, config = resume_case
    run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)
    records = _read_records(config)
    records[0].pop(field)
    config.records_file.write_text(json.dumps(records), encoding="utf-8")

    for _ in range(2):
        assert run_conversion_pipeline(FakeClient([]), config)["skipped"] == 1
        saved = _read_records(config)[0]
        assert saved[field] is None
        assert saved["attempts"] == records[0]["attempts"]
        assert json.loads(config.success_file.read_text(encoding="utf-8")) == []


def test_unrecorded_output_stays_unverified_until_overwrite(resume_case):
    source_path, config = resume_case
    config.output_dir.mkdir()
    output_path = config.output_dir / source_path.name
    output_path.write_text("# Legacy output.\n", encoding="utf-8")

    assert run_conversion_pipeline(FakeClient([]), config)["skipped"] == 1
    saved = _read_records(config)[0]
    assert saved["source_sha256"] is None
    assert saved["output_sha256"] is None
    assert saved["verification_seed"] is None
    assert output_path.read_text(encoding="utf-8") == "# Legacy output.\n"

    config.overwrite = True
    assert run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)["success"] == 1


@pytest.mark.parametrize("replace_output", ["missing", "overwrite"])
def test_regeneration_preserves_attempts_and_artifacts(resume_case, replace_output):
    _, config = resume_case
    run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)
    original_record = _read_records(config)[0]
    artifacts = {
        Path(original_record["attempts"][0][key]): Path(original_record["attempts"][0][key]).read_bytes()
        for key in ("prompt_path", "candidate_path")
    }
    if replace_output == "missing":
        Path(original_record["output_path"]).unlink()
    else:
        config.overwrite = True

    assert run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)["success"] == 1
    saved = _read_records(config)[0]
    assert saved["attempts_used"] == 2
    assert saved["attempts"][0] == original_record["attempts"][0]
    assert saved["attempts"][1]["attempt"] == 2
    for path, content in artifacts.items():
        assert path.read_bytes() == content


def test_failed_rerun_keeps_history_and_excludes_stale_success(resume_case, monkeypatch):
    _, config = resume_case
    run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)
    original_record = _read_records(config)[0]
    config.overwrite = True
    monkeypatch.setattr(pipeline, "verify_candidate", lambda *args, **kwargs: VerificationResult(False, "Mismatch."))
    assert run_conversion_pipeline(FakeClient([BAD_GENERATION]), config)["failed"] == 1
    saved = _read_records(config)[0]
    assert saved["attempts_used"] == 2
    assert saved["attempts"][0] == original_record["attempts"][0]
    assert json.loads(config.success_file.read_text(encoding="utf-8")) == []
    config.overwrite = False
    assert run_conversion_pipeline(FakeClient([]), config)["skipped"] == 1


def test_orphan_artifacts_are_not_overwritten(resume_case):
    _, config = resume_case
    prompt_path = config.artifacts_dir / "prompts" / "sample.attempt_1.txt"
    candidate_path = config.artifacts_dir / "candidates" / "sample.attempt_2.py"
    for path in (prompt_path, candidate_path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("orphan evidence", encoding="utf-8")

    assert run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)["success"] == 1
    assert _read_records(config)[0]["attempts"][0]["attempt"] == 3
    for path in (prompt_path, candidate_path):
        assert path.read_text(encoding="utf-8") == "orphan evidence"


def test_partial_rerun_preserves_later_records(resume_case):
    source_path, config = resume_case
    later_source = source_path.with_name("z_later.py")
    later_source.write_text(SOURCE_CODE, encoding="utf-8")
    run_conversion_pipeline(FakeClient([GOOD_GENERATION, GOOD_GENERATION]), config)
    original_records = _read_records(config)
    config.overwrite = True

    class InterruptedClient:
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                raise KeyboardInterrupt
            return GOOD_GENERATION

    with pytest.raises(KeyboardInterrupt):
        run_conversion_pipeline(InterruptedClient(), config)
    saved = _read_records(config)
    assert len(saved) == 2
    assert saved[0]["attempts_used"] == 2
    assert saved[1] == original_records[1]
    assert len(json.loads(config.success_file.read_text(encoding="utf-8"))) == 2


def test_removed_source_invalidates_retained_success(resume_case):
    source_path, config = resume_case
    run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)
    source_path.unlink()
    assert run_conversion_pipeline(FakeClient([]), config)["total"] == 0
    assert _read_records(config)[0]["status"] == "skipped"
    assert json.loads(config.success_file.read_text(encoding="utf-8")) == []


def test_interruption_before_first_result_removes_stale_success(resume_case):
    source_path, config = resume_case
    run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)
    source_path.write_text(SOURCE_CODE + "\n# Changed source.\n", encoding="utf-8")
    config.overwrite = True

    class InterruptedClient:
        def generate(self, *args, **kwargs):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_conversion_pipeline(InterruptedClient(), config)
    assert _read_records(config)[0]["status"] == "skipped"
    assert json.loads(config.success_file.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("root", ["input_dir", "output_dir"])
def test_changed_root_preserves_old_provenance_without_reuse(resume_case, root):
    source_path, config = resume_case
    run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)
    original_record = _read_records(config)[0]
    original_path = getattr(config, root) / source_path.name
    new_root = source_path.parent.parent / "different_root"
    new_root.mkdir()
    (new_root / source_path.name).write_bytes(original_path.read_bytes())
    setattr(config, root, new_root)

    assert run_conversion_pipeline(FakeClient([]), config)["skipped"] == 1
    saved = _read_records(config)[0]
    assert saved["source_path"] == original_record["source_path"]
    assert saved["output_path"] == original_record["output_path"]
    assert saved["source_sha256"] == original_record["source_sha256"]
    assert saved["output_sha256"] == original_record["output_sha256"]
    assert json.loads(config.success_file.read_text(encoding="utf-8")) == []


def test_malformed_records_are_not_replaced(resume_case):
    _, config = resume_case
    config.artifacts_dir.mkdir()
    config.records_file.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot resume"):
        run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)
    assert config.records_file.read_text(encoding="utf-8") == "{broken"
    assert not (config.artifacts_dir / "prompts").exists()


@pytest.mark.parametrize("changed_file", ["source", "candidate"])
def test_changes_during_verification_cannot_be_saved_as_success(resume_case, monkeypatch, changed_file):
    source_path, config = resume_case

    def mutate_during_verification(source, candidate, **kwargs):
        path = source if changed_file == "source" else candidate
        path.write_text("# Changed during verification.\n", encoding="utf-8")
        return VerificationResult(True, "Matched.")

    monkeypatch.setattr(pipeline, "verify_candidate", mutate_during_verification)
    assert run_conversion_pipeline(FakeClient([GOOD_GENERATION]), config)["failed"] == 1
    assert "changed during verification" in _read_records(config)[0]["final_error"]
    assert not (config.output_dir / source_path.name).exists()


def test_pipeline_loads_records_once(resume_case, monkeypatch):
    source_path, config = resume_case
    source_path.with_name("second.py").write_text(SOURCE_CODE, encoding="utf-8")
    original_load = pipeline._load_records
    calls = []

    def load_once(config):
        calls.append(config)
        return original_load(config)

    monkeypatch.setattr(pipeline, "_load_records", load_once)
    assert run_conversion_pipeline(FakeClient([GOOD_GENERATION, GOOD_GENERATION]), config)["success"] == 2
    assert len(calls) == 1
