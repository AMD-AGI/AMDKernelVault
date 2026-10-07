# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

import hashlib
import json
import stat
from dataclasses import replace
from pathlib import Path

import pytest

from triton_kernel_gen.contracts import (
    GenerationSettings,
    ModelResponse,
    PreparedTask,
    TaskSpec,
    VerificationResult,
)
from triton_kernel_gen.errors import ReferenceValidationError
from triton_kernel_gen.pipeline import RecordWriteError, _atomic_json, run_task

CODE = "def module_fn(x):\n    return x\n"
OTHER_CODE = "def module_fn(x):\n    return x + 0\n"


def completion(code=CODE):
    return ModelResponse(
        f"```python\n{code}```",
        "generator-model",
        "stop",
        {"completion_tokens": 12},
        "saved reasoning",
    )


def passed():
    return VerificationResult(
        compile_success=True,
        correctness_success=True,
        runtime_success=True,
        timing_success=True,
        message="passed",
        correctness_cases=[{"case_id": "a", "success": True}, {"case_id": "b", "success": True}],
        performance_cases=[
            {"case_id": "p", "baseline_ms": 1.0, "candidate_ms": 2.0, "speedup": 0.5}
        ],
        environment={"gpu": "fake"},
    )


class Client:
    def __init__(self, responses, events=None, name="generation", hook=None):
        self.responses = iter(responses)
        self.calls = []
        self.events = events if events is not None else []
        self.name = name
        self.hook = hook

    def generate(self, messages, **settings):
        self.events.append(self.name)
        self.calls.append((messages, settings))
        if self.hook:
            self.hook()
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


class Verifier:
    def __init__(self, results=None, events=None, prepare_error=None, hook=None):
        self.results = iter(results if results is not None else [passed()])
        self.events = events if events is not None else []
        self.prepare_error = prepare_error
        self.hook = hook
        self.calls = []

    def prepare(self, task, work_dir):
        self.events.append("prepare")
        if self.prepare_error:
            raise self.prepare_error
        return PreparedTask(
            task,
            work_dir,
            {"correctness_count": 2, "performance_count": 1, "environment": {"gpu": "fake"}},
        )

    def verify(self, prepared, candidate_path):
        self.events.append("verify")
        self.calls.append(candidate_path)
        if self.hook:
            self.hook(candidate_path)
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def task(tmp_path):
    files = {}
    for name, content in {
        "module.py": "class Model: pass\n",
        "functional.py": CODE,
        "cases.py": "# external case provider\n",
    }.items():
        path = tmp_path / name
        path.write_text(content)
        files[name] = path
    return TaskSpec(
        "operation",
        files["module.py"],
        files["functional.py"],
        files["cases.py"],
        provenance={"source": "synthetic test"},
    )


@pytest.fixture
def settings(tmp_path):
    return GenerationSettings(tmp_path / "outputs", tmp_path / "artifacts", max_attempts=4)


def reflector(events=None, hook=None):
    return Client(
        [
            ModelResponse(
                "Use the actual strides.", "reflector-model", reasoning="reflection reasoning"
            )
        ]
        * 10,
        events,
        "reflection",
        hook,
    )


def test_validates_reference_before_generation_and_keeps_slower_candidate(task, settings):
    events = []
    generator = Client([completion()], events)
    reflection = reflector(events)
    verifier = Verifier(events=events)
    record = run_task(task, generator, reflection, verifier, settings)
    assert events == ["prepare", "generation", "verify"]
    assert record["status"] == "success"
    assert record["counts"] == {
        "generation_calls": 1,
        "reflection_calls": 0,
        "verification_calls": 1,
        "accepted_candidates": 1,
    }
    accepted = record["accepted_candidates"][0]
    assert accepted["verification"]["performance_cases"][0]["speedup"] == 0.5
    assert Path(accepted["output_path"]).read_text() == CODE
    assert (
        hashlib.sha256(Path(accepted["output_path"]).read_bytes()).hexdigest()
        == accepted["candidate_sha256"]
    )
    assert stat.S_IMODE(Path(accepted["output_path"]).stat().st_mode) == 0o444
    assert record == json.loads(Path(record["record_path"]).read_text())


@pytest.mark.parametrize(
    "failure_stage,expected",
    [
        ("reference", "reference_failed"),
        ("infrastructure", "infrastructure_error"),
        ("input_modified", "input_modified"),
    ],
)
def test_reference_failures_do_not_generate(task, settings, failure_stage, expected):
    generator = Client([])
    verifier = Verifier(
        prepare_error=ReferenceValidationError(
            "trusted boundary failed", failure_stage=failure_stage
        )
    )
    record = run_task(task, generator, reflector(), verifier, settings)
    assert record["status"] == expected
    assert generator.calls == [] and record["attempts"] == []
    assert record["reference_failure_stage"] == failure_stage
    assert Path(record["record_path"]).is_file()


def test_prepare_storage_failure_is_infrastructure_failure(task, settings):
    record = run_task(
        task,
        Client([]),
        reflector(),
        Verifier(prepare_error=OSError("scratch disk failed")),
        settings,
    )
    assert record["status"] == "infrastructure_error"


def test_missing_source_is_reference_failure(task, settings):
    task.functional_path.unlink()
    record = run_task(task, Client([]), reflector(), Verifier(), settings)
    assert record["status"] == "reference_failed"


def test_compile_failure_triggers_separate_reflection_before_retry(task, settings):
    events = []
    failed = replace(
        passed(), compile_success=False, failure_stage="compile", message="invalid launch"
    )
    generator = Client([completion(), completion(OTHER_CODE)], events)
    record = run_task(
        task, generator, reflector(events), Verifier([failed, passed()], events), settings
    )
    assert events == ["prepare", "generation", "verify", "reflection", "generation", "verify"]
    assert record["status"] == "success"
    assert record["attempts"][0]["status"] == "verification_failed"
    assert "invalid launch" in generator.calls[1][0][1]["content"]
    assert "Use the actual strides" in generator.calls[1][0][1]["content"]
    assert record["attempts"][0]["reflection"]["response"]["reasoning"] == "reflection reasoning"


def test_invalid_response_is_recorded_and_reflected(task, settings):
    generator = Client([ModelResponse("no code", "model", reasoning="trace"), completion()])
    verifier = Verifier()
    record = run_task(task, generator, reflector(), verifier, settings)
    assert record["status"] == "success"
    assert record["attempts"][0]["status"] == "parse_failed"
    assert record["attempts"][0]["generation"]["text"] == "no code"
    assert record["attempts"][0]["generation"]["reasoning"] == "trace"
    assert len(verifier.calls) == 1


@pytest.mark.parametrize("reason", ["length", "max_tokens"])
def test_truncated_response_does_not_run_even_with_complete_fence(task, settings, reason):
    verifier = Verifier()
    record = run_task(
        task,
        Client([replace(completion(), finish_reason=reason)]),
        reflector(),
        verifier,
        replace(settings, max_attempts=1),
    )
    assert record["status"] == "exhausted"
    assert record["attempts"][0]["status"] == "parse_failed"
    assert verifier.calls == []


def test_budget_exhaustion_skips_final_reflection(task, settings):
    failure = replace(passed(), correctness_success=False, failure_stage="correctness")
    reflection = reflector()
    record = run_task(
        task,
        Client([completion()] * 2),
        reflection,
        Verifier([failure] * 2),
        replace(settings, max_attempts=2),
    )
    assert record["status"] == "exhausted"
    assert len(record["attempts"]) == 2
    assert len(reflection.calls) == 1
    assert "reflection" not in record["attempts"][-1]


def test_variant_search_keeps_separate_immutable_outputs(task, settings):
    reflection = reflector()
    record = run_task(
        task,
        Client([completion(), completion(OTHER_CODE)]),
        reflection,
        Verifier([passed(), passed()]),
        replace(settings, num_variants=2),
    )
    assert record["status"] == "success"
    first, second = record["accepted_candidates"]
    assert first["output_path"] != second["output_path"]
    assert Path(first["output_path"]).read_text() == CODE
    assert Path(second["output_path"]).read_text() == OTHER_CODE
    assert record["attempts"][0]["reflection"]["purpose"] == "variant"
    assert '"speedup": 0.5' in reflection.calls[0][0][1]["content"]
    assert len(reflection.calls) == 1


def test_duplicates_do_not_count_as_distinct_variants(task, settings):
    record = run_task(
        task,
        Client([completion()] * 2),
        reflector(),
        Verifier([passed(), passed()]),
        replace(settings, max_attempts=2, num_variants=2),
    )
    assert record["status"] == "partial_success"
    assert len(record["accepted_candidates"]) == 1
    assert record["attempts"][1]["status"] == "duplicate"
    assert record["counts"]["verification_calls"] == 2


@pytest.mark.parametrize("stage", ["compile", "runtime", "correctness", "timing"])
def test_all_success_dimensions_are_required(task, settings, stage):
    result = replace(passed(), **{stage + "_success": False}, failure_stage=stage)
    record = run_task(
        task,
        Client([completion()]),
        reflector(),
        Verifier([result]),
        replace(settings, max_attempts=1),
    )
    assert record["status"] == "exhausted" and not record["accepted_candidates"]
    assert record["attempts"][0]["verification"]["success"] is False


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("worker failed"),
        replace(passed(), compile_success=False, failure_stage="infrastructure"),
    ],
)
def test_infrastructure_failures_stop_without_reflection(task, settings, failure):
    reflection = reflector()
    record = run_task(task, Client([completion()]), reflection, Verifier([failure]), settings)
    assert record["status"] == "infrastructure_error"
    assert reflection.calls == []
    assert record["counts"]["generation_calls"] == 1


def test_provider_failure_has_no_fabricated_verification(task, settings):
    record = run_task(
        task, Client([RuntimeError("request failed")]), reflector(), Verifier(), settings
    )
    assert record["status"] == "provider_error"
    assert record["counts"]["verification_calls"] == 0
    assert "verification" not in record["attempts"][0]
    assert (
        json.loads((Path(record["run_dir"]) / "attempts" / "attempt-0001.json").read_text())[
            "error"
        ]["message"]
        == "request failed"
    )


def test_reflection_provider_failure_preserves_accepted_output(task, settings):
    record = run_task(
        task,
        Client([completion()]),
        Client([RuntimeError("reflection failed")]),
        Verifier(),
        replace(settings, num_variants=2),
    )
    assert record["status"] == "provider_error"
    assert len(record["accepted_candidates"]) == 1
    assert Path(record["accepted_candidates"][0]["output_path"]).is_file()
    assert record["attempts"][0]["reflection"]["error"]["phase"] == "reflection"


def test_reruns_verify_again_and_do_not_overwrite_previous_record(task, settings):
    first = run_task(task, Client([completion()]), reflector(), Verifier(), settings)
    previous = Path(first["record_path"]).read_bytes()
    verifier = Verifier()
    second = run_task(task, Client([completion()]), reflector(), verifier, settings)
    assert first["run_dir"] != second["run_dir"]
    assert (
        first["accepted_candidates"][0]["output_path"]
        != second["accepted_candidates"][0]["output_path"]
    )
    assert Path(first["record_path"]).read_bytes() == previous
    assert len(verifier.calls) == 1


def test_source_and_dependency_snapshots_preserve_names_and_hashes(task, settings, tmp_path):
    helper = tmp_path / "helper.py"
    helper.write_text("HELPER_MARKER = 1\n")
    seed = tmp_path / "seed.py"
    seed.write_text("SEED_MARKER = 1\n")
    task = replace(task, dependency_paths=(helper,), seed_kernel_path=seed)
    generator = Client([completion()])
    record = run_task(task, generator, reflector(), Verifier(), settings)
    assert len(record["sources"]) == 5
    for source in record["sources"].values():
        snapshot = Path(source["snapshot_path"])
        assert snapshot.name == source["filename"]
        assert stat.S_IMODE(snapshot.stat().st_mode) == 0o444
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == source["sha256"]
    text = generator.calls[0][0][1]["content"]
    assert "HELPER_MARKER" in text and "SEED_MARKER" in text
    assert str(tmp_path) not in text


def test_raw_seed_is_context_without_pipeline_syntax_or_execution_checks(task, settings, tmp_path):
    seed = tmp_path / "cuda_seed.py"
    seed.write_text("import unsupported_cuda_module\nCUDA-only source awaiting porting\n")
    generator = Client([completion()])
    verifier = Verifier()
    record = run_task(
        replace(task, seed_kernel_path=seed), generator, reflector(), verifier, settings
    )
    assert record["status"] == "success"
    assert record["sources"]["seed_kernel"]["purpose"] == "unverified_source_context"
    assert "baseline_kernel" not in record["sources"]
    assert "CUDA-only source awaiting porting" in generator.calls[0][0][1]["content"]
    assert len(verifier.calls) == 1


def test_baseline_snapshot_context_and_verifier_metadata_remain_distinct(task, settings, tmp_path):
    baseline = tmp_path / "amd_baseline.py"
    baseline.write_text("AMD_BASELINE_MARKER = 1\n")
    seed = tmp_path / "cuda_seed.py"
    seed.write_text("CUDA_SEED_MARKER = 1\n")

    class BaselineVerifier(Verifier):
        def prepare(self, prepared_task, work_dir):
            prepared = super().prepare(prepared_task, work_dir)
            assert prepared_task.baseline_kernel_path == baseline
            assert prepared_task.seed_kernel_path == seed
            prepared.reference_record["baseline_validation"] = {"success": True}
            return prepared

    result = passed()
    result.performance_cases[0].update(
        {"baseline_kind": "baseline_kernel", "baseline_kernel_ms": 1.0}
    )
    generator = Client([completion(), completion(OTHER_CODE)])
    reflection = reflector()
    record = run_task(
        replace(task, seed_kernel_path=seed, baseline_kernel_path=baseline),
        generator,
        reflection,
        BaselineVerifier([result, result]),
        replace(settings, num_variants=2),
    )
    assert record["status"] == "success"
    assert record["reference"]["baseline_validation"] == {"success": True}
    assert record["sources"]["baseline_kernel"]["purpose"] == "optional_latency_baseline"
    assert record["sources"]["seed_kernel"]["purpose"] == "unverified_source_context"
    source = record["sources"]["baseline_kernel"]
    snapshot = Path(source["snapshot_path"])
    assert snapshot.name == baseline.name
    assert snapshot.read_bytes() == baseline.read_bytes()
    assert source["sha256"] == hashlib.sha256(baseline.read_bytes()).hexdigest()
    assert stat.S_IMODE(snapshot.stat().st_mode) == 0o444
    for messages in (generator.calls[0][0], reflection.calls[0][0]):
        assert "Verified Triton baseline" in messages[1]["content"]
        assert "AMD_BASELINE_MARKER" in messages[1]["content"]
        assert "Raw Triton seed (unverified source context)" in messages[1]["content"]
        assert "CUDA_SEED_MARKER" in messages[1]["content"]
        assert str(tmp_path) not in messages[1]["content"]
    recorded_case = record["accepted_candidates"][0]["verification"]["performance_cases"][0]
    assert recorded_case["baseline_kind"] == "baseline_kernel"
    assert recorded_case["baseline_kernel_ms"] == 1.0


def test_baseline_failure_stops_before_generation(task, settings, tmp_path):
    baseline = tmp_path / "baseline.py"
    baseline.write_text("BASELINE = 1\n")
    generator = Client([])
    record = run_task(
        replace(task, baseline_kernel_path=baseline),
        generator,
        reflector(),
        Verifier(prepare_error=ReferenceValidationError("baseline validation failed")),
        settings,
    )
    assert record["status"] == "reference_failed"
    assert generator.calls == []
    assert "baseline_kernel" in record["sources"]


def test_baseline_modification_during_generation_stops_verification(task, settings, tmp_path):
    baseline = tmp_path / "baseline.py"
    baseline.write_text("BASELINE = 1\n")
    generator = Client([completion()], hook=lambda: baseline.write_text("changed"))
    verifier = Verifier()
    record = run_task(
        replace(task, baseline_kernel_path=baseline),
        generator,
        reflector(),
        verifier,
        settings,
    )
    assert record["status"] == "input_modified"
    assert verifier.calls == []


def test_source_modification_during_generation_stops_verification(task, settings):
    generator = Client([completion()], hook=lambda: task.functional_path.write_text("changed"))
    verifier = Verifier()
    record = run_task(task, generator, reflector(), verifier, settings)
    assert record["status"] == "input_modified"
    assert verifier.calls == []
    assert record["attempts"][0]["generation"]["model"] == "generator-model"


def test_source_modification_during_raising_model_call_has_integrity_status(task, settings):
    generator = Client(
        [RuntimeError("request failed")], hook=lambda: task.functional_path.write_text("changed")
    )
    record = run_task(task, generator, reflector(), Verifier(), settings)
    assert record["status"] == "input_modified"


@pytest.mark.parametrize(
    "error",
    [
        ReferenceValidationError("reference failed"),
        OSError("worker unavailable"),
        RuntimeError("worker failed"),
    ],
)
def test_reference_modification_during_failed_prepare_has_integrity_status(task, settings, error):
    class MutatingVerifier(Verifier):
        def prepare(self, prepared_task, work_dir):
            prepared_task.functional_path.write_text("changed")
            raise error

    generator = Client([])
    record = run_task(task, generator, reflector(), MutatingVerifier(), settings)
    assert record["status"] == "input_modified"
    assert generator.calls == []


def test_candidate_modification_cannot_be_retained(task, settings):
    def mutate(path):
        path.chmod(0o600)
        path.write_text(OTHER_CODE)

    record = run_task(task, Client([completion()]), reflector(), Verifier(hook=mutate), settings)
    assert record["status"] == "input_modified"
    assert not record["accepted_candidates"]


def test_reference_modification_cannot_be_retained(task, settings):
    record = run_task(
        task,
        Client([completion()]),
        reflector(),
        Verifier(hook=lambda path: task.case_provider_path.write_text("changed")),
        settings,
    )
    assert record["status"] == "input_modified"
    assert not record["accepted_candidates"]


def test_records_preserve_actual_case_counts_and_model_metadata(task, settings):
    record = run_task(task, Client([completion()]), reflector(), Verifier(), settings)
    assert record["reference"]["correctness_count"] == 2
    verification = record["attempts"][0]["verification"]
    assert verification["correctness_case_count"] == 2
    assert verification["performance_case_count"] == 1
    assert len(verification["correctness_cases"]) == 2
    assert record["attempts"][0]["generation"]["usage"] == {"completion_tokens": 12}
    assert record["provenance"] == task.provenance


@pytest.mark.parametrize("identifier", ["..", "../escape", "/absolute", "a/b", "a\\b", "", "."])
def test_unsafe_task_identifiers_fail_before_calls(task, settings, identifier):
    generator = Client([])
    with pytest.raises(ValueError):
        run_task(replace(task, task_id=identifier), generator, reflector(), Verifier(), settings)
    assert not generator.calls


@pytest.mark.parametrize(
    "changes",
    [
        {"max_attempts": 0},
        {"max_attempts": True},
        {"num_variants": 0},
        {"num_variants": 5},
        {"temperature": float("nan")},
        {"max_tokens": -1},
        {"history_char_limit": 0},
    ],
)
def test_invalid_settings_fail_before_calls(task, settings, changes):
    with pytest.raises(ValueError):
        run_task(task, Client([]), reflector(), Verifier(), replace(settings, **changes))


def test_caller_can_choose_directory_under_checkout(task, settings, tmp_path):
    (tmp_path / ".git").mkdir()
    record = run_task(task, Client([completion()]), reflector(), Verifier(), settings)
    assert record["status"] == "success"


def test_input_file_cannot_be_an_output_directory(task, settings):
    with pytest.raises(ValueError):
        run_task(
            task,
            Client([]),
            reflector(),
            Verifier(),
            replace(settings, output_dir=task.functional_path),
        )
    assert task.functional_path.read_text() == CODE


def test_task_output_symlink_fails_before_generation(task, settings, tmp_path):
    settings.output_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (settings.output_dir / task.task_id).symlink_to(outside, target_is_directory=True)
    generator = Client([])
    with pytest.raises(ValueError, match="symbolic link"):
        run_task(task, generator, reflector(), Verifier(), settings)
    assert generator.calls == []
    assert list(outside.iterdir()) == []


def test_atomic_json_keeps_prior_record_when_replace_fails(tmp_path, monkeypatch):
    path = tmp_path / "record.json"
    path.write_text('{"prior": true}\n')

    def fail_replace(*args):
        raise OSError("disk error")

    monkeypatch.setattr("triton_kernel_gen.pipeline.os.replace", fail_replace)
    with pytest.raises(RecordWriteError):
        _atomic_json(path, {"next": True})
    assert json.loads(path.read_text()) == {"prior": True}
    assert list(tmp_path.glob("*.tmp")) == []
