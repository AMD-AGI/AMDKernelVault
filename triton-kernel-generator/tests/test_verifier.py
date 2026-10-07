# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Exercise real subprocess boundaries with a CPU-only GPU runtime double."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest

from triton_kernel_gen.contracts import TaskSpec, VerificationSettings
from triton_kernel_gen.verifier import ReferenceValidationError, Verifier

_CPU_RUNTIME = """
import sys
import types
import torch
import triton_kernel_gen.worker as worker

triton = types.ModuleType('triton')
runtime = types.ModuleType('triton.runtime')
jit_module = types.ModuleType('triton.runtime.jit')
class JITFunction:
    def __init__(self, function):
        self.fn = function
    def run(self, *args, **kwargs):
        kwargs.pop('warmup', None)
        return self.fn(*args, **kwargs)
    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.run(*args, **kwargs)
jit_module.JITFunction = JITFunction
triton.jit = JITFunction
sys.modules.update({'triton': triton, 'triton.runtime': runtime, 'triton.runtime.jit': jit_module})
worker.runtime_environment = lambda settings: ({
    'execution': 'CPU runtime double, not GPU evidence',
    'triton': '3.3.0', 'target_arch': 'gfx942', 'hip': 'mock',
}, torch.device('cpu'))
def bench(fn, settings):
    fn()
    fn()
    return 2.0 if sys.argv[1] == 'benchmark' else 1.0
worker.bench_ms = bench
raise SystemExit(worker.main())
"""

_ORIGINAL = """
import torch
class Model(torch.nn.Module):
    def forward(self, x):
        return x + 1
"""

_FUNCTIONAL = """
import torch
def module_fn(x):
    return x + 1
class Model(torch.nn.Module):
    def forward(self, x, fn=module_fn):
        return fn(x)
"""

_PROVIDER = """
import torch
from triton_kernel_gen import Case
def get_cases(task_id, module, *, kind):
    assert task_id == 'add_one'
    return [Case(
        kind + str(index), args=(torch.arange(12.0).reshape(3, 4)[1:, ::2],),
        metadata={'source': 'external synthetic test', 'index': index}
    ) for index in range(2 if kind == 'correctness' else 1)]
"""

_CANDIDATE = """
import torch
import triton
@triton.jit
def kernel(x, output):
    output.copy_(x + 1)
def module_fn(x):
    output = torch.empty_like(x)
    kernel[(1,)](x, output)
    return output
"""


class CPUVerifier(Verifier):
    """Replace GPU mechanics inside child processes, without changing validation."""

    def _worker_command(self, mode, request_path):
        return [sys.executable, "-c", _CPU_RUNTIME, mode, str(request_path)]


def _write(path: Path, source: str) -> Path:
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


@pytest.fixture
def task(tmp_path):
    return TaskSpec(
        "add_one",
        _write(tmp_path / "original.py", _ORIGINAL),
        _write(tmp_path / "functional.py", _FUNCTIONAL),
        _write(tmp_path / "provider.py", _PROVIDER),
    )


@pytest.fixture
def verifier():
    return CPUVerifier(VerificationSettings(timeout_seconds=30))


def test_real_subprocess_pipeline_freezes_cases_and_accepts_slow_correct_candidate(
    task, verifier, tmp_path
):
    prepared = verifier.prepare(task, tmp_path / "verification")
    assert prepared.reference_record["correctness_count"] == 2
    assert prepared.reference_record["performance_count"] == 1
    first = prepared.reference_record["cases"][0]
    assert first["inputs"]["tensors"][0]["stride"] == [4, 2]
    assert first["inputs"]["tensors"][0]["storage_offset"] == 4
    assert first["metadata"]["source"] == "external synthetic test"
    # Public record edits cannot change the private authority or tolerance.
    prepared.reference_record["cases"][0]["atol"] = 10000
    candidate = _write(tmp_path / "candidate.py", _CANDIDATE)
    result = verifier.verify(prepared, candidate)
    assert result.success
    assert result.runtime_success
    assert result.performance_cases[0]["baseline_ms"] == 1.0
    assert result.performance_cases[0]["candidate_ms"] == 2.0
    assert result.performance_cases[0]["speedup"] == 0.5
    assert result.correctness_cases[0]["atol"] == 1e-3
    assert "CPU runtime double" in result.environment["execution"]
    requests = list(prepared.work_dir.glob("candidate-*/request.json"))
    assert len(requests) == 2
    for request in requests:
        payload = json.loads(request.read_text())
        assert "task" not in payload
        assert "expected" not in payload
        assert "atol" not in payload["cases"][0]
        assert not (request.parent / "expected").exists()


def test_numerical_failure_never_starts_timing(task, verifier, tmp_path):
    prepared = verifier.prepare(task, tmp_path / "verification")
    candidate = _write(tmp_path / "candidate.py", _CANDIDATE.replace("x + 1", "x + 10"))
    result = verifier.verify(prepared, candidate)
    assert result.compile_success and result.runtime_success
    assert not result.correctness_success and not result.timing_success
    assert result.failure_stage == "correctness"
    assert not list(prepared.work_dir.glob("candidate-benchmark-*"))


def test_seeded_random_output_replays_after_benchmark(task, verifier, tmp_path):
    task.module_path.write_text(_ORIGINAL.replace("x + 1", "x + torch.rand_like(x)"))
    task.functional_path.write_text(_FUNCTIONAL.replace("x + 1", "x + torch.rand_like(x)"))
    candidate = _write(
        tmp_path / "candidate.py", _CANDIDATE.replace("x + 1", "x + torch.rand_like(x)")
    )
    prepared = verifier.prepare(task, tmp_path / "verification")
    result = verifier.verify(prepared, candidate)
    assert result.success
    assert result.performance_cases[0]["timed_comparison"]["success"]


def test_environment_changes_invalidate_the_baseline(task, tmp_path):
    class ChangedEnvironment(CPUVerifier):
        def _worker_command(self, mode, request_path):
            script = _CPU_RUNTIME.replace(
                "'target_arch': 'gfx942'",
                "'target_arch': ('gfx942' if sys.argv[1] == 'reference' else 'gfx950')",
            )
            return [sys.executable, "-c", script, mode, str(request_path)]

    verifier = ChangedEnvironment(VerificationSettings(timeout_seconds=30))
    prepared = verifier.prepare(task, tmp_path / "verification")
    result = verifier.verify(prepared, _write(tmp_path / "candidate.py", _CANDIDATE))
    assert not result.success
    assert result.failure_stage == "infrastructure"
    assert "target_arch" in result.message


@pytest.mark.parametrize("changed_mode", ["execute", "benchmark"])
def test_parent_checks_native_devices_before_and_after_timing(task, tmp_path, changed_mode):
    class ChangedOutputDevice(CPUVerifier):
        def _worker_command(self, mode, request_path):
            runtime = _CPU_RUNTIME
            if mode == changed_mode:
                # Only the native device descriptors use a GPU runtime double.
                injection = """
original_metadata = worker.tree_metadata
def metadata(value):
    result = original_metadata(value)
    for tensor in result['tensors']:
        tensor['device'] = 'cuda:0'
    for storage in result['storages']:
        storage['device'] = 'cuda:0'
    return result
worker.tree_metadata = metadata
raise SystemExit(worker.main())
"""
                runtime = runtime.replace("raise SystemExit(worker.main())", injection)
            return [sys.executable, "-c", runtime, mode, str(request_path)]

    verifier = ChangedOutputDevice(VerificationSettings(timeout_seconds=30))
    prepared = verifier.prepare(task, tmp_path / "verification")
    result = verifier.verify(prepared, _write(tmp_path / "candidate.py", _CANDIDATE))
    assert result.compile_success and result.runtime_success
    assert not result.correctness_success
    assert result.failure_stage == "correctness"
    assert "native device mismatch" in result.message


def test_stage1_failure_occurs_before_any_candidate_worker(task, verifier, tmp_path):
    task.functional_path.write_text(_FUNCTIONAL.replace("x + 1", "x + 2"))
    with pytest.raises(ReferenceValidationError, match="default functional reference differs"):
        verifier.prepare(task, tmp_path / "verification")
    assert not list((tmp_path / "verification").glob("candidate-*"))


@pytest.mark.parametrize("modified", ["source", "expected", "source_deleted"])
def test_frozen_source_and_expected_files_cannot_change(task, verifier, tmp_path, modified):
    prepared = verifier.prepare(task, tmp_path / "verification")
    if modified == "source":
        task.case_provider_path.write_text("raise RuntimeError('changed')")
    elif modified == "source_deleted":
        task.module_path.unlink()
    else:
        root = Path(prepared.state["reference_artifacts"])
        (root / "expected" / "correctness_0" / "manifest.json").write_text("{}")
    result = verifier.verify(prepared, _write(tmp_path / "candidate.py", _CANDIDATE))
    assert not result.success
    assert result.failure_stage == "input_modified"
    assert not list(prepared.work_dir.glob("candidate-*"))


def test_baseline_kernel_sets_measured_baseline_before_generation(task, verifier, tmp_path):
    from dataclasses import replace

    baseline = _write(tmp_path / "baseline.py", _CANDIDATE)
    prepared = verifier.prepare(
        replace(task, baseline_kernel_path=baseline), tmp_path / "verification"
    )
    performance = [
        case for case in prepared.reference_record["cases"] if case["kind"] == "performance"
    ]
    assert performance[0]["baseline_kind"] == "baseline_kernel"
    assert performance[0]["reference_ms"] == 1.0
    assert performance[0]["baseline_ms"] == performance[0]["baseline_kernel_ms"] == 2.0
    assert prepared.reference_record["baseline_validation"]["runtime_success"]


def test_invalid_baseline_kernel_rejects_preparation(task, verifier, tmp_path):
    from dataclasses import replace

    baseline = _write(tmp_path / "baseline.py", _CANDIDATE.replace("x + 1", "x - 1"))
    with pytest.raises(ReferenceValidationError, match="baseline kernel failed"):
        verifier.prepare(replace(task, baseline_kernel_path=baseline), tmp_path / "verification")


def test_raw_seed_kernel_is_context_and_does_not_execute(task, verifier, tmp_path):
    from dataclasses import replace

    seed = _write(
        tmp_path / "cuda_seed.py", "raise RuntimeError('This CUDA source cannot run on AMD.')"
    )
    prepared = verifier.prepare(replace(task, seed_kernel_path=seed), tmp_path / "verification")
    assert str(seed.resolve()) in prepared.reference_record["source_hashes"]
    assert not list(prepared.work_dir.glob("candidate-*"))
    performance = [
        case for case in prepared.reference_record["cases"] if case["kind"] == "performance"
    ]
    assert performance[0]["baseline_kind"] == "functional_reference"


def test_declared_helper_directory_works_in_oracle(task, verifier, tmp_path):
    from dataclasses import replace

    directory = tmp_path / "helpers"
    directory.mkdir()
    helper = _write(directory / "offset_helper.py", "OFFSET = 1")
    task.functional_path.write_text(
        "from offset_helper import OFFSET\n" + _FUNCTIONAL.replace("x + 1", "x + OFFSET")
    )
    prepared = verifier.prepare(
        replace(task, dependency_paths=(helper,)), tmp_path / "verification"
    )
    assert str(helper.resolve()) in prepared.reference_record["source_hashes"]
    helper.write_text("OFFSET = 2")
    result = verifier.verify(prepared, _write(tmp_path / "candidate.py", _CANDIDATE))
    assert result.failure_stage == "input_modified"


def test_package_relative_helpers_remain_available_during_reference_calls(task, verifier, tmp_path):
    from dataclasses import replace

    package = tmp_path / "external_package"
    package.mkdir()
    initializer = _write(package / "__init__.py", "# External package fixture.\n")
    helper = _write(package / "operation.py", "def add_one(x):\n    return x + 1\n")
    original = _write(
        package / "original.py",
        """
        import torch
        class Model(torch.nn.Module):
            def forward(self, x):
                from .operation import add_one
                return add_one(x)
    """,
    )
    functional = _write(
        package / "functional.py",
        """
        import torch
        def module_fn(x):
            from .operation import add_one
            return add_one(x)
        class Model(torch.nn.Module):
            def forward(self, x, fn=module_fn):
                return fn(x)
    """,
    )
    task = replace(
        task,
        module_path=original,
        functional_path=functional,
        dependency_paths=(initializer, helper),
    )
    prepared = verifier.prepare(task, tmp_path / "verification")
    assert prepared.reference_record["stage1_success"]
    result = verifier.verify(prepared, _write(tmp_path / "candidate.py", _CANDIDATE))
    assert result.success
    assert not list(package.rglob("*.pyc"))


def test_worker_scrubs_credentials_and_uses_empty_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    monkeypatch.setenv("CUSTOM_SECRET", "must-not-leak")
    monkeypatch.setenv("CPATH", "excluded-even-when-allowlisted")

    class InspectEnvironment(Verifier):
        def _worker_command(self, mode, request_path):
            script = """
import json, os
from pathlib import Path
Path('report.json').write_text(json.dumps({'success': True, 'environment': dict(os.environ)}))
"""
            return [sys.executable, "-c", script]

    inspector = InspectEnvironment(
        VerificationSettings(timeout_seconds=10, excluded_environment=("CPATH",))
    )
    root = tmp_path / "worker"
    report = inspector._run_worker("reference", root, {})
    environment = report["environment"]
    assert "OPENAI_API_KEY" not in environment
    assert "CUSTOM_SECRET" not in environment
    assert "CPATH" not in environment
    assert environment["HOME"] == str(root / "home")
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_worker_timeout_kills_its_process_group_and_preserves_stage(tmp_path):
    class SleepingWorker(Verifier):
        def _worker_command(self, mode, request_path):
            script = """
import json, time
from pathlib import Path
Path('report.json').write_text(json.dumps({'success': False, 'compile_success': True, 'phase': 'runtime'}))
time.sleep(60)
"""
            return [sys.executable, "-c", script]

    verifier = SleepingWorker(VerificationSettings(timeout_seconds=0.3))
    report = verifier._run_worker("execute", tmp_path / "worker", {})
    assert report["timed_out"]
    assert report["compile_success"]
    assert report["failure_stage"] == "runtime"
    assert report["worker_exit_code"] < 0


def test_symlink_report_is_rejected(tmp_path):
    class SymlinkWorker(Verifier):
        def _worker_command(self, mode, request_path):
            script = """
from pathlib import Path
Path('other.json').write_text('{"success": true}')
Path('report.json').symlink_to('other.json')
"""
            return [sys.executable, "-c", script]

    report = SymlinkWorker(VerificationSettings())._run_worker("execute", tmp_path / "worker", {})
    assert not report["success"]
    assert report["failure_stage"] == "infrastructure"


def test_fifo_report_is_rejected_without_waiting_for_a_writer(tmp_path):
    class FifoWorker(Verifier):
        def _worker_command(self, mode, request_path):
            return [sys.executable, "-c", "import os; os.mkfifo('report.json')"]

    report = FifoWorker(VerificationSettings(timeout_seconds=2))._run_worker(
        "execute", tmp_path / "worker", {}
    )
    assert not report["success"]
    assert report["failure_stage"] == "infrastructure"


@pytest.mark.parametrize(
    "field,value",
    [
        ("atol", float("nan")),
        ("rtol", -1),
        ("warmup_ms", -1),
        ("measure_ms", 0),
        ("timeout_seconds", 0),
        ("gpu", True),
        ("seed", -1),
        ("target_arch", "sm90"),
    ],
)
def test_invalid_verification_settings_fail_before_worker(field, value):
    with pytest.raises(ValueError):
        Verifier(VerificationSettings(**{field: value}))
