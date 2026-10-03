# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Check task loading and the zero-cost command preview."""

import json
from unittest.mock import patch

import pytest

from triton_kernel_gen.cli import main
from triton_kernel_gen.tasks import load_tasks


def manifest(tmp_path):
    for name in ("module.py", "functional.py", "cases.py", "helper.py"):
        (tmp_path / name).write_text(
            "raise RuntimeError('This input must not execute during a preview.')\n"
        )
    value = {
        "schema_version": 1,
        "tasks": [
            {
                "task_id": "example",
                "module": "module.py",
                "functional": "functional.py",
                "case_provider": "cases.py",
                "dependencies": ["helper.py"],
                "provenance": {"source": "caller-supplied"},
            }
        ],
    }
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps(value))
    return path, value


def arguments(path, tmp_path):
    return [
        "--manifest",
        str(path),
        "--output-dir",
        str(tmp_path / "out"),
        "--artifacts-dir",
        str(tmp_path / "artifacts"),
        "--endpoint",
        "http://model-host/v1/chat/completions",
        "--model-id",
        "served-model",
        "--model-family",
        "gpt-oss-120b",
        "--max-attempts",
        "5",
    ]


def test_manifest_paths_and_declared_dependencies(tmp_path):
    path, _ = manifest(tmp_path)
    tasks = load_tasks(path)
    assert tasks[0].module_path == tmp_path / "module.py"
    assert tasks[0].dependency_paths == (tmp_path / "helper.py",)
    assert tasks[0].provenance == {"source": "caller-supplied"}


def test_preview_does_not_execute_inputs_or_contact_provider(tmp_path, capsys):
    path, _ = manifest(tmp_path)
    with patch("requests.Session.post", side_effect=AssertionError("No network during preview.")):
        assert main(arguments(path, tmp_path) + ["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["maximum_model_requests"] == 9
    assert plan["verification"]["rtol"] == 1e-4
    assert plan["verification"]["atol"] == 1e-3
    assert plan["verification"]["warmup_ms"] == 25
    assert plan["verification"]["measure_ms"] == 100
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", ".hidden", "x" * 129])
def test_unsafe_task_ids_rejected(tmp_path, bad_id):
    path, value = manifest(tmp_path)
    value["tasks"][0]["task_id"] = bad_id
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_tasks(path)


def test_duplicate_task_ids_rejected(tmp_path):
    path, value = manifest(tmp_path)
    value["tasks"].append(value["tasks"][0])
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="repeats"):
        load_tasks(path)


def test_boolean_manifest_version_rejected(tmp_path):
    path, value = manifest(tmp_path)
    value["schema_version"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="schema_version"):
        load_tasks(path)


def test_invalid_budget_fails_before_execution(tmp_path, capsys):
    path, _ = manifest(tmp_path)
    assert main(arguments(path, tmp_path) + ["--num-variants", "6"]) == 2
    assert "variant count" in capsys.readouterr().err


def test_container_identity_is_recorded_when_supplied(tmp_path, capsys, monkeypatch):
    path, _ = manifest(tmp_path)
    monkeypatch.setenv("TRITON_GEN_CONTAINER_REFERENCE", "example-image:local")
    monkeypatch.setenv("TRITON_GEN_CONTAINER_IMAGE_ID", "sha256:example")
    assert main(arguments(path, tmp_path) + ["--dry-run"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["container"] == {"reference": "example-image:local", "image_id": "sha256:example"}


def test_cli_does_not_contain_a_default_three_attempt_limit(tmp_path, capsys):
    path, _ = manifest(tmp_path)
    args = arguments(path, tmp_path)
    del args[-2:]
    with pytest.raises(SystemExit) as error:
        main(args + ["--dry-run"])
    assert error.value.code == 2
