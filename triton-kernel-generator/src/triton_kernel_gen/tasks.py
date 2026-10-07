# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Read external task manifests without preparing or modifying datasets."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .contracts import TaskSpec

TASK_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def validate_task(task: TaskSpec) -> TaskSpec:
    if not isinstance(task.task_id, str) or TASK_ID.fullmatch(task.task_id) is None:
        raise ValueError(
            "The task ID must start with a letter or digit and use letters, digits, dots, underscores, or hyphens."
        )
    paths = [
        task.module_path,
        task.functional_path,
        task.case_provider_path,
        *task.dependency_paths,
    ]
    if task.seed_kernel_path:
        paths.append(task.seed_kernel_path)
    if task.baseline_kernel_path:
        paths.append(task.baseline_kernel_path)
    for path in paths:
        if not path.is_file() or path.suffix != ".py":
            raise ValueError(f"The external code path must be an existing Python file: {path}")
    if task.module_path.resolve() == task.functional_path.resolve():
        raise ValueError(
            "The original module and standardized functional reference must be separate files."
        )
    if not isinstance(task.provenance, dict):
        raise ValueError("Task provenance must be a JSON object.")
    json.dumps(task.provenance, allow_nan=False)
    return task


def load_tasks(manifest: Path) -> list[TaskSpec]:
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
    ):
        raise ValueError("The task manifest requires schema_version=1.")
    rows = value.get("tasks")
    if not isinstance(rows, list) or not rows:
        raise ValueError("The task manifest requires a nonempty tasks list.")
    base = manifest.resolve().parent

    def path(raw: object) -> Path:
        if not isinstance(raw, str) or not raw:
            raise ValueError("Each task code path must be a nonempty string.")
        item = Path(raw).expanduser()
        return (base / item).resolve() if not item.is_absolute() else item.resolve()

    tasks = []
    identifiers = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each task entry must be a JSON object.")
        try:
            dependencies = row.get("dependencies", [])
            if not isinstance(dependencies, list):
                raise ValueError("Task dependencies must be a list of Python paths.")
            task = TaskSpec(
                task_id=row["task_id"],
                module_path=path(row["module"]),
                functional_path=path(row["functional"]),
                case_provider_path=path(row["case_provider"]),
                seed_kernel_path=path(row["seed_kernel"]) if row.get("seed_kernel") else None,
                provenance=row.get("provenance", {}),
                dependency_paths=tuple(path(item) for item in dependencies),
                baseline_kernel_path=path(row["baseline_kernel"])
                if row.get("baseline_kernel")
                else None,
            )
        except KeyError as exc:
            raise ValueError(f"The task entry is missing {exc.args[0]}.") from None
        validate_task(task)
        if task.task_id in identifiers:
            raise ValueError(f"The task manifest repeats task ID {task.task_id}.")
        identifiers.add(task.task_id)
        tasks.append(task)
    return tasks
