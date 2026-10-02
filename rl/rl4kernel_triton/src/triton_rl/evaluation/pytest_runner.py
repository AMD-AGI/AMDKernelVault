# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Seed one pytest process and save its status separately from its output."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest


def install_timing_aliases() -> None:
    """Support the historical imports used by external reference tests."""
    from triton_rl.evaluation import timing

    sys.modules["performance_utils_pytest"] = timing
    for prefix in ("tb_eval", "geak_eval"):
        parent = None
        for name in (prefix, f"{prefix}.perf", f"{prefix}.perf.ROCm"):
            module = sys.modules.get(name)
            if module is None:
                module = ModuleType(name)
                module.__path__ = []
                sys.modules[name] = module
            if parent is not None:
                setattr(parent, name.rsplit(".", 1)[-1], module)
            parent = module
        setattr(parent, "performance_utils_pytest", timing)
        sys.modules[f"{prefix}.perf.ROCm.performance_utils_pytest"] = timing


def seed_process(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Report:
    def __init__(self, seed: int):
        self.seed = seed
        self.collected = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.collection_errors = 0
        self.exceptions: list[str] = []

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item):
        seed_process(self.seed)

    def pytest_collection_finish(self, session):
        self.collected = len(session.items)

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors += 1

    def pytest_runtest_makereport(self, item, call):
        if call.excinfo is not None:
            self.exceptions.append(call.excinfo.type.__name__)

    def pytest_runtest_logreport(self, report):
        if report.failed:
            self.failed += 1
        elif report.skipped:
            self.skipped += 1
        elif report.when == "call" and report.passed:
            self.passed += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference-parent", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--mode", choices=("correctness", "performance"), required=True)
    args = parser.parse_args()
    sys.path.append(str(args.reference_parent))
    install_timing_aliases()
    seed_process(args.seed)
    report = Report(args.seed)
    selection = [str(args.file)]
    if args.mode == "correctness":
        selection += ["-k", "not test_performance and not test_save_performance_results"]
    else:
        selection = [
            f"{args.file}::test_performance",
            f"{args.file}::test_save_performance_results",
        ]
    args.report.with_suffix(".started.json").write_text(
        '{"execution_started": true}', encoding="utf-8"
    )
    code = pytest.main(
        [
            *selection,
            "-q",
            "-s",
            "-p",
            "no:cacheprovider",
            "-c",
            "/dev/null",
            "--rootdir",
            str(args.file.parent),
            "--confcutdir",
            str(args.file.parent),
            "-o",
            "addopts=",
        ],
        plugins=[report],
    )
    counters = {name: value for name, value in vars(report).items() if name != "seed"}
    args.report.write_text(json.dumps({**counters, "exit_code": int(code)}), encoding="utf-8")
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
