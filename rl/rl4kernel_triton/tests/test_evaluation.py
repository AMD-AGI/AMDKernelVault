# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify path controls, pytest failures, seeds, and worker cleanup on the CPU."""

import asyncio
import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
from aiohttp import web

from triton_rl.contracts import EvaluationResult
from triton_rl.evaluation.protocol import (
    REFERENCE_SEPARATOR,
    ProtocolError,
    resolve_reference,
    validate_request,
)
from triton_rl.evaluation.server import InfrastructureError, ServerConfig, create_app, run_worker
from triton_rl.evaluation.verification import ReferenceError
from triton_rl.evaluation.worker import CandidateFailure, _run_stage, load_outputs, prepare_sources


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="triton-rl-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_reference_paths_reject_escape_and_symlinks(self):
        reference_root = self.root / "references"
        reference_root.mkdir()
        (self.root / "outside.py").write_text("pass")
        (reference_root / "inside.py").write_text("pass")
        (reference_root / "link.py").symlink_to(self.root / "outside.py")
        self.assertEqual(
            resolve_reference(reference_root, "inside.py"), reference_root / "inside.py"
        )
        for filename in (
            "../outside.py",
            "/outside.py",
            "link.py",
            "a\\b.py",
            "a//b.py",
            "./inside.py",
            "inside.txt",
            "missing.py",
        ):
            with self.subTest(filename=filename), self.assertRaises(ProtocolError):
                resolve_reference(reference_root, filename)

    def test_request_validates_tolerances_without_override(self):
        request = {
            "protocol_version": 1,
            "code": "pass",
            "filename": "a.py",
            "atol": 1e-6,
            "rtol": 2e-5,
        }
        self.assertEqual(validate_request(request)["atol"], 1e-6)
        self.assertEqual(validate_request(request)["rtol"], 2e-5)
        for value in (True, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ProtocolError):
                validate_request({**request, "atol": value})

    def test_source_uses_only_external_tests(self):
        reference = self.root / "external.py"
        reference.write_text(
            "def kernel():\n    return 1\n"
            + REFERENCE_SEPARATOR
            + "\ndef test_external():\n    assert kernel() == 1\n"
        )
        generated = "def kernel():\n    return 2\ndef test_cheat():\n    pass\ntest_cheat()\n"
        reference_path, candidate_path = prepare_sources(reference, generated, self.root)
        self.assertIn("def kernel():\n    return 1", reference_path.read_text())
        self.assertIn("_triton_rl_seed_process(42)", reference_path.read_text())
        self.assertIn("test_external", candidate_path.read_text())
        self.assertNotIn("test_cheat", candidate_path.read_text())

    def test_missing_test_section_is_reference_failure(self):
        reference = self.root / "external.py"
        reference.write_text("pass\n")
        with self.assertRaises(ReferenceError):
            prepare_sources(reference, "pass", self.root)

    def test_candidate_syntax_failure_has_no_compilation_reward(self):
        reference = self.root / "external.py"
        reference.write_text("pass\n" + REFERENCE_SEPARATOR + "\ndef test_reference():\n    pass\n")
        with self.assertRaises(CandidateFailure) as failure:
            prepare_sources(reference, "def broken(:", self.root)
        self.assertFalse(failure.exception.result.compiled)
        self.assertFalse(failure.exception.result.correct)

    def _stage(self, *, reference=False, mode="correctness"):
        path = self.root / "test_kernel.py"
        path.write_text("pass")
        return _run_stage(
            path,
            reference_parent=self.root,
            mode=mode,
            reference=reference,
            seed=42,
            deadline=time.monotonic() + 30,
            perf_dir=self.root / "perf",
        )

    def test_candidate_timeout_and_reference_timeout_differ(self):
        with patch(
            "triton_rl.evaluation.worker.subprocess.run",
            side_effect=subprocess.TimeoutExpired("pytest", 1),
        ):
            with self.assertRaises(CandidateFailure) as failure:
                self._stage()
            self.assertEqual(failure.exception.result.error_type, "timeout")
            with self.assertRaises(ReferenceError):
                self._stage(reference=True)

    def test_pytest_exit_codes_override_saved_output_claims(self):
        def run(*args, **kwargs):
            (self.root / "correctness_report.json").write_text(
                json.dumps(
                    {
                        "collected": 1,
                        "passed": 0,
                        "failed": 1,
                        "skipped": 0,
                        "collection_errors": 0,
                        "exceptions": ["AssertionError"],
                        "exit_code": 1,
                    }
                )
            )
            return SimpleNamespace(returncode=1)

        with patch("triton_rl.evaluation.worker.subprocess.run", side_effect=run):
            with self.assertRaises(CandidateFailure) as failure:
                self._stage()
            self.assertTrue(failure.exception.result.compiled)
            self.assertFalse(failure.exception.result.correct)
            with self.assertRaises(ReferenceError):
                self._stage(reference=True)

    def test_no_tests_or_skipped_tests_cannot_succeed(self):
        for collected, passed, skipped, exit_code in ((0, 0, 0, 5), (1, 0, 1, 0)):

            def run(*args, **kwargs):
                (self.root / "correctness_report.json").write_text(
                    json.dumps(
                        {
                            "collected": collected,
                            "passed": passed,
                            "failed": 0,
                            "skipped": skipped,
                            "collection_errors": 0,
                            "exceptions": [],
                            "exit_code": exit_code,
                        }
                    )
                )
                return SimpleNamespace(returncode=exit_code)

            with (
                self.subTest(exit_code=exit_code),
                patch("triton_rl.evaluation.worker.subprocess.run", side_effect=run),
            ):
                with self.assertRaises(ReferenceError):
                    self._stage()

    def test_native_candidate_crash_does_not_become_infrastructure_error(self):
        def run(*args, **kwargs):
            (self.root / "correctness_report.started.json").write_text(
                '{"execution_started": true}'
            )
            return SimpleNamespace(returncode=-11)

        with patch("triton_rl.evaluation.worker.subprocess.run", side_effect=run):
            with self.assertRaises(CandidateFailure) as failure:
                self._stage()
            self.assertEqual(failure.exception.result.error_type, "execution_error")
            with self.assertRaises(ReferenceError):
                self._stage(reference=True)

    def test_candidate_exit_after_runner_start_is_a_candidate_failure(self):
        def run(*args, **kwargs):
            (self.root / "correctness_report.started.json").write_text(
                '{"execution_started": true}'
            )
            return SimpleNamespace(returncode=11)

        with patch("triton_rl.evaluation.worker.subprocess.run", side_effect=run):
            with self.assertRaises(CandidateFailure) as failure:
                self._stage()
            self.assertFalse(failure.exception.result.compiled)
            with self.assertRaises(ReferenceError):
                self._stage(reference=True)

    def test_separate_pytest_processes_receive_identical_seeds(self):
        import torch

        source = """import random
import numpy as np
import torch
from tb_eval.perf.ROCm.performance_utils_pytest import do_bench_config

def test_output():
    assert do_bench_config().warm_up == 25
    value = torch.rand(3) + random.random() + np.random.random()
    torch.save({'_CALL_SUCCESS_': torch.tensor(True), 'value': value}, __file__.replace('.', '_') + '.pt')
"""
        outputs = []
        for name in ("reference", "candidate"):
            folder = self.root / name
            folder.mkdir()
            path = folder / "test_kernel.py"
            path.write_text(source)
            report = _run_stage(
                path,
                reference_parent=self.root,
                mode="correctness",
                reference=True,
                seed=17,
                deadline=time.monotonic() + 60,
                perf_dir=folder / "perf",
            )
            self.assertEqual(report["passed"], 1)
            outputs.append(load_outputs(path, reference=True))
        torch.testing.assert_close(outputs[0]["value"], outputs[1]["value"], atol=0, rtol=0)

    def test_candidate_initialization_cannot_shift_external_test_seeds(self):
        import torch

        reference = self.root / "external.py"
        tests = """
import random
import numpy as np
module_value = torch.rand(3)

def test_output():
    value = torch.rand(3) + random.random() + np.random.random()
    torch.save({'_CALL_SUCCESS_': torch.tensor(True), 'module_value': module_value, 'value': value}, __file__.replace('.', '_') + '.pt')
"""
        reference.write_text("import torch\n" + REFERENCE_SEPARATOR + tests)
        paths = prepare_sources(
            reference, "import torch\nnoise = torch.rand(20)\n", self.root, seed=17
        )
        outputs = []
        for path in paths:
            _run_stage(
                path,
                reference_parent=self.root,
                mode="correctness",
                reference=True,
                seed=17,
                deadline=time.monotonic() + 60,
                perf_dir=path.parent / "perf",
            )
            outputs.append(load_outputs(path, reference=True))
        for key in ("module_value", "value"):
            torch.testing.assert_close(outputs[0][key], outputs[1][key], atol=0, rtol=0)


class SupervisorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="triton-rl-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "task.py").write_text("pass")
        self.config = ServerConfig(self.root, gpu=3, work_root=self.root)
        self.request = {
            "protocol_version": 1,
            "code": "pass",
            "filename": "task.py",
            "atol": 0,
            "rtol": 0,
        }

    def test_relative_configuration_paths_become_absolute(self):
        relative = Path(os.path.relpath(self.root))
        config = ServerConfig(relative, gpu=0, work_root=relative)
        self.assertEqual(config.work_root, self.root)
        self.assertEqual(config.reference_root, self.root)

    async def test_supervisor_uses_json_gpu_assignment_and_unique_directory(self):
        captured = []

        async def start(*args, **kwargs):
            captured.append((args, kwargs))
            result_path = Path(args[args.index("--result") + 1])
            request_path = Path(args[args.index("--request") + 1])
            self.assertEqual(json.loads(request_path.read_text())["request"], self.request)
            result_path.write_text(json.dumps({"result": EvaluationResult(True, False).to_dict()}))
            return SimpleNamespace(pid=12345, returncode=0, wait=AsyncMock(return_value=0))

        with (
            patch("triton_rl.evaluation.server.asyncio.create_subprocess_exec", side_effect=start),
            patch("triton_rl.evaluation.server._kill_group") as kill,
        ):
            await run_worker(self.config, self.request)
            await run_worker(self.config, self.request)
        self.assertNotEqual(captured[0][1]["cwd"], captured[1][1]["cwd"])
        self.assertFalse(captured[0][1]["cwd"].exists())
        self.assertEqual(captured[0][1]["env"]["ROCR_VISIBLE_DEVICES"], "3")
        self.assertTrue(captured[0][1]["start_new_session"])
        self.assertEqual(kill.call_count, 2)

    async def test_supervisor_timeout_kills_process_group_and_cleans_files(self):
        process = SimpleNamespace(pid=12345, returncode=0, wait=AsyncMock(return_value=0))

        async def timeout(awaitable, **kwargs):
            await awaitable
            raise asyncio.TimeoutError()

        with (
            patch(
                "triton_rl.evaluation.server.asyncio.create_subprocess_exec", return_value=process
            ),
            patch(
                "triton_rl.evaluation.server.asyncio.wait_for",
                side_effect=timeout,
            ),
            patch("triton_rl.evaluation.server._kill_group") as kill,
        ):
            with self.assertRaises(InfrastructureError):
                await run_worker(self.config, self.request)
        kill.assert_called_once_with(12345)
        self.assertEqual(list(self.root.iterdir()), [self.root / "task.py"])

    async def test_server_serializes_all_clients(self):
        active = 0
        maximum = 0

        async def evaluate(config, request):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            return {"protocol_version": 1, "result": EvaluationResult(True, False).to_dict()}

        runner = web.AppRunner(create_app(self.config, evaluator=evaluate))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/evaluate"
        try:
            async with aiohttp.ClientSession() as session:

                async def post():
                    async with session.post(url, json=self.request) as response:
                        return response.status

                self.assertEqual(await asyncio.gather(post(), post()), [200, 200])
            self.assertEqual(maximum, 1)
        finally:
            await runner.cleanup()

    async def test_http_disconnect_cancels_running_evaluation(self):
        started, cancelled = asyncio.Event(), asyncio.Event()
        calls = 0

        async def evaluate(config, request):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    cancelled.set()
                    raise
            return {"protocol_version": 1, "result": EvaluationResult(True, False).to_dict()}

        runner = web.AppRunner(
            create_app(self.config, evaluator=evaluate), handler_cancellation=True
        )
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}/evaluate"
        try:
            async with aiohttp.ClientSession() as session:
                request = asyncio.create_task(session.post(url, json=self.request))
                await started.wait()
                request.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await request
                await asyncio.wait_for(cancelled.wait(), 1)
                async with session.post(url, json=self.request) as response:
                    self.assertEqual(response.status, 200)
        finally:
            await runner.cleanup()
