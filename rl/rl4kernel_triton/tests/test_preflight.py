# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check runtime preflight behavior without importing GPU libraries."""

import contextlib
import io
import unittest
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from unittest.mock import Mock, patch

from triton_rl import preflight


@dataclass
class FakeSample:
    class Status(Enum):
        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    prompt: str = ""
    tokens: list = field(default_factory=list)
    response: str = ""
    response_length: int = 0
    label: object = None
    reward: object = None
    loss_mask: object = None
    rollout_log_probs: object = None
    status: Status = Status.PENDING
    metadata: dict = field(default_factory=dict)


class FakeGenerateState:
    def __init__(self, args):
        raise AssertionError("Preflight must not create a model or tokenizer.")


class FakeRuntimeComponent:
    def __init__(self, *args, **kwargs):
        raise AssertionError("Preflight must not construct a bridge or allocator.")


async def fake_post(url, payload, use_http2=False, max_retries=60):
    raise AssertionError("Preflight must not send a request.")


class NoHardwareTorch:
    def __init__(self, hip="6.3"):
        self.version = SimpleNamespace(hip=hip)

    @property
    def cuda(self):
        raise AssertionError("Preflight accessed hardware when the check was disabled.")


def runtime_modules(torch=None):
    return {
        "torch": NoHardwareTorch() if torch is None else torch,
        "slime.rollout.sglang_rollout": SimpleNamespace(GenerateState=FakeGenerateState),
        "slime.utils.types": SimpleNamespace(Sample=FakeSample),
        "slime.utils.http_utils": SimpleNamespace(post=fake_post),
        "mbridge": SimpleNamespace(AutoBridge=FakeRuntimeComponent),
        "slime_plugins.mbridge": SimpleNamespace(),
        "vllm.device_allocator.cumem": SimpleNamespace(CuMemAllocator=FakeRuntimeComponent),
        "megatron.core": SimpleNamespace(),
        "megatron.training": SimpleNamespace(),
        "sglang": SimpleNamespace(),
        "ray": SimpleNamespace(),
        "aiohttp": SimpleNamespace(),
        "pytest": SimpleNamespace(),
        "numpy": SimpleNamespace(),
        "triton": SimpleNamespace(__version__="3.3.0"),
    }


def importer_for(modules):
    def import_module(name):
        if name not in modules:
            raise ModuleNotFoundError(name)
        value = modules[name]
        if isinstance(value, Exception):
            raise value
        return value

    return Mock(side_effect=import_module)


class PreflightTests(unittest.TestCase):
    def test_evaluator_without_gpu_keeps_all_dependency_checks(self):
        importer = importer_for(runtime_modules())
        report = preflight.run_preflight("evaluator", no_gpu=True, gpu=27, importer=importer)
        self.assertTrue(report.passed)
        self.assertTrue(report.gpu_skipped)
        self.assertEqual(
            [call.args[0] for call in importer.call_args_list],
            ["torch", "aiohttp", "pytest", "numpy", "triton"],
        )
        self.assertFalse(any(check.name == "GPU availability" for check in report.checks))

    def test_trainer_checks_the_rollout_api_without_creating_runtime_state(self):
        importer = importer_for(runtime_modules())
        report = preflight.run_preflight("trainer", no_gpu=True, importer=importer)
        self.assertTrue(report.passed)
        self.assertEqual(
            {call.args[0] for call in importer.call_args_list},
            {
                "torch",
                "slime.rollout.sglang_rollout",
                "slime.utils.types",
                "slime.utils.http_utils",
                "mbridge",
                "slime_plugins.mbridge",
                "vllm.device_allocator.cumem",
                "megatron.core",
                "megatron.training",
                "sglang",
                "ray",
            },
        )
        self.assertTrue(any(check.name == "Slime Sample API" for check in report.checks))

    def test_bridge_and_allocator_require_callable_apis(self):
        for name, attribute in (
            ("mbridge", "AutoBridge"),
            ("vllm.device_allocator.cumem", "CuMemAllocator"),
        ):
            for value in (None, object(), 1):
                with self.subTest(module=name, value=value):
                    modules = runtime_modules()
                    modules[name] = (
                        SimpleNamespace()
                        if value is None
                        else SimpleNamespace(**{attribute: value})
                    )
                    report = preflight.run_preflight(
                        "trainer", no_gpu=True, importer=importer_for(modules)
                    )
                    check = next(
                        item for item in report.checks if item.name == f"{name}.{attribute} API"
                    )
                    self.assertFalse(report.passed)
                    self.assertFalse(check.passed)
                    self.assertIn("must be callable", check.message)

    def test_bridge_and_allocator_factories_are_not_called(self):
        def factory(*args, **kwargs):
            raise AssertionError("Preflight must not call a runtime factory.")

        modules = runtime_modules()
        modules["mbridge"] = SimpleNamespace(AutoBridge=factory)
        modules["vllm.device_allocator.cumem"] = SimpleNamespace(CuMemAllocator=factory)
        report = preflight.run_preflight("trainer", no_gpu=True, importer=importer_for(modules))
        self.assertTrue(report.passed)

    def test_cpu_or_unknown_build_fails_even_without_gpu_checks(self):
        for role in ("trainer", "evaluator"):
            for hip in (None, "", "  ", 6.3):
                with self.subTest(role=role, hip=hip):
                    modules = runtime_modules(NoHardwareTorch(hip))
                    report = preflight.run_preflight(
                        role, no_gpu=True, importer=importer_for(modules)
                    )
                    self.assertFalse(report.passed)
                    self.assertTrue(
                        any("must use a HIP build" in check.message for check in report.checks)
                    )

    def test_each_required_import_can_fail_without_stopping_other_checks(self):
        for role in ("trainer", "evaluator"):
            successful_importer = importer_for(runtime_modules())
            preflight.run_preflight(role, no_gpu=True, importer=successful_importer)
            names = [call.args[0] for call in successful_importer.call_args_list]
            for missing in names:
                with self.subTest(role=role, missing=missing):
                    modules = runtime_modules()
                    del modules[missing]
                    importer = importer_for(modules)
                    report = preflight.run_preflight(role, no_gpu=True, importer=importer)
                    self.assertFalse(report.passed)
                    self.assertEqual(importer.call_count, len(names))
                    failure = next(check for check in report.checks if check.name == missing)
                    self.assertFalse(failure.passed)
                    self.assertIn("ModuleNotFoundError", failure.message)

    def test_broken_native_import_becomes_a_reported_failure(self):
        modules = runtime_modules()
        modules["triton"] = OSError("The native library cannot load.")
        report = preflight.run_preflight("evaluator", no_gpu=True, importer=importer_for(modules))
        self.assertFalse(report.passed)
        self.assertTrue(
            any(
                check.message
                == "Cannot import triton: OSError. Details: The native library cannot load."
                for check in report.checks
            )
        )

    def test_import_error_identifies_a_missing_transitive_dependency(self):
        modules = runtime_modules()
        modules["sglang"] = ModuleNotFoundError(
            "No module named 'missing_backend'", name="missing_backend"
        )
        report = preflight.run_preflight("trainer", no_gpu=True, importer=importer_for(modules))
        failure = next(check for check in report.checks if check.name == "sglang")
        self.assertFalse(report.passed)
        self.assertIn("missing_backend", failure.message)

    def test_evaluator_requires_the_exact_triton_version_without_gpu_checks(self):
        for version in ("3.2.0", "3.3.0+custom", None):
            with self.subTest(version=version):
                modules = runtime_modules()
                modules["triton"] = SimpleNamespace(__version__=version)
                report = preflight.run_preflight(
                    "evaluator", no_gpu=True, importer=importer_for(modules)
                )
                self.assertFalse(report.passed)
                message = next(
                    check.message for check in report.checks if check.name == "Triton version"
                )
                self.assertIn("must equal 3.3.0", message)
                self.assertIn(repr(version), message)

    def test_incompatible_generate_state_constructor_fails(self):
        class IncompatibleState:
            def __init__(self, args, extra_required_argument):
                pass

        for state in (None, object, IncompatibleState):
            with self.subTest(state=state):
                modules = runtime_modules()
                modules["slime.rollout.sglang_rollout"] = SimpleNamespace(GenerateState=state)
                report = preflight.run_preflight(
                    "trainer", no_gpu=True, importer=importer_for(modules)
                )
                self.assertFalse(report.passed)
                self.assertTrue(
                    any(
                        check.name == "Slime GenerateState API" and not check.passed
                        for check in report.checks
                    )
                )

    def test_sample_must_supply_fields_and_status_values(self):
        @dataclass
        class MissingFields:
            prompt: str = ""

        @dataclass
        class MissingStatus(FakeSample):
            Status = SimpleNamespace(PENDING="pending")

        for sample, expected in (
            (object, "Sample dataclass"),
            (MissingFields, "rollout_log_probs"),
            (MissingStatus, "ABORTED"),
        ):
            with self.subTest(sample=sample):
                modules = runtime_modules()
                modules["slime.utils.types"] = SimpleNamespace(Sample=sample)
                report = preflight.run_preflight(
                    "trainer", no_gpu=True, importer=importer_for(modules)
                )
                self.assertFalse(report.passed)
                self.assertTrue(any(expected in check.message for check in report.checks))

    def test_post_must_be_async_and_accept_the_rollout_arguments(self):
        def synchronous_post(url, payload, use_http2=False):
            pass

        async def missing_http2(url, payload):
            pass

        for post in (None, synchronous_post, missing_http2):
            with self.subTest(post=post):
                modules = runtime_modules()
                modules["slime.utils.http_utils"] = SimpleNamespace(post=post)
                report = preflight.run_preflight(
                    "trainer", no_gpu=True, importer=importer_for(modules)
                )
                self.assertFalse(report.passed)
                self.assertTrue(
                    any(
                        check.name == "Slime post API" and not check.passed
                        for check in report.checks
                    )
                )

    def test_selected_visible_gpu_is_queried(self):
        cuda = SimpleNamespace(
            is_available=Mock(return_value=True),
            device_count=Mock(return_value=2),
            get_device_name=Mock(return_value="AMD Instinct MI300X"),
        )
        torch = SimpleNamespace(version=SimpleNamespace(hip="6.3"), cuda=cuda)
        report = preflight.run_preflight(
            "evaluator", gpu=1, importer=importer_for(runtime_modules(torch))
        )
        self.assertTrue(report.passed)
        self.assertFalse(report.gpu_skipped)
        cuda.get_device_name.assert_called_once_with(1)

    def test_invisible_or_out_of_range_gpu_fails_before_device_query(self):
        for available, count, gpu in ((False, 2, 0), (True, 0, 0), (True, 2, 2), (True, 2, 3)):
            with self.subTest(available=available, count=count, gpu=gpu):
                cuda = SimpleNamespace(
                    is_available=Mock(return_value=available),
                    device_count=Mock(return_value=count),
                    get_device_name=Mock(),
                )
                torch = SimpleNamespace(version=SimpleNamespace(hip="6.3"), cuda=cuda)
                report = preflight.run_preflight(
                    "evaluator", gpu=gpu, importer=importer_for(runtime_modules(torch))
                )
                self.assertFalse(report.passed)
                cuda.get_device_name.assert_not_called()

    def test_gpu_driver_failure_becomes_a_reported_failure(self):
        cuda = SimpleNamespace(is_available=Mock(side_effect=RuntimeError("The driver failed.")))
        torch = SimpleNamespace(version=SimpleNamespace(hip="6.3"), cuda=cuda)
        report = preflight.run_preflight("evaluator", importer=importer_for(runtime_modules(torch)))
        self.assertFalse(report.passed)
        self.assertTrue(any("RuntimeError" in check.message for check in report.checks))

    def test_invalid_role_or_index_fails_before_imports(self):
        importer = Mock()
        for role, gpu in (("unknown", 0), ("trainer", -1), ("evaluator", True), ("evaluator", "0")):
            with self.subTest(role=role, gpu=gpu):
                with self.assertRaises(ValueError):
                    preflight.run_preflight(role, gpu=gpu, importer=importer)
        importer.assert_not_called()


class PreflightCliTests(unittest.TestCase):
    def invoke(self, argv, modules):
        output = io.StringIO()
        errors = io.StringIO()
        with (
            patch.object(preflight.importlib, "import_module", importer_for(modules)),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(errors),
        ):
            result = preflight.main(argv)
        return result, output.getvalue(), errors.getvalue()

    def test_skipped_gpu_checks_are_explicit_on_success(self):
        status, output, errors = self.invoke(["evaluator", "--no-gpu"], runtime_modules())
        self.assertEqual(status, 0)
        self.assertIn("GPU checks were skipped", output)
        self.assertIn("does not validate GPU availability", output)
        self.assertNotIn("GPU 0 is visible", output)
        self.assertEqual(errors, "")

    def test_missing_dependency_returns_nonzero(self):
        modules = runtime_modules()
        del modules["pytest"]
        status, output, errors = self.invoke(["evaluator", "--no-gpu"], modules)
        self.assertEqual(status, 1)
        self.assertIn("Cannot import pytest", output)
        self.assertIn("preflight failed", errors)

    def test_trainer_reports_api_scope_and_required_pin(self):
        status, output, _ = self.invoke(["trainer", "--no-gpu"], runtime_modules())
        self.assertEqual(status, 0)
        self.assertIn(preflight.SLIME_COMMIT, output)
        self.assertIn("do not verify the installed source commit", output)

    def test_cli_rejects_negative_gpu_index_before_imports(self):
        importer = Mock()
        with (
            patch.object(preflight.importlib, "import_module", importer),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                preflight.main(["evaluator", "--gpu", "-1"])
        self.assertEqual(raised.exception.code, 2)
        importer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
