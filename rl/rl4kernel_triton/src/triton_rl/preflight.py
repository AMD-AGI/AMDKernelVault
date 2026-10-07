# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check runtime imports, API compatibility, and optional GPU availability."""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Sequence

SLIME_COMMIT = "5f781608ba28738fc73f44fa12efef1cdb408ee2"
TRITON_VERSION = "3.3.0"

_TRAINER_MODULES = (
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
)
_EVALUATOR_MODULES = ("aiohttp", "pytest", "numpy", "triton")
_SAMPLE_FIELDS = frozenset(
    {
        "prompt",
        "tokens",
        "response",
        "response_length",
        "label",
        "reward",
        "loss_mask",
        "rollout_log_probs",
        "status",
        "metadata",
    }
)
_SAMPLE_STATUSES = ("PENDING", "COMPLETED", "TRUNCATED", "ABORTED")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class PreflightReport:
    role: str
    checks: tuple[CheckResult, ...]
    gpu_skipped: bool

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


def _generate_state_error(module: Any) -> str | None:
    state = getattr(module, "GenerateState", None)
    if not inspect.isclass(state) or state.__init__ is object.__init__:
        return "Slime must provide a GenerateState class with an args constructor."
    try:
        inspect.signature(state.__init__).bind(None, object())
    except (TypeError, ValueError):
        return "GenerateState must accept one args object."
    return None


def _sample_error(module: Any) -> str | None:
    sample = getattr(module, "Sample", None)
    if not inspect.isclass(sample) or not is_dataclass(sample):
        return "Slime must provide the Sample dataclass."
    missing_fields = sorted(_SAMPLE_FIELDS - {field.name for field in fields(sample)})
    if missing_fields:
        return "Sample lacks required fields: " + ", ".join(missing_fields) + "."
    statuses = getattr(sample, "Status", None)
    missing_statuses = [name for name in _SAMPLE_STATUSES if not hasattr(statuses, name)]
    if missing_statuses:
        return "Sample.Status lacks required values: " + ", ".join(missing_statuses) + "."
    return None


def _post_error(module: Any) -> str | None:
    post = getattr(module, "post", None)
    if not callable(post) or not inspect.iscoroutinefunction(post):
        return "Slime must provide an asynchronous post function."
    try:
        inspect.signature(post).bind("http://localhost/generate", {}, use_http2=False)
    except (TypeError, ValueError):
        return "Slime post must accept url, payload, and use_http2."
    return None


def _gpu_check(torch: Any, gpu: int) -> CheckResult:
    try:
        cuda = torch.cuda
        if not cuda.is_available():
            return CheckResult("GPU availability", False, "PyTorch cannot access a GPU.")
        count = cuda.device_count()
        if type(count) is not int or count < 1:
            return CheckResult("GPU availability", False, "PyTorch reports no visible GPU.")
        if gpu >= count:
            return CheckResult(
                "GPU availability",
                False,
                f"GPU index {gpu} is invalid. PyTorch reports {count} visible GPUs.",
            )
        name = cuda.get_device_name(gpu)
    except Exception as error:
        return CheckResult(
            "GPU availability",
            False,
            f"PyTorch cannot query GPU {gpu}: {type(error).__name__}.",
        )
    return CheckResult("GPU availability", True, f"GPU {gpu} is visible: {name}.")


def run_preflight(
    role: str,
    *,
    no_gpu: bool = False,
    gpu: int = 0,
    importer: Callable[[str], Any] | None = None,
) -> PreflightReport:
    """Check the selected runtime without creating a model or tokenizer.

    The import checks verify API compatibility, not the installed Git commit.
    The container build controls the Slime source pin.
    GPU checks query availability and the selected device.
    They do not execute a kernel.
    """
    if role not in ("trainer", "evaluator"):
        raise ValueError("The role must be trainer or evaluator.")
    if type(gpu) is not int or gpu < 0:
        raise ValueError("The GPU index must be a nonnegative integer.")
    importer = importlib.import_module if importer is None else importer
    checks: list[CheckResult] = []
    modules: dict[str, Any] = {}
    required = _TRAINER_MODULES if role == "trainer" else _EVALUATOR_MODULES
    for name in ("torch", *required):
        try:
            modules[name] = importer(name)
        except Exception as error:
            detail = " ".join(str(error).split())
            message = f"Cannot import {name}: {type(error).__name__}."
            if detail:
                message += f" Details: {detail}"
            checks.append(CheckResult(name, False, message))
        else:
            checks.append(CheckResult(name, True, f"Imported {name}."))

    torch = modules.get("torch")
    hip_ok = False
    if torch is not None:
        hip = getattr(getattr(torch, "version", None), "hip", None)
        hip_ok = isinstance(hip, str) and bool(hip.strip())
        checks.append(
            CheckResult(
                "PyTorch HIP build",
                hip_ok,
                f"PyTorch reports HIP {hip}." if hip_ok else "PyTorch must use a HIP build.",
            )
        )

    if role == "trainer":
        api_checks = (
            (
                "slime.rollout.sglang_rollout",
                "Slime GenerateState API",
                _generate_state_error,
                "GenerateState accepts the required constructor arguments.",
            ),
            (
                "slime.utils.types",
                "Slime Sample API",
                _sample_error,
                "Sample provides the required fields and status values.",
            ),
            (
                "slime.utils.http_utils",
                "Slime post API",
                _post_error,
                "Slime post accepts the required asynchronous call.",
            ),
        )
        for name, label, checker, success in api_checks:
            if name in modules:
                error = checker(modules[name])
                checks.append(CheckResult(label, error is None, error or success))
        for name, attribute in (
            ("mbridge", "AutoBridge"),
            ("vllm.device_allocator.cumem", "CuMemAllocator"),
        ):
            if name in modules:
                qualified_name = f"{name}.{attribute}"
                supported = callable(getattr(modules[name], attribute, None))
                checks.append(
                    CheckResult(
                        f"{qualified_name} API",
                        supported,
                        f"{qualified_name} is callable."
                        if supported
                        else f"{qualified_name} must be callable.",
                    )
                )
    elif "triton" in modules:
        installed = getattr(modules["triton"], "__version__", None)
        matches = installed == TRITON_VERSION
        checks.append(
            CheckResult(
                "Triton version",
                matches,
                f"Triton version {TRITON_VERSION} is installed."
                if matches
                else f"Triton version must equal {TRITON_VERSION}. The installed version is {installed!r}.",
            )
        )

    if not no_gpu and hip_ok:
        checks.append(_gpu_check(torch, gpu))
    return PreflightReport(role, tuple(checks), gpu_skipped=no_gpu)


def _gpu_index(value: str) -> int:
    try:
        index = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("The GPU index must be a nonnegative integer.") from None
    if index < 0:
        raise argparse.ArgumentTypeError("The GPU index must be a nonnegative integer.")
    return index


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check the trainer or evaluator runtime.")
    parser.add_argument(
        "role", choices=("trainer", "evaluator"), help="Select the runtime to check."
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU queries. Keep the HIP build and dependency checks.",
    )
    parser.add_argument(
        "--gpu", type=_gpu_index, default=0, help="Select a visible GPU index. The default is 0."
    )
    args = parser.parse_args(argv)
    report = run_preflight(args.role, no_gpu=args.no_gpu, gpu=args.gpu)
    for check in report.checks:
        label = "PASS" if check.passed else "FAIL"
        print(f"{label}: {check.message}")
    if args.role == "trainer":
        print(f"Required Slime source: THUDM/slime@{SLIME_COMMIT}.")
        print("The API checks do not verify the installed source commit.")
    if report.gpu_skipped:
        print("SKIP: GPU checks were skipped. This run does not validate GPU availability.")
    else:
        print("The GPU check queries availability. It does not execute a kernel.")
    if report.passed:
        print(f"The {args.role} preflight passed.")
        return 0
    print(f"The {args.role} preflight failed.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
