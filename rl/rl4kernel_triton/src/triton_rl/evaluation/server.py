# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve one serialized execution worker inside a dedicated Docker container."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import signal
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

from triton_rl.evaluation.protocol import (
    MAX_REQUEST_BYTES,
    MAX_RESPONSE_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    resolve_reference,
    subprocess_environment,
    validate_request,
    validate_result,
)


class InfrastructureError(RuntimeError):
    """The service cannot establish a candidate result."""


@dataclass(frozen=True)
class ServerConfig:
    reference_root: Path
    gpu: int
    timeout_seconds: float = 1500.0
    seed: int = 42
    work_root: Path = Path("/tmp")

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_root", self.reference_root.resolve())
        object.__setattr__(self, "work_root", self.work_root.resolve())
        if not self.reference_root.is_dir():
            raise ValueError("The reference root must be an existing directory.")
        if type(self.gpu) is not int or self.gpu < 0:
            raise ValueError("The GPU index must be a nonnegative integer.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("The worker timeout must be finite and positive.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("The seed must be an integer from 0 through 4294967295.")
        if not self.work_root.is_dir() or "." in str(self.work_root.resolve()):
            raise ValueError(
                "The work root must exist and contain no periods in its absolute path."
            )


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


async def run_worker(config: ServerConfig, request: dict) -> dict:
    """Keep all candidate execution outside the HTTP server process."""
    with tempfile.TemporaryDirectory(prefix="triton-rl-", dir=config.work_root) as directory:
        root = Path(directory)
        request_path, result_path = root / "request.json", root / "result.json"
        request_path.write_text(
            json.dumps(
                {
                    "request": request,
                    "reference_root": str(config.reference_root.resolve()),
                    "timeout_seconds": config.timeout_seconds,
                    "seed": config.seed,
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        env = subprocess_environment()
        env["ROCR_VISIBLE_DEVICES"] = str(config.gpu)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["PYTHONHASHSEED"] = str(config.seed)
        env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        env.pop("PYTEST_ADDOPTS", None)
        env["TRITON_CACHE_DIR"] = str(root / "triton-cache")
        env["TMPDIR"] = str(root)
        with (root / "worker.log").open("wb") as log:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "triton_rl.evaluation.worker",
                "--request",
                str(request_path),
                "--result",
                str(result_path),
                cwd=root,
                env=env,
                stdout=log,
                stderr=log,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(process.wait(), timeout=config.timeout_seconds + 10)
            except asyncio.TimeoutError as exc:
                raise InfrastructureError("The worker exceeded its supervisor timeout.") from exc
            finally:
                _kill_group(process.pid)
                await process.wait()
        if process.returncode != 0 or not result_path.is_file():
            raise InfrastructureError("The worker exited without a valid result file.")
        if result_path.stat().st_size > MAX_RESPONSE_BYTES:
            raise InfrastructureError("The worker result exceeds the size limit.")
        try:
            envelope = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise InfrastructureError("The worker result contains invalid JSON.") from exc
        if not isinstance(envelope, dict):
            raise InfrastructureError("The worker result must be an object.")
        if "error" in envelope:
            error = envelope["error"]
            message = (
                error.get("message", "The worker failed.")
                if isinstance(error, dict)
                else str(error)
            )
            raise InfrastructureError(message)
        try:
            result = validate_result(envelope["result"])
        except (KeyError, ProtocolError) as exc:
            raise InfrastructureError("The worker returned an invalid result.") from exc
        return {"protocol_version": PROTOCOL_VERSION, "result": result.to_dict()}


def create_app(config: ServerConfig, *, evaluator=None) -> web.Application:
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    lock = asyncio.Lock()
    execute = evaluator or run_worker

    async def evaluate(request: web.Request) -> web.Response:
        try:
            payload = validate_request(await request.json())
            resolve_reference(config.reference_root, payload["filename"])
        except (ValueError, OSError) as exc:
            return web.json_response(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "error": {
                        "type": "invalid_request",
                        "message": str(exc),
                    },
                },
                status=400,
            )
        try:
            async with lock:
                result = await execute(config, payload)
            return web.json_response(result, dumps=lambda value: json.dumps(value, allow_nan=False))
        except InfrastructureError as exc:
            return web.json_response(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "error": {
                        "type": "infrastructure_error",
                        "message": str(exc),
                    },
                },
                status=500,
            )

    async def health(_: web.Request) -> web.Response:
        return web.json_response(
            {"protocol_version": PROTOCOL_VERSION, "status": "ready", "gpu": config.gpu}
        )

    app.router.add_post("/evaluate", evaluate)
    app.router.add_get("/health", health)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--timeout-seconds", type=float, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--work-root", type=Path, default=Path("/tmp"))
    args = parser.parse_args()
    if os.environ.get("TRITON_RL_DEDICATED_CONTAINER") != "1" or not Path("/.dockerenv").is_file():
        parser.error(
            "Start this service inside dedicated Docker with TRITON_RL_DEDICATED_CONTAINER=1."
        )
    try:
        config = ServerConfig(
            args.reference_root, args.gpu, args.timeout_seconds, args.seed, args.work_root
        )
    except ValueError as exc:
        parser.error(str(exc))
    web.run_app(
        create_app(config),
        host=args.host,
        port=args.port,
        access_log=None,
        handler_cancellation=True,
    )


if __name__ == "__main__":
    main()
