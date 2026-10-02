# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send candidate source to explicitly configured execution services."""

from __future__ import annotations

import asyncio
import json
import math
import os
from urllib.parse import urlsplit
from weakref import WeakKeyDictionary, WeakValueDictionary

import aiohttp

from triton_rl.contracts import EvaluationResult
from triton_rl.evaluation.protocol import (
    MAX_RESPONSE_BYTES,
    PROTOCOL_VERSION,
    ProtocolError,
    validate_request,
    validate_result,
)


class SandboxError(RuntimeError):
    """A service failure must stop training instead of creating a reward."""


class SandboxTransportError(SandboxError):
    """The client could not exchange a response with the service."""


class SandboxProtocolError(SandboxError):
    """The service returned an invalid execution response."""


class SandboxRemoteError(SandboxError):
    """The service reported an infrastructure or reference failure."""


class SandboxClient:
    """Share endpoint reservations across clients within one event loop."""

    _pools: WeakKeyDictionary = WeakKeyDictionary()

    def __init__(self, urls: list[str], *, timeout_seconds: float = 1560.0):
        if not urls:
            raise ValueError("Configure at least one execution endpoint.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("The execution timeout must be finite and positive.")
        self.urls = tuple(self._endpoint(url) for url in urls)
        if len(set(self.urls)) != len(self.urls):
            raise ValueError("Each execution endpoint must appear once.")
        self.timeout_seconds = timeout_seconds
        self._session: aiohttp.ClientSession | None = None
        self._closed = False

    def _pool(self) -> asyncio.Queue[str]:
        loop = asyncio.get_running_loop()
        pools = self._pools.setdefault(loop, WeakValueDictionary())
        key = tuple(sorted(self.urls))
        if key not in pools:
            if any(set(key).intersection(existing) for existing in pools):
                raise SandboxError(
                    "Concurrent clients must use identical or disjoint execution endpoint sets."
                )
            queue: asyncio.Queue[str] = asyncio.Queue()
            for url in key:
                queue.put_nowait(url)
            pools[key] = queue
        return pools[key]

    @staticmethod
    def _endpoint(value: str) -> str:
        value = value.strip().rstrip("/")
        parsed = urlsplit(value)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError("Each execution endpoint must use an HTTP or HTTPS URL.")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError(
                "Execution endpoints cannot contain credentials, queries, or fragments."
            )
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("The execution endpoint contains an invalid port.") from exc
        return value + "/evaluate"

    @classmethod
    def from_env(cls) -> "SandboxClient":
        raw = os.environ.get("TRITON_RL_SANDBOX_URLS", "")
        urls = [url.strip() for url in raw.split(",")]
        if not raw or any(not url for url in urls):
            raise ValueError("Set TRITON_RL_SANDBOX_URLS to comma-separated execution base URLs.")
        try:
            timeout = float(os.environ.get("TRITON_RL_SANDBOX_TIMEOUT_SECONDS", "1560"))
        except ValueError as exc:
            raise ValueError("TRITON_RL_SANDBOX_TIMEOUT_SECONDS must be numeric.") from exc
        return cls(urls, timeout_seconds=timeout)

    async def evaluate(
        self, code: str, filename: str, *, atol: float, rtol: float
    ) -> EvaluationResult:
        if self._closed:
            raise SandboxError("The execution client is closed.")
        payload = validate_request(
            {
                "protocol_version": PROTOCOL_VERSION,
                "code": code,
                "filename": filename,
                "atol": atol,
                "rtol": rtol,
            }
        )
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                trust_env=False,
            )
        available = self._pool()
        endpoint = await available.get()
        try:
            async with self._session.post(
                endpoint, json=payload, allow_redirects=False
            ) as response:
                data = bytearray()
                async for chunk in response.content.iter_chunked(65536):
                    data.extend(chunk)
                    if len(data) > MAX_RESPONSE_BYTES:
                        raise SandboxProtocolError("The execution response exceeds the size limit.")
                try:
                    body = json.loads(data)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise SandboxProtocolError(
                        "The execution service returned invalid JSON."
                    ) from exc
                if response.status != 200:
                    detail = body.get("error", {}) if isinstance(body, dict) else {}
                    message = (
                        detail.get("message", "The execution service failed.")
                        if isinstance(detail, dict)
                        else str(detail)
                    )
                    raise SandboxRemoteError(f"HTTP {response.status}: {message}")
                if not isinstance(body, dict) or set(body) != {"protocol_version", "result"}:
                    raise SandboxProtocolError("The execution response has an invalid envelope.")
                if (
                    type(body["protocol_version"]) is not int
                    or body["protocol_version"] != PROTOCOL_VERSION
                ):
                    raise SandboxProtocolError("The execution protocol version does not match.")
                try:
                    return validate_result(body["result"])
                except ProtocolError as exc:
                    raise SandboxProtocolError(str(exc)) from exc
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise SandboxTransportError(
                f"The execution request failed: {type(exc).__name__}."
            ) from exc
        finally:
            available.put_nowait(endpoint)

    async def close(self) -> None:
        self._closed = True
        if self._session is not None:
            await self._session.close()

    async def __aenter__(self) -> "SandboxClient":
        if self._closed:
            raise SandboxError("The execution client is closed.")
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
