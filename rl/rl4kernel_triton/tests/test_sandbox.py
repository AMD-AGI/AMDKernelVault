# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise HTTP failures and shared endpoint capacity without executing code."""

import asyncio
import os
import unittest
from unittest.mock import patch

from aiohttp import web

from triton_rl.contracts import EvaluationResult
from triton_rl.sandbox import (
    SandboxClient,
    SandboxProtocolError,
    SandboxRemoteError,
    SandboxTransportError,
)


def success():
    return web.json_response(
        {"protocol_version": 1, "result": EvaluationResult(True, True, speedup=2).to_dict()}
    )


class SandboxTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.runners = []

    async def asyncTearDown(self):
        for runner in self.runners:
            await runner.cleanup()

    async def endpoint(self, handler):
        app = web.Application()
        app.router.add_post("/evaluate", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        self.runners.append(runner)
        return f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    async def test_source_uses_json_without_execution(self):
        code = "print('quoted \\n text')\n# $(touch /not-a-command) `not-a-command`"
        captured = []

        async def handle(request):
            captured.append(await request.json())
            return success()

        async with SandboxClient([await self.endpoint(handle)]) as client:
            result = await client.evaluate(code, "nested/task.py", atol=1e-3, rtol=1e-4)
        self.assertEqual(captured[0]["code"], code)
        self.assertEqual(captured[0]["filename"], "nested/task.py")
        self.assertEqual(captured[0]["rtol"], 1e-4)
        self.assertEqual(result.speedup, 2)
        self.assertTrue(client._session.closed)

    async def test_remote_failure_does_not_become_reward(self):
        async def handle(request):
            return web.json_response({"error": {"message": "Reference tests failed."}}, status=500)

        async with SandboxClient([await self.endpoint(handle)]) as client:
            with self.assertRaisesRegex(SandboxRemoteError, "Reference tests failed"):
                await client.evaluate("pass", "task.py", atol=0, rtol=0)

    async def test_invalid_json_and_result_are_protocol_errors(self):
        responses = [
            web.Response(text="not JSON"),
            web.json_response(
                {
                    "protocol_version": 1,
                    "result": {"compiled": True, "correct": True, "speedup": float("nan")},
                }
            ),
        ]

        async def handle(request):
            return responses.pop(0)

        async with SandboxClient([await self.endpoint(handle)]) as client:
            for _ in range(2):
                with self.assertRaises(SandboxProtocolError):
                    await client.evaluate("pass", "task.py", atol=0, rtol=0)

    async def test_timeout_raises_and_returns_capacity(self):
        first = True

        async def handle(request):
            nonlocal first
            if first:
                first = False
                await asyncio.sleep(0.1)
            return success()

        async with SandboxClient([await self.endpoint(handle)], timeout_seconds=0.03) as client:
            with self.assertRaises(SandboxTransportError):
                await client.evaluate("pass", "task.py", atol=0, rtol=0)
            self.assertTrue((await client.evaluate("pass", "task.py", atol=0, rtol=0)).correct)

    async def test_clients_share_capacity_across_two_endpoints(self):
        active = set()
        both_active = asyncio.Event()
        release = asyncio.Event()

        def handler(index):
            async def handle(request):
                self.assertNotIn(index, active)
                active.add(index)
                if len(active) == 2:
                    both_active.set()
                await release.wait()
                active.remove(index)
                return success()

            return handle

        urls = [await self.endpoint(handler(index)) for index in range(2)]
        async with SandboxClient(urls) as first, SandboxClient(list(reversed(urls))) as second:
            tasks = [
                asyncio.create_task(client.evaluate("pass", "task.py", atol=0, rtol=0))
                for client in (first, second)
            ]
            try:
                await asyncio.wait_for(both_active.wait(), 1)
                self.assertEqual(active, {0, 1})
            finally:
                release.set()
                await asyncio.gather(*tasks)

    async def test_cancellation_releases_shared_reservation(self):
        started, release = asyncio.Event(), asyncio.Event()
        calls = 0

        async def handle(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                started.set()
                await release.wait()
            return success()

        urls = [await self.endpoint(handle)]
        async with SandboxClient(urls) as first, SandboxClient(urls) as second:
            task = asyncio.create_task(first.evaluate("pass", "task.py", atol=0, rtol=0))
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            release.set()
            result = await asyncio.wait_for(second.evaluate("pass", "task.py", atol=0, rtol=0), 1)
            self.assertTrue(result.correct)

    def test_environment_requires_explicit_urls(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TRITON_RL_SANDBOX_URLS"):
                SandboxClient.from_env()
        with patch.dict(
            os.environ, {"TRITON_RL_SANDBOX_URLS": "https://worker.invalid, "}, clear=True
        ):
            with self.assertRaises(ValueError):
                SandboxClient.from_env()

    def test_rejects_unsafe_endpoint_components(self):
        for url in (
            "file:///tmp/worker",
            "http://user:secret@worker",
            "http://worker?key=value",
            "http://worker#fragment",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                SandboxClient([url])
