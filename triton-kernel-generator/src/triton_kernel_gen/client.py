# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Client for a configured chat-completion inference service."""

from __future__ import annotations

import math
import os
from typing import Any
from urllib.parse import urlparse

import requests

from .contracts import ModelResponse


class ModelServiceError(RuntimeError):
    """The model service did not return a valid generation."""


def validate_endpoint(endpoint: str) -> None:
    if not isinstance(endpoint, str) or not endpoint or any(c.isspace() for c in endpoint):
        raise ValueError("The inference endpoint must be a URL without whitespace.")
    parsed = urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("The inference endpoint must use HTTP or HTTPS.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError(
            "The inference endpoint must not contain credentials, queries, or fragments."
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("The inference endpoint has an invalid port.") from exc
    if port == 0:
        raise ValueError("The inference endpoint port must be positive.")


class ChatClient:
    """Send one explicit request per generation without hidden model retries."""

    def __init__(
        self,
        endpoint: str,
        model_id: str,
        *,
        api_key_env: str | None = "TRITON_GEN_API_KEY",
        timeout_seconds: float = 300,
        request_options: dict[str, Any] | None = None,
        session: Any = None,
    ) -> None:
        validate_endpoint(endpoint)
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Supply the model identifier served by the inference endpoint.")
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("The inference timeout must be positive and finite.")
        options = dict(request_options or {})
        reserved = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "stream",
            "n",
            "api_key",
            "authorization",
            "headers",
        }
        if reserved.intersection(options):
            raise ValueError(
                "Additional request options must not replace the recorded generation settings."
            )
        self.endpoint = endpoint
        self.model_id = model_id
        self.timeout_seconds = float(timeout_seconds)
        self.request_options = options
        self._api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self._session = session or requests.Session()

    def generate(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> ModelResponse:
        if not messages or any(
            not isinstance(item, dict)
            or item.get("role") not in {"system", "user", "assistant"}
            or not isinstance(item.get("content"), str)
            for item in messages
        ):
            raise ValueError("Supply nonempty chat messages with text content.")
        if isinstance(temperature, bool) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("The sampling temperature must be finite and nonnegative.")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("The output token limit must be a positive integer.")
        payload = {
            **self.request_options,
            "model": self.model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        try:
            response = self._session.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout_seconds,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise ModelServiceError(
                f"The inference request failed: {type(exc).__name__}."
            ) from None
        if response.status_code != 200:
            # Provider error bodies can echo credentials or private routing details.
            raise ModelServiceError(f"The inference endpoint returned HTTP {response.status_code}.")
        try:
            result = response.json()
            choices = result["choices"]
            if not isinstance(choices, list) or len(choices) != 1:
                raise ValueError("Expected one completion.")
            message = choices[0]["message"]
            text = message["content"]
            reasoning = message.get("reasoning_content", message.get("reasoning"))
            if not isinstance(text, str) or not text.strip():
                raise ValueError("The completion contains no text.")
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError("The reasoning field is not text.")
            model = result.get("model", self.model_id)
            usage = result.get("usage")
            if usage is None:
                usage = {}
            finish_reason = choices[0].get("finish_reason")
            if not isinstance(model, str) or not model.strip() or not isinstance(usage, dict):
                raise ValueError("Invalid model response metadata.")
            if finish_reason is not None and not isinstance(finish_reason, str):
                raise ValueError("Invalid finish reason.")
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelServiceError(
                "The inference endpoint returned an invalid completion object."
            ) from None
        return ModelResponse(text, model, finish_reason, usage, reasoning)

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "ChatClient":
        return self

    def __exit__(self, *unused: Any) -> None:
        self.close()
