# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from typing import Any

import anthropic
import openai
from tenacity import retry, stop_after_attempt, wait_random_exponential


class BaseModel(ABC):
    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], **kwargs) -> str:
        raise NotImplementedError


class StandardOpenAIModel(BaseModel):
    _use_original_gpt5_options = True

    def __init__(self, model_id: str = "gpt-4o", api_key: str | None = None):
        if api_key is None:
            raise ValueError("No API key provided.")
        self.model_id = model_id
        self.client = openai.OpenAI(api_key=api_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict[str, Any]], temperature: float = 0.0, max_tokens: int = 12000, **_: Any) -> str:
        request = {"model": self.model_id, "messages": messages, "n": 1, "stream": False}
        if self._use_original_gpt5_options and re.fullmatch(
            r"gpt-5(?:-(?:mini|nano))?(?:-2025-\d{2}-\d{2})?", self.model_id
        ):
            request["max_completion_tokens"] = max_tokens
        else:
            request["temperature"] = temperature
            request["max_tokens"] = max_tokens
        response = self.client.chat.completions.create(**request)
        if not response.choices:
            raise ValueError("No response choices returned from the API.")
        return response.choices[0].message.content or ""


class OpenAIModel(StandardOpenAIModel):
    """Keep the legacy constructor while using the public OpenAI API."""

    def __init__(
        self,
        model_id: str = "gpt-4o",
        model_api_version: str | None = None,
        api_key: str | None = None,
    ):
        # The public API does not use the legacy model_api_version argument.
        super().__init__(model_id=model_id, api_key=api_key)

    def generate(self, messages: list[dict[str, Any]], temperature: float = 1.0, max_tokens: int = 12000, **kwargs: Any) -> str:
        return super().generate(
            messages, temperature=temperature, max_tokens=min(max_tokens, 16000), **kwargs
        )


class StandardClaudeModel(BaseModel):
    def __init__(self, model_id: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        if api_key is None:
            raise ValueError("No API key provided.")
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=api_key, base_url="https://api.anthropic.com")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict[str, Any]], temperature: float = 0.0, max_tokens: int = 16000, **_: Any) -> str:
        system_blocks = []
        conversation = []
        for message in messages:
            if message.get("role") == "system":
                content = message["content"]
                if isinstance(content, str):
                    system_blocks.append({"type": "text", "text": content})
                else:
                    system_blocks.extend(content)
            else:
                conversation.append(message)
        request = {
            "model": self.model_id,
            "messages": conversation,
            "temperature": temperature,
            "max_tokens": min(max_tokens, 16000),
        }
        if system_blocks:
            request["system"] = system_blocks
        response = self.client.messages.create(**request)
        if not response.content:
            raise ValueError("No response content returned from the API.")
        return response.content[0].text


class ClaudeModel(StandardClaudeModel):
    """Keep the legacy class name for the public Anthropic API."""

    def generate(self, messages: list[dict[str, Any]], temperature: float = 1.0, max_tokens: int = 16000, **kwargs: Any) -> str:
        return super().generate(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )


class GeminiModel(StandardOpenAIModel):
    _use_original_gpt5_options = False

    def __init__(self, model_id: str = "gemini-2.5-pro", api_key: str | None = None):
        if api_key is None:
            raise ValueError("No API key provided.")
        self.model_id = model_id
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    def generate(self, messages: list[dict[str, Any]], temperature: float = 1.0, max_tokens: int = 30000, **kwargs: Any) -> str:
        return super().generate(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        )


def create_model_client(provider: str, model_id: str, api_key: str | None) -> BaseModel:
    normalized = provider.strip().lower()
    providers = {
        "openai": (StandardOpenAIModel, ("OPENAI_API_KEY",)),
        "standard-openai": (StandardOpenAIModel, ("OPENAI_API_KEY",)),
        "claude": (StandardClaudeModel, ("ANTHROPIC_API_KEY",)),
        "standard-claude": (StandardClaudeModel, ("ANTHROPIC_API_KEY",)),
        "gemini": (GeminiModel, ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    }
    if normalized not in providers:
        raise ValueError(f"Unsupported provider '{provider}'. Expected one of {list(providers)}.")

    model_class, provider_key_names = providers[normalized]
    key_names = ("TORCH2HIP_API_KEY", "TORCH_MODU2FUNC_API_KEY", *provider_key_names)
    resolved_api_key = api_key or next(
        (os.environ[name] for name in key_names if os.environ.get(name)), None
    )
    if not resolved_api_key:
        raise ValueError(
            f"An API key is required. Pass --api-key or set one of {', '.join(key_names)}."
        )
    return model_class(api_key=resolved_api_key, model_id=model_id)
