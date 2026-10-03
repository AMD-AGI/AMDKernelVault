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
    _use_original_gpt5_parameters = True

    def __init__(self, model_id: str = "gpt-4o", api_key: str | None = None):
        if api_key is None:
            raise ValueError("No API key provided.")
        self.model_id = model_id
        self.client = openai.OpenAI(api_key=api_key)

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict[str, Any]], temperature: float = 0.0, max_tokens: int = 12000, **_: Any) -> str:
        if self._use_original_gpt5_parameters and re.fullmatch(
            r"gpt-5(?:-(?:mini|nano))?(?:-2025-\d{2}-\d{2})?", self.model_id
        ):
            generation_options = {"max_completion_tokens": max_tokens}
        else:
            generation_options = {"temperature": temperature, "max_tokens": max_tokens}
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            n=1,
            stream=False,
            **generation_options,
        )
        if not response.choices:
            raise ValueError("No response choices returned from the API.")
        return response.choices[0].message.content or ""


class StandardClaudeModel(BaseModel):
    def __init__(self, model_id: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        if api_key is None:
            raise ValueError("No API key provided.")
        self.model_id = model_id
        self.client = anthropic.Anthropic(api_key=api_key, base_url="https://api.anthropic.com")

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def generate(self, messages: list[dict[str, Any]], temperature: float = 0.0, max_tokens: int = 16000, **_: Any) -> str:
        response = self.client.messages.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=min(max_tokens, 16000),
        )
        if not response.content:
            raise ValueError("No response content returned from the API.")
        return response.content[0].text


# Preserve the existing import names for the public provider clients.
OpenAIModel = StandardOpenAIModel
ClaudeModel = StandardClaudeModel


class GeminiModel(StandardOpenAIModel):
    _use_original_gpt5_parameters = False

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
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )


def create_model_client(provider: str, model_id: str, api_key: str | None) -> BaseModel:
    providers = {
        "openai": (OpenAIModel, ("OPENAI_API_KEY",)),
        "standard-openai": (StandardOpenAIModel, ("OPENAI_API_KEY",)),
        "claude": (ClaudeModel, ("ANTHROPIC_API_KEY",)),
        "standard-claude": (StandardClaudeModel, ("ANTHROPIC_API_KEY",)),
        "gemini": (GeminiModel, ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    }
    normalized = provider.strip().lower()
    if normalized not in providers:
        raise ValueError(f"Unsupported provider '{provider}'. Expected one of {list(providers)}.")

    model_class, provider_env_vars = providers[normalized]
    key_env_vars = (
        "PY_HIP_KERNEL2KERNEL_API_KEY",
        "HIP2HIP_API_KEY",
        "TORCH2HIP_API_KEY",
        "TORCH_MODU2FUNC_API_KEY",
        *provider_env_vars,
    )
    resolved_api_key = api_key
    for name in key_env_vars:
        if resolved_api_key:
            break
        resolved_api_key = os.getenv(name)
    if not resolved_api_key:
        raise ValueError(
            f"An API key is required. Pass --api-key or set one of: {', '.join(key_env_vars)}."
        )

    return model_class(api_key=resolved_api_key, model_id=model_id)
