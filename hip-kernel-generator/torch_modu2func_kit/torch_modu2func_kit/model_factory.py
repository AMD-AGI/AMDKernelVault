# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os

from models import (
    ClaudeModel,
    GeminiModel,
    OpenAIModel,
    StandardClaudeModel,
    StandardOpenAIModel,
)


_PROVIDER_SETTINGS = {
    "openai": (OpenAIModel, "gpt-4o", "OPENAI_API_KEY"),
    "standard-openai": (StandardOpenAIModel, "gpt-4o", "OPENAI_API_KEY"),
    "claude": (ClaudeModel, "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
    "standard-claude": (StandardClaudeModel, "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
    "gemini": (GeminiModel, "gemini-2.5-pro", "GEMINI_API_KEY"),
}


def create_model_client(provider: str, model_id: str | None, api_key: str | None):
    normalized = provider.strip().lower()
    if normalized not in _PROVIDER_SETTINGS:
        raise ValueError(
            f"Unsupported provider '{provider}'. Expected one of {list(_PROVIDER_SETTINGS)}."
        )
    model_class, default_model, key_variable = _PROVIDER_SETTINGS[normalized]
    resolved_api_key = (
        api_key or os.getenv("TORCH_MODU2FUNC_API_KEY") or os.getenv(key_variable)
    )
    if not resolved_api_key:
        raise ValueError(
            "An API key is required. "
            f"Pass --api-key or set TORCH_MODU2FUNC_API_KEY or {key_variable}."
        )
    return model_class(api_key=resolved_api_key, model_id=model_id or default_model)
