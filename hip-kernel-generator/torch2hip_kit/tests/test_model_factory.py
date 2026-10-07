# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from unittest.mock import Mock

import pytest

import torch2hip_kit.model_clients as model_clients
import torch2hip_kit.model_factory as model_factory
from torch2hip_kit.cli import build_parser


API_KEY_NAMES = (
    "TORCH2HIP_API_KEY",
    "TORCH_MODU2FUNC_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
)


@pytest.fixture(autouse=True)
def clear_api_keys(monkeypatch) -> None:
    for name in API_KEY_NAMES:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize(
    ("provider", "class_name"),
    [
        ("openai", "StandardOpenAIModel"),
        ("standard-openai", "StandardOpenAIModel"),
        ("claude", "StandardClaudeModel"),
        ("standard-claude", "StandardClaudeModel"),
        ("gemini", "GeminiModel"),
        ("  OpEnAI  ", "StandardOpenAIModel"),
    ],
)
def test_factory_routes_provider_aliases(monkeypatch, provider, class_name) -> None:
    constructor = Mock()
    monkeypatch.setattr(model_clients, class_name, constructor)

    client = model_factory.create_model_client(provider, "model-test", "explicit-key")

    constructor.assert_called_once_with(model_id="model-test", api_key="explicit-key")
    assert client is constructor.return_value


@pytest.mark.parametrize(
    ("provider", "key_name", "class_name"),
    [
        ("openai", "OPENAI_API_KEY", "StandardOpenAIModel"),
        ("standard-openai", "OPENAI_API_KEY", "StandardOpenAIModel"),
        ("claude", "ANTHROPIC_API_KEY", "StandardClaudeModel"),
        ("standard-claude", "ANTHROPIC_API_KEY", "StandardClaudeModel"),
        ("gemini", "GEMINI_API_KEY", "GeminiModel"),
        ("gemini", "GOOGLE_API_KEY", "GeminiModel"),
    ],
)
def test_factory_accepts_provider_environment_key(monkeypatch, provider, key_name, class_name) -> None:
    monkeypatch.setenv(key_name, "provider-key")
    constructor = Mock()
    monkeypatch.setattr(model_clients, class_name, constructor)

    model_factory.create_model_client(provider, "model-test", None)

    constructor.assert_called_once_with(model_id="model-test", api_key="provider-key")


@pytest.mark.parametrize(
    ("explicit_key", "torch2hip_key", "legacy_key", "expected_key"),
    [
        ("explicit-key", "kit-key", "legacy-key", "explicit-key"),
        (None, "kit-key", "legacy-key", "kit-key"),
        (None, "", "legacy-key", "legacy-key"),
        (None, "", "", "provider-key"),
    ],
)
def test_factory_preserves_key_priority(
    monkeypatch, explicit_key, torch2hip_key, legacy_key, expected_key
) -> None:
    monkeypatch.setenv("TORCH2HIP_API_KEY", torch2hip_key)
    monkeypatch.setenv("TORCH_MODU2FUNC_API_KEY", legacy_key)
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")
    constructor = Mock()
    monkeypatch.setattr(model_clients, "StandardOpenAIModel", constructor)

    model_factory.create_model_client("openai", "model-test", explicit_key)

    constructor.assert_called_once_with(model_id="model-test", api_key=expected_key)


def test_gemini_key_precedes_google_key(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    constructor = Mock()
    monkeypatch.setattr(model_clients, "GeminiModel", constructor)

    model_factory.create_model_client("gemini", "model-test", None)

    constructor.assert_called_once_with(model_id="model-test", api_key="gemini-key")


@pytest.mark.parametrize(
    ("provider", "unrelated_key", "expected_key"),
    [
        ("openai", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"),
        ("claude", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
        ("gemini", "OPENAI_API_KEY", "GEMINI_API_KEY"),
    ],
)
def test_factory_rejects_unrelated_provider_key(monkeypatch, provider, unrelated_key, expected_key) -> None:
    monkeypatch.setenv(unrelated_key, "unrelated-key")

    with pytest.raises(ValueError, match=expected_key):
        model_factory.create_model_client(provider, "model-test", None)


def test_factory_rejects_unknown_provider_before_key_lookup() -> None:
    with pytest.raises(ValueError, match="Unsupported provider 'unknown'"):
        model_factory.create_model_client("unknown", "model-test", None)


def test_model_factory_keeps_compatibility_exports() -> None:
    for name in model_factory.__all__:
        assert getattr(model_factory, name) is getattr(model_clients, name)


def test_cli_defaults_to_public_openai_model() -> None:
    args = build_parser().parse_args(
        ["--module-dir", "modules", "--functional-dir", "functions", "--output-dir", "outputs"]
    )

    assert args.provider == "openai"
    assert args.model_id == "gpt-4o"
