# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import anthropic
import openai
import pytest

from models import ClaudeModel, GeminiModel, OpenAIModel, StandardClaudeModel, StandardOpenAIModel
from torch_modu2func_kit.cli import build_parser
from torch_modu2func_kit.model_factory import create_model_client


PROVIDERS = [
    ("openai", OpenAIModel, "gpt-4o", "OPENAI_API_KEY"),
    ("standard-openai", StandardOpenAIModel, "gpt-4o", "OPENAI_API_KEY"),
    ("claude", ClaudeModel, "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
    ("standard-claude", StandardClaudeModel, "claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
    ("gemini", GeminiModel, "gemini-2.5-pro", "GEMINI_API_KEY"),
]


@pytest.fixture
def sdk_clients(monkeypatch):
    for variable in ("TORCH_MODU2FUNC_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(variable, raising=False)
    openai_client = Mock()
    openai_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="generated code"))]
    )
    claude_client = Mock()
    claude_client.messages.create.return_value = SimpleNamespace(
        content=[SimpleNamespace(text="generated code")]
    )
    openai_factory = Mock(return_value=openai_client)
    claude_factory = Mock(return_value=claude_client)
    monkeypatch.setattr(openai, "OpenAI", openai_factory)
    monkeypatch.setattr(anthropic, "Anthropic", claude_factory)
    return openai_factory, claude_factory


@pytest.mark.parametrize("provider,model_class,default_model,key_variable", PROVIDERS)
def test_provider_defaults_and_explicit_model(sdk_clients, provider, model_class, default_model, key_variable):
    client = create_model_client(provider, None, "explicit-key")
    assert isinstance(client, model_class)
    assert client.model_id == default_model
    custom = create_model_client(provider.upper(), "custom-public-model", "explicit-key")
    assert custom.model_id == "custom-public-model"


@pytest.mark.parametrize("provider,model_class,default_model,key_variable", PROVIDERS)
@pytest.mark.parametrize("source", ["explicit", "shared", "provider"])
def test_api_key_precedence(sdk_clients, monkeypatch, provider, model_class, default_model, key_variable, source):
    monkeypatch.setenv(key_variable, "provider-key")
    if source != "provider":
        monkeypatch.setenv("TORCH_MODU2FUNC_API_KEY", "shared-key")
    explicit_key = "explicit-key" if source == "explicit" else None
    create_model_client(provider, None, explicit_key)
    constructor = sdk_clients[1] if "claude" in provider else sdk_clients[0]
    assert constructor.call_args.kwargs["api_key"] == source + "-key"


def test_missing_key_and_unknown_provider(sdk_clients):
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        create_model_client("gemini", None, None)
    with pytest.raises(ValueError, match="Unsupported provider"):
        create_model_client("unknown", None, None)
    for constructor in sdk_clients:
        constructor.assert_not_called()


def test_openai_legacy_arguments_use_public_client(sdk_clients):
    client = OpenAIModel("gpt-4o", "legacy-version", "public-key")
    sdk_clients[0].assert_called_once_with(api_key="public-key")
    assert isinstance(client, StandardOpenAIModel)
    assert client.model_api_version == "legacy-version"
    assert client.generate([{"role": "user", "content": "Convert this code."}]) == "generated code"
    request = sdk_clients[0].return_value.chat.completions.create.call_args.kwargs
    assert request["model"] == "gpt-4o"
    assert request["temperature"] == 1.0
    assert request["max_tokens"] == 5000


@pytest.mark.parametrize("model_class", [StandardOpenAIModel, OpenAIModel])
@pytest.mark.parametrize("model_id", [
    "gpt-5",
    "gpt-5-2025-08-07",
    "gpt-5-mini",
    "gpt-5-mini-2025-08-07",
    "gpt-5-nano",
    "gpt-5-nano-2025-08-07",
])
def test_original_gpt5_payload(sdk_clients, model_class, model_id):
    client = model_class(model_id=model_id, api_key="public-key")
    messages = [{"role": "user", "content": "Convert this code."}]
    assert client.generate(
        messages, temperature=0.2, max_tokens=1536, presence_penalty=0.3, frequency_penalty=0.6
    ) == "generated code"
    sdk_clients[0].return_value.chat.completions.create.assert_called_once_with(
        model=model_id,
        messages=messages,
        n=1,
        stream=False,
        max_completion_tokens=1536,
        presence_penalty=0.3,
        frequency_penalty=0.6,
    )


@pytest.mark.parametrize("model_class", [StandardOpenAIModel, OpenAIModel])
@pytest.mark.parametrize("model_id", [
    "gpt-4o",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.1-2025-11-13",
    "gpt-5.2-2025-12-11",
    "gpt-5-pro",
    "gpt-5-chat-latest",
    "gpt-5-codex",
    "gpt-5-mini-latest",
    "gpt-5-nano-preview",
    "gpt-5-2026-01-01",
    "gpt-5-mini-2026-01-01",
    "gpt-5-nano-2026-01-01",
    "gpt-5-2025-8-7",
    "gpt-5-2025-08-07-extra",
    "gpt-5-mini-2025-08-07-extra",
    "gpt-5-nano-2025-08-07-extra",
    "custom/gpt-5",
    "custom-public-model",
    "GPT-5",
    "gpt-5 ",
])
def test_other_openai_models_keep_existing_payload(sdk_clients, model_class, model_id):
    client = model_class(model_id=model_id, api_key="public-key")
    messages = [{"role": "user", "content": "Convert this code."}]
    assert client.generate(
        messages, temperature=0.2, max_tokens=1536, presence_penalty=0.3, frequency_penalty=0.6
    ) == "generated code"
    sdk_clients[0].return_value.chat.completions.create.assert_called_once_with(
        model=model_id,
        messages=messages,
        temperature=0.2,
        n=1,
        stream=False,
        max_tokens=1536,
        presence_penalty=0.3,
        frequency_penalty=0.6,
    )


@pytest.mark.parametrize("model_class,temperature", [(StandardOpenAIModel, 0), (OpenAIModel, 1.0)])
def test_openai_default_payload_stays_unchanged(sdk_clients, model_class, temperature):
    client = model_class(api_key="public-key")
    messages = [{"role": "user", "content": "Convert this code."}]
    assert client.generate(messages) == "generated code"
    sdk_clients[0].return_value.chat.completions.create.assert_called_once_with(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
        n=1,
        stream=False,
        max_tokens=5000,
        presence_penalty=0,
        frequency_penalty=0,
    )


def test_claude_legacy_arguments_use_public_client(sdk_clients):
    client = ClaudeModel(api_key="public-key")
    sdk_clients[1].assert_called_once_with(api_key="public-key", base_url="https://api.anthropic.com")
    assert isinstance(client, StandardClaudeModel)
    assert client.generate(
        [{"role": "user", "content": "Convert this code."}], max_tokens=1024, max_completion_tokens=2048
    ) == "generated code"
    request = sdk_clients[1].return_value.messages.create.call_args.kwargs
    assert request["model"] == "claude-sonnet-4-20250514"
    assert request["temperature"] == 1.0
    assert request["max_tokens"] == 1024
    assert "max_completion_tokens" not in request
    assert "system" not in request


@pytest.mark.parametrize("model_class", [ClaudeModel, StandardClaudeModel])
def test_claude_extracts_system_messages(sdk_clients, model_class):
    system_block = {"type": "text", "text": "Preserve the model output."}
    user_message = {"role": "user", "content": "Convert this code."}
    messages = [
        {"role": "system", "content": "Use PyTorch functions."},
        {"role": "system", "content": [system_block]},
        user_message,
    ]
    client = model_class(api_key="public-key")
    assert client.generate(messages) == "generated code"
    request = sdk_clients[1].return_value.messages.create.call_args.kwargs
    assert request["messages"] == [user_message]
    assert request["system"] == [
        {"type": "text", "text": "Use PyTorch functions."}, system_block
    ]
    assert len(messages) == 3


def test_gemini_uses_public_endpoint_and_chat_response(sdk_clients):
    client = GeminiModel(api_key="public-key")
    sdk_clients[0].assert_called_once_with(
        api_key="public-key", base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    messages = [{"role": "user", "content": "Convert this code."}]
    assert client.generate(messages, temperature=0.2, max_tokens=1024) == "generated code"
    request = sdk_clients[0].return_value.chat.completions.create.call_args.kwargs
    assert request["model"] == "gemini-2.5-pro"
    assert request["messages"] == messages
    assert request["temperature"] == 0.2
    assert request["max_tokens"] == 1024
    sdk_clients[0].return_value.chat.completions.create.return_value = SimpleNamespace(choices=[])
    with pytest.raises(ValueError, match="no response choices"):
        GeminiModel.generate.__wrapped__(client, messages)


@pytest.mark.parametrize("model_id", ["gemini-2.5-pro", "gpt-5"])
def test_gemini_payload_stays_independent_of_openai_model_rules(sdk_clients, model_id):
    client = GeminiModel(model_id=model_id, api_key="public-key")
    messages = [{"role": "user", "content": "Convert this code."}]
    assert client.generate(
        messages, temperature=0.2, max_tokens=1536, presence_penalty=0.3, frequency_penalty=0.6
    ) == "generated code"
    sdk_clients[0].return_value.chat.completions.create.assert_called_once_with(
        model=model_id,
        messages=messages,
        temperature=0.2,
        max_tokens=1536,
        top_p=0.95,
        presence_penalty=0.3,
        frequency_penalty=0.6,
    )


@pytest.mark.parametrize("provider,model_class,default_model,key_variable", PROVIDERS)
def test_cli_selects_provider_default(sdk_clients, provider, model_class, default_model, key_variable):
    args = build_parser().parse_args([
        "--input-dir", "input", "--output-dir", "output", "--provider", provider, "--api-key", "public-key"
    ])
    client = create_model_client(args.provider, args.model_id, args.api_key)
    assert client.model_id == default_model
