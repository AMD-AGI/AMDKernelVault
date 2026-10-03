# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import socket
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from py_hip_kernel2kernel_kit import cli, model_clients, model_factory


PACKAGE_KEY_NAMES = (
    "PY_HIP_KERNEL2KERNEL_API_KEY",
    "HIP2HIP_API_KEY",
    "TORCH2HIP_API_KEY",
    "TORCH_MODU2FUNC_API_KEY",
)
PROVIDER_KEY_NAMES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@pytest.fixture(autouse=True)
def isolated_providers(monkeypatch):
    for name in (*PACKAGE_KEY_NAMES, *PROVIDER_KEY_NAMES):
        monkeypatch.delenv(name, raising=False)

    def deny_network(*args, **kwargs):
        raise AssertionError("Provider tests must not access the network.")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    openai_constructor = Mock()
    anthropic_constructor = Mock()
    monkeypatch.setattr(model_clients.openai, "OpenAI", openai_constructor)
    monkeypatch.setattr(model_clients.openai, "AzureOpenAI", Mock(side_effect=deny_network))
    monkeypatch.setattr(model_clients.anthropic, "Anthropic", anthropic_constructor)
    return {"openai": openai_constructor, "anthropic": anthropic_constructor}


@pytest.mark.parametrize(
    ("provider", "expected_type", "sdk_name", "endpoint_kwargs"),
    [
        ("openai", model_clients.StandardOpenAIModel, "openai", {}),
        ("standard-openai", model_clients.StandardOpenAIModel, "openai", {}),
        ("claude", model_clients.StandardClaudeModel, "anthropic", {"base_url": "https://api.anthropic.com"}),
        ("standard-claude", model_clients.StandardClaudeModel, "anthropic", {"base_url": "https://api.anthropic.com"}),
        ("gemini", model_clients.GeminiModel, "openai", {"base_url": GEMINI_BASE_URL}),
        ("  OpEnAi  ", model_clients.StandardOpenAIModel, "openai", {}),
    ],
)
def test_factory_uses_public_provider(provider, expected_type, sdk_name, endpoint_kwargs, isolated_providers):
    client = model_clients.create_model_client(provider, "selected-model", "explicit-key")

    assert type(client) is expected_type
    assert client.model_id == "selected-model"
    isolated_providers[sdk_name].assert_called_once_with(api_key="explicit-key", **endpoint_kwargs)
    other_sdk = "anthropic" if sdk_name == "openai" else "openai"
    isolated_providers[other_sdk].assert_not_called()


def test_existing_import_names_alias_public_clients():
    assert model_factory.OpenAIModel is model_clients.StandardOpenAIModel
    assert model_factory.ClaudeModel is model_clients.StandardClaudeModel
    assert model_factory.create_model_client is model_clients.create_model_client


@pytest.mark.parametrize(
    ("model_class", "expected_model"),
    [
        (model_clients.OpenAIModel, "gpt-4o"),
        (model_clients.ClaudeModel, "claude-sonnet-4-20250514"),
        (model_clients.GeminiModel, "gemini-2.5-pro"),
    ],
)
def test_public_model_defaults(model_class, expected_model):
    assert model_class(api_key="test-key").model_id == expected_model


@pytest.mark.parametrize(
    ("provider", "key_name", "sdk_name"),
    [
        ("openai", "OPENAI_API_KEY", "openai"),
        ("standard-openai", "OPENAI_API_KEY", "openai"),
        ("claude", "ANTHROPIC_API_KEY", "anthropic"),
        ("standard-claude", "ANTHROPIC_API_KEY", "anthropic"),
        ("gemini", "GEMINI_API_KEY", "openai"),
        ("gemini", "GOOGLE_API_KEY", "openai"),
    ],
)
def test_provider_environment_key(provider, key_name, sdk_name, monkeypatch, isolated_providers):
    monkeypatch.setenv(key_name, "provider-key")

    model_clients.create_model_client(provider, "selected-model", None)

    assert isolated_providers[sdk_name].call_args.kwargs["api_key"] == "provider-key"


@pytest.mark.parametrize("key_index", range(len(PACKAGE_KEY_NAMES)))
def test_package_key_precedence(key_index, monkeypatch, isolated_providers):
    for name in PACKAGE_KEY_NAMES[key_index:]:
        monkeypatch.setenv(name, name)
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")

    model_clients.create_model_client("openai", "selected-model", None)

    assert isolated_providers["openai"].call_args.kwargs["api_key"] == PACKAGE_KEY_NAMES[key_index]


def test_explicit_key_precedes_environment(monkeypatch, isolated_providers):
    for name in (*PACKAGE_KEY_NAMES, *PROVIDER_KEY_NAMES):
        monkeypatch.setenv(name, "environment-key")

    model_clients.create_model_client("openai", "selected-model", "explicit-key")

    isolated_providers["openai"].assert_called_once_with(api_key="explicit-key")


def test_empty_package_key_uses_provider_key(monkeypatch, isolated_providers):
    monkeypatch.setenv(PACKAGE_KEY_NAMES[0], "")
    monkeypatch.setenv("OPENAI_API_KEY", "provider-key")

    model_clients.create_model_client("openai", "selected-model", "")

    isolated_providers["openai"].assert_called_once_with(api_key="provider-key")


def test_gemini_key_precedes_google_key(monkeypatch, isolated_providers):
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    model_clients.create_model_client("gemini", "selected-model", None)

    isolated_providers["openai"].assert_called_once_with(api_key="gemini-key", base_url=GEMINI_BASE_URL)


@pytest.mark.parametrize("provider", ["openai", "standard-openai", "claude", "standard-claude", "gemini"])
def test_missing_key_rejects_client_creation(provider, isolated_providers):
    with pytest.raises(ValueError, match="An API key is required"):
        model_clients.create_model_client(provider, "selected-model", None)

    for constructor in isolated_providers.values():
        constructor.assert_not_called()


def test_provider_does_not_use_another_provider_key(monkeypatch, isolated_providers):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        model_clients.create_model_client("claude", "selected-model", None)

    isolated_providers["anthropic"].assert_not_called()


def test_unsupported_provider_precedes_missing_key(isolated_providers):
    with pytest.raises(ValueError, match="Unsupported provider 'unknown'"):
        model_clients.create_model_client("unknown", "selected-model", None)

    for constructor in isolated_providers.values():
        constructor.assert_not_called()


@pytest.mark.parametrize("provider", ["openai", "standard-openai", "gemini"])
def test_openai_compatible_generation(provider, isolated_providers):
    sdk_client = isolated_providers["openai"].return_value
    sdk_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="generated code"))]
    )
    client = model_clients.create_model_client(provider, "selected-model", "test-key")
    messages = [{"role": "user", "content": "Optimize the kernel."}]

    assert client.generate(messages, temperature=0.25, max_tokens=2345) == "generated code"

    sdk_client.chat.completions.create.assert_called_once_with(
        model="selected-model",
        messages=messages,
        temperature=0.25,
        n=1,
        stream=False,
        max_tokens=2345,
    )


@pytest.mark.parametrize("provider", ["openai", "standard-openai"])
@pytest.mark.parametrize("model_id", [
    "gpt-5", "gpt-5-2025-08-07", "gpt-5-mini", "gpt-5-mini-2025-08-07",
    "gpt-5-nano", "gpt-5-nano-2025-08-07",
])
def test_original_gpt5_generation_parameters(provider, model_id, isolated_providers):
    sdk_client = isolated_providers["openai"].return_value
    sdk_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="generated code"))]
    )
    client = model_clients.create_model_client(provider, model_id, "test-key")
    messages = [{"role": "user", "content": "Optimize the kernel."}]

    assert client.generate(messages, temperature=0.25, max_tokens=2345) == "generated code"

    sdk_client.chat.completions.create.assert_called_once_with(
        model=model_id, messages=messages, n=1, stream=False, max_completion_tokens=2345
    )


@pytest.mark.parametrize(("provider", "model_id"), [
    ("standard-openai", "gpt-4o"),
    ("standard-openai", "gpt-5.1"),
    ("standard-openai", "gpt-5.1-2025-11-13"),
    ("standard-openai", "gpt-5.2"),
    ("standard-openai", "gpt-5.2-2025-12-11"),
    ("standard-openai", "gpt-5-chat-latest"),
    ("standard-openai", "gpt-5-custom"),
    ("standard-openai", "selected-model"),
    ("gemini", "gemini-2.5-pro"),
    ("gemini", "gpt-5"),
])
def test_other_models_keep_existing_generation_parameters(provider, model_id, isolated_providers):
    sdk_client = isolated_providers["openai"].return_value
    sdk_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="generated code"))]
    )
    client = model_clients.create_model_client(provider, model_id, "test-key")
    messages = [{"role": "user", "content": "Optimize the kernel."}]

    assert client.generate(messages, temperature=0.25, max_tokens=2345) == "generated code"

    sdk_client.chat.completions.create.assert_called_once_with(
        model=model_id, messages=messages, temperature=0.25, n=1, stream=False, max_tokens=2345
    )


@pytest.mark.parametrize("provider", ["claude", "standard-claude"])
def test_claude_generation(provider, isolated_providers):
    sdk_client = isolated_providers["anthropic"].return_value
    sdk_client.messages.create.return_value = SimpleNamespace(content=[SimpleNamespace(text="generated code")])
    client = model_clients.create_model_client(provider, "selected-model", "test-key")
    messages = [{"role": "user", "content": "Optimize the kernel."}]

    assert client.generate(messages, temperature=0.25, max_tokens=2345) == "generated code"

    sdk_client.messages.create.assert_called_once_with(
        model="selected-model", messages=messages, temperature=0.25, max_tokens=2345
    )


@pytest.mark.parametrize("num_workers", [1, 2])
def test_cli_records_provider_and_model(num_workers, monkeypatch, tmp_path):
    clients = [Mock(), Mock()]
    factory = Mock(side_effect=clients)
    pipeline = Mock(return_value={"total": 0, "success": 0, "failed": 0, "skipped": 0})
    monkeypatch.setattr(cli, "create_model_client", factory)
    monkeypatch.setattr(cli, "run_optimization_pipeline", pipeline)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "py-hip-kernel2kernel",
            "--baseline-hip-dir", str(tmp_path / "baseline"),
            "--module-dir", str(tmp_path / "module"),
            "--functional-dir", str(tmp_path / "functional"),
            "--output-dir", str(tmp_path / "output"),
            "--artifacts-dir", str(tmp_path / "artifacts"),
            "--provider", "gemini",
            "--model-id", "gemini-2.5-pro",
            "--api-key", "test-key",
            "--num-workers", str(num_workers),
        ],
    )

    cli.main()

    supplied_client, config = pipeline.call_args.args
    assert config.provider == "gemini"
    assert config.model_id == "gemini-2.5-pro"
    assert config.num_workers == num_workers
    if num_workers == 1:
        assert supplied_client is clients[0]
        factory.assert_called_once_with("gemini", "gemini-2.5-pro", "test-key")
    else:
        factory.assert_not_called()
        assert supplied_client() is clients[0]
        assert supplied_client() is clients[1]
        assert factory.call_count == 2
        factory.assert_called_with("gemini", "gemini-2.5-pro", "test-key")


def test_cli_defaults_to_public_openai_model():
    args = cli.build_parser().parse_args(
        ["--baseline-hip-dir", "baseline", "--module-dir", "module", "--functional-dir", "functional", "--output-dir", "output"]
    )

    assert args.provider == "openai"
    assert args.model_id == "gpt-4o"
