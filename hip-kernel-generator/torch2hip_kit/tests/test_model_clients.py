# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from torch2hip_kit import model_clients


@pytest.fixture
def sdk_constructors(monkeypatch):
    openai_constructor = Mock()
    anthropic_constructor = Mock()
    monkeypatch.setattr(model_clients.openai, "OpenAI", openai_constructor)
    monkeypatch.setattr(model_clients.anthropic, "Anthropic", anthropic_constructor)
    return openai_constructor, anthropic_constructor


@pytest.mark.parametrize("class_name", ["StandardOpenAIModel", "OpenAIModel"])
def test_openai_classes_use_public_client_and_default_model(sdk_constructors, class_name) -> None:
    openai_constructor, anthropic_constructor = sdk_constructors

    model = getattr(model_clients, class_name)(api_key="test-key")

    openai_constructor.assert_called_once_with(api_key="test-key")
    anthropic_constructor.assert_not_called()
    assert model.model_id == "gpt-4o"


def test_openai_legacy_constructor_keeps_positional_arguments(sdk_constructors) -> None:
    openai_constructor, _ = sdk_constructors

    model = model_clients.OpenAIModel("model-test", "legacy-version", "test-key")

    openai_constructor.assert_called_once_with(api_key="test-key")
    assert model.model_id == "model-test"


@pytest.mark.parametrize("class_name", ["StandardClaudeModel", "ClaudeModel"])
def test_claude_classes_use_public_client_and_default_model(sdk_constructors, class_name) -> None:
    openai_constructor, anthropic_constructor = sdk_constructors

    model = getattr(model_clients, class_name)(api_key="test-key")

    anthropic_constructor.assert_called_once_with(api_key="test-key", base_url="https://api.anthropic.com")
    openai_constructor.assert_not_called()
    assert model.model_id == "claude-sonnet-4-20250514"


def test_gemini_uses_public_compatible_endpoint_and_default_model(sdk_constructors) -> None:
    openai_constructor, anthropic_constructor = sdk_constructors

    model = model_clients.GeminiModel(api_key="test-key")

    openai_constructor.assert_called_once_with(
        api_key="test-key", base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    anthropic_constructor.assert_not_called()
    assert model.model_id == "gemini-2.5-pro"


@pytest.mark.parametrize("class_name", ["StandardOpenAIModel", "OpenAIModel", "GeminiModel"])
def test_openai_compatible_generation_forwards_request_and_reads_text(sdk_constructors, class_name) -> None:
    openai_constructor, _ = sdk_constructors
    create = openai_constructor.return_value.chat.completions.create
    create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="generated kernel"))])
    model = getattr(model_clients, class_name)(model_id="model-test", api_key="test-key")
    messages = [{"role": "user", "content": "Convert this module."}]

    result = model.generate(messages, temperature=0.25, max_tokens=1234)

    assert result == "generated kernel"
    create.assert_called_once_with(
        model="model-test", messages=messages, temperature=0.25, n=1, stream=False, max_tokens=1234
    )


@pytest.mark.parametrize("class_name", ["StandardOpenAIModel", "OpenAIModel"])
@pytest.mark.parametrize(
    "model_id",
    [
        "gpt-5", "gpt-5-2025-08-07", "gpt-5-mini", "gpt-5-mini-2025-08-07",
        "gpt-5-nano", "gpt-5-nano-2025-08-07",
    ],
)
def test_original_gpt5_uses_completion_budget_and_default_sampling(
    sdk_constructors, class_name, model_id
) -> None:
    openai_constructor, _ = sdk_constructors
    create = openai_constructor.return_value.chat.completions.create
    create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="HIP code"))])
    model = getattr(model_clients, class_name)(model_id=model_id, api_key="test-key")
    messages = [{"role": "user", "content": "Convert this module."}]

    assert model.generate(messages, temperature=0.25, max_tokens=1234) == "HIP code"

    create.assert_called_once_with(
        model=model_id, messages=messages, n=1, stream=False, max_completion_tokens=1234
    )


@pytest.mark.parametrize("model_id", ["gpt-4o", "gpt-5.1", "gpt-5.2", "gpt-5-chat-latest", "local-model"])
def test_other_openai_models_keep_existing_request_parameters(sdk_constructors, model_id) -> None:
    openai_constructor, _ = sdk_constructors
    create = openai_constructor.return_value.chat.completions.create
    create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="HIP code"))])
    model = model_clients.StandardOpenAIModel(model_id=model_id, api_key="test-key")
    messages = [{"role": "user", "content": "Convert this module."}]

    assert model.generate(messages, temperature=0.25, max_tokens=1234) == "HIP code"

    create.assert_called_once_with(
        model=model_id, messages=messages, temperature=0.25, n=1, stream=False, max_tokens=1234
    )


def test_gemini_keeps_its_request_parameters_for_arbitrary_model_names(sdk_constructors) -> None:
    openai_constructor, _ = sdk_constructors
    create = openai_constructor.return_value.chat.completions.create
    create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="HIP code"))])
    model = model_clients.GeminiModel(model_id="gpt-5", api_key="test-key")
    messages = [{"role": "user", "content": "Convert this module."}]

    assert model.generate(messages, temperature=0.25, max_tokens=1234) == "HIP code"

    create.assert_called_once_with(
        model="gpt-5", messages=messages, temperature=0.25, n=1, stream=False, max_tokens=1234
    )


@pytest.mark.parametrize("class_name", ["StandardClaudeModel", "ClaudeModel"])
def test_claude_generation_forwards_request_and_reads_text(sdk_constructors, class_name) -> None:
    _, anthropic_constructor = sdk_constructors
    create = anthropic_constructor.return_value.messages.create
    create.return_value = SimpleNamespace(content=[SimpleNamespace(text="generated kernel")])
    model = getattr(model_clients, class_name)(model_id="model-test", api_key="test-key")
    messages = [{"role": "user", "content": "Convert this module."}]

    result = model.generate(messages, temperature=0.25, max_tokens=1234)

    assert result == "generated kernel"
    create.assert_called_once_with(model="model-test", messages=messages, temperature=0.25, max_tokens=1234)


@pytest.mark.parametrize("class_name", ["StandardClaudeModel", "ClaudeModel"])
def test_claude_extracts_system_messages_without_changing_input(sdk_constructors, class_name) -> None:
    _, anthropic_constructor = sdk_constructors
    create = anthropic_constructor.return_value.messages.create
    create.return_value = SimpleNamespace(content=[SimpleNamespace(text="generated kernel")])
    model = getattr(model_clients, class_name)(model_id="model-test", api_key="test-key")
    structured_system = [{"type": "text", "text": "Use HIP."}]
    messages = [
        {"role": "system", "content": "Convert the module."},
        {"role": "system", "content": structured_system},
        {"role": "user", "content": "Input module."},
        {"role": "assistant", "content": "Previous attempt."},
        {"role": "user", "content": "Correct the error."},
    ]

    assert model.generate(messages, temperature=0.25, max_tokens=1234) == "generated kernel"

    create.assert_called_once_with(
        model="model-test",
        messages=messages[2:],
        temperature=0.25,
        max_tokens=1234,
        system=[
            {"type": "text", "text": "Convert the module."},
            {"type": "text", "text": "Use HIP."},
        ],
    )
    assert len(messages) == 5
    assert messages[0] == {"role": "system", "content": "Convert the module."}
    assert structured_system == [{"type": "text", "text": "Use HIP."}]


@pytest.mark.parametrize(
    ("class_name", "temperature", "max_tokens"),
    [("StandardOpenAIModel", 0.0, 12000), ("OpenAIModel", 1.0, 12000), ("GeminiModel", 1.0, 30000)],
)
def test_openai_compatible_classes_keep_generation_defaults(
    sdk_constructors, class_name, temperature, max_tokens
) -> None:
    openai_constructor, _ = sdk_constructors
    create = openai_constructor.return_value.chat.completions.create
    create.return_value = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="generated kernel"))])
    model = getattr(model_clients, class_name)(model_id="model-test", api_key="test-key")
    messages = [{"role": "user", "content": "Convert this module."}]

    assert model.generate(messages) == "generated kernel"

    create.assert_called_once_with(
        model="model-test", messages=messages, temperature=temperature, n=1, stream=False, max_tokens=max_tokens
    )


@pytest.mark.parametrize(("class_name", "temperature"), [("StandardClaudeModel", 0.0), ("ClaudeModel", 1.0)])
def test_claude_classes_keep_generation_defaults(sdk_constructors, class_name, temperature) -> None:
    _, anthropic_constructor = sdk_constructors
    create = anthropic_constructor.return_value.messages.create
    create.return_value = SimpleNamespace(content=[SimpleNamespace(text="generated kernel")])
    model = getattr(model_clients, class_name)(model_id="model-test", api_key="test-key")
    messages = [{"role": "user", "content": "Convert this module."}]

    assert model.generate(messages) == "generated kernel"

    create.assert_called_once_with(model="model-test", messages=messages, temperature=temperature, max_tokens=16000)


@pytest.mark.parametrize(
    "class_name", ["StandardOpenAIModel", "OpenAIModel", "StandardClaudeModel", "ClaudeModel", "GeminiModel"]
)
def test_direct_constructor_rejects_missing_key(sdk_constructors, class_name) -> None:
    openai_constructor, anthropic_constructor = sdk_constructors

    with pytest.raises(ValueError, match="No API key provided"):
        getattr(model_clients, class_name)()

    openai_constructor.assert_not_called()
    anthropic_constructor.assert_not_called()
