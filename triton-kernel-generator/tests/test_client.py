# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Check the inference boundary without contacting a model service."""

from unittest.mock import Mock

import pytest
import requests

from triton_kernel_gen.client import ChatClient, ModelServiceError, validate_endpoint


def completion(**overrides):
    return {
        "model": "served-model",
        "choices": [
            {
                "message": {"content": "kernel code", "reasoning_content": "analysis"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"completion_tokens": 12},
        **overrides,
    }


def client(session=None, **options):
    return ChatClient(
        "http://model-host:8000/v1/chat/completions",
        "requested-model",
        session=session or Mock(),
        **options,
    )


def test_preserves_generation_and_usage(monkeypatch):
    monkeypatch.setenv("TRITON_GEN_API_KEY", "test-only-key")
    session = Mock()
    session.post.return_value.status_code = 200
    session.post.return_value.json.return_value = completion()
    model = client(session, request_options={"top_p": 0.9})
    result = model.generate([{"role": "user", "content": "task"}], temperature=0.7, max_tokens=100)
    assert result.text == "kernel code"
    assert result.reasoning == "analysis"
    assert result.model == "served-model"
    assert result.usage == {"completion_tokens": 12}
    sent = session.post.call_args.kwargs
    assert sent["headers"]["Authorization"] == "Bearer test-only-key"
    assert sent["json"]["model"] == "requested-model"
    assert sent["json"]["top_p"] == 0.9
    assert sent["json"]["stream"] is False
    assert sent["allow_redirects"] is False
    session.post.assert_called_once()


def test_anonymous_local_endpoint_does_not_require_key(monkeypatch):
    monkeypatch.delenv("TRITON_GEN_API_KEY", raising=False)
    session = Mock()
    session.post.return_value.status_code = 200
    session.post.return_value.json.return_value = completion()
    client(session).generate([{"role": "user", "content": "task"}], temperature=0, max_tokens=100)
    assert "Authorization" not in session.post.call_args.kwargs["headers"]


def test_http_failure_does_not_echo_private_response():
    session = Mock()
    session.post.return_value.status_code = 401
    session.post.return_value.text = "private diagnostic with a secret"
    with pytest.raises(ModelServiceError, match="HTTP 401") as error:
        client(session).generate(
            [{"role": "user", "content": "task"}], temperature=0, max_tokens=100
        )
    assert "secret" not in str(error.value)
    session.post.return_value.json.assert_not_called()


def test_transport_failure_does_not_echo_url_credentials():
    session = Mock()
    session.post.side_effect = requests.Timeout("private upstream details")
    with pytest.raises(ModelServiceError, match="Timeout") as error:
        client(session).generate(
            [{"role": "user", "content": "task"}], temperature=0, max_tokens=100
        )
    assert "private" not in str(error.value)
    session.post.assert_called_once()


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        completion(choices=[]),
        completion(choices=[{}, {}]),
        completion(choices=[{"message": {"content": None}}]),
        completion(model=""),
        completion(usage=[]),
    ],
)
def test_invalid_completions_fail(response):
    session = Mock()
    session.post.return_value.status_code = 200
    session.post.return_value.json.return_value = response
    with pytest.raises(ModelServiceError):
        client(session).generate(
            [{"role": "user", "content": "task"}], temperature=0, max_tokens=100
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "file:///tmp/model",
        "http://user:pass@host",
        "https://host?key=secret",
        "http://host/#x",
        "http://host:0",
        "http://host:99999",
        "http://host:bad",
        "http://bad host/",
    ],
)
def test_invalid_endpoint_is_rejected(endpoint):
    with pytest.raises(ValueError):
        validate_endpoint(endpoint)


@pytest.mark.parametrize(
    "key", ["model", "messages", "max_tokens", "temperature", "stream", "n", "api_key", "headers"]
)
def test_request_options_cannot_change_recorded_fields(key):
    with pytest.raises(ValueError):
        client(request_options={key: "override"})
