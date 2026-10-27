"""Tests for MockCandidate and AnthropicCandidate (the latter via an injected fake)."""

from __future__ import annotations

import pytest

from llm_eval_benchmark.candidates import AnthropicCandidate, MockCandidate

from .conftest import make_task


class TestMockCandidate:
    def test_strong_profile_returns_strong_response(self, exact_match_task):
        candidate = MockCandidate(name="careful-prompt", response_key="strong")
        assert candidate.respond(exact_match_task) == exact_match_task.mock_responses["strong"]

    def test_weak_profile_returns_weak_response(self, exact_match_task):
        candidate = MockCandidate(name="rushed-prompt", response_key="weak")
        assert candidate.respond(exact_match_task) == exact_match_task.mock_responses["weak"]

    def test_invalid_response_key_raises(self):
        with pytest.raises(ValueError, match="response_key"):
            MockCandidate(name="x", response_key="medium")

    def test_respond_raises_clear_error_when_task_missing_key(self):
        task = make_task(mock_responses={"strong": "only strong here"})
        candidate = MockCandidate(name="rushed-prompt", response_key="weak")
        with pytest.raises(KeyError, match="weak"):
            candidate.respond(task)

    def test_name_is_stored(self):
        candidate = MockCandidate(name="careful-prompt", response_key="strong")
        assert candidate.name == "careful-prompt"


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeMessage:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


class _FakeMessagesEndpoint:
    def __init__(self, text: str):
        self._text = text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeMessage(self._text)


class _FakeAnthropicClient:
    def __init__(self, text: str):
        self.messages = _FakeMessagesEndpoint(text)


class TestAnthropicCandidate:
    def test_respond_returns_extracted_text(self):
        client = _FakeAnthropicClient("Paris is the capital of France.")
        candidate = AnthropicCandidate(client=client, model="claude-test-model")

        response = candidate.respond(make_task(prompt="What is the capital of France?"))

        assert response == "Paris is the capital of France."

    def test_sends_model_and_prompt(self):
        client = _FakeAnthropicClient("ok")
        candidate = AnthropicCandidate(client=client, model="claude-test-model")

        candidate.respond(make_task(prompt="Hello?"))

        call = client.messages.calls[0]
        assert call["model"] == "claude-test-model"
        assert call["messages"] == [{"role": "user", "content": "Hello?"}]
        assert "system" not in call

    def test_includes_system_prompt_when_given(self):
        client = _FakeAnthropicClient("ok")
        candidate = AnthropicCandidate(
            client=client, model="claude-test-model", system_prompt="Be concise."
        )

        candidate.respond(make_task(prompt="Hello?"))

        assert client.messages.calls[0]["system"] == "Be concise."

    def test_default_name_includes_model(self):
        candidate = AnthropicCandidate(client=_FakeAnthropicClient("ok"), model="claude-test-model")
        assert candidate.name == "anthropic:claude-test-model"

    def test_custom_name_overrides_default(self):
        candidate = AnthropicCandidate(
            client=_FakeAnthropicClient("ok"), model="claude-test-model", name="my-candidate"
        )
        assert candidate.name == "my-candidate"
