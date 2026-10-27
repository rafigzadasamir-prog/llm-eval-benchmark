"""Tests for the Anthropic-backed LLMJudge, exercised only against fakes.

No test in this module (or anywhere else in the suite) constructs a real
anthropic.Anthropic() client or makes a network call -- only the lightweight
FakeAnthropicClient below, which mimics the minimal
`client.messages.create(...) -> message` shape the real SDK exposes.
"""

from __future__ import annotations

import pytest

from llm_eval_benchmark.judge_anthropic import JudgeParsingError, LLMJudge
from llm_eval_benchmark.suite import Heuristics, ScoringSpec

from .conftest import make_task


class _FakeTextBlock:
    def __init__(self, text: str):
        self.text = text


class _FakeMessage:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


class _FakeMessagesEndpoint:
    """Returns one queued response per call; records every call's kwargs."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("FakeMessagesEndpoint ran out of queued responses")
        return _FakeMessage(self._responses.pop(0))


class _FakeAnthropicClient:
    def __init__(self, responses: list[str]):
        self.messages = _FakeMessagesEndpoint(responses)


def _task():
    return make_task(
        category="instruction_following",
        prompt="Say just the word 'done'.",
        scoring=ScoringSpec(
            method="llm_judge",
            rubric="A perfect response is just the single word 'done'.",
            heuristics=Heuristics(max_words=1, must_include_any=("done",)),
        ),
        mock_responses={"strong": "Done", "weak": "I am done with the task."},
    )


class TestSuccessfulParsing:
    def test_parses_valid_json_on_first_attempt(self):
        client = _FakeAnthropicClient(['{"score": 0.9, "rationale": "Close enough."}'])
        judge = LLMJudge(client=client)

        result = judge.judge(_task(), "Done")

        assert result.score == 0.9
        assert result.rationale == "Close enough."
        assert result.method == "llm_judge"
        assert len(client.messages.calls) == 1

    def test_parses_json_wrapped_in_markdown_fences(self):
        fenced = 'Here you go:\n```json\n{"score": 1.0, "rationale": "Perfect."}\n```'
        client = _FakeAnthropicClient([fenced])
        judge = LLMJudge(client=client)

        result = judge.judge(_task(), "Done")

        assert result.score == 1.0
        assert result.rationale == "Perfect."

    def test_sends_prompt_and_rubric_to_the_client(self):
        task = _task()
        client = _FakeAnthropicClient(['{"score": 0.5, "rationale": "ok"}'])
        judge = LLMJudge(client=client, model="claude-test-model")

        judge.judge(task, "Done")

        call = client.messages.calls[0]
        assert call["model"] == "claude-test-model"
        assert task.prompt in call["messages"][0]["content"]
        assert task.scoring.rubric in call["messages"][0]["content"]


class TestRetryLogic:
    def test_retries_after_malformed_json_and_then_succeeds(self):
        client = _FakeAnthropicClient(
            [
                "this is not json at all",
                '{"score": 0.8, "rationale": "Recovered on retry."}',
            ]
        )
        judge = LLMJudge(client=client, max_retries=2)

        result = judge.judge(_task(), "Done")

        assert result.score == 0.8
        assert len(client.messages.calls) == 2

    def test_retries_after_out_of_range_score(self):
        client = _FakeAnthropicClient(
            [
                '{"score": 1.7, "rationale": "Out of range."}',
                '{"score": 0.6, "rationale": "Valid now."}',
            ]
        )
        judge = LLMJudge(client=client, max_retries=2)

        result = judge.judge(_task(), "Done")

        assert result.score == 0.6
        assert len(client.messages.calls) == 2

    def test_retries_after_missing_keys(self):
        client = _FakeAnthropicClient(
            [
                '{"verdict": "great"}',
                '{"score": 0.4, "rationale": "Now valid."}',
            ]
        )
        judge = LLMJudge(client=client, max_retries=2)

        result = judge.judge(_task(), "Done")

        assert result.score == 0.4

    def test_raises_judge_parsing_error_after_exhausting_retries(self):
        client = _FakeAnthropicClient(["not json", "still not json", "nope"])
        judge = LLMJudge(client=client, max_retries=2)

        with pytest.raises(JudgeParsingError, match="after 3 attempt"):
            judge.judge(_task(), "Done")

        assert len(client.messages.calls) == 3

    def test_default_max_retries_allows_two_follow_up_attempts(self):
        client = _FakeAnthropicClient(["bad", "bad", "bad"])
        judge = LLMJudge(client=client)  # default max_retries

        with pytest.raises(JudgeParsingError):
            judge.judge(_task(), "Done")

        assert len(client.messages.calls) == 3  # 1 initial + 2 retries
