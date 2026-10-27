"""Candidates: the thing being benchmarked (a model, or a prompt configuration)."""

from __future__ import annotations

from typing import Any, Protocol

from .anthropic_utils import extract_text
from .suite import Task


class Candidate(Protocol):
    """Anything that can produce a response to a task's prompt."""

    name: str

    def respond(self, task: Task) -> str: ...


class MockCandidate:
    """Deterministic, offline candidate that replays hand-authored canned responses.

    Every task in the bundled suite ships two author-written responses:
    ``mock_responses["strong"]`` (a careful, instruction-following, faithful
    answer) and ``mock_responses["weak"]`` (a plausible but flawed answer --
    wrong facts, ignored formatting constraints, or hallucinated summary
    details). A ``MockCandidate`` simply replays one of those two sets, which
    is what lets the whole pipeline (run -> score -> aggregate -> report)
    be exercised with zero network access, in CI or anywhere else.
    """

    def __init__(self, name: str, response_key: str):
        if response_key not in ("strong", "weak"):
            raise ValueError(f"response_key must be 'strong' or 'weak', got {response_key!r}")
        self.name = name
        self.response_key = response_key

    def respond(self, task: Task) -> str:
        try:
            return task.mock_responses[self.response_key]
        except KeyError as exc:
            raise KeyError(
                f"Task {task.id!r} has no mock_responses entry for {self.response_key!r}"
            ) from exc


class AnthropicCandidate:
    """Real candidate backed by the Anthropic Messages API.

    Additive: this class never constructs a real ``anthropic.Anthropic()``
    client itself -- the caller injects one (see ``run.py`` for the real
    wiring, and ``tests/test_candidates.py`` for an injected fake). It is
    exercised in the automated test suite only against fakes, never a real
    network call.
    """

    def __init__(
        self,
        client: Any,
        model: str,
        name: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 1024,
    ):
        self._client = client
        self._model = model
        self.name = name or f"anthropic:{model}"
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

    def respond(self, task: Task) -> str:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": [{"role": "user", "content": task.prompt}],
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        message = self._client.messages.create(**kwargs)
        return extract_text(message)
