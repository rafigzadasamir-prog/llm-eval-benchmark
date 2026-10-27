"""LLM-as-judge scorer backed by the real Anthropic Messages API.

This is the "real" counterpart to ``RuleBasedJudge``: it asks a Claude model
to rate an open-ended response against the task's rubric and return a
structured ``{"score": float, "rationale": str}`` JSON object, with a small
retry loop in case the model's output isn't valid JSON.

It never constructs a real ``anthropic.Anthropic()`` client itself -- the
caller passes in a client (see ``run.py`` for the real wiring, and
``tests/test_llm_judge.py`` for injected fakes). This keeps the automated
test suite fully offline: ``LLMJudge`` is exercised only against fakes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Protocol

from .anthropic_utils import extract_text
from .scoring import ScoreResult
from .suite import Task

DEFAULT_MODEL = "claude-3-5-haiku-20241022"
DEFAULT_MAX_RETRIES = 2
DEFAULT_MAX_TOKENS = 300

JUDGE_SYSTEM_PROMPT = (
    "You are a strict, consistent evaluator of AI assistant responses. "
    "Given a task, a scoring rubric, and the assistant's response, output your "
    "judgment as a single JSON object with keys 'score' (a float between 0 and 1) "
    "and 'rationale' (a short, one to two sentence explanation). "
    "Output ONLY the JSON object with no other text, no markdown fences."
)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class JudgeParsingError(RuntimeError):
    """Raised when the judge model's output cannot be parsed after all retries."""


class AnthropicMessagesClient(Protocol):
    """The minimal shape of ``anthropic.Anthropic().messages`` relied upon here."""

    def create(self, **kwargs: Any) -> Any: ...


def _build_user_prompt(task: Task, response: str) -> str:
    return (
        f"Task given to the assistant:\n{task.prompt}\n\n"
        f"Rubric:\n{task.scoring.rubric}\n\n"
        f"Assistant's response:\n{response}\n\n"
        'Respond only with JSON: {"score": <float between 0 and 1>, "rationale": "<explanation>"}'
    )


def _parse_judgment(raw_text: str) -> tuple[float, str]:
    match = _JSON_OBJECT_RE.search(raw_text)
    if not match:
        raise ValueError(f"No JSON object found in judge output: {raw_text!r}")

    payload = json.loads(match.group(0))
    if "score" not in payload or "rationale" not in payload:
        raise ValueError(f"Judge JSON is missing required keys: {payload!r}")

    score = float(payload["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Judge score out of range [0, 1]: {score}")

    return score, str(payload["rationale"])


class LLMJudge:
    """LLM-as-judge scorer. Accepts any client exposing ``.messages.create(...)``."""

    def __init__(
        self,
        client: Any,
        model: str = DEFAULT_MODEL,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self._client = client
        self._model = model
        self._max_retries = max_retries
        self._max_tokens = max_tokens

    def judge(self, task: Task, response: str) -> ScoreResult:
        last_error: Exception | None = None
        for _attempt in range(self._max_retries + 1):
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": _build_user_prompt(task, response)}],
            )
            raw_text = extract_text(message)
            try:
                score, rationale = _parse_judgment(raw_text)
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                continue
            return ScoreResult(score, rationale, "llm_judge")

        raise JudgeParsingError(
            f"Could not parse a valid judgment for task {task.id!r} after "
            f"{self._max_retries + 1} attempt(s): {last_error}"
        )
