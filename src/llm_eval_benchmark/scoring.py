"""Deterministic rule-based scoring and the offline RuleBasedJudge fallback.

Two scoring paths exist:

- ``RuleBasedScorer`` resolves the deterministic methods (``exact_match``,
  ``regex``, ``numeric_tolerance``) with pure string/number logic -- no model
  calls, no network, fully reproducible.
- ``RuleBasedJudge`` resolves ``llm_judge`` tasks *offline* using the
  hand-authored ``heuristics`` bundled with each task. It is the fallback used
  whenever no Anthropic API key is configured (including in CI), and the
  real ``LLMJudge`` (see ``judge_anthropic.py``) is a drop-in replacement for
  it that shares the same ``Judge`` protocol.
"""

from __future__ import annotations

import dataclasses
import re
from typing import Protocol

from .suite import DETERMINISTIC_METHODS, Task

_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


@dataclasses.dataclass(frozen=True)
class ScoreResult:
    """The outcome of scoring a single candidate response to a single task."""

    score: float
    rationale: str
    method: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be within [0, 1], got {self.score}")


class Judge(Protocol):
    """Anything that can resolve an ``llm_judge`` task: real or rule-based."""

    def judge(self, task: Task, response: str) -> ScoreResult: ...


def _extract_last_number(text: str) -> float | None:
    """Return the last number appearing in ``text``, treating it as the final answer."""
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return float(matches[-1].replace(",", ""))


def _contains_whole_phrase(response: str, phrase: str) -> bool:
    """Case-insensitive whole-word/phrase containment check."""
    pattern = r"\b" + re.escape(phrase.strip().lower()) + r"\b"
    return re.search(pattern, response.lower()) is not None


class RuleBasedScorer:
    """Deterministic, offline scorer for exact_match / regex / numeric_tolerance tasks."""

    def score(self, task: Task, response: str) -> ScoreResult:
        spec = task.scoring
        if spec.method not in DETERMINISTIC_METHODS:
            raise ValueError(f"RuleBasedScorer cannot handle scoring method {spec.method!r}")
        if spec.method == "exact_match":
            return self._score_exact_match(spec, response)
        if spec.method == "regex":
            return self._score_regex(spec, response)
        return self._score_numeric_tolerance(spec, response)

    @staticmethod
    def _score_exact_match(spec, response: str) -> ScoreResult:
        candidates = [spec.reference_answer, *spec.aliases]
        for candidate in candidates:
            if candidate and _contains_whole_phrase(response, candidate):
                return ScoreResult(
                    1.0, f"Response contains expected answer {candidate!r}.", "rule_based"
                )
        return ScoreResult(
            0.0,
            f"Response did not contain expected answer {spec.reference_answer!r} (or its aliases).",
            "rule_based",
        )

    @staticmethod
    def _score_regex(spec, response: str) -> ScoreResult:
        if re.search(spec.pattern, response, re.IGNORECASE):
            return ScoreResult(
                1.0, f"Response matched expected pattern {spec.pattern!r}.", "rule_based"
            )
        return ScoreResult(
            0.0, f"Response did not match expected pattern {spec.pattern!r}.", "rule_based"
        )

    @staticmethod
    def _score_numeric_tolerance(spec, response: str) -> ScoreResult:
        expected = float(spec.reference_answer)
        actual = _extract_last_number(response)
        if actual is None:
            return ScoreResult(0.0, "No numeric answer found in response.", "rule_based")
        if abs(actual - expected) <= spec.tolerance:
            return ScoreResult(
                1.0,
                f"Extracted value {actual} is within tolerance ({spec.tolerance}) of "
                f"expected {expected}.",
                "rule_based",
            )
        return ScoreResult(
            0.0,
            f"Extracted value {actual} differs from expected {expected} by more than "
            f"tolerance {spec.tolerance}.",
            "rule_based",
        )


class RuleBasedJudge:
    """Offline heuristic judge used whenever no real LLM judge is configured.

    Each heuristic field defined on the task contributes exactly one pass/fail
    check; the resulting score is the fraction of checks that passed. This
    keeps the fallback fully deterministic and makes every task's offline
    grading behavior easy to reason about and unit test.
    """

    def judge(self, task: Task, response: str) -> ScoreResult:
        heuristics = task.scoring.heuristics
        lower = response.lower()
        checks: list[tuple[str, bool]] = []

        if heuristics.must_include_any:
            ok = any(phrase.lower() in lower for phrase in heuristics.must_include_any)
            checks.append((f"includes any of {list(heuristics.must_include_any)}", ok))

        if heuristics.must_include_all:
            ok = all(phrase.lower() in lower for phrase in heuristics.must_include_all)
            checks.append((f"includes all of {list(heuristics.must_include_all)}", ok))

        if heuristics.forbidden_phrases:
            ok = not any(phrase.lower() in lower for phrase in heuristics.forbidden_phrases)
            checks.append((f"avoids forbidden phrases {list(heuristics.forbidden_phrases)}", ok))

        if heuristics.min_words is not None or heuristics.max_words is not None:
            n_words = len(response.split())
            ok = True
            if heuristics.min_words is not None:
                ok = ok and n_words >= heuristics.min_words
            if heuristics.max_words is not None:
                ok = ok and n_words <= heuristics.max_words
            checks.append((f"word count {n_words} within bounds", ok))

        if heuristics.min_lines is not None or heuristics.max_lines is not None:
            n_lines = len([line for line in response.strip().splitlines() if line.strip()])
            ok = True
            if heuristics.min_lines is not None:
                ok = ok and n_lines >= heuristics.min_lines
            if heuristics.max_lines is not None:
                ok = ok and n_lines <= heuristics.max_lines
            checks.append((f"line count {n_lines} within bounds", ok))

        if not checks:
            # Suite validation guarantees every llm_judge task defines at
            # least one heuristic, but guard against programmatic misuse.
            return ScoreResult(
                0.5, "No heuristics defined for this task; neutral score.", "rule_based_judge"
            )

        passed = sum(1 for _, ok in checks if ok)
        score = passed / len(checks)
        rationale = "; ".join(f"{'PASS' if ok else 'FAIL'}: {desc}" for desc, ok in checks)
        return ScoreResult(score, rationale, "rule_based_judge")


def score_task(task: Task, response: str, judge: Judge) -> ScoreResult:
    """Dispatch a task to the deterministic scorer or the given judge, as appropriate."""
    if task.scoring.method in DETERMINISTIC_METHODS:
        return RuleBasedScorer().score(task, response)
    if task.scoring.method == "llm_judge":
        return judge.judge(task, response)
    raise ValueError(f"Unknown scoring method: {task.scoring.method!r}")
