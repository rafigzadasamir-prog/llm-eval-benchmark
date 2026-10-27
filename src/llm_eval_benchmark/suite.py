"""Loading and validation of the benchmark suite (tasks + scoring specs)."""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any

import yaml

VALID_CATEGORIES = {
    "factual_qa",
    "arithmetic_reasoning",
    "instruction_following",
    "summarization_faithfulness",
}

# Methods a RuleBasedScorer can resolve without any LLM call.
DETERMINISTIC_METHODS = {"exact_match", "regex", "numeric_tolerance"}
VALID_METHODS = DETERMINISTIC_METHODS | {"llm_judge"}

DEFAULT_SUITE_PATH = Path(__file__).parent / "data" / "benchmark_suite.yaml"


class SuiteValidationError(ValueError):
    """Raised when a benchmark suite file fails schema validation."""


@dataclasses.dataclass(frozen=True)
class Heuristics:
    """Deterministic fallback checks used by RuleBasedJudge for llm_judge tasks."""

    must_include_any: tuple[str, ...] = ()
    must_include_all: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    min_words: int | None = None
    max_words: int | None = None
    min_lines: int | None = None
    max_lines: int | None = None

    def is_empty(self) -> bool:
        return not (
            self.must_include_any
            or self.must_include_all
            or self.forbidden_phrases
            or self.min_words is not None
            or self.max_words is not None
            or self.min_lines is not None
            or self.max_lines is not None
        )


@dataclasses.dataclass(frozen=True)
class ScoringSpec:
    """How a task's response should be scored."""

    method: str
    reference_answer: str | None = None
    aliases: tuple[str, ...] = ()
    pattern: str | None = None
    tolerance: float = 0.0
    rubric: str | None = None
    heuristics: Heuristics = dataclasses.field(default_factory=Heuristics)


@dataclasses.dataclass(frozen=True)
class Task:
    """A single benchmark task."""

    id: str
    category: str
    prompt: str
    scoring: ScoringSpec
    mock_responses: dict[str, str]


def _require(condition: bool, task_id: str, message: str) -> None:
    if not condition:
        raise SuiteValidationError(f"Task {task_id!r}: {message}")


def _parse_heuristics(task_id: str, raw: dict[str, Any] | None) -> Heuristics:
    raw = raw or {}
    allowed_keys = {
        "must_include_any",
        "must_include_all",
        "forbidden_phrases",
        "min_words",
        "max_words",
        "min_lines",
        "max_lines",
    }
    unknown = set(raw) - allowed_keys
    _require(not unknown, task_id, f"unknown heuristics keys: {sorted(unknown)}")
    return Heuristics(
        must_include_any=tuple(raw.get("must_include_any", [])),
        must_include_all=tuple(raw.get("must_include_all", [])),
        forbidden_phrases=tuple(raw.get("forbidden_phrases", [])),
        min_words=raw.get("min_words"),
        max_words=raw.get("max_words"),
        min_lines=raw.get("min_lines"),
        max_lines=raw.get("max_lines"),
    )


def _parse_scoring(task_id: str, raw: dict[str, Any]) -> ScoringSpec:
    method = raw.get("method")
    _require(
        method in VALID_METHODS,
        task_id,
        f"scoring.method must be one of {sorted(VALID_METHODS)}, got {method!r}",
    )

    reference_answer = raw.get("reference_answer")
    aliases = tuple(raw.get("aliases", []))
    pattern = raw.get("pattern")
    tolerance = float(raw.get("tolerance", 0.0))
    rubric = raw.get("rubric")
    heuristics = _parse_heuristics(task_id, raw.get("heuristics"))

    if method == "exact_match":
        _require(
            bool(reference_answer),
            task_id,
            "scoring.reference_answer is required for method 'exact_match'",
        )
    elif method == "regex":
        _require(bool(pattern), task_id, "scoring.pattern is required for method 'regex'")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise SuiteValidationError(
                f"Task {task_id!r}: invalid regex pattern {pattern!r}: {exc}"
            ) from exc
    elif method == "numeric_tolerance":
        _require(
            bool(reference_answer),
            task_id,
            "scoring.reference_answer is required for method 'numeric_tolerance'",
        )
        try:
            float(reference_answer)
        except (TypeError, ValueError) as exc:
            raise SuiteValidationError(
                f"Task {task_id!r}: reference_answer {reference_answer!r} is not numeric"
            ) from exc
        _require(tolerance >= 0, task_id, f"scoring.tolerance must be >= 0, got {tolerance}")
    else:  # llm_judge
        _require(bool(rubric), task_id, "scoring.rubric is required for method 'llm_judge'")
        _require(
            not heuristics.is_empty(),
            task_id,
            "scoring.heuristics must define at least one check for method 'llm_judge' "
            "(needed for the offline RuleBasedJudge fallback)",
        )

    return ScoringSpec(
        method=method,
        reference_answer=reference_answer,
        aliases=aliases,
        pattern=pattern,
        tolerance=tolerance,
        rubric=rubric,
        heuristics=heuristics,
    )


def _parse_task(raw: dict[str, Any]) -> Task:
    task_id = raw.get("id")
    _require(bool(task_id), "<unknown>", "missing required field 'id'")
    _require(isinstance(task_id, str), task_id, "field 'id' must be a string")

    category = raw.get("category")
    _require(
        category in VALID_CATEGORIES,
        task_id,
        f"category must be one of {sorted(VALID_CATEGORIES)}, got {category!r}",
    )

    prompt = raw.get("prompt")
    _require(bool(prompt), task_id, "missing required field 'prompt'")

    scoring_raw = raw.get("scoring")
    _require(isinstance(scoring_raw, dict), task_id, "missing required field 'scoring'")
    scoring = _parse_scoring(task_id, scoring_raw)

    mock_responses = raw.get("mock_responses")
    _require(isinstance(mock_responses, dict), task_id, "missing required field 'mock_responses'")
    for key in ("strong", "weak"):
        _require(
            bool(mock_responses.get(key)),
            task_id,
            f"mock_responses.{key} must be a non-empty string",
        )

    return Task(
        id=task_id,
        category=category,
        prompt=prompt,
        scoring=scoring,
        mock_responses=dict(mock_responses),
    )


def parse_suite(data: dict[str, Any], *, source: str = "<data>") -> list[Task]:
    """Parse and validate an already-loaded suite dict into a list of Task objects."""
    if not isinstance(data, dict) or "tasks" not in data:
        raise SuiteValidationError(
            f"{source}: top-level document must be a mapping with a 'tasks' list"
        )

    raw_tasks = data["tasks"]
    if not isinstance(raw_tasks, list) or not raw_tasks:
        raise SuiteValidationError(f"{source}: 'tasks' must be a non-empty list")

    tasks = [_parse_task(raw) for raw in raw_tasks]

    seen_ids: set[str] = set()
    for task in tasks:
        if task.id in seen_ids:
            raise SuiteValidationError(f"{source}: duplicate task id {task.id!r}")
        seen_ids.add(task.id)

    return tasks


def load_suite(path: str | Path | None = None) -> list[Task]:
    """Load and validate the benchmark suite from a YAML file.

    Defaults to the bundled suite shipped with the package.
    """
    suite_path = Path(path) if path is not None else DEFAULT_SUITE_PATH
    if not suite_path.exists():
        raise FileNotFoundError(f"Benchmark suite file not found: {suite_path}")
    data = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
    return parse_suite(data, source=str(suite_path))
