"""Shared fixtures: small hand-built tasks for unit tests that don't need the
full bundled suite, plus a fixture for loading the real bundled suite."""

from __future__ import annotations

import pytest

from llm_eval_benchmark.suite import Heuristics, ScoringSpec, Task, load_suite


def make_task(
    id: str = "t-1",
    category: str = "factual_qa",
    prompt: str = "What is 2 + 2?",
    scoring: ScoringSpec | None = None,
    mock_responses: dict[str, str] | None = None,
) -> Task:
    if scoring is None:
        scoring = ScoringSpec(method="exact_match", reference_answer="4")
    if mock_responses is None:
        mock_responses = {"strong": "4", "weak": "5"}
    return Task(
        id=id, category=category, prompt=prompt, scoring=scoring, mock_responses=mock_responses
    )


@pytest.fixture
def exact_match_task() -> Task:
    return make_task(
        id="fqa-test",
        category="factual_qa",
        prompt="What is the capital of Testland?",
        scoring=ScoringSpec(
            method="exact_match", reference_answer="Testville", aliases=("TV City",)
        ),
        mock_responses={
            "strong": "The capital is Testville.",
            "weak": "The capital is Nowhere.",
        },
    )


@pytest.fixture
def regex_task() -> Task:
    return make_task(
        id="fqa-regex",
        category="factual_qa",
        prompt="What year is it in the story?",
        scoring=ScoringSpec(method="regex", pattern=r"\b2042\b"),
        mock_responses={"strong": "It is the year 2042.", "weak": "It is the year 2043."},
    )


@pytest.fixture
def numeric_task() -> Task:
    return make_task(
        id="math-test",
        category="arithmetic_reasoning",
        prompt="What is 6 times 7?",
        scoring=ScoringSpec(method="numeric_tolerance", reference_answer="42", tolerance=0),
        mock_responses={"strong": "6 times 7 is 42.", "weak": "6 times 7 is 48."},
    )


@pytest.fixture
def numeric_tolerance_task() -> Task:
    return make_task(
        id="math-tol",
        category="arithmetic_reasoning",
        prompt="What is 10 divided by 3, to one decimal place?",
        scoring=ScoringSpec(method="numeric_tolerance", reference_answer="3.3", tolerance=0.1),
        mock_responses={"strong": "Approximately 3.3", "weak": "Approximately 5.0"},
    )


@pytest.fixture
def llm_judge_task() -> Task:
    return make_task(
        id="instr-test",
        category="instruction_following",
        prompt="Reply with just the word 'done'.",
        scoring=ScoringSpec(
            method="llm_judge",
            rubric="A perfect response is just the single word 'done'.",
            heuristics=Heuristics(max_words=1, must_include_any=("done",)),
        ),
        mock_responses={
            "strong": "Done",
            "weak": "I have completed the task, so the answer is done.",
        },
    )


@pytest.fixture
def bundled_suite():
    return load_suite()
