"""Tests for benchmark suite loading and schema validation."""

from __future__ import annotations

import pytest

from llm_eval_benchmark.suite import (
    VALID_CATEGORIES,
    SuiteValidationError,
    parse_suite,
)


def _minimal_task(**overrides) -> dict:
    task = {
        "id": "t-1",
        "category": "factual_qa",
        "prompt": "What is 2 + 2?",
        "scoring": {"method": "exact_match", "reference_answer": "4"},
        "mock_responses": {"strong": "4", "weak": "5"},
    }
    task.update(overrides)
    return task


class TestBundledSuite:
    def test_loads_30_to_40_tasks(self, bundled_suite):
        assert 30 <= len(bundled_suite) <= 40

    def test_all_categories_present(self, bundled_suite):
        categories = {task.category for task in bundled_suite}
        assert categories == VALID_CATEGORIES

    def test_ids_are_unique(self, bundled_suite):
        ids = [task.id for task in bundled_suite]
        assert len(ids) == len(set(ids))

    def test_every_task_has_two_mock_responses(self, bundled_suite):
        for task in bundled_suite:
            assert set(task.mock_responses) >= {"strong", "weak"}
            assert task.mock_responses["strong"].strip()
            assert task.mock_responses["weak"].strip()

    def test_llm_judge_tasks_have_rubric_and_heuristics(self, bundled_suite):
        llm_judge_tasks = [t for t in bundled_suite if t.scoring.method == "llm_judge"]
        assert llm_judge_tasks, "expected at least one llm_judge task"
        for task in llm_judge_tasks:
            assert task.scoring.rubric
            assert not task.scoring.heuristics.is_empty()

    def test_deterministic_tasks_have_reference_or_pattern(self, bundled_suite):
        for task in bundled_suite:
            if task.scoring.method in ("exact_match", "numeric_tolerance"):
                assert task.scoring.reference_answer
            elif task.scoring.method == "regex":
                assert task.scoring.pattern


class TestParseSuiteValidation:
    def test_valid_minimal_suite_parses(self):
        tasks = parse_suite({"tasks": [_minimal_task()]})
        assert len(tasks) == 1
        assert tasks[0].id == "t-1"

    def test_missing_tasks_key_raises(self):
        with pytest.raises(SuiteValidationError, match="'tasks'"):
            parse_suite({})

    def test_empty_tasks_list_raises(self):
        with pytest.raises(SuiteValidationError, match="non-empty"):
            parse_suite({"tasks": []})

    def test_duplicate_ids_raise(self):
        with pytest.raises(SuiteValidationError, match="duplicate task id"):
            parse_suite({"tasks": [_minimal_task(), _minimal_task()]})

    def test_invalid_category_raises(self):
        with pytest.raises(SuiteValidationError, match="category"):
            parse_suite({"tasks": [_minimal_task(category="not_a_real_category")]})

    def test_invalid_method_raises(self):
        task = _minimal_task(scoring={"method": "vibes_based", "reference_answer": "4"})
        with pytest.raises(SuiteValidationError, match="scoring.method"):
            parse_suite({"tasks": [task]})

    def test_exact_match_without_reference_answer_raises(self):
        task = _minimal_task(scoring={"method": "exact_match"})
        with pytest.raises(SuiteValidationError, match="reference_answer"):
            parse_suite({"tasks": [task]})

    def test_regex_without_pattern_raises(self):
        task = _minimal_task(scoring={"method": "regex"})
        with pytest.raises(SuiteValidationError, match="pattern"):
            parse_suite({"tasks": [task]})

    def test_regex_with_invalid_pattern_raises(self):
        task = _minimal_task(scoring={"method": "regex", "pattern": "("})
        with pytest.raises(SuiteValidationError, match="invalid regex"):
            parse_suite({"tasks": [task]})

    def test_numeric_tolerance_with_non_numeric_reference_raises(self):
        task = _minimal_task(
            scoring={"method": "numeric_tolerance", "reference_answer": "not-a-number"}
        )
        with pytest.raises(SuiteValidationError, match="not numeric"):
            parse_suite({"tasks": [task]})

    def test_numeric_tolerance_with_negative_tolerance_raises(self):
        task = _minimal_task(
            scoring={"method": "numeric_tolerance", "reference_answer": "4", "tolerance": -1}
        )
        with pytest.raises(SuiteValidationError, match="tolerance"):
            parse_suite({"tasks": [task]})

    def test_llm_judge_without_rubric_raises(self):
        task = _minimal_task(scoring={"method": "llm_judge", "heuristics": {"max_words": 1}})
        with pytest.raises(SuiteValidationError, match="rubric"):
            parse_suite({"tasks": [task]})

    def test_llm_judge_without_heuristics_raises(self):
        task = _minimal_task(scoring={"method": "llm_judge", "rubric": "Be nice."})
        with pytest.raises(SuiteValidationError, match="heuristics"):
            parse_suite({"tasks": [task]})

    def test_llm_judge_with_unknown_heuristic_key_raises(self):
        task = _minimal_task(
            scoring={
                "method": "llm_judge",
                "rubric": "Be nice.",
                "heuristics": {"max_vibes": 1},
            }
        )
        with pytest.raises(SuiteValidationError, match="unknown heuristics keys"):
            parse_suite({"tasks": [task]})

    def test_missing_mock_responses_raises(self):
        task = _minimal_task(mock_responses={"strong": "4"})
        with pytest.raises(SuiteValidationError, match="mock_responses"):
            parse_suite({"tasks": [task]})

    def test_missing_prompt_raises(self):
        task = _minimal_task(prompt="")
        with pytest.raises(SuiteValidationError, match="prompt"):
            parse_suite({"tasks": [task]})
