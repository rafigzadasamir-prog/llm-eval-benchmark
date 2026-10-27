"""Tests for aggregate(): per-category and overall mean scoring."""

from __future__ import annotations

import pytest

from llm_eval_benchmark.aggregate import aggregate
from llm_eval_benchmark.runner import TaskResult
from llm_eval_benchmark.scoring import ScoreResult

from .conftest import make_task


def _result(task_id: str, category: str, score: float) -> TaskResult:
    task = make_task(id=task_id, category=category)
    return TaskResult(task=task, response="r", score=ScoreResult(score, "r", "rule_based"))


def test_overall_score_is_mean_of_all_results():
    results = [_result("a", "factual_qa", 1.0), _result("b", "factual_qa", 0.0)]
    report = aggregate("candidate-x", results)
    assert report.overall_score == 0.5
    assert report.candidate_name == "candidate-x"


def test_category_scores_are_grouped_correctly():
    results = [
        _result("a", "factual_qa", 1.0),
        _result("b", "factual_qa", 0.5),
        _result("c", "arithmetic_reasoning", 0.0),
    ]
    report = aggregate("candidate-x", results)

    assert report.category_scores["factual_qa"].mean_score == 0.75
    assert report.category_scores["factual_qa"].n_tasks == 2
    assert report.category_scores["arithmetic_reasoning"].mean_score == 0.0
    assert report.category_scores["arithmetic_reasoning"].n_tasks == 1


def test_overall_score_weights_every_task_equally_across_categories():
    # 3 perfect factual_qa tasks + 1 zero arithmetic task -> overall is still
    # the mean over all 4 tasks, not an average-of-category-means.
    results = [
        _result("a", "factual_qa", 1.0),
        _result("b", "factual_qa", 1.0),
        _result("c", "factual_qa", 1.0),
        _result("d", "arithmetic_reasoning", 0.0),
    ]
    report = aggregate("candidate-x", results)
    assert report.overall_score == 0.75


def test_results_are_preserved_on_the_report():
    results = [_result("a", "factual_qa", 1.0)]
    report = aggregate("candidate-x", results)
    assert report.results == tuple(results)


def test_empty_results_raises():
    with pytest.raises(ValueError, match="empty"):
        aggregate("candidate-x", [])
