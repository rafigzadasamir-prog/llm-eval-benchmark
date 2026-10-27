"""Tests for compare(): side-by-side comparison of two candidates' reports."""

from __future__ import annotations

from llm_eval_benchmark.aggregate import aggregate
from llm_eval_benchmark.compare import compare
from llm_eval_benchmark.runner import TaskResult
from llm_eval_benchmark.scoring import ScoreResult

from .conftest import make_task


def _result(task_id: str, category: str, score: float) -> TaskResult:
    task = make_task(id=task_id, category=category)
    return TaskResult(task=task, response="r", score=ScoreResult(score, "r", "rule_based"))


def test_candidate_a_wins_a_category_it_scores_higher_in():
    report_a = aggregate("a", [_result("1", "factual_qa", 1.0)])
    report_b = aggregate("b", [_result("1", "factual_qa", 0.0)])

    comparison = compare(report_a, report_b)

    assert len(comparison.category_comparisons) == 1
    cat = comparison.category_comparisons[0]
    assert cat.category == "factual_qa"
    assert cat.winner == "a"
    assert comparison.overall_winner == "a"


def test_candidate_b_wins_when_it_scores_higher():
    report_a = aggregate("a", [_result("1", "factual_qa", 0.2)])
    report_b = aggregate("b", [_result("1", "factual_qa", 0.8)])

    comparison = compare(report_a, report_b)

    assert comparison.category_comparisons[0].winner == "b"
    assert comparison.overall_winner == "b"


def test_tie_when_scores_are_equal():
    report_a = aggregate("a", [_result("1", "factual_qa", 0.5)])
    report_b = aggregate("b", [_result("1", "factual_qa", 0.5)])

    comparison = compare(report_a, report_b)

    assert comparison.category_comparisons[0].winner == "tie"
    assert comparison.overall_winner == "tie"


def test_per_category_winners_can_differ_from_overall_winner():
    report_a = aggregate(
        "a",
        [
            _result("1", "factual_qa", 1.0),
            _result("2", "arithmetic_reasoning", 0.0),
        ],
    )
    report_b = aggregate(
        "b",
        [
            _result("1", "factual_qa", 0.0),
            _result("2", "arithmetic_reasoning", 1.0),
        ],
    )

    comparison = compare(report_a, report_b)
    winners = {c.category: c.winner for c in comparison.category_comparisons}

    assert winners["factual_qa"] == "a"
    assert winners["arithmetic_reasoning"] == "b"
    # Overall scores are tied (0.5 each) even though no category tied.
    assert comparison.overall_winner == "tie"


def test_missing_category_in_one_report_is_treated_as_zero():
    report_a = aggregate("a", [_result("1", "factual_qa", 1.0)])
    report_b = aggregate("b", [_result("1", "arithmetic_reasoning", 1.0)])

    comparison = compare(report_a, report_b)
    winners = {
        c.category: (c.score_a, c.score_b, c.winner) for c in comparison.category_comparisons
    }

    assert winners["factual_qa"] == (1.0, 0.0, "a")
    assert winners["arithmetic_reasoning"] == (0.0, 1.0, "b")


def test_comparison_report_retains_candidate_names():
    report_a = aggregate("careful-prompt", [_result("1", "factual_qa", 1.0)])
    report_b = aggregate("rushed-prompt", [_result("1", "factual_qa", 0.0)])

    comparison = compare(report_a, report_b)

    assert comparison.name_a == "careful-prompt"
    assert comparison.name_b == "rushed-prompt"
