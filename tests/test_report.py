"""Tests for report rendering: text and JSON, single-candidate and comparison."""

from __future__ import annotations

import json

from llm_eval_benchmark.aggregate import aggregate
from llm_eval_benchmark.compare import compare
from llm_eval_benchmark.report import (
    comparison_to_dict,
    render_comparison_text,
    render_json_report,
    render_text_report,
    report_to_dict,
)
from llm_eval_benchmark.runner import TaskResult
from llm_eval_benchmark.scoring import ScoreResult

from .conftest import make_task


def _result(
    task_id: str, category: str, score: float, prompt: str = "p", response: str = "r"
) -> TaskResult:
    task = make_task(id=task_id, category=category, prompt=prompt)
    score_result = ScoreResult(score, "because reasons", "rule_based")
    return TaskResult(task=task, response=response, score=score_result)


class TestRenderTextReport:
    def test_includes_candidate_name_and_overall_score(self):
        report = aggregate("my-candidate", [_result("1", "factual_qa", 1.0)])
        text = render_text_report(report)
        assert "my-candidate" in text
        assert "100.0%" in text

    def test_includes_per_category_breakdown(self):
        report = aggregate(
            "x", [_result("1", "factual_qa", 1.0), _result("2", "arithmetic_reasoning", 0.5)]
        )
        text = render_text_report(report)
        assert "factual_qa" in text
        assert "arithmetic_reasoning" in text
        assert "50.0%" in text

    def test_includes_example_transcripts(self):
        result = _result("task-42", "factual_qa", 1.0, prompt="What?", response="Answer!")
        report = aggregate("x", [result])
        text = render_text_report(report, n_examples=1)
        assert "task-42" in text
        assert "What?" in text
        assert "Answer!" in text
        assert "because reasons" in text

    def test_zero_examples_omits_transcripts_section(self):
        report = aggregate("x", [_result("1", "factual_qa", 1.0)])
        text = render_text_report(report, n_examples=0)
        assert "Example transcripts" not in text


class TestReportToDict:
    def test_structure_round_trips_through_json(self):
        report = aggregate("x", [_result("1", "factual_qa", 1.0)])
        payload = json.loads(render_json_report(report))
        assert payload["candidate_name"] == "x"
        assert payload["overall_score"] == 1.0
        assert payload["category_scores"]["factual_qa"]["n_tasks"] == 1
        assert payload["results"][0]["task_id"] == "1"
        assert payload["results"][0]["score"] == 1.0

    def test_report_to_dict_matches_json_rendering(self):
        report = aggregate("x", [_result("1", "factual_qa", 0.75)])
        assert report_to_dict(report) == json.loads(render_json_report(report))


class TestComparisonRendering:
    def test_render_comparison_text_lists_both_candidates_and_winner(self):
        report_a = aggregate("careful-prompt", [_result("1", "factual_qa", 1.0)])
        report_b = aggregate("rushed-prompt", [_result("1", "factual_qa", 0.0)])
        comparison = compare(report_a, report_b)

        text = render_comparison_text(comparison)

        assert "careful-prompt" in text
        assert "rushed-prompt" in text
        assert "factual_qa" in text
        assert "careful-prompt" in text.splitlines()[-1]  # overall winner line

    def test_comparison_to_dict_structure(self):
        report_a = aggregate("a", [_result("1", "factual_qa", 1.0)])
        report_b = aggregate("b", [_result("1", "factual_qa", 0.0)])
        comparison = compare(report_a, report_b)

        payload = comparison_to_dict(comparison)

        assert payload["name_a"] == "a"
        assert payload["name_b"] == "b"
        assert payload["overall_winner"] == "a"
        assert payload["categories"][0]["category"] == "factual_qa"
        assert payload["categories"][0]["winner"] == "a"
