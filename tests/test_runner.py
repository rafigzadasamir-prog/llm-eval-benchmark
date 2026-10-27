"""Tests for run_suite: wiring a candidate and judge over a list of tasks."""

from __future__ import annotations

from llm_eval_benchmark.runner import run_suite
from llm_eval_benchmark.scoring import ScoreResult

from .conftest import make_task


class _EchoCandidate:
    name = "echo"

    def respond(self, task):
        return task.mock_responses["strong"]


class _StubJudge:
    def __init__(self):
        self.seen = []

    def judge(self, task, response):
        self.seen.append((task.id, response))
        return ScoreResult(0.42, "stub", "rule_based_judge")


def test_run_suite_preserves_task_order(exact_match_task, numeric_task, llm_judge_task):
    tasks = [exact_match_task, numeric_task, llm_judge_task]
    results = run_suite(tasks, _EchoCandidate(), _StubJudge())

    assert [r.task.id for r in results] == [t.id for t in tasks]


def test_run_suite_uses_deterministic_scorer_for_exact_match(exact_match_task):
    results = run_suite([exact_match_task], _EchoCandidate(), _StubJudge())
    assert results[0].score.method == "rule_based"
    assert results[0].score.score == 1.0


def test_run_suite_uses_judge_for_llm_judge_tasks(llm_judge_task):
    judge = _StubJudge()
    results = run_suite([llm_judge_task], _EchoCandidate(), judge)
    assert results[0].score.score == 0.42
    assert judge.seen == [(llm_judge_task.id, llm_judge_task.mock_responses["strong"])]


def test_run_suite_records_actual_response_text(exact_match_task):
    results = run_suite([exact_match_task], _EchoCandidate(), _StubJudge())
    assert results[0].response == exact_match_task.mock_responses["strong"]


def test_run_suite_handles_empty_task_list():
    assert run_suite([], _EchoCandidate(), _StubJudge()) == []


def test_run_suite_calls_candidate_once_per_task():
    calls = []

    class CountingCandidate:
        name = "counter"

        def respond(self, task):
            calls.append(task.id)
            return "x"

    tasks = [make_task(id="a"), make_task(id="b"), make_task(id="c")]
    run_suite(tasks, CountingCandidate(), _StubJudge())
    assert calls == ["a", "b", "c"]
