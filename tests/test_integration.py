"""End-to-end pipeline test: suite -> run -> score -> aggregate -> compare.

Uses the real bundled suite and the two offline MockCandidate profiles with
the RuleBasedJudge fallback -- i.e. exactly the fully offline path CI runs.
This is a regression test tied to the hand-authored suite content: it
asserts the "careful" candidate beats the "rushed" one in every category,
which is true by construction (every task's "strong" response was authored
to satisfy its own scoring rule; "weak" was authored to violate it) but is
worth pinning down so a future suite edit can't silently break it.
"""

from __future__ import annotations

from llm_eval_benchmark.aggregate import aggregate
from llm_eval_benchmark.candidates import MockCandidate
from llm_eval_benchmark.compare import compare
from llm_eval_benchmark.runner import run_suite
from llm_eval_benchmark.scoring import RuleBasedJudge
from llm_eval_benchmark.suite import load_suite


def test_full_offline_pipeline_careful_beats_rushed_everywhere():
    tasks = load_suite()
    judge = RuleBasedJudge()

    careful = MockCandidate(name="careful-prompt", response_key="strong")
    rushed = MockCandidate(name="rushed-prompt", response_key="weak")

    report_careful = aggregate(careful.name, run_suite(tasks, careful, judge))
    report_rushed = aggregate(rushed.name, run_suite(tasks, rushed, judge))

    assert report_careful.overall_score > report_rushed.overall_score

    comparison = compare(report_careful, report_rushed)
    for cat in comparison.category_comparisons:
        assert cat.winner == "careful-prompt", f"expected careful-prompt to win {cat.category}"

    assert comparison.overall_winner == "careful-prompt"


def test_careful_candidate_scores_highly_but_not_suspiciously_perfect():
    tasks = load_suite()
    careful = MockCandidate(name="careful-prompt", response_key="strong")
    report = aggregate(careful.name, run_suite(tasks, careful, RuleBasedJudge()))

    # The "strong" responses are mostly correct by construction, but one
    # arithmetic task's "strong" response deliberately contains a slip
    # (see math-03 in the suite), so this should not be a perfect 1.0 --
    # which also guards against the demo looking artificially flawless.
    assert 0.9 <= report.overall_score < 1.0


def test_rushed_candidate_scores_poorly():
    tasks = load_suite()
    rushed = MockCandidate(name="rushed-prompt", response_key="weak")
    report = aggregate(rushed.name, run_suite(tasks, rushed, RuleBasedJudge()))

    assert report.overall_score < 0.3


def test_every_task_is_scored_exactly_once_per_run():
    tasks = load_suite()
    candidate = MockCandidate(name="careful-prompt", response_key="strong")
    results = run_suite(tasks, candidate, RuleBasedJudge())
    assert len(results) == len(tasks)
    assert {r.task.id for r in results} == {t.id for t in tasks}
