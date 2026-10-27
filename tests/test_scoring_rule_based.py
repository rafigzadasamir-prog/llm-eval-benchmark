"""Tests for the deterministic RuleBasedScorer and the score_task dispatcher."""

from __future__ import annotations

import pytest

from llm_eval_benchmark.scoring import RuleBasedScorer, ScoreResult, score_task
from llm_eval_benchmark.suite import ScoringSpec

from .conftest import make_task


class TestScoreResult:
    def test_rejects_score_above_one(self):
        with pytest.raises(ValueError, match="0, 1"):
            ScoreResult(score=1.5, rationale="oops", method="rule_based")

    def test_rejects_negative_score(self):
        with pytest.raises(ValueError, match="0, 1"):
            ScoreResult(score=-0.1, rationale="oops", method="rule_based")

    def test_accepts_boundary_scores(self):
        assert ScoreResult(0.0, "r", "rule_based").score == 0.0
        assert ScoreResult(1.0, "r", "rule_based").score == 1.0


class TestExactMatch:
    def test_matches_reference_answer_embedded_in_sentence(self, exact_match_task):
        result = RuleBasedScorer().score(exact_match_task, "The capital is Testville, apparently.")
        assert result.score == 1.0
        assert result.method == "rule_based"

    def test_matches_alias(self, exact_match_task):
        result = RuleBasedScorer().score(exact_match_task, "I'd guess TV City, but not certain.")
        assert result.score == 1.0

    def test_wrong_answer_scores_zero(self, exact_match_task):
        result = RuleBasedScorer().score(exact_match_task, "The capital is Nowhere.")
        assert result.score == 0.0

    def test_match_is_case_insensitive(self):
        task = make_task(scoring=ScoringSpec(method="exact_match", reference_answer="Paris"))
        result = RuleBasedScorer().score(task, "the capital is PARIS.")
        assert result.score == 1.0

    def test_match_requires_whole_word_not_substring(self):
        # "Au" should not spuriously match inside "August" or "Austria".
        task = make_task(scoring=ScoringSpec(method="exact_match", reference_answer="Au"))
        result = RuleBasedScorer().score(task, "It happened in Austria, in August.")
        assert result.score == 0.0


class TestRegex:
    def test_matching_pattern_scores_one(self, regex_task):
        result = RuleBasedScorer().score(regex_task, "It is the year 2042.")
        assert result.score == 1.0

    def test_non_matching_pattern_scores_zero(self, regex_task):
        result = RuleBasedScorer().score(regex_task, "It is the year 2043.")
        assert result.score == 0.0

    def test_alternation_pattern(self):
        task = make_task(scoring=ScoringSpec(method="regex", pattern=r"\b(descend|downward)\b"))
        assert RuleBasedScorer().score(task, "The answer is downward motion.").score == 1.0
        assert RuleBasedScorer().score(task, "The answer is upward motion.").score == 0.0


class TestNumericTolerance:
    def test_exact_match_scores_one(self, numeric_task):
        assert RuleBasedScorer().score(numeric_task, "6 times 7 is 42.").score == 1.0

    def test_wrong_number_scores_zero(self, numeric_task):
        assert RuleBasedScorer().score(numeric_task, "6 times 7 is 48.").score == 0.0

    def test_no_number_found_scores_zero(self, numeric_task):
        result = RuleBasedScorer().score(numeric_task, "I don't know.")
        assert result.score == 0.0
        assert "No numeric answer" in result.rationale

    def test_uses_last_number_in_response(self, numeric_task):
        # Reasoning trace ending in the right final answer should still pass.
        trace = "6 times 7: first think of 6 groups of 7, that's 42."
        result = RuleBasedScorer().score(numeric_task, trace)
        assert result.score == 1.0

    def test_tolerance_within_bound_scores_one(self, numeric_tolerance_task):
        result = RuleBasedScorer().score(numeric_tolerance_task, "Approximately 3.25")
        assert result.score == 1.0

    def test_tolerance_outside_bound_scores_zero(self, numeric_tolerance_task):
        result = RuleBasedScorer().score(numeric_tolerance_task, "Approximately 5.0")
        assert result.score == 0.0

    def test_handles_comma_thousands_separator(self):
        task = make_task(
            scoring=ScoringSpec(method="numeric_tolerance", reference_answer="1000", tolerance=0)
        )
        assert RuleBasedScorer().score(task, "The total is 1,000.").score == 1.0


class TestRuleBasedScorerRejectsLlmJudge:
    def test_raises_for_llm_judge_method(self, llm_judge_task):
        with pytest.raises(ValueError, match="cannot handle"):
            RuleBasedScorer().score(llm_judge_task, "Done")


class TestScoreTaskDispatch:
    def test_dispatches_deterministic_method_without_calling_judge(self, numeric_task):
        class ExplodingJudge:
            def judge(self, task, response):
                raise AssertionError("judge should not be called for a deterministic task")

        result = score_task(numeric_task, "6 times 7 is 42.", ExplodingJudge())
        assert result.score == 1.0
        assert result.method == "rule_based"

    def test_dispatches_llm_judge_method_to_judge(self, llm_judge_task):
        class StubJudge:
            def judge(self, task, response):
                return ScoreResult(0.75, "stub rationale", "rule_based_judge")

        result = score_task(llm_judge_task, "Done", StubJudge())
        assert result.score == 0.75
        assert result.rationale == "stub rationale"
