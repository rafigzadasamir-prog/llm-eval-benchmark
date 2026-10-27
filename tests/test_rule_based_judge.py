"""Tests for the offline RuleBasedJudge heuristic fallback."""

from __future__ import annotations

from llm_eval_benchmark.scoring import RuleBasedJudge
from llm_eval_benchmark.suite import Heuristics, ScoringSpec

from .conftest import make_task


def _judge_task(heuristics: Heuristics, prompt: str = "Say something.", rubric: str = "Be good."):
    return make_task(
        category="instruction_following",
        scoring=ScoringSpec(method="llm_judge", rubric=rubric, heuristics=heuristics),
        prompt=prompt,
        mock_responses={"strong": "ok", "weak": "ok"},
    )


class TestMustIncludeAny:
    def test_passes_when_one_phrase_present(self):
        task = _judge_task(Heuristics(must_include_any=("cat", "dog")))
        result = RuleBasedJudge().judge(task, "I have a dog.")
        assert result.score == 1.0

    def test_fails_when_no_phrase_present(self):
        task = _judge_task(Heuristics(must_include_any=("cat", "dog")))
        result = RuleBasedJudge().judge(task, "I have a fish.")
        assert result.score == 0.0


class TestMustIncludeAll:
    def test_passes_only_when_all_phrases_present(self):
        task = _judge_task(Heuristics(must_include_all=("1.", "2.")))
        assert RuleBasedJudge().judge(task, "1. first\n2. second").score == 1.0
        assert RuleBasedJudge().judge(task, "1. first only").score == 0.0


class TestForbiddenPhrases:
    def test_passes_when_none_present(self):
        task = _judge_task(Heuristics(forbidden_phrases=("however", "but")))
        assert RuleBasedJudge().judge(task, "This is a clean sentence.").score == 1.0

    def test_fails_when_any_present(self):
        task = _judge_task(Heuristics(forbidden_phrases=("however", "but")))
        assert RuleBasedJudge().judge(task, "This is good, however imperfect.").score == 0.0


class TestWordCountBounds:
    def test_max_words_pass_and_fail(self):
        task = _judge_task(Heuristics(max_words=2))
        assert RuleBasedJudge().judge(task, "Two words").score == 1.0
        assert RuleBasedJudge().judge(task, "This is way too many words").score == 0.0

    def test_min_words_pass_and_fail(self):
        task = _judge_task(Heuristics(min_words=3))
        assert RuleBasedJudge().judge(task, "One two three four").score == 1.0
        assert RuleBasedJudge().judge(task, "Two").score == 0.0

    def test_min_and_max_together(self):
        task = _judge_task(Heuristics(min_words=2, max_words=4))
        assert RuleBasedJudge().judge(task, "Exactly three words").score == 1.0
        assert RuleBasedJudge().judge(task, "Just").score == 0.0
        assert RuleBasedJudge().judge(task, "This has way too many words in it").score == 0.0


class TestLineCountBounds:
    def test_exact_line_count(self):
        task = _judge_task(Heuristics(min_lines=3, max_lines=3))
        assert RuleBasedJudge().judge(task, "one\ntwo\nthree").score == 1.0
        assert RuleBasedJudge().judge(task, "just one line").score == 0.0

    def test_blank_lines_are_not_counted(self):
        task = _judge_task(Heuristics(min_lines=2, max_lines=2))
        result = RuleBasedJudge().judge(task, "one\n\ntwo\n")
        assert result.score == 1.0


class TestMultipleHeuristicsCombine:
    def test_partial_credit_when_some_checks_fail(self):
        task = _judge_task(Heuristics(max_words=1, must_include_any=("cold",)))
        # Satisfies the content check but not the brevity check -> 1 of 2.
        result = RuleBasedJudge().judge(task, "I think it is cold outside today.")
        assert result.score == 0.5
        assert "PASS" in result.rationale
        assert "FAIL" in result.rationale

    def test_full_credit_when_all_checks_pass(self):
        task = _judge_task(Heuristics(max_words=1, must_include_any=("cold",)))
        result = RuleBasedJudge().judge(task, "Cold")
        assert result.score == 1.0

    def test_zero_credit_when_all_checks_fail(self):
        task = _judge_task(Heuristics(max_words=1, must_include_any=("cold",)))
        result = RuleBasedJudge().judge(task, "It is definitely quite warm today, I would say.")
        assert result.score == 0.0


class TestNoHeuristicsDefined:
    def test_returns_neutral_score(self):
        task = _judge_task(Heuristics())
        result = RuleBasedJudge().judge(task, "anything")
        assert result.score == 0.5
        assert result.method == "rule_based_judge"


class TestRationaleMethodTag:
    def test_method_is_rule_based_judge(self):
        task = _judge_task(Heuristics(must_include_any=("x",)))
        result = RuleBasedJudge().judge(task, "x")
        assert result.method == "rule_based_judge"
