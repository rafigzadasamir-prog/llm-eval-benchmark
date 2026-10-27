"""Tests for the python -m llm_eval_benchmark.run CLI.

These all run fully offline: no ANTHROPIC_API_KEY is set (explicitly removed
via monkeypatch), so --judge auto always resolves to RuleBasedJudge and the
only usable candidates are the bundled MockCandidate profiles.
"""

from __future__ import annotations

import json

import pytest

from llm_eval_benchmark.candidates import MockCandidate
from llm_eval_benchmark.run import build_candidate, build_judge, main
from llm_eval_benchmark.scoring import RuleBasedJudge


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


class TestBuildCandidate:
    def test_known_mock_names_resolve(self):
        assert isinstance(build_candidate("careful-prompt"), MockCandidate)
        assert isinstance(build_candidate("rushed-prompt"), MockCandidate)

    def test_careful_prompt_uses_strong_responses(self):
        candidate = build_candidate("careful-prompt")
        assert candidate.response_key == "strong"

    def test_rushed_prompt_uses_weak_responses(self):
        candidate = build_candidate("rushed-prompt")
        assert candidate.response_key == "weak"

    def test_unknown_name_raises_system_exit(self):
        with pytest.raises(SystemExit, match="Unknown candidate"):
            build_candidate("not-a-real-candidate")

    def test_anthropic_candidate_without_api_key_raises_system_exit(self):
        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
            build_candidate("anthropic:claude-3-5-haiku-20241022")

    def test_anthropic_candidate_without_model_raises_system_exit(self):
        with pytest.raises(SystemExit, match="model"):
            build_candidate("anthropic:")


class TestBuildJudge:
    def test_rule_based_preference_returns_rule_based_judge(self):
        assert isinstance(build_judge("rule_based"), RuleBasedJudge)

    def test_auto_without_api_key_returns_rule_based_judge(self):
        assert isinstance(build_judge("auto"), RuleBasedJudge)

    def test_llm_preference_without_api_key_raises(self):
        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
            build_judge("llm")


class TestMainSingleCandidate:
    def test_exits_zero_and_prints_report(self, capsys):
        exit_code = main(["--candidate", "careful-prompt"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "careful-prompt" in captured.out
        assert "Overall score" in captured.out

    def test_missing_candidate_returns_error_code(self, capsys):
        exit_code = main([])
        captured = capsys.readouterr()

        assert exit_code == 2
        assert "error" in captured.err

    def test_list_candidates(self, capsys):
        exit_code = main(["--list-candidates"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "careful-prompt" in captured.out
        assert "rushed-prompt" in captured.out

    def test_json_format_is_valid_json(self, capsys):
        main(["--candidate", "careful-prompt", "--format", "json"])
        captured = capsys.readouterr()

        payload = json.loads(captured.out)
        assert payload["candidate_name"] == "careful-prompt"
        assert 0.0 <= payload["overall_score"] <= 1.0

    def test_output_file_is_written(self, tmp_path, capsys):
        output_path = tmp_path / "report.txt"
        exit_code = main(["--candidate", "careful-prompt", "--output", str(output_path)])

        assert exit_code == 0
        assert output_path.exists()
        assert "careful-prompt" in output_path.read_text(encoding="utf-8")


class TestMainComparison:
    def test_compare_two_mock_candidates(self, capsys):
        exit_code = main(["--candidate", "careful-prompt", "--compare", "rushed-prompt"])
        captured = capsys.readouterr()

        assert exit_code == 0
        assert "Comparison" in captured.out
        assert "careful-prompt" in captured.out
        assert "rushed-prompt" in captured.out

    def test_careful_prompt_beats_rushed_prompt_overall(self, capsys):
        main(["--candidate", "careful-prompt", "--compare", "rushed-prompt", "--format", "json"])
        captured = capsys.readouterr()

        payload = json.loads(captured.out)
        assert payload["comparison"]["overall_winner"] == "careful-prompt"
        assert payload["candidate_a"]["overall_score"] > payload["candidate_b"]["overall_score"]

    def test_comparison_json_structure(self, capsys):
        main(["--candidate", "careful-prompt", "--compare", "rushed-prompt", "--format", "json"])
        captured = capsys.readouterr()

        payload = json.loads(captured.out)
        assert set(payload) == {"candidate_a", "candidate_b", "comparison"}
        assert len(payload["comparison"]["categories"]) == 4
