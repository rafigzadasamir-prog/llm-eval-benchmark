"""CLI entry point: ``python -m llm_eval_benchmark.run``.

Runs the bundled benchmark suite against one candidate (or two, for a
side-by-side comparison) and prints a report. With no API key configured,
everything resolves through ``MockCandidate`` and ``RuleBasedJudge`` --
fully offline, fully deterministic.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .aggregate import aggregate
from .candidates import AnthropicCandidate, Candidate, MockCandidate
from .compare import compare
from .report import (
    comparison_to_dict,
    render_comparison_text,
    render_json_report,
    render_text_report,
    report_to_dict,
)
from .runner import run_suite
from .scoring import Judge, RuleBasedJudge
from .suite import load_suite

# Two deterministic mock "prompt configurations" bundled for demos and CI:
# careful-prompt replays each task's hand-authored "strong" response,
# rushed-prompt replays the "weak" one. See candidates.py / the suite file.
MOCK_CANDIDATE_RESPONSE_KEYS = {
    "careful-prompt": "strong",
    "rushed-prompt": "weak",
}


def build_candidate(name: str) -> Candidate:
    """Resolve a candidate name from the CLI into a Candidate instance."""
    if name in MOCK_CANDIDATE_RESPONSE_KEYS:
        return MockCandidate(name=name, response_key=MOCK_CANDIDATE_RESPONSE_KEYS[name])

    if name.startswith("anthropic:"):
        model = name.split(":", 1)[1]
        if not model:
            raise SystemExit(
                "Candidate 'anthropic:<model-id>' requires a model id after the colon."
            )
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(
                f"Candidate {name!r} requires the ANTHROPIC_API_KEY environment variable to be set."
            )
        try:
            import anthropic
        except ImportError as exc:
            raise SystemExit("The 'anthropic' package is required for live candidates.") from exc
        client = anthropic.Anthropic(api_key=api_key)
        return AnthropicCandidate(client=client, model=model, name=name)

    raise SystemExit(
        f"Unknown candidate {name!r}. Known mock candidates: "
        f"{sorted(MOCK_CANDIDATE_RESPONSE_KEYS)}. "
        "Use 'anthropic:<model-id>' (with ANTHROPIC_API_KEY set) for a live candidate."
    )


def _build_llm_judge():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit(
            "--judge llm requires the ANTHROPIC_API_KEY environment variable to be set."
        )
    try:
        import anthropic
    except ImportError as exc:
        raise SystemExit("The 'anthropic' package is required for the LLM judge.") from exc

    from .judge_anthropic import LLMJudge

    client = anthropic.Anthropic(api_key=api_key)
    return LLMJudge(client=client)


def build_judge(preference: str = "auto") -> Judge:
    """Resolve the judge to use for llm_judge tasks.

    "auto" uses the real Anthropic-backed judge if ANTHROPIC_API_KEY is set,
    and falls back to the fully offline RuleBasedJudge otherwise.
    """
    if preference == "rule_based":
        return RuleBasedJudge()
    if preference == "llm":
        return _build_llm_judge()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return _build_llm_judge()
    return RuleBasedJudge()


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m llm_eval_benchmark.run",
        description="Run the bundled LLM evaluation benchmark against one or two candidates.",
    )
    parser.add_argument("--candidate", help="Candidate name to evaluate (see --list-candidates).")
    parser.add_argument(
        "--compare",
        help="Second candidate name; if given, produces a side-by-side comparison.",
    )
    parser.add_argument(
        "--suite",
        type=Path,
        default=None,
        help="Path to a benchmark suite YAML file (default: bundled suite).",
    )
    parser.add_argument(
        "--judge",
        choices=["auto", "rule_based", "llm"],
        default="auto",
        help=(
            "How to resolve llm_judge tasks (default: auto -> LLM judge if "
            "ANTHROPIC_API_KEY is set, else rule-based)."
        ),
    )
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument(
        "--examples",
        type=int,
        default=2,
        help="Number of example transcripts to include (one per category).",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Write the report to this file instead of stdout."
    )
    parser.add_argument(
        "--list-candidates", action="store_true", help="List known candidate names and exit."
    )
    return parser.parse_args(argv)


def _render_single(report, fmt: str, n_examples: int) -> str:
    if fmt == "json":
        return render_json_report(report)
    return render_text_report(report, n_examples)


def _render_comparison(report_a, report_b, comparison, fmt: str, n_examples: int) -> str:
    if fmt == "json":
        return json.dumps(
            {
                "candidate_a": report_to_dict(report_a),
                "candidate_b": report_to_dict(report_b),
                "comparison": comparison_to_dict(comparison),
            },
            indent=2,
        )
    return "\n\n".join(
        [
            render_text_report(report_a, n_examples),
            render_text_report(report_b, n_examples),
            render_comparison_text(comparison),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.list_candidates:
        print("Available mock candidates (fully offline):")
        for name in sorted(MOCK_CANDIDATE_RESPONSE_KEYS):
            print(f"  - {name}")
        print("Live candidate: anthropic:<model-id> (requires ANTHROPIC_API_KEY)")
        return 0

    if not args.candidate:
        print("error: --candidate is required (or pass --list-candidates)", file=sys.stderr)
        return 2

    tasks = load_suite(args.suite)
    judge = build_judge(args.judge)

    candidate_a = build_candidate(args.candidate)
    report_a = aggregate(candidate_a.name, run_suite(tasks, candidate_a, judge))

    if args.compare:
        candidate_b = build_candidate(args.compare)
        report_b = aggregate(candidate_b.name, run_suite(tasks, candidate_b, judge))
        comparison = compare(report_a, report_b)
        output = _render_comparison(report_a, report_b, comparison, args.format, args.examples)
    else:
        output = _render_single(report_a, args.format, args.examples)

    if args.output:
        args.output.write_text(output, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
