"""Render AggregateReport / ComparisonReport objects as text or JSON."""

from __future__ import annotations

import json

from .aggregate import AggregateReport
from .compare import ComparisonReport
from .runner import TaskResult


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _render_transcript(result: TaskResult) -> list[str]:
    return [
        f"  [{result.task.category}] task={result.task.id}",
        f"    prompt:    {result.task.prompt}",
        f"    response:  {result.response}",
        f"    score:     {result.score.score:.2f} ({result.score.method})",
        f"    rationale: {result.score.rationale}",
        "",
    ]


def render_text_report(report: AggregateReport, n_examples: int = 2) -> str:
    """Render a human-readable report: overall score, per-category scores, example transcripts."""
    lines = [f"=== LLM Eval Benchmark Report: {report.candidate_name} ===", ""]
    n_tasks = len(report.results)
    lines.append(f"Overall score: {_format_pct(report.overall_score)} ({n_tasks} tasks)")
    lines.append("")
    lines.append("Per-category scores:")
    for category, cat_score in sorted(report.category_scores.items()):
        lines.append(f"  - {category}: {_format_pct(cat_score.mean_score)} (n={cat_score.n_tasks})")

    if n_examples > 0:
        lines.append("")
        lines.append("Example transcripts (one per category):")
        shown_categories: set[str] = set()
        for result in report.results:
            if result.task.category in shown_categories:
                continue
            shown_categories.add(result.task.category)
            lines.extend(_render_transcript(result))
            if len(shown_categories) >= n_examples:
                break

    return "\n".join(lines)


def render_comparison_text(comparison: ComparisonReport) -> str:
    """Render a side-by-side comparison table of two candidates, with a per-category winner."""
    lines = [
        f"=== Comparison: {comparison.name_a} vs {comparison.name_b} ===",
        "",
        f"{'Category':<28}{comparison.name_a:>18}{comparison.name_b:>18}  Winner",
    ]
    for c in comparison.category_comparisons:
        lines.append(
            f"{c.category:<28}{_format_pct(c.score_a):>18}{_format_pct(c.score_b):>18}  {c.winner}"
        )
    lines.append("")
    lines.append(
        f"{'OVERALL':<28}{_format_pct(comparison.overall_score_a):>18}"
        f"{_format_pct(comparison.overall_score_b):>18}  {comparison.overall_winner}"
    )
    return "\n".join(lines)


def report_to_dict(report: AggregateReport) -> dict:
    """Convert an AggregateReport into a plain, JSON-serializable dict."""
    return {
        "candidate_name": report.candidate_name,
        "overall_score": report.overall_score,
        "category_scores": {
            category: {"mean_score": cs.mean_score, "n_tasks": cs.n_tasks}
            for category, cs in report.category_scores.items()
        },
        "results": [
            {
                "task_id": r.task.id,
                "category": r.task.category,
                "prompt": r.task.prompt,
                "response": r.response,
                "score": r.score.score,
                "method": r.score.method,
                "rationale": r.score.rationale,
            }
            for r in report.results
        ],
    }


def comparison_to_dict(comparison: ComparisonReport) -> dict:
    """Convert a ComparisonReport into a plain, JSON-serializable dict."""
    return {
        "name_a": comparison.name_a,
        "name_b": comparison.name_b,
        "overall_score_a": comparison.overall_score_a,
        "overall_score_b": comparison.overall_score_b,
        "overall_winner": comparison.overall_winner,
        "categories": [
            {
                "category": c.category,
                "score_a": c.score_a,
                "score_b": c.score_b,
                "winner": c.winner,
            }
            for c in comparison.category_comparisons
        ],
    }


def render_json_report(report: AggregateReport) -> str:
    return json.dumps(report_to_dict(report), indent=2)
