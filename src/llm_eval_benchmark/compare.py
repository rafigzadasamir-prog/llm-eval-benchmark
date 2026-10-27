"""Side-by-side comparison of two candidates' aggregate reports."""

from __future__ import annotations

import dataclasses

from .aggregate import AggregateReport

# Scores are floating-point means; treat differences this small as a tie
# rather than letting float noise pick an arbitrary "winner".
_TIE_EPSILON = 1e-9


@dataclasses.dataclass(frozen=True)
class CategoryComparison:
    category: str
    score_a: float
    score_b: float
    winner: str  # candidate_a's name, candidate_b's name, or "tie"


@dataclasses.dataclass(frozen=True)
class ComparisonReport:
    name_a: str
    name_b: str
    category_comparisons: tuple[CategoryComparison, ...]
    overall_score_a: float
    overall_score_b: float
    overall_winner: str


def _pick_winner(score_a: float, score_b: float, name_a: str, name_b: str) -> str:
    if abs(score_a - score_b) <= _TIE_EPSILON:
        return "tie"
    return name_a if score_a > score_b else name_b


def compare(report_a: AggregateReport, report_b: AggregateReport) -> ComparisonReport:
    """Compare two candidates' aggregate reports, category by category and overall.

    Both reports are expected to come from running the same suite, but this
    tolerates one candidate's results covering a category the other's don't
    (e.g. a custom suite subset) by treating a missing category as a 0.0.
    """
    categories = sorted(set(report_a.category_scores) | set(report_b.category_scores))

    comparisons = []
    for category in categories:
        score_a = (
            report_a.category_scores[category].mean_score
            if category in report_a.category_scores
            else 0.0
        )
        score_b = (
            report_b.category_scores[category].mean_score
            if category in report_b.category_scores
            else 0.0
        )
        winner = _pick_winner(score_a, score_b, report_a.candidate_name, report_b.candidate_name)
        comparisons.append(CategoryComparison(category, score_a, score_b, winner))

    overall_winner = _pick_winner(
        report_a.overall_score,
        report_b.overall_score,
        report_a.candidate_name,
        report_b.candidate_name,
    )

    return ComparisonReport(
        name_a=report_a.candidate_name,
        name_b=report_b.candidate_name,
        category_comparisons=tuple(comparisons),
        overall_score_a=report_a.overall_score,
        overall_score_b=report_b.overall_score,
        overall_winner=overall_winner,
    )
