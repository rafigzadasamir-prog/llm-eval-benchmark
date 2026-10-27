"""Aggregate per-task results into per-category and overall scores."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Sequence

from .runner import TaskResult


@dataclasses.dataclass(frozen=True)
class CategoryScore:
    category: str
    mean_score: float
    n_tasks: int


@dataclasses.dataclass(frozen=True)
class AggregateReport:
    candidate_name: str
    overall_score: float
    category_scores: dict[str, CategoryScore]
    results: tuple[TaskResult, ...]


def aggregate(candidate_name: str, results: Sequence[TaskResult]) -> AggregateReport:
    """Compute per-category and overall mean scores for a run's results."""
    if not results:
        raise ValueError("Cannot aggregate an empty list of results.")

    by_category: dict[str, list[float]] = defaultdict(list)
    for result in results:
        by_category[result.task.category].append(result.score.score)

    category_scores = {
        category: CategoryScore(category, sum(scores) / len(scores), len(scores))
        for category, scores in sorted(by_category.items())
    }
    overall_score = sum(result.score.score for result in results) / len(results)

    return AggregateReport(
        candidate_name=candidate_name,
        overall_score=overall_score,
        category_scores=category_scores,
        results=tuple(results),
    )
