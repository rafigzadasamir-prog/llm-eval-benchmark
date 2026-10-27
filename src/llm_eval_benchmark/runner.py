"""Run a candidate against a list of tasks and score each response."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from .candidates import Candidate
from .scoring import Judge, ScoreResult, score_task
from .suite import Task


@dataclasses.dataclass(frozen=True)
class TaskResult:
    """One task's prompt/response/score, kept together for reporting."""

    task: Task
    response: str
    score: ScoreResult


def run_suite(tasks: Sequence[Task], candidate: Candidate, judge: Judge) -> list[TaskResult]:
    """Run ``candidate`` over every task, scoring each response as it comes in."""
    results = []
    for task in tasks:
        response = candidate.respond(task)
        score = score_task(task, response, judge)
        results.append(TaskResult(task=task, response=response, score=score))
    return results
