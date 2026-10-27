"""llm-eval-benchmark: a small, dependency-light LLM evaluation and benchmarking harness."""

from .aggregate import AggregateReport, CategoryScore, aggregate
from .candidates import AnthropicCandidate, Candidate, MockCandidate
from .compare import CategoryComparison, ComparisonReport, compare
from .judge_anthropic import JudgeParsingError, LLMJudge
from .runner import TaskResult, run_suite
from .scoring import Judge, RuleBasedJudge, RuleBasedScorer, ScoreResult, score_task
from .suite import Heuristics, ScoringSpec, SuiteValidationError, Task, load_suite

__version__ = "0.1.0"

__all__ = [
    "AggregateReport",
    "AnthropicCandidate",
    "Candidate",
    "CategoryComparison",
    "CategoryScore",
    "ComparisonReport",
    "Heuristics",
    "Judge",
    "JudgeParsingError",
    "LLMJudge",
    "MockCandidate",
    "RuleBasedJudge",
    "RuleBasedScorer",
    "ScoreResult",
    "ScoringSpec",
    "SuiteValidationError",
    "Task",
    "TaskResult",
    "__version__",
    "aggregate",
    "compare",
    "load_suite",
    "run_suite",
    "score_task",
]
