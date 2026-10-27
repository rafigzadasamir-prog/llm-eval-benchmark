# llm-eval-benchmark

A small, dependency-light **LLM evaluation and benchmarking harness**: it runs one or more
"candidates" (a model, or a prompt configuration) against a hand-written benchmark suite,
scores every response, aggregates the scores, and prints a report — including a side-by-side
comparison of two candidates with a per-category winner.

It runs fully offline out of the box (no API key, no network access), and the same pipeline
plugs into the real Anthropic API as a drop-in upgrade for both the candidate and the judge.

## Why evaluation and benchmarking matter for LLM apps

Shipping an LLM feature without an eval harness means every prompt tweak, model upgrade, or
system-prompt change is a guess. A benchmarking harness turns "I think the new prompt is
better" into a number you can diff:

- **Regression detection.** A prompt change that fixes one case can silently break three
  others. Running the same suite before and after turns that into a visible category-level
  score drop instead of a support ticket next week.
- **Model/version upgrades aren't free upgrades.** A newer or cheaper model is not
  automatically a better fit for your task; a suite that covers your actual task mix (factual
  recall, arithmetic, instruction-following, faithful summarization) tells you whether a
  switch is safe.
- **"Looks good to me" doesn't scale.** Eyeballing a handful of chat transcripts catches
  obvious breakage but not systematic weaknesses (e.g. consistently ignoring a formatting
  instruction, or quietly hallucinating a number in a summary). Deterministic + LLM-judged
  scoring catches both.
- **Deterministic scoring where it's possible, judged scoring where it's not.** Factual
  answers and arithmetic have a right answer you can check with code. Instruction-following
  and summarization faithfulness don't reduce to a string match — that's what the
  LLM-as-judge pattern (and its offline heuristic fallback) is for.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌─────────────┐     ┌──────────┐
│   SUITE      │     │     RUN      │     │     SCORE      │     │  AGGREGATE   │     │  REPORT  │
│ (YAML file,  │ --> │ candidate    │ --> │ RuleBasedScorer│ --> │ per-category │ --> │ text /   │
│ 40 hand-     │     │ .respond()   │     │ (exact_match,  │     │ + overall    │     │ json,    │
│ written      │     │ for each     │     │  regex,        │     │ mean scores  │     │ example  │
│ tasks)       │     │ task         │     │  numeric_      │     │              │     │ trans-   │
│              │     │              │     │  tolerance)    │     │              │     │ cripts,  │
│              │     │              │     │       or       │     │              │     │ 2-way    │
│              │     │              │     │ Judge          │     │              │     │ compare  │
│              │     │              │     │ (LLMJudge /    │     │              │     │          │
│              │     │              │     │  RuleBasedJudge)│    │              │     │          │
└─────────────┘     └──────────────┘     └───────────────┘     └─────────────┘     └──────────┘
```

- **Suite** (`suite.py`) — loads and validates `data/benchmark_suite.yaml`: 40 tasks across
  `factual_qa`, `arithmetic_reasoning`, `instruction_following`, `summarization_faithfulness`.
  Each task carries its scoring method, and two hand-authored canned responses
  (`mock_responses.strong` / `.weak`) used by the offline mock candidates.
- **Run** (`runner.py`, `candidates.py`) — a `Candidate` produces a response per task.
  `MockCandidate` replays a canned response (fully offline); `AnthropicCandidate` calls the
  real Anthropic Messages API.
- **Score** (`scoring.py`, `judge_anthropic.py`) — deterministic methods
  (`exact_match`/`regex`/`numeric_tolerance`) are resolved by `RuleBasedScorer`, no model
  calls involved. `llm_judge` tasks are resolved by a `Judge`: either the offline
  `RuleBasedJudge` (heuristic checks authored per-task) or the real `LLMJudge` (asks Claude
  for a structured `{score, rationale}` JSON verdict against the task's rubric, with a retry
  loop for malformed output).
- **Aggregate** (`aggregate.py`) — per-category and overall mean scores.
- **Report** (`report.py`) — renders text or JSON, including example transcripts; `compare.py`
  produces a category-by-category and overall winner between two candidates.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Running the benchmark

Fully offline, no API key needed — list the bundled candidates, run one, or compare two:

```bash
python -m llm_eval_benchmark.run --list-candidates

python -m llm_eval_benchmark.run --candidate careful-prompt

python -m llm_eval_benchmark.run --candidate careful-prompt --compare rushed-prompt
```

Useful flags: `--format {text,json}`, `--output <path>`, `--examples <n>` (example
transcripts per report), `--suite <path>` (use your own suite file), `--judge
{auto,rule_based,llm}`.

### Real example report (rule-based / offline mode)

This is actual output from `python -m llm_eval_benchmark.run --candidate careful-prompt
--compare rushed-prompt --examples 1`, captured from a real run of this repository (not
fabricated):

```
=== LLM Eval Benchmark Report: careful-prompt ===

Overall score: 97.5% (40 tasks)

Per-category scores:
  - arithmetic_reasoning: 90.0% (n=10)
  - factual_qa: 100.0% (n=10)
  - instruction_following: 100.0% (n=10)
  - summarization_faithfulness: 100.0% (n=10)

Example transcripts (one per category):
  [factual_qa] task=fqa-01
    prompt:    What is the capital of France?
    response:  The capital of France is Paris.
    score:     1.00 (rule_based)
    rationale: Response contains expected answer 'Paris'.


=== LLM Eval Benchmark Report: rushed-prompt ===

Overall score: 11.2% (40 tasks)

Per-category scores:
  - arithmetic_reasoning: 0.0% (n=10)
  - factual_qa: 0.0% (n=10)
  - instruction_following: 40.0% (n=10)
  - summarization_faithfulness: 5.0% (n=10)

Example transcripts (one per category):
  [factual_qa] task=fqa-01
    prompt:    What is the capital of France?
    response:  I believe the capital of France is Lyon, though I could be mixing it up with another city.
    score:     0.00 (rule_based)
    rationale: Response did not contain expected answer 'Paris' (or its aliases).


=== Comparison: careful-prompt vs rushed-prompt ===

Category                        careful-prompt     rushed-prompt  Winner
arithmetic_reasoning                     90.0%              0.0%  careful-prompt
factual_qa                              100.0%              0.0%  careful-prompt
instruction_following                   100.0%             40.0%  careful-prompt
summarization_faithfulness              100.0%              5.0%  careful-prompt

OVERALL                                  97.5%             11.2%  careful-prompt
```

`careful-prompt` and `rushed-prompt` are both `MockCandidate`s: they replay the suite's
hand-authored `strong`/`weak` responses, so this comparison illustrates the harness's
mechanics (scoring, aggregation, category-level winners) deterministically rather than
comparing two real models. Note `careful-prompt` isn't a perfect 100% — one arithmetic task's
"strong" response deliberately contains a calculation slip (`math-03`), so the rule-based
scorer catching a mistake in the "good" candidate is itself part of the demonstration.

An `instruction_following` transcript (LLM-judge task, scored offline by `RuleBasedJudge`)
from the same run looks like this:

```
  [instruction_following] task=instr-01
    prompt:    Respond with exactly one word: the opposite of 'hot'.
    response:  Cold
    score:     1.00 (rule_based_judge)
    rationale: PASS: includes any of ['cold']; PASS: word count 1 within bounds
```

## Plugging in a real model and a real judge

Set `ANTHROPIC_API_KEY` and the CLI automatically upgrades:

```bash
export ANTHROPIC_API_KEY=sk-ant-...

# Judge defaults to the real LLMJudge whenever a key is present (override with --judge).
python -m llm_eval_benchmark.run --candidate "anthropic:claude-3-5-haiku-20241022"

python -m llm_eval_benchmark.run \
  --candidate "anthropic:claude-3-5-haiku-20241022" \
  --compare careful-prompt
```

- Any candidate name of the form `anthropic:<model-id>` resolves to `AnthropicCandidate`,
  which calls the real Messages API.
- `--judge auto` (the default) uses `LLMJudge` (real Anthropic-backed scoring with retry on
  malformed JSON) whenever `ANTHROPIC_API_KEY` is set, and `RuleBasedJudge` otherwise.
  `--judge rule_based` / `--judge llm` force one or the other.

Programmatically, both are just another implementation of the same small protocols
(`Candidate.respond(task) -> str`, `Judge.judge(task, response) -> ScoreResult`), so you can
wire in any other model/provider the same way `AnthropicCandidate` and `LLMJudge` do — see
`src/llm_eval_benchmark/candidates.py` and `src/llm_eval_benchmark/judge_anthropic.py`.

## Testing

```bash
pip install -r requirements.txt   # includes pytest and ruff
ruff check .
pytest -q
```

The automated suite (122 tests) runs with **zero network calls**: `MockCandidate` and
`RuleBasedJudge` cover the offline path, and `AnthropicCandidate`/`LLMJudge` are tested only
against small injected fake clients (see `tests/test_candidates.py` and
`tests/test_llm_judge.py`) that mimic the Anthropic SDK's `client.messages.create(...)`
shape — no real `anthropic.Anthropic()` client is ever constructed in a test. CI
(`.github/workflows/ci.yml`) runs on Python 3.11 with `ANTHROPIC_API_KEY` unset.

## Project structure

```
llm-eval-benchmark/
├── src/llm_eval_benchmark/
│   ├── __init__.py            # public API re-exports
│   ├── suite.py                # suite schema, YAML loading, validation
│   ├── candidates.py           # Candidate protocol, MockCandidate, AnthropicCandidate
│   ├── scoring.py               # ScoreResult, RuleBasedScorer, RuleBasedJudge, score_task
│   ├── judge_anthropic.py      # LLMJudge (real Anthropic-backed judge, JSON parse + retry)
│   ├── anthropic_utils.py      # shared Anthropic response-text extraction helper
│   ├── runner.py                # run_suite(): candidate x judge -> TaskResult list
│   ├── aggregate.py             # per-category / overall mean scoring
│   ├── compare.py               # two-candidate comparison + per-category winner
│   ├── report.py                # text / JSON report rendering
│   ├── run.py                   # CLI: python -m llm_eval_benchmark.run
│   └── data/
│       └── benchmark_suite.yaml # 40 hand-written tasks, 4 categories
├── tests/                      # pytest suite (unit + integration), fully offline
├── .github/workflows/ci.yml    # Python 3.11, ruff + pytest, no secrets/network
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## License

MIT — see [LICENSE](LICENSE).
