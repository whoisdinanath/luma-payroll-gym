# Luma Payroll Gym

Evaluate agents on fictional payroll work: select the right dated reference, compute a payslip, and ask for clarification when records do not determine an answer.

- Dashboard: run the FastAPI server below. The old static hosted page is a historical snapshot, not the active dashboard.
- [Live sandbox](https://luma-payroll-sandbox.whoisdinanath.workers.dev)
- [Evaluation report](reports/evaluation.md)
- [Code guide](reports/architecture.md)
- [Requirements and remaining work](reports/requirements.md)

## Why payroll

One narrow workflow combines useful lookups, checkable arithmetic, contribution caps, and genuinely unresolved records. Fictional rules make the oracle reproducible without depending on changing law. Insurance payout was the strongest alternative with a smaller calculator; payroll offers more numerical interactions. Expense reimbursement risks being too easy, while real statutory tax or legal work adds interpretation and source-maintenance overhead.

This measures controlled rule application, not readiness to run real statutory payroll.

## Setup

Python 3.12 is recommended. Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync --extra environment --extra server
uv run luma-gym validate data/cases.json
uv run luma-gym audit-tasks data/cases.json
```

The calculator and generator require no API key. Hosted model calls use `OPENROUTER_API_KEY` from `~/.env`, or `--env-file` to select another file. `.env.example` shows the variable name. Do not commit keys.

## Sandbox quickstart

Fetch three tasks, submit deliberately invalid answers, and print zero scores:

```bash
uv run python scripts/quickstart.py
```

This needs no key and makes no judge calls. For your own agent:

1. `GET /tasks?n=3&tier=all` returns a run ID, written rules, source documents, and reference schedules.
2. `POST /submit` accepts `{"run_id":"...","answers":[{"id":"...","answer":{...}}]}`.
3. `GET /runs/<run_id>` retrieves saved scores.

Expected answers never appear in fetched task payloads. Scores include per-component marks and governing clauses. Run IDs are opaque bearer capabilities, valid for one day. The API accepts at most 10 tasks and 12 caller requests per minute. Concurrent submissions are claimed atomically; repeated completed submissions return cached results. No caller API keys are accepted or stored.

The same asynchronous grader serves evaluation and the sandbox. Correct clarification answers with nonempty explanations use the same constrained judge. Empty explanations receive no explanation credit. Hosted judge spending has a $0.50 ceiling reserved inside the project-wide $10 allowance. When exhausted, qualitative scoring returns an explicit service error rather than silently changing the rubric. Cold starts and deployment propagation may temporarily delay requests.

Run the local API with:

```bash
uv run --extra environment --extra server uvicorn luma_gym.server:app --port 8000
```

Open http://127.0.0.1:8000/ for the dashboard. FastAPI renders ordinary HTML templates from `templates/` and reads saved JSON results on each request. There is no HTML-generation script or browser JavaScript framework. Task cells link to a separate transcript page. Refreshing shows the latest saved metrics. Hosting the new FastAPI dashboard is separate from the existing Cloudflare sandbox deployment.

## Evaluate

Plan a three-task tool pilot without inference spending:

```bash
uv run --extra environment python -m luma_gym.eval \
  --model z-ai/glm-5.3-flash --tasks data/case-pilot.json --limit 3
```

Add `--execute` to call the model. Use `--mode single` for inline documents and references. Each task gets three rollouts. Change `--output results/my-run` for a different experiment. The configured model is always a CLI argument.

Runs save config, progress, transcripts, metrics, and scores. Completed rollouts are skipped on resume. Unfinished calls require review so a crash does not silently buy duplicate work. Native request logs retain usage and provider-reported cost; missing usage and judge errors are infrastructure failures, not wrong model answers.

New paid work uses `results/spending.json`, locking reservations across processes. `--max-cost` can lower the $10 project ceiling. The current unattended protocol allows at most 8192 completion tokens per request so an upper cost can be reserved; this supersedes earlier uncapped pilots. Provider pricing and token reporting still need verification, and truncated answers must be reported. The old uncapped execution path has been removed.

Five requested configurations are DeepSeek V4.1 Flash, MiMo V2.6 Pro, Gemini 3.1 Flash-Lite, GLM-5.3 Flash, and GPT-5.6 Luna. Luna did not accept temperature 0.7 on the available route. Its provider-default run is visibly labelled and is not claimed to have identical sampling controls. See [model plan](reports/model-plan.md).

## Environment and tools

```python
from luma_gym import load_environment

env = load_environment(mode="tools", tasks_path="data/case-pilot.json")
```

The optional environment dependency pins verifiers 0.1.11, matching the assignment's Environment, SingleTurnEnv, ToolEnv, and Rubric API. Current upstream uses a different API.

Three tools list case documents, read one case document, and retrieve a dated reference row. Tool mode withholds source records and numerical rates from the prompt. Single-turn receives identical sources inline. Case schedules vary across tasks, so memorizing the default rates is insufficient. Documents are synthetic structured records, not PDFs.

Operational limits are eight turns, 120 seconds per rollout, ten seconds per tool, and 24 tool calls. Invalid arguments produce generic tool errors; tool and infrastructure timeouts are not silently turned into answer errors.

## Rules, generator, and scoring

- [Payroll rules](rules/payroll.md) are authoritative, separate from code. Current identifier: 3.
- `reference.py` uses Decimal and cent-level half-up rounding. Exact expected monetary answers do not permit an extra cent of tolerance.
- `tasks.py` holds a simple Task class and deterministic input generation. `cases.py` varies authoritative reference schedules. Ground truth is computed independently in Python, never by the evaluated or trained model.
- Retrieval, calculation, and trap tiers cover effective dates, caps, missing inputs, contract conflicts, approved amendments, and draft/future amendments.
- `data/cases.json` contains 100 current tasks. `data/seed_tasks.json` and older datasets remain historical artifacts under their own rule IDs. Unknown versions are rejected.

Generation controls include tier mix, boundary frequency, deductions, and matching contracts. Same seed and controls reproduce the same data. Semantic signatures reject known equivalences across splits instead of trusting seeds alone. Current case-specific training splits are in `data/training-splits/`; final-held-out tasks must not be used for tuning. Transfer scenarios are separate in `data/transfer.json`.

The trusted reference enables unlimited labelled practice without human labelling per instance, but only where the written rules and input facts determine an answer. It does not solve interpretation of actual law, unreliable fact gathering, or qualitative communication.

[Scoring](rules/scoring.md) combines exact correctness (0.70), field accuracy (0.25), and clause match (0.05), with wrong calculation credit scaled below 0.20. Wrong decisions and wrong trap reasons receive zero. Correct clarification adds a 0.10 qualitative explanation component behind the deterministic gate. The [judge prompt](rules/judge.md) is pinned, constrained to three criteria, and evaluated for repeated-judgment disagreement.

See [judge reliability](reports/judge-reliability.md), [failure analysis](reports/failure-analysis.md), and [human review requirements](reports/human-review.md). An automated integer-cent cross-check found zero divergences in 10 computation examples; that is not the assignment's independent human audit.

## Results and limitations

Results are generated progressively and preserved, including failures and scaffold corrections. [Evaluation](reports/evaluation.md) reports current counts, means, and sample SD; partial runs are not a final fair ranking. The dashboard includes modes, tiers, component scores, transcripts, cost, and latency. Scripted baselines use the current shared grader.

**Local RL training has not run.** No learning curves, KL sweep, frozen-reference experiment, or held-out improvement is claimed. [Training plan](reports/training-plan.md) lists the required protocol. Model and algorithm selection wait for the user's GPU availability and must include reasons and alternatives. The Loom recording and independent human fidelity audit also need the user's participation.

Other reports: [trade-offs](reports/trade-offs.md), [edge cases](reports/edge-cases.md), [walkthrough outline](reports/walkthrough.md). The full 18-page assignment transcription is local in `docs/ai-training-gym-assignment.md`; reread relevant phases before each milestone.

## Checks and AI usage

Focused tests are local in Git-ignored `tests/`, per user preference:

```bash
uv run --extra environment --extra server python -m unittest discover -s tests -v
```

OpenCode assisted with domain comparison, fictional rule design, implementation, experiments, transcript review, and documentation. Explicit seed answers and transfer scenarios were AI-assisted, not independently human-audited. Ground-truth generation uses the Python reference, not LLM labels. No results or training evidence are fabricated.
