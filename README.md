# Luma Payroll Gym

A verifiable gym for one slice of payroll work under a fictional rule set: look up a rate for a given month, compute a monthly payslip, or ask for clarification when the records do not determine an answer. Every answer is computed by a reference implementation of the written rules, rates differ per case so they must be looked up, and caps and conflicting records make natural traps. The same grader serves the evaluation, the RL trainer, and a live sandbox.

- Dashboard: https://luma-payroll-sandbox.whoisdinanath.workers.dev
- Sandbox API: the same URL (`/tasks`, `/submit`, `/runs/<run_id>`)
- Report (evaluation, RL training, failure analysis, transfer test, edge cases, trade-offs, AI usage): https://luma-payroll-sandbox.whoisdinanath.workers.dev/report
- Trained adapters and training logs: https://huggingface.co/whoisdinanath/MiniCPM5-1B-Luma-Payroll-GRPO

## Stack

Python 3.12 with uv. `verifiers` 0.1.11 for the environment. OpenRouter for hosted models and the judge (GLM 5.3 Flash). PyTorch, Transformers, and PEFT to train MiniCPM5-1B with LoRA on a 6 GB RTX 3050; a few evaluations ran on Modal L4 GPUs (`scripts/modal_eval.py`). FastAPI and Jinja templates render the dashboard; one Cloudflare Python Worker with D1 serves the static site and the sandbox API.

## Architecture

The code is in `src/` and installs as the `luma_gym` package.

```text
rules/payroll.md        rule set, version 3, written before the code
  reference.py          Decimal reference; each step cites its clause (R1 to R9)
  cases.py              generator 4: seed -> tasks, ground truth from reference.py, never from a model
  data/v4/              train, development, test, hosted evaluation, SFT, and demonstration sets
  data/seed_tasks.json  21 hand-checked tasks across the three tiers
environment.py          load_environment(): single-turn or tool mode (verifiers)
scoring.py, judge.py    rubric 8, one grader shared by every path below
  eval.py               hosted runs -> results/<run>/config.json, transcript.jsonl, metrics.json, scores.json
  sft.py, train.py      SFT and GRPO; local.py evaluates the local model
  server.py, deploy/    sandbox API (Cloudflare Worker and D1)
  dashboard.py          reads saved results; scripts/export_site.py builds the static site
```

## Task schema

Every task is a JSON object validated on load (`src/tasks.py`): `id`, `slice`, `difficulty` (1 retrieval, 2 calculation, 3 trap), `prompt`, `record` (payroll record, contract, amendment, and the case's reference schedule), `ground_truth`, `verifier`, `tags`, `schema_version`, `ruleset_version`, `seed`, and `retrieval_field` for retrieval tasks. Unknown rule or schema versions are rejected. Ground truth is a payslip with two-decimal money strings, one retrieved value, or a clarification reason code, each with its governing clauses.

## Rules and ground truth

`rules/payroll.md` is the authoritative rule set. `reference.py` implements it with Decimal arithmetic and half-up rounding, and every step cites its clause. Money must match to the cent with no tolerance, because R2 rounds every amount to the cent.

The reference implementation is the verifier: from any seed, the generator produces new tasks with exact answers and no human labelling. This holds only where the written rules and the record decide the answer. It cannot read real statute, judge whether a document can be trusted, or judge the quality of an explanation (the judge does that), and it inherits any misreading of the rules. A hand audit of 21 generated tasks against the rule text found no divergence (`reports/fidelity-audit.json`).

## Reward (rubric 8)

| Outcome | Score |
| --- | --- |
| Accepted: right decision, every substantive field exact | 0.70 exact + 0.25 field accuracy + 0.05 clause match: 0.95 or 1.0 |
| Rejected answer with the right decision | -0.5 + 0.2 × share of computed values right (the value, for a retrieval): -0.5 to -0.3 |
| Any other rejected answer: wrong value or reason, calculating through a trap, refusing a payslip the rules determine, malformed | -0.5 |

Correct clarifications score 0.9 × the deterministic score + 0.1 × a judge score for the explanation. A rejected answer can never score above -0.3, so partial credit ranks wrong answers without rewarding them. Refusing a payslip the rules determine costs the same as getting every computed field wrong. The judge's 0.10 weight keeps its opinion from outweighing the deterministic score. Rubrics 4 to 7 each fixed something RL exploited in the version before; rubric 8 prices a refusal like a wrong answer once the calculator made exact payslips reachable. `rules/scoring.md` gives what each score means under the rules, why each trap is a trap, and the rubric history; `rules/judge.md` is the judge prompt.

## Environment and tools

```python
from luma_gym import load_environment

env = load_environment(mode="tools", tasks_path="data/v4/eval.json")
```

Single-turn mode puts the record and reference schedule in the prompt. Tool mode hides them behind `list_documents`, `read_document`, and `lookup_reference`, and adds `calculate`, which evaluates a short formula exactly and holds no rates or rules (`calculator=False` removes it). Limits: 8 turns, 120 seconds per rollout, 10 seconds per tool call, and 24 tool calls per rollout. The answer must be one JSON object in the format given in `rules/scoring.md`.

## How to run

```bash
uv sync --extra environment --extra server
uv run luma-gym validate data/v4/test.json
uv run luma-gym generate --seed 4041 --calculations 48 --traps 48 --retrievals 12 --output tasks.json
```

The last command reproduces `data/v4/eval.json` exactly. Difficulty options: `--strata`, `--reasons`, `--conflicts`, `--contracts`.

Evaluate a model (needs `OPENROUTER_API_KEY` in `~/.env`, or `--env-file`; see `.env.example`):

```bash
uv run --extra environment python -m luma_gym.eval --model google/gemini-3.1-flash-lite-preview \
  --tasks data/v4/eval.json --mode single --rollouts 3 --max-cost 2.00 --output results/my-run --execute
```

Without `--execute` it prints the plan. Finished rollouts are skipped on resume; every paid request reserves its worst-case cost in `results/spending.json`, and the run stops cleanly at the cap.

Dashboard and tests:

```bash
uv run --extra environment --extra server uvicorn luma_gym.server:app --port 8000
uv run --extra environment --extra server python -m unittest discover tests
```

Training needs a CUDA GPU. Model weights are not in the repository; configs record the base model revision and each adapter's hash.

```bash
uv sync --extra local
uv run --extra local python -m luma_gym.sft --output results/v4/sft
uv run --extra local python -m luma_gym.sft --output results/v4/sft --merge
uv run --extra local python -m luma_gym.train --tasks data/v4/train.json --demos data/v4/demos.json --mode calculator \
  --base results/v4/sft/merged --max-new-tokens 384 --beta 0.04 --steps 60 --output results/v4/rl-0.04
uv run --extra local python -m luma_gym.local --tasks data/v4/test.json --demos data/v4/demos.json --mode calculator \
  --base results/v4/sft/merged --adapter results/v4/rl-0.04/checkpoints/step-60 --rollouts 4 --max-new-tokens 384 \
  --limit 1000 --output results/v4/test-rl-0.04-calculator
uv run --extra environment python scripts/score_local.py results/v4/test-rl-0.04-calculator --tasks data/v4/test.json
```

`luma_gym.local` options `--zero-shot` (no worked examples) and `--mode tools --tool-demos` (examples shown as tool calls) reproduce the checks in section 6 of the report; `scripts/modal_eval.py` runs the same evaluations on Modal GPUs.

## Sandbox quickstart

```bash
uv run python scripts/quickstart.py
```

It fetches three tasks, submits deliberately malformed answers, and prints their scores (-0.5 each). No key is needed. To test your own agent:

- `GET /tasks?n=10&tier=2` returns a run ID, the rule-set and rubric versions, the seed, the tier mix, and the tasks without answers.
- `POST /submit` with `{"run_id": ..., "answers": [{"id": ..., "answer": {...}}]}` returns each task's score, its components, and the clauses behind each mark.
- `GET /runs/<run_id>` returns the saved result.

Abuse and cost: a run holds at most 10 tasks, each caller gets 12 requests per minute, and run IDs expire after 24 hours. A repeated submission returns the saved score, a changed one is rejected, and concurrent submissions are claimed atomically in D1. No caller API keys are accepted or stored. Only correct clarifications reach the paid judge, which has its own \$0.50 ceiling; once it is spent, a submission that needs the judge gets HTTP 503 instead of a score without it. The first request after idle time can be slow while the Worker cold starts.

## AI usage

OpenCode was used for coding assistance. All changes were reviewed.
