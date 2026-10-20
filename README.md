# Luma Payroll Gym

A verifiable gym for one slice of payroll work under a fictional rule set: look up a rate for a given month, compute a monthly payslip, or ask for clarification when the records do not determine an answer. Every answer is computed by a reference implementation of the written rules, rates differ per case so they must be looked up, and caps and conflicting records make natural traps. The same grader serves the evaluation, the RL trainer, and a live sandbox.

- Dashboard: https://luma-payroll-sandbox.whoisdinanath.workers.dev
- Sandbox API: the same URL (`/tasks`, `/submit`, `/runs/<run_id>`)
- Sandbox guide (tiers, request/answer formats, runnable client, rewards, and limits): https://luma-payroll-sandbox.whoisdinanath.workers.dev/sandbox-guide
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
| Rejected payslip with the right decision | -0.5 + 0.2 × share of computed values right: -0.5 to -0.3 |
| Any other rejected answer: wrong value or reason, calculating through a trap, refusing a payslip the rules determine, malformed | -0.5 |

Correct clarifications score 0.9 × the deterministic score + 0.1 × a judge score for the explanation. A rejected answer can never score above -0.3, so partial credit ranks wrong answers without rewarding them. Refusing a payslip the rules determine costs the same as getting every computed field wrong. The judge's 0.10 weight keeps its opinion from outweighing the deterministic score. Rubrics 4 to 7 each fixed something RL exploited in the version before; rubric 8 prices a refusal like a wrong answer once the calculator made exact payslips reachable. `rules/scoring.md` gives what each score means under the rules, why each trap is a trap, and the rubric history; `rules/judge.md` is the judge prompt.

## Environment and tools

```python
from luma_gym import load_environment

env = load_environment(mode="tools", tasks_path="data/v4/eval.json")
```

Single-turn mode puts the record and reference schedule in the prompt. Tool mode hides them behind `list_documents`, `read_document`, and `lookup_reference`, and adds `calculate`, which evaluates a short formula exactly and holds no rates or rules (`calculator=False` removes it). Limits: 8 turns, 120 seconds per rollout, 10 seconds per tool call, and 24 tool calls per rollout. The answer must be one JSON object in the format given in `rules/scoring.md`.

## How to run

For the dashboard, sandbox, hosted evaluation, and tests (no GPU needed):

```bash
uv sync --extra server --extra environment
uv run python -m unittest discover -s tests -q
uv run uvicorn luma_gym.server:app --port 8000
```

Open http://localhost:8000. For training and local inference, use `uv sync --all-extras` and a CUDA GPU. MiniCPM5-1B downloads from Hugging Face on first use. `requirements.txt` is the pinned all-extras pip alternative; `pip install -r requirements.txt` includes the larger training stack. Use `uv run` before the commands below if the project environment is not activated.

Tasks:

```bash
luma-gym validate data/v4/test.json
luma-gym generate --seed 4041 --calculations 48 --traps 48 --retrievals 12 --output tasks.json
```

The second command reproduces `data/v4/eval.json` exactly. Difficulty options: `--strata`, `--reasons`, `--conflicts`, `--contracts`.

Evaluate a hosted model (needs `OPENROUTER_API_KEY` in `~/.env`, or `--env-file`; see `.env.example`):

```bash
python -m luma_gym.eval --model google/gemini-3.1-flash-lite-preview --tasks data/v4/eval.json \
  --mode single --rollouts 3 --max-cost 2.00 --output results/my-run --execute
```

Without `--execute` it prints the plan. Finished rollouts are skipped on resume; every paid request reserves its worst-case cost in `results/spending.json`, and the run stops cleanly at `--max-cost`.

Dashboard and tests:

```bash
uvicorn luma_gym.server:app --port 8000
python -m unittest discover tests
```

SFT, RL, and evaluation of the local model:

```bash
python -m luma_gym.sft --output results/v4/sft
python -m luma_gym.sft --output results/v4/sft --merge
python -m luma_gym.train --tasks data/v4/train.json --demos data/v4/demos.json --mode calculator \
  --base results/v4/sft/merged --max-new-tokens 384 --beta 0.04 --steps 60 --output results/v4/rl-0.04
python -m luma_gym.local --tasks data/v4/test.json --demos data/v4/demos.json --mode calculator \
  --base results/v4/sft/merged --adapter results/v4/rl-0.04/checkpoints/step-60 --rollouts 4 \
  --max-new-tokens 384 --limit 1000 --output results/v4/test-rl-0.04-calculator
python scripts/score_local.py results/v4/test-rl-0.04-calculator --tasks data/v4/test.json
```

The payslip-focused run adds `--calc-prompts 6 --trap-prompts 1 --balance-halves --lr 5e-5 --steps 40`. On `luma_gym.local`, `--zero-shot` drops the worked examples and `--mode tools --tool-demos` shows them as tool calls.

The fresh confirmation is reproducible with the existing Modal setup:

```bash
python -m scripts.confirm_training
modal run scripts/modal_eval.py --fresh
python -m scripts.confirm_training --summarize
```

The Modal command is paid and refuses to relaunch saved runs automatically. It compares three fixed checkpoints on seed 51021, with a 3,000-second limit per L4 invocation, capped CPU/memory, no automatic retries, and progress saved to a Modal volume. Summarizing is CPU-only and uses the shared programmatic verifier, without charging an explanation judge. Trained weights are hosted separately at the Hugging Face link above; the repository contains configurations, evaluation outputs, and training traces.

## Sandbox quickstart

```bash
python scripts/quickstart.py
```

It fetches three payslip tasks and submits an exact answer, one with net pay off by a cent, and a refusal, then prints each score with its components and clauses (1.00, -0.35, -0.50). No key is needed. To test your own agent:

- `GET /tasks?n=10&tier=2` returns a run ID, the rule-set and rubric versions, the seed, the tier mix, and the tasks without answers.
- `POST /submit` with `{"run_id": ..., "answers": [{"id": ..., "answer": {...}}]}` returns each task's score, its components, and the clauses behind each mark.
- `GET /runs/<run_id>` returns the saved result.

For a real model demonstration, run `uv run python scripts/demo_model.py --output results/my-demo`, then `uv run python scripts/demo_score.py --input results/my-demo`. The first command fetches a fresh task and saves the actual model response; the second submits that unchanged answer and saves the live reward. The model call uses your OpenRouter key; the public sandbox itself requires no client key. See the [sandbox guide](reports/sandbox-api.md) for tiers, answer schemas, limits, and a standalone client.

## Training objective

The goal is to make correct answers more likely without changing the model too far in one update.

1. **SFT teaches the starting procedure:** learn from solved examples of reading payroll records and using the calculator.
2. **GRPO learns from attempts:** generate several answers to the same task, score them, and compare their rewards. Better-than-average attempts are encouraged; worse ones are discouraged. If every reward is tied, that group provides no reward preference.
3. **PPO clipping limits each update:** reduce the incentive to make very large changes to an answer's token probabilities in one update.
4. **The KL penalty limits drift:** compare the changing model with its frozen starting reference. Beta controls how strongly that difference is penalized.

We use LoRA adapters and 4-bit base weights to fit training on a 6 GB GPU. GRPO does not need a separate value model, which also saves memory. The reward comes from rubric 8; the calculator only evaluates the arithmetic the model writes. It does not choose the rates or decide whether a payslip is justified. The objective is implemented in `src/train.py`; the full report is supplied separately as a submission PDF.

## Next steps

- Train on varied workplace wording, not just one fixed instruction.
- Include cases with several issues, so the model must check missing inputs before salary conflicts.
- Reduce dependence on worked examples and train with the same document tools used at evaluation.
- Repeat the fresh-case check across training seeds before treating one strong run as a reliable result.

## AI usage

OpenCode was used for coding assistance. All changes were reviewed.
