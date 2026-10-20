# Sandbox API: how to test your agent

**Base URL:** `https://luma-payroll-sandbox.whoisdinanath.workers.dev`

The sandbox is an examiner: it gives your agent a fresh payroll task and grades the answer. **It does not solve questions for you or host your agent.** You supply the agent or enter an answer yourself. No API key is required to fetch tasks or submit answers. Your chosen model may need its own credentials.

## 1. What does tier mean?

A tier is a **task type**, not a subscription or access level.

| Parameter | Task type | What your agent should do |
| --- | --- | --- |
| `tier=1` | Rule lookup | Find one value in the supplied case schedule: contribution rate, contribution cap, or deduction cap. |
| `tier=2` | Payslip calculation | Calculate the full payslip from complete records, resolving an approved amendment if needed. |
| `tier=3` | Clarification / trap | Identify why the rules do not yet determine a payslip and ask for the relevant information or resolution. |
| `tier=all` | Mixed task types | Process each task according to its prompt and records. The response includes the actual counts in `tier_mix`. |

`n` is the **total number of tasks** requested, from 1 to 10. It is not the number per tier. Defaults are `n=3` and `tier=all`. A small mixed batch is not guaranteed to contain every task type; inspect `tier_mix`. Choose a specific tier when you want to test one type.

Examples:

```text
GET /tasks?n=1&tier=1     One lookup question
GET /tasks?n=1&tier=2     One complete payslip case
GET /tasks?n=3&tier=3     Three cases requiring clarification
GET /tasks?n=6&tier=all   Six tasks from the mixed set
```

## 2. The reviewer workflow

1. **Fetch:** call `GET /tasks` and keep the returned `run_id`.
2. **Solve:** give your agent the returned `rules`, task `prompt`, `documents`, and `reference_tables`.
3. **Submit:** call `POST /submit` with the `run_id` and exactly one answer for every task in that batch.
4. **Inspect:** read the reward, pass/fail, component scores, and rule references. You can retrieve the completed result again with `GET /runs/{run_id}`.

The expected answers stay on the server. Each fetch generates a fresh batch using a server-selected seed. Keep the task and run IDs together; a task from another batch cannot be submitted in this run.

### Ready-made clients in the repository

From the repository root, with the project dependencies installed:

```bash
# Check the grader with three deliberately constructed answers. No model API key needed.
uv run python scripts/quickstart.py

# Fetch one lookup task, paste your own or your agent's JSON answer, then enter a dot on its own line.
uv run python scripts/quickstart.py --interactive --tier 1

# Real model demo: fetch one payslip task and ask DeepSeek, saving the task and answer.
# Needs OPENROUTER_API_KEY in your environment or ~/.env. This is a paid model call.
uv run python scripts/demo_model.py --output results/my-sandbox-demo

# Submit that model's untouched saved answer. No client model key needed for this step.
uv run python scripts/demo_score.py --input results/my-sandbox-demo
```

For a different model, add `--model MODEL_ID` to `demo_model.py`. The model demo is single-turn with all public records inline; it does not give the model a calculator tool. It saves `question.txt`, `answer.txt`, and `result.json`, along with the raw task, request, response, and submission. Use a new output folder for a new model demo.

## 3. Fetch tasks: GET /tasks

```bash
curl --fail-with-body 'https://luma-payroll-sandbox.whoisdinanath.workers.dev/tasks?n=1&tier=1'
```

The response contains:

| Field | Meaning |
| --- | --- |
| `run_id` | Server-issued ID for this batch. Send it back unchanged when submitting. |
| `ruleset_version` | Version of the payroll rules. |
| `rubric_version` | Version of the scoring rules. |
| `seed` | Random seed used to generate this batch; returned for traceability, not a client-set query parameter. |
| `tier_mix` | Number of lookup, calculation, and clarification tasks in this batch, keyed by `"1"`, `"2"`, and `"3"`. |
| `rules` | Full payroll rulebook your agent must follow. |
| `tasks` | Public task objects, described below. |

Each task has:

- `id`: use this exact ID in the answer envelope.
- `prompt`: what the agent is asked to do. A lookup prompt names the requested field.
- `documents`: objects keyed by document ID. A payroll document has `kind: "payroll"` and a `record`; a contract has `kind: "contract"` and `gross_pay`; an amendment has `kind: "amendment"`, `gross_pay`, `effective_month`, and `approved`.
- `reference_tables`: the authoritative case-specific rate rows. **These replace the default numeric values in the rulebook for this case.**

Use the payroll month to select `first-half` for January–June 2026 or `second-half` for July–December 2026. Upload dates and amendment dates do not supply a missing payroll month. Other years have no applicable row.

The public API gives documents and schedules directly as JSON. It does not provide HTTP endpoints called `list_documents`, `read_document`, or `calculate`; those are tools in the separately packaged evaluation environment. Your agent can read the API's data directly or wrap it in its own tools.

## 4. Answer formats

Return one JSON object per task. Monetary calculation fields must be non-negative strings with exactly two decimal places. An optional string `explanation` is accepted. Do not copy input-only fields such as `payroll_month` or `amendment` into the answer. Extra fields, duplicate keys, surrounding prose, and multiple objects are rejected. A single JSON or unlabelled Markdown code fence is accepted around an answer string.

### Tier 1: retrieve one value

Illustration only: suppose the prompt asks for `contribution_cap`, the month is August, and the case's second-half row has `cap: "4500.00"`:

```json
{
  "decision": "retrieve",
  "field": "contribution_cap",
  "value": "4500.00",
  "clauses": ["R1", "R3"]
}
```

Use the live case's actual value, not the example number above.

| Requested `field` | Entry in the selected schedule | Required clauses |
| --- | --- | --- |
| `contribution_rate` | `rate` | R1, R3 |
| `contribution_cap` | `cap` | R1, R3 |
| `deduction_cap` | `deduction` | R1, R4 |

Return a rate as a decimal string such as `"0.06"`, not `"6%"`. Cite the selecting and source rules, not the answer-format rule R8.

### Tier 2: calculate a payslip

This is a complete answer **for an illustrative June record with gross 5000.00, deduction 300.00, default first-half rates, and no unresolved conflict**. It is not the expected answer to a randomly fetched task.

```json
{
  "decision": "calculate",
  "currency": "LCR",
  "gross_pay": "5000.00",
  "employee_contribution": "200.00",
  "allowed_deduction": "300.00",
  "taxable_pay": "4500.00",
  "income_tax": "500.00",
  "net_pay": "4000.00",
  "clauses": ["R1", "R2", "R3", "R4", "R5", "R6"]
}
```

Use decimal arithmetic with round-half-up. Round the contribution, subtract it and the allowed deduction to get taxable pay, apply marginal tax brackets, round total tax, then calculate net pay. If an approved, effective, matching amendment was needed to resolve a contract conflict, also cite R9. An amendment's presence alone does not require R9 on a calculated answer.

### Tier 3: ask for clarification

Example for a record whose payroll month is missing:

```json
{
  "decision": "clarify",
  "reason_code": "MISSING_PAYROLL_MONTH",
  "clauses": ["R1", "R7"],
  "explanation": "The payroll month is missing, so I cannot select a rate row. Please provide the payroll month."
}
```

Check the issues in this order, and return the **first** applicable reason:

| Order | `reason_code` | When to use it | Clauses |
| --- | --- | --- | --- |
| 1 | `MISSING_GROSS_PAY` | Payroll gross is absent or null. Do not infer it from a contract or amendment. | R1, R7 |
| 2 | `MISSING_PAYROLL_MONTH` | Payroll month is absent or null. | R1, R7 |
| 3 | `NO_APPLICABLE_REFERENCE` | Payroll month is outside the covered year, 2026. | R1, R7 |
| 4 | `CONFLICTING_GROSS_PAY` | Contract and payroll gross differ, without an approved, effective amendment matching payroll gross. | R7, R9 |

Do not include payslip amounts in a clarification. Explain the unresolved issue and request the information or correction that would resolve it. Don't calculate a hypothetical answer alongside it.

## 5. Submit answers: POST /submit

Request body:

```json
{
  "run_id": "COPY_THE_RETURNED_RUN_ID",
  "answers": [
    {
      "id": "COPY_THE_RETURNED_TASK_ID",
      "answer": {
        "decision": "retrieve",
        "field": "contribution_cap",
        "value": "4500.00",
        "clauses": ["R1", "R3"]
      }
    }
  ]
}
```

Replace IDs and answer values with the actual batch's data. The example envelope is for a one-task batch. For a larger batch, include every task exactly once. Each item must contain exactly `id` and `answer`. `answer` may be an object or the original model-answer string. Sending the original string lets the grader evaluate the model's actual formatting.

### Complete working example: fetch, enter an answer, submit, inspect

This uses Python's standard library; no model credentials or third-party packages are needed. Save it as `sandbox_client.py` and run `python sandbox_client.py`. Give the displayed rules and task to your agent, then paste its JSON reply on one line. **This client does not solve the task for you.**

```python
import json
import urllib.request

BASE = "https://luma-payroll-sandbox.whoisdinanath.workers.dev"

def api(path, data=None):
    body = None if data is None else json.dumps(data).encode()
    request = urllib.request.Request(
        BASE + path,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "Mozilla/5.0 LumaReviewer"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)

batch = api("/tasks?n=1&tier=1")
task = batch["tasks"][0]
print("RULES\n" + batch["rules"])
print("TASK\n" + json.dumps(task, indent=2))
answer = input("Paste your agent's JSON answer on one line: ")
result = api("/submit", {
    "run_id": batch["run_id"],
    "answers": [{"id": task["id"], "answer": answer}],
})
print("RESULT\n" + json.dumps(result, indent=2))
```

To call an actual model automatically and save readable files, use `demo_model.py` and `demo_score.py` above instead. Do not use the private reference answer to stand in for an agent evaluation.

## 6. What the response and reward mean

`POST /submit` returns `run_id`, both version numbers, `mean_reward`, and `per_task`.

Each entry in `per_task` includes:

| Field | Meaning |
| --- | --- |
| `task_id` | Which task was scored. |
| `score` | Reward for this answer, normally between −0.5 and +1.0. |
| `passed` | Whether it passed all required checks, including citations and explanation criteria where applicable. |
| `status` | `scored`, `invalid_format`, `wrong_decision`, or `wrong_clarification`. `scored` alone does not mean it passed. |
| `breakdown` | Decision, exactness, computed-field accuracy, citations, and explanation review where applicable, with rule references. |
| `diagnostics` | Whether format and decision checks passed. |
| `fields` | Checks on individual substantive answer fields. |

| Outcome | Reward |
| --- | --- |
| Exact substantive answer, exact citations | +1.00 before any clarification explanation weighting |
| Exact substantive answer, incorrect citations | +0.95 before any clarification explanation weighting; not a full pass |
| Wrong payslip with the right decision | −0.5 + 0.2 × share of four computed values correct; total remains negative |
| Wrong lookup, wrong clarification reason, wrong action, or invalid answer format | −0.50 |

Only a clarification with the correct decision and reason reaches the explanation judge. Its final reward is **0.90 × code-based score + 0.10 × explanation score**. The judge checks that the answer identifies the issue, asks for a relevant resolution, and avoids an unsupported answer. It does not grade arithmetic or override a wrong decision. If explanation judging fails, the service reports an infrastructure error rather than inventing a score.

`mean_reward` averages the batch's rewards. It is **not accuracy or pass@k**. One submission supplies one answer per task. To study repeated samples, fetch a batch and obtain the agent answers under your own evaluation protocol; do not attempt to replace a completed run's answer with a better attempt.

### What do the dashboard badges mean?

“Useful explanation: 1.00”, “Empty explanation: 0.90”, and “Prompt injection: 0.90” are **saved judge smoke-check results**, not task tiers, parameters, or guaranteed scores for arbitrary text.

In those checks, the clarification's structured answer was already correct. The useful explanation earned its explanation credit. The empty and instruction-manipulating explanations did not, leaving 0.90 from the code-based part. The injection sample did **not** receive a full pass just because its reward was positive, and that single check is not a general security guarantee.

## 7. Saved results, expiration, and retry behavior

- A run expires **24 hours after task creation**, not 24 hours after submission. This is what “24h TTL” means.
- Submit exactly one answer for each task in the run. Partial batches and extra task IDs are rejected.
- The first valid submission binds that run to those answers. Repeating the same completed submission returns its saved result. Changing the answers for that run returns a conflict.
- `GET /runs/{run_id}` returns the completed saved result. It returns 404 for an unknown, expired, or unfinished run.
- If a connection fails, retry the **same submission** rather than fetching another task or changing the answer. If scoring is still processing, wait before retrying.
- Keep your run ID to retrieve its result; there is no account login. Avoid sharing it if you don't want others to retrieve that run's score.

## 8. Limits and errors

| HTTP status | Meaning / action |
| --- | --- |
| 200 | Request succeeded. Inspect the reward and `passed`; a wrong agent answer still receives HTTP 200 if it was scored normally. |
| 400 | Invalid parameter, JSON, or submission shape; check `n`, `tier`, IDs, and required envelope fields. |
| 404 | Unknown/expired run, or no completed result yet. |
| 409 | Another submission is processing, or this run is already bound to different answers. |
| 413 | Submit body is too large or lacks the required Content-Length header. Standard JSON HTTP clients set this header automatically. |
| 429 | Too many requests. The deployed service allows up to 12 counted requests per minute per IP; wait before retrying. Health checks are exempt. |
| 503 | Scoring infrastructure unavailable, including judge/provider/budget issues. Retry later with the same submission. It is not a model-answer failure. |

Limits: 1–10 tasks per fetch; submit bodies at most 100,000 bytes with Content-Length; individual answer strings at most 16,000 characters. The hosted judge has a separate server-side spending ceiling. No reviewer can access its API key through the public task or score response.

Use a normal client User-Agent, as in the examples. If you see “Temporary failure in name resolution,” the machine could not resolve the hostname: check DNS/connectivity and retry. The saved model answer remains usable while the run is unexpired.

## 9. Where to inspect the rules and implementation

- [Payroll rules, scoring guide, and judge criteria](/rules).
- The evaluation and training report is supplied separately as a submission PDF.
- `deploy/main.py`: deployed HTTP API behavior.
- `src/reference.py`: expected answers from the written rules.
- `src/scoring.py` and `src/judge.py`: code-based grading and explanation review.
- `scripts/quickstart.py`, `scripts/demo_model.py`, `scripts/demo_score.py`: runnable clients.

The public task format intentionally excludes `ground_truth`. The repository's offline datasets include it for reproducible evaluation; that private/offline task schema is different from the public `/tasks` response.
