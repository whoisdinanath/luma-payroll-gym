import asyncio
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

from luma_gym.budget import Budget
from luma_gym.hosted import load_key
import luma_gym.judge
from luma_gym.judge import HostedJudge, JudgeError, judge_explanation
from luma_gym.tasks import Task


ROOT = Path(__file__).resolve().parents[1]
EFFORT = sys.argv[1] if len(sys.argv) > 1 else "low"
OUTPUT = ROOT / f"results/judge-reliability-{EFFORT}-effort"
REPEATS = 3


def examples():
    tasks = {item["id"]: Task.from_dict(item) for item in json.loads((ROOT / "data/cases.json").read_text())}
    rows = []
    for row in json.loads((ROOT / "results/judge-reliability-structured/assessments.json").read_text()):
        rows.append({"set": "controlled", "name": f"{row['task_id']} {row['kind']}", "task": tasks[row["task_id"]],
                     "explanation": row["explanation"], "old": [item["criteria"] for item in row["assessments"]]})
    for row in json.loads((ROOT / "results/judge-actual-reliability/assessments.json").read_text()):
        rows.append({"set": "actual", "name": f"{row['model']} {row['id']}", "task": tasks[row["id"].rsplit(":", 1)[0]],
                     "explanation": row["explanation"], "old": [item["criteria"] for item in row["judgments"]]})
    train = [Task.from_dict(item) for item in json.loads((ROOT / "data/training-splits/train.json").read_text())]
    for line in (ROOT / "results/rl-rubric-6-beta-0.04/judge-transcript.jsonl").read_text().splitlines()[::-1]:
        event = json.loads(line)
        if event.get("event") == "request":
            asked = json.loads(event["messages"][-1]["content"])
            task = next(task for task in train if task.record == asked["record"])
            rows.append({"set": "failed in training", "name": task.id, "task": task,
                         "explanation": asked["explanation"], "old": []})
            break
    return rows


async def main():
    luma_gym.judge.REASONING_EFFORT = EFFORT
    OUTPUT.mkdir(parents=True, exist_ok=False)
    prompt = (ROOT / "rules/judge.md").read_text()
    rows = examples()
    limit = asyncio.Semaphore(6)
    async with AsyncOpenAI(api_key=load_key(Path.home() / ".env", "OPENROUTER_API_KEY"),
                           base_url="https://openrouter.ai/api/v1", timeout=120, max_retries=0) as native:
        with (OUTPUT / "transcript.jsonl").open("a") as log:
            judge = HostedJudge(native, log, Budget())

            async def ask(row):
                async with limit:
                    try:
                        return (await judge_explanation(judge, "z-ai/glm-5.3-flash", row["task"], row["explanation"], prompt))["criteria"]
                    except JudgeError as error:
                        return {"error": str(error)}

            answers = await asyncio.gather(*(ask(row) for row in rows for _ in range(REPEATS)))
    report = []
    for index, row in enumerate(rows):
        new = answers[index * REPEATS:(index + 1) * REPEATS]
        valid = [json.dumps(item, sort_keys=True) for item in new if "error" not in item]
        old = [json.dumps(item, sort_keys=True) for item in row["old"]]
        report.append({"set": row["set"], "name": row["name"], "explanation": row["explanation"], "new": new, "old": row["old"],
                       "failures": len(new) - len(valid), "stable": len(set(valid)) == 1 and len(valid) == REPEATS,
                       "matches_old": bool(old) and bool(valid) and Counter(valid).most_common(1)[0][0] == Counter(old).most_common(1)[0][0]})
    responses = [json.loads(line) for line in (OUTPUT / "transcript.jsonl").read_text().splitlines()]
    responses = [event["response"] for event in responses if event.get("event") == "response"]
    tokens = [(item.get("usage") or {}).get("completion_tokens", 0) for item in responses]
    summary = {"reasoning_effort": EFFORT, "examples": len(rows), "repeats": REPEATS,
               "failures": sum(row["failures"] for row in report),
               "unstable_examples": [row["name"] for row in report if not row["stable"]],
               "changed_from_old": [row["name"] for row in report if row["old"] and not row["matches_old"]],
               "length_stops": sum(item["choices"][0].get("finish_reason") == "length" for item in responses),
               "median_completion_tokens": statistics.median(tokens), "max_completion_tokens": max(tokens),
               "cost_usd": judge.cost, "calls": judge.calls}
    (OUTPUT / "assessments.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    for row in report:
        if not row["stable"] or (row["old"] and not row["matches_old"]) or row["set"] == "failed in training":
            print(row["set"], row["name"], "| new", row["new"], "| old", row["old"][:1], "|", row["explanation"][:120])


asyncio.run(main())
