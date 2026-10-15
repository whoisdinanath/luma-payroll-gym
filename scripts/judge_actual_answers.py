import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from luma_gym.budget import Budget
from luma_gym.eval import RecordingClient
from luma_gym.hosted import load_key
from luma_gym.judge import JUDGE_PRICE, judge_explanation
from luma_gym.scoring import parse_answer
from luma_gym.tasks import Task


root = Path(__file__).resolve().parents[1]


async def main():
    tasks = {item["id"]: Task.from_dict(item) for item in json.loads((root / "data/cases.json").read_text())}
    examples = []
    for name in ("gemini", "mimo", "glm"):
        path = root / f"results/comparison-{name}-single/transcript.jsonl"
        if not path.exists():
            continue
        found = 0
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event["event"] != "output" or not event["output"]:
                continue
            output = event["output"]
            task = tasks[output["info"]["task_id"]]
            if task.difficulty != 3:
                continue
            try:
                answer = parse_answer(output["completion"][-1]["content"])
            except ValueError:
                continue
            if answer.get("explanation") and answer.get("reason_code") == task.ground_truth["reason_code"]:
                examples.append((name, event["id"], task, answer["explanation"]))
                found += 1
            if found == 5:
                break
    directory = root / "results/judge-actual-reliability"
    directory.mkdir(exist_ok=True)
    saved = directory / "assessments.json"
    results = json.loads(saved.read_text()) if saved.exists() else []
    completed = {(row["model"], row["id"]) for row in results}
    async with AsyncOpenAI(api_key=load_key(Path.home() / ".env", "OPENROUTER_API_KEY"),
                           base_url="https://openrouter.ai/api/v1", max_retries=0, timeout=90) as native:
        with (directory / "transcript.jsonl").open("a") as log:
            client = RecordingClient(native, log, JUDGE_PRICE, Budget())
            for name, identity, task, text in examples:
                if (name, identity) in completed:
                    continue
                judgments = []
                for repeat in range(3):
                    try:
                        judgments.append(await judge_explanation(client, "z-ai/glm-5.3-flash", task, text,
                                                                (root / "rules/judge.md").read_text()))
                    except ValueError as error:
                        judgments.append({"error": str(error), "score": None})
                results.append({"model": name, "id": identity, "explanation": text, "judgments": judgments})
                (directory / "assessments.json").write_text(json.dumps(results, indent=2) + "\n")
                print(name, identity, [item["score"] for item in judgments], flush=True)
            valid = [row for row in results if all("criteria" in item for item in row["judgments"])]
            disagreements = sum(len({json.dumps(item["criteria"], sort_keys=True) for item in row["judgments"]}) > 1 for row in valid)
            report = {"examples": len(results), "repeats": 3, "disagreement_rate": disagreements / len(valid) if valid else None,
                      "complete_valid_examples": len(valid), "malformed_or_failed_examples": len(results) - len(valid),
                      "recorded_cost_usd": client.cost, "selection": "first five correct trap explanations per model"}
            (directory / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
            print(report)


asyncio.run(main())
