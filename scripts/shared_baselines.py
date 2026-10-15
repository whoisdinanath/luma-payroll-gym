import asyncio
import json
from pathlib import Path

from luma_gym.baselines import baseline_answers
from luma_gym.judge import grade_answer
from luma_gym.scoring import RUBRIC_VERSION
from luma_gym.tasks import Task


ROOT = Path(__file__).resolve().parents[1]


async def score_set(path, example=None):
    tasks = [Task.from_dict(item) for item in json.loads(path.read_text())]
    results = {}
    for task in tasks:
        for name, answer in baseline_answers(task, example).items():
            if name == "reference_sanity_check":
                name = "reference_with_empty_explanation"
            result = await grade_answer(task, json.dumps(answer))
            results.setdefault(name, []).append(result)
    summary = {name: {"mean": sum(row["score"] for row in rows) / len(rows),
                      "pass_rate": sum(row["passed"] for row in rows) / len(rows)} for name, rows in results.items()}
    return {"ruleset_version": "3", "rubric_version": RUBRIC_VERSION, "tasks": len(tasks), "summary": summary,
            "results": results}


async def main():
    old = await score_set(ROOT / "data/cases.json")
    (ROOT / "reports/shared-baselines.json").write_text(json.dumps(old, indent=2) + "\n")
    demo = json.loads((ROOT / "data/v4/demos.json").read_text())[0]["ground_truth"]
    report = {}
    for name in ("eval", "train", "test"):
        report[name] = await score_set(ROOT / f"data/v4/{name}.json", demo)
    (ROOT / "reports/baselines-v4.json").write_text(json.dumps(report, indent=2) + "\n")
    rows = {name: {key: round(value["mean"], 3) for key, value in item["summary"].items()} for name, item in report.items()}
    rows["old eval (cases.json)"] = {key: round(value["mean"], 3) for key, value in old["summary"].items()}
    print(json.dumps(rows, indent=2))


asyncio.run(main())
