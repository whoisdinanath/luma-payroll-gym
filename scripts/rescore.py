import argparse
import asyncio
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

from luma_gym.hosted import read_events
from luma_gym.judge import grade_answer
from luma_gym.tasks import Task


class SavedJudge:
    def __init__(self, assessment):
        self.assessment = assessment

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        message = SimpleNamespace(content=json.dumps(self.assessment["criteria"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


async def rescore(task, text, old):
    saved = (old or {}).get("breakdown", {}).get("explanation")
    judge = SavedJudge(saved) if saved and saved.get("source") != "empty-explanation" else None
    result = await grade_answer(task, text, judge, prompt="replayed saved judgment")
    if saved and "explanation" in result["breakdown"]:
        result["breakdown"]["explanation"] = saved
    return result


def answer_texts(directory):
    if (directory / "generations.jsonl").exists():
        return {row["id"]: row["final_answer"] for row in read_events(directory / "generations.jsonl")}
    texts = {}
    for event in read_events(directory / "transcript.jsonl"):
        if event.get("event") == "output" and event.get("output"):
            completion = event["output"].get("completion") or []
            final = completion[-1] if completion else None
            texts[event["id"]] = final["content"] if final and final.get("role") == "assistant" else None
    return texts


async def main():
    parser = argparse.ArgumentParser(description="Re-score saved answers under the current rubric, replaying saved judgments")
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tasks", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.directory / "config.json").read_text())
    if config.get("taskset_sha256") != hashlib.sha256(args.tasks.read_bytes()).hexdigest():
        raise SystemExit(f"{args.directory.name}: task file does not match the run")
    tasks = {item["id"]: Task.from_dict(item) for item in json.loads(args.tasks.read_text())}
    texts = answer_texts(args.directory)
    path = args.directory / "scores.json"
    rows = json.loads(path.read_text())
    version = next(row["score"]["rubric_version"] for row in rows if row.get("score"))
    history = args.directory / f"scores-rubric-{version}.json"
    if not history.exists():
        history.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        if row.get("score"):
            row["score"] = await rescore(tasks[row["id"].rsplit(":", 1)[0]], texts[row["id"]], row["score"])
    path.write_text(json.dumps(rows, indent=2) + "\n")
    scored = [row for row in rows if row.get("score")]
    rewards = [row["score"]["score"] for row in scored]
    metrics_path = args.directory / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update({"rubric_version": scored[0]["score"]["rubric_version"], "mean_reward": statistics.mean(rewards),
                    "sd_reward": statistics.stdev(rewards) if len(rewards) > 1 else None})
    if "pass_rate" in metrics:
        metrics["pass_rate"] = statistics.mean(row["score"]["passed"] for row in scored)
    if "passed" in metrics:
        metrics["passed"] = sum(row["score"]["passed"] for row in scored)
    if "statuses" in metrics:
        metrics["statuses"] = {status: sum(row["score"]["status"] == status for row in scored)
                               for status in {row["score"]["status"] for row in scored}}
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"{args.directory.name}: rubric {version} -> {metrics['rubric_version']}, mean {metrics['mean_reward']:.4f}")


asyncio.run(main())
