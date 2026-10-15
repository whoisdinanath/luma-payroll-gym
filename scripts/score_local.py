import argparse
import asyncio
import json
import statistics
from pathlib import Path

from openai import AsyncOpenAI

from luma_gym.budget import Budget
from luma_gym.eval import RecordingClient
from luma_gym.hosted import load_key, read_events
from luma_gym.judge import JUDGE_PRICE, grade_answer
from luma_gym.tasks import Task


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--tasks", type=Path, default=Path("data/case-pilot.json"))
    args = parser.parse_args()
    tasks = {item["id"]: Task.from_dict(item) for item in json.loads(args.tasks.read_text())}
    generations = read_events(args.directory / "generations.jsonl")
    results = []
    existing_path = args.directory / "scores.json"
    existing = {row["id"]: row for row in json.loads(existing_path.read_text())} if existing_path.exists() else {}
    metrics_path = args.directory / "metrics.json"
    previous_cost = json.loads(metrics_path.read_text()).get("judge_cost_usd", 0) if metrics_path.exists() and existing else 0
    async with AsyncOpenAI(api_key=load_key(Path.home() / ".env", "OPENROUTER_API_KEY"),
                           base_url="https://openrouter.ai/api/v1", timeout=90, max_retries=0) as native:
        with (args.directory / "judge-transcript.jsonl").open("a") as log:
            judge = RecordingClient(native, log, JUDGE_PRICE, Budget())
            for row in generations:
                if row["id"] in existing:
                    results.append(existing[row["id"]])
                    continue
                result = await grade_answer(tasks[row["task_id"]], row["final_answer"], judge,
                                            prompt=Path("rules/judge.md").read_text())
                results.append({"id": row["id"], "tier": row["tier"], "score": result,
                                "truncated": row["truncated"], "seconds": row["seconds"]})
                existing_path.write_text(json.dumps(results, indent=2) + "\n")
            rewards = [row["score"]["score"] for row in results]
            seeds = sorted({tasks[row["task_id"]].seed for row in generations if tasks[row["task_id"]].seed is not None})
            metrics = {"rollouts": len(results), "tasks": str(args.tasks), "generator_seeds": seeds,
                       "mean_reward": statistics.mean(rewards),
                       "sd_reward": statistics.stdev(rewards) if len(rewards) > 1 else None,
                       "passed": sum(row["score"]["passed"] for row in results),
                       "truncated": sum(row["truncated"] for row in results),
                       "judge_cost_usd": judge.cost + previous_cost, "local_inference_api_cost_usd": 0,
                       "total_generation_seconds": sum(row["seconds"] for row in results),
                       "statuses": {status: sum(row["score"]["status"] == status for row in results)
                                    for status in {row["score"]["status"] for row in results}}}
            for name, value in (("scores", results), ("metrics", metrics)):
                (args.directory / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
            print(json.dumps(metrics, indent=2))


asyncio.run(main())
