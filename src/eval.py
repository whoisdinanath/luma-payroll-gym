import argparse
import asyncio
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from verifiers.clients import OpenAIChatCompletionsClient
from verifiers.utils.save_utils import make_serializable

from luma_gym.environment import load_environment, read_rules
from luma_gym.hosted import append_event, catalog_prices, load_key, read_events, request_json
from luma_gym.judge import JUDGE_PRICE, grade_answer
from luma_gym.scoring import RUBRIC_VERSION
from luma_gym.tasks import Task
from luma_gym.budget import LIMIT, Budget
from luma_gym import RULESET_VERSION


RETRY_WAITS = (2, 8, 30)


class RecordingClient(OpenAIChatCompletionsClient):
    def __init__(self, client, log, price, budget=None):
        super().__init__(client)
        self.log = log
        self.price = price
        self.cost = 0.0
        self.known_cost = True
        self.budget = budget
        self.retries = 0

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        for wait in (*RETRY_WAITS, None):
            try:
                return await self.record_request(prompt, model, sampling_args, tools, **kwargs)
            except (APIStatusError, APIConnectionError, APITimeoutError) as failure:
                status = getattr(failure, "status_code", None)
                if wait is None or (status is not None and status != 429 and status < 500):
                    raise
                self.retries += 1
                append_event(self.log, {"event": "retry", "http_status": status, "wait_seconds": wait,
                                        "message": str(failure)[:300]})
                await asyncio.sleep(wait)

    async def record_request(self, prompt, model, sampling_args, tools=None, **kwargs):
        started = time.monotonic()
        reserved = 0
        if self.budget:
            input_bound = len(json.dumps(prompt).encode()) + len(json.dumps(tools).encode()) + 2048
            reserved = input_bound * self.price["input"] + 8192 * self.price["output"]
            self.budget.reserve(reserved)
            sampling_args = {**sampling_args, "max_tokens": 8192}
        append_event(self.log, {"event": "request", "messages": prompt, "tools": tools})
        try:
            response = await super().get_native_response(prompt, model, sampling_args, tools, **kwargs)
        except APIStatusError as failure:
            if self.budget and failure.status_code in (400, 401, 402, 403, 404, 422, 429):
                self.budget.settle(reserved, 0)
            append_event(self.log, {"event": "provider_error", "http_status": failure.status_code,
                                    "message": str(failure)[:1500]})
            raise
        data = response.model_dump(mode="json")
        usage = data.get("usage") or {}
        cost = usage.get("cost")
        source = "provider"
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            source = "catalog_estimate"
            if type(usage.get("prompt_tokens")) is int and type(usage.get("completion_tokens")) is int:
                cost = usage["prompt_tokens"] * self.price["input"] + usage["completion_tokens"] * self.price["output"]
            else:
                cost = None
        self.known_cost = self.known_cost and cost is not None
        self.cost += cost or 0
        append_event(self.log, {"event": "response", "response": data, "usage": usage,
                                "cost_usd": cost, "cost_source": source,
                                "latency_seconds": time.monotonic() - started})
        if self.budget:
            if cost is None:
                raise ValueError("Unknown provider cost; reservation retained")
            self.budget.settle(reserved, cost)
        return response


def save_summary(directory, events):
    results = [event for event in events if event["event"] == "finished"]
    scored = [event for event in results if event.get("score") is not None]
    scores = [event["score"]["score"] for event in scored]
    metrics = {
        "completed_rollouts": len(results), "scored_rollouts": len(scored),
        "infrastructure_failures": sum(event.get("error") is not None for event in results),
        "recorded_cost_usd": sum(event["cost_usd"] for event in results),
        "unknown_cost_rollouts": sum(not event["known_cost"] for event in results),
        "retried_rollouts": sum(event.get("retries", 0) > 0 for event in results),
        "mean_reward": statistics.mean(scores) if scores else None,
        "sd_reward": statistics.stdev(scores) if len(scores) > 1 else None,
        "pass_rate": sum(event["score"]["passed"] for event in scored) / len(scored) if scored else None,
    }
    metrics["tiers"] = {}
    for tier in (1, 2, 3):
        values = [event["score"]["score"] for event in scored if event["tier"] == tier]
        if values:
            metrics["tiers"][str(tier)] = {"mean": statistics.mean(values),
                                           "sd": statistics.stdev(values) if len(values) > 1 else None}
    metrics["mean_latency_seconds"] = statistics.mean(event["latency_seconds"] for event in results) if results else None
    for name, data in (("metrics", metrics), ("scores", results)):
        (directory / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")


async def evaluate(args):
    settings = json.loads(args.config.read_text())
    prices = catalog_prices(request_json(settings["catalog_url"]), [args.model])
    env = load_environment(mode=args.mode, tasks_path=args.tasks, calculator=not args.no_calculator)
    rows = env.get_dataset().to_list()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("Task limit must be positive")
    if args.limit:
        rows = rows[:args.limit]
    if args.task_ids:
        rows = [row for row in rows if row["info"]["task_id"] in args.task_ids]
        if len(rows) != len(set(args.task_ids)):
            raise ValueError("Unknown or repeated task IDs")
    if args.rollouts < 3:
        raise ValueError("Use at least three rollouts")
    config = {
        "model": args.model, "mode": args.mode, "rollouts": args.rollouts,
        "temperature": None if args.provider_default_temperature else 0.7,
        "base_url": settings["base_url"], "price": prices[args.model],
        "rubric_version": RUBRIC_VERSION, "ruleset_version": RULESET_VERSION,
        "judge_model": "z-ai/glm-5.3-flash",
        "judge_prompt_sha256": hashlib.sha256(Path("rules/judge.md").read_bytes()).hexdigest(),
        "taskset_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256(read_rules().encode()).hexdigest(),
        "prompts_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
        "tool_schema_sha256": hashlib.sha256(json.dumps(
            [tool.model_dump(mode="json") for tool in (env.tool_defs or [])], sort_keys=True,
        ).encode()).hexdigest(),
        "task_ids": [row["info"]["task_id"] for row in rows],
        "generator_seeds": sorted({item["seed"] for item in json.loads(args.tasks.read_text()) if item["seed"] is not None}),
        "completion_token_ceiling": 8192,
        "max_cost": args.max_cost,
    }
    print(json.dumps({"planned_rollouts": len(rows) * args.rollouts, "config": config}, indent=2))
    if not args.execute:
        return
    key = load_key(args.env_file, settings["key_env"])
    budget = Budget(limit=args.max_cost)
    print("Shared spending:", budget.change(), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Run configuration changed; use another output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    events = read_events(args.output / "progress.jsonl")
    started = {event["id"] for event in events if event["event"] == "started"}
    finished = {event["id"] for event in events if event["event"] == "finished"}
    if started - finished:
        raise ValueError("Unfinished rollout found; inspect transcripts and cost before retrying")
    async with AsyncOpenAI(api_key=key, base_url=settings["base_url"], max_retries=0, timeout=110) as native:
        with (args.output / "progress.jsonl").open("a") as progress:
            with (args.output / "transcript.jsonl").open("a") as transcript:
                judge_client = RecordingClient(native, transcript, JUDGE_PRICE, budget)
                env = load_environment(mode=args.mode, tasks_path=args.tasks, judge_client=judge_client,
                                       judge_prompt=Path("rules/judge.md").read_text(), calculator=not args.no_calculator)
                for row in rows:
                    for rollout in range(args.rollouts):
                        identity = f"{row['info']['task_id']}:{rollout}"
                        if identity in finished:
                            continue
                        append_event(progress, {"event": "started", "id": identity})
                        append_event(transcript, {"event": "rollout", "id": identity})
                        client = RecordingClient(native, transcript, prices[args.model], budget)
                        score = None
                        error = None
                        output = None
                        start = time.monotonic()
                        judge_cost_before, judge_retries_before = judge_client.cost, judge_client.retries
                        try:
                            sampling = {"extra_body": {"provider": {"require_parameters": True}}}
                            sampling["temperature"] = None if args.provider_default_temperature else 0.7
                            output = await env.run_rollout(
                                row, client, args.model,
                                sampling,
                                state_columns=["payroll_score", "payroll_tool_calls", "payroll_tool_errors"],
                            )
                            output = json.loads(json.dumps(output, default=make_serializable))
                            if output.get("error"):
                                error = "EnvironmentError"
                            else:
                                score = output.get("payroll_score")
                                if score is None:
                                    error = "MissingScore"
                        except Exception as failure:
                            error = type(failure).__name__
                            client.known_cost = False
                        if error == "MissingScore" and output:
                            private = {item["id"]: item for item in json.loads(args.tasks.read_text())}
                            completion = output.get("completion") or []
                            text = completion[-1].get("content") if completion else None
                            try:
                                score = await grade_answer(Task.from_dict(private[row["info"]["task_id"]]),
                                                           text, judge_client, prompt=Path("rules/judge.md").read_text())
                                error = None
                                output["judge_recovery"] = True
                            except ValueError:
                                pass
                        event = {
                            "event": "finished", "id": identity, "task_id": row["info"]["task_id"],
                            "tier": row["info"]["tier"], "rollout": rollout,
                            "cost_usd": client.cost + judge_client.cost - judge_cost_before,
                            "known_cost": client.known_cost and judge_client.known_cost,
                            "latency_seconds": time.monotonic() - start, "score": score, "error": error,
                            "retries": client.retries + judge_client.retries - judge_retries_before,
                            "token_usage": output.get("token_usage") if output else None,
                            "tool_calls": output.get("payroll_tool_calls") if output else None,
                            "tool_errors": output.get("payroll_tool_errors") if output else None,
                            "truncated": output.get("is_truncated") if output else None,
                        }
                        append_event(progress, event)
                        append_event(transcript, {"event": "output", "id": identity, "output": output})
                        events.append(event)
                        save_summary(args.output, events)
                        print(f"{identity} reward={score['score'] if score else None} cost=${client.cost:.6f} error={error}", flush=True)
                        if error or not client.known_cost:
                            raise ValueError("Rollout failed or its cost is unknown; see saved files")
                        await asyncio.sleep(60 / settings["requests_per_minute"])


def main():
    parser = argparse.ArgumentParser(description="Evaluate a hosted model and save transcripts")
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=["single", "tools"], default="tools")
    parser.add_argument("--tasks", type=Path, default=Path("data/cases.json"))
    parser.add_argument("--task-ids", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--config", type=Path, default=Path("configs/openrouter-models.json"))
    parser.add_argument("--env-file", type=Path, default=Path.home() / ".env")
    parser.add_argument("--output", type=Path, default=Path("results/tool-pilot"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-cost", type=float, default=LIMIT)
    parser.add_argument("--provider-default-temperature", action="store_true")
    parser.add_argument("--no-calculator", action="store_true", help="Tool mode without the calculate tool (ablation)")
    args = parser.parse_args()
    try:
        asyncio.run(evaluate(args))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
