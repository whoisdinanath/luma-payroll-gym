"""Fetch a live sandbox task, ask a model, and save the complete demo evidence."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from luma_gym.budget import Budget
from luma_gym.hosted import catalog_prices, load_key, request_json


ROOT = Path(__file__).resolve().parents[1]


def save(directory, name, value):
    (directory / name).write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://luma-payroll-sandbox.whoisdinanath.workers.dev")
    parser.add_argument("--model", default="deepseek/deepseek-v4.1-flash")
    parser.add_argument("--tier", choices=("1", "2", "3"), default="2")
    parser.add_argument("--output", type=Path, default=ROOT / "results/sandbox-demo")
    args = parser.parse_args()
    directory = args.output
    if directory.exists() and any(directory.iterdir()):
        parser.error("Output folder already contains a demo. Use --output results/sandbox-demo-2 for a new one.")
    key = os.environ.get("OPENROUTER_API_KEY") or load_key(Path.home() / ".env", "OPENROUTER_API_KEY")
    api = "https://openrouter.ai/api/v1"
    catalog = request_json(api + "/models")
    price = catalog_prices(catalog, [args.model])[args.model]
    print("1. Fetching one fresh task from the live sandbox...", flush=True)
    url = args.url.rstrip("/")
    fetched = httpx.get(f"{url}/tasks?n=1&tier={args.tier}",
                        headers={"User-Agent": "Mozilla/5.0 LumaDemo"}, timeout=60, follow_redirects=True)
    fetched.raise_for_status()
    batch = fetched.json()
    task = batch["tasks"][0]
    messages = [
        {"role": "system", "content": (
            "Solve the task using only the supplied payroll rules and case data. "
            "Return one JSON object in the R6, R7, or R8 format. "
            "For clarification, include a short explanation and ask for the relevant resolution.\n\n"
            + batch["rules"]
        )},
        {"role": "user", "content": task["prompt"] + "\n\nSource documents:\n"
         + json.dumps(task["documents"], indent=2) + "\n\nCase-specific rate schedules:\n"
         + json.dumps(task["reference_tables"], indent=2)},
    ]
    directory.mkdir(parents=True, exist_ok=True)
    save(directory, "task.json", batch)
    question = "\n\n".join(f"{message['role'].upper()}\n{message['content']}" for message in messages)
    (directory / "question.txt").write_text(question + "\n")
    print("\nQUESTION SENT TO THE MODEL\n" + messages[1]["content"], flush=True)
    payload = {
        "model": args.model, "messages": messages, "temperature": 0.7, "max_tokens": 8192,
        "provider": {"require_parameters": True, "max_price": {
            "prompt": price["input"] * 1e6, "completion": price["output"] * 1e6}},
    }
    save(directory, "request.json", payload)
    reservation = (len(json.dumps(messages).encode()) + 2048) * price["input"] + 8192 * price["output"]
    budget = Budget(ROOT / "results/spending.json")
    budget.reserve(reservation)
    print(f"\n2. Asking {args.model}...", flush=True)
    # No reference solver, answer repair, or automatic retries: preserve the model's actual output.
    response = request_json(api + "/chat/completions", payload, key=key)
    save(directory, "response.json", response)
    cost = (response.get("usage") or {}).get("cost")
    if type(cost) not in (int, float) or cost < 0:
        raise ValueError("Provider cost unavailable; budget reservation retained. Response saved.")
    budget.settle(reservation, cost)
    answer = response["choices"][0]["message"].get("content")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Model returned no answer text. Raw response saved; no submission made.")
    (directory / "answer.txt").write_text(answer)
    save(directory, "submission.json", {
        "run_id": batch["run_id"], "answers": [{"id": task["id"], "answer": answer}],
    })
    save(directory, "demo.json", {
        "sandbox_url": url, "model": args.model, "tier": args.tier,
        "created_at": datetime.now(timezone.utc).isoformat(), "cost_usd": cost,
        "finish_reason": response["choices"][0].get("finish_reason"),
        "mode": "single-turn; public documents and schedules inline; no calculator tool",
    })
    print("\nMODEL'S ACTUAL ANSWER\n" + answer)
    print(f"\nModel call cost: ${cost:.6f}")
    print(f"Saved question, response, answer, and submission in {directory}")
    print(f"\nNext: uv run python scripts/demo_score.py --input {directory}")


if __name__ == "__main__":
    main()
