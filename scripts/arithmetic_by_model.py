import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

from luma_gym.hosted import read_events
from luma_gym.tasks import Task


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["employee_contribution", "taxable_pay", "income_tax", "net_pay"]
HOSTED = ["llama-1b", "llama-3b", "gemma-4b", "qwen-7b", "gemini", "glm", "mimo", "luna-default", "deepseek"]


def answers(directory):
    found, current = {}, None
    for event in read_events(directory / "transcript.jsonl"):
        if event["event"] == "rollout":
            current = event["id"]
        elif event["event"] == "response" and current and event["response"].get("model") != "z-ai/glm-5.3-flash":
            found[current] = event["response"]["choices"][0]["message"].get("content")
        elif event["event"] == "output" and event.get("output") and event["output"].get("completion"):
            found[event["id"]] = event["output"]["completion"][-1].get("content")
    finished = {event["id"] for event in read_events(directory / "progress.jsonl") if event["event"] == "finished"}
    return {identity: text for identity, text in found.items() if identity in finished}


def lenient(text):
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def measure(rows, tasks):
    attempts = exact = consistent = 0
    right = dict.fromkeys(FIELDS, 0)
    payslips = 0
    for task_id, text in rows:
        task = tasks[task_id]
        if task.difficulty != 2:
            continue
        payslips += 1
        answer = lenient(text)
        if not answer or answer.get("decision") != "calculate":
            continue
        attempts += 1
        hits = [answer.get(field) == task.ground_truth[field] for field in FIELDS]
        exact += all(hits)
        for field, hit in zip(FIELDS, hits):
            right[field] += hit
        try:
            values = {key: Decimal(answer[key]) for key in ["gross_pay", "allowed_deduction"] + FIELDS}
            consistent += values["taxable_pay"] == values["gross_pay"] - values["employee_contribution"] - values["allowed_deduction"]
        except (KeyError, InvalidOperation, TypeError):
            pass
    return {"parser": "lenient",
            "payslip_answers": payslips, "attempted": attempts, "all_four_right": exact,
            "fields_right": right, "own_taxable_consistent": consistent}


def main():
    eval_tasks = {item["id"]: Task.from_dict(item) for item in json.loads((ROOT / "data/v4/eval.json").read_text())}
    report = {}
    for name in HOSTED:
        directory = ROOT / "results/v4" / f"hosted-{name}-single"
        if (directory / "progress.jsonl").exists():
            rows = [(identity.split(":")[0], text) for identity, text in answers(directory).items()]
            report[name] = measure(rows, eval_tasks)
    test_tasks = {item["id"]: Task.from_dict(item) for item in json.loads((ROOT / "data/v4/test.json").read_text())}
    rows = [(row["task_id"], row["final_answer"]) for row in read_events(ROOT / "results/v4/test-raw-single/generations.jsonl")]
    report["minicpm-1b (local, test set)"] = measure(rows, test_tasks)
    (ROOT / "reports/arithmetic-by-model.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'model':30s} {'attempted':>9s} {'all 4':>6s} {'contrib':>8s} {'taxable':>8s} {'tax':>6s} {'net':>6s} {'self-consistent':>15s}")
    for name, row in report.items():
        n = row["attempted"] or 1
        fields = row["fields_right"]
        print(f"{name:30s} {row['attempted']:9d} {row['all_four_right'] / n:6.0%} "
              + " ".join(f"{fields[field] / n:{width}.0%}" for field, width in zip(FIELDS, (8, 8, 6, 6)))
              + f" {row['own_taxable_consistent'] / n:15.0%}")
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
