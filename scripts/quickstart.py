import argparse
import json
import urllib.request
from decimal import Decimal

from luma_gym.reference import calculate_payroll


parser = argparse.ArgumentParser()
parser.add_argument("--url", default="https://luma-payroll-sandbox.whoisdinanath.workers.dev")
parser.add_argument("--interactive", action="store_true", help="Fetch one task and submit an answer you paste")
parser.add_argument("--tier", choices=("1", "2", "3", "all"), default="2", help="Task type for interactive mode")
args = parser.parse_args()


def call(path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    headers = {"User-Agent": "Mozilla/5.0 LumaQuickstart", "Content-Type": "application/json"}
    with urllib.request.urlopen(urllib.request.Request(args.url + path, data=body, headers=headers), timeout=60) as response:
        return json.load(response)


def record(task):
    found = {"reference_tables": task["reference_tables"]}
    for document in task["documents"].values():
        if document["kind"] == "payroll":
            found.update(document["record"])
        elif document["kind"] == "contract":
            found["contract_gross_pay"] = document["gross_pay"]
        else:
            found["amendment"] = {key: value for key, value in document.items() if key != "kind"}
    return found


if args.interactive:
    batch = call(f"/tasks?n=1&tier={args.tier}")
    task = batch["tasks"][0]
    print("\nQUESTION FOR YOUR AGENT\n")
    print(batch["rules"])
    print("\nTask:", task["prompt"])
    print("\nSource documents:")
    print(json.dumps(task["documents"], indent=2))
    print("\nRate schedules:")
    print(json.dumps(task["reference_tables"], indent=2))
    print("\nReturn one R6, R7, or R8 JSON answer using the supplied rules.")
    print("\nGive everything above to your agent, then paste its answer below.")
    print("End with a line containing only a dot (.) and press Enter. No answer is generated for you.")
    lines = []
    try:
        while True:
            line = input()
            if line.strip() == ".":
                break
            lines.append(line)
    except EOFError:
        pass
    answer = "\n".join(lines).strip()
    if not answer:
        raise SystemExit("No answer supplied; nothing submitted.")
    result = call("/submit", {"run_id": batch["run_id"], "answers": [{"id": task["id"], "answer": answer}]})
    print("\nLIVE SCORE\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0)

batch = call("/tasks?n=3&tier=2")
print("run", batch["run_id"], "| rule set", batch["ruleset_version"], "| seed", batch["seed"], "| tier mix", batch["tier_mix"])
exact = calculate_payroll(record(batch["tasks"][0]))
cent_off = calculate_payroll(record(batch["tasks"][1]))
cent_off["net_pay"] = str(Decimal(cent_off["net_pay"]) + Decimal("0.01"))
refusal = {"decision": "clarify", "reason_code": "MISSING_GROSS_PAY", "clauses": ["R1", "R7"]}
answers = [{"id": task["id"], "answer": answer} for task, answer in zip(batch["tasks"], (exact, cent_off, refusal))]
result = call("/submit", {"run_id": batch["run_id"], "answers": answers})
for label, item in zip(("exact", "net pay off by a cent", "refused"), result["per_task"]):
    parts = ", ".join(f"{name} {part['score']:.2f}" for name, part in item["breakdown"].items())
    print(f"{label:22s} score {item['score']:+.2f} | {parts} | clauses {' '.join(item['breakdown']['exact']['clauses'])}")
