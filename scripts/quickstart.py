import argparse
import json
import urllib.request
from decimal import Decimal

from luma_gym.reference import calculate_payroll


parser = argparse.ArgumentParser()
parser.add_argument("--url", default="https://luma-payroll-sandbox.whoisdinanath.workers.dev")
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
