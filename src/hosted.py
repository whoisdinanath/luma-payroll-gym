import json
import math
import os
import urllib.request
from pathlib import Path


def request_json(url, payload=None, key=None):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=body, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def load_key(path, key_name=None):
    values = dict(os.environ)
    if path:
        for line in Path(path).expanduser().read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            name, separator, value = line.partition("=")
            if separator:
                values[name.strip()] = value.strip().strip("\"'")
    name = key_name or "OPENROUTER_API_KEY"
    if values.get(name):
        return values[name]
    raise ValueError(f"Missing {name}")


def messages_for(task, rules):
    contract = (
        "Allowed answer fields: calculate uses decision, currency, gross_pay, employee_contribution, "
        "allowed_deduction, taxable_pay, income_tax, net_pay, clauses; retrieve uses decision, field, value, clauses; "
        "clarify uses decision, reason_code, clauses. An optional explanation string is permitted. "
        "Do not include input-only fields such as payroll_month, contract_gross_pay, or amendment. "
        "For clarification include a short explanation naming the unresolved blocker and asking for its resolution. "
    )
    return [
        {"role": "system", "content": (
            "Follow the supplied rules. Return one JSON object without Markdown fences or surrounding prose. "
            "Use exactly the required answer fields and only the clauses that directly govern the answer. "
            "Use the citation requirements in the supplied rules. "
            "Money must be strings with two decimal places. " + contract + "\n\n" + rules
        )},
        {"role": "user", "content": task.prompt + "\nRecord: " + json.dumps(task.record)},
    ]


def catalog_prices(catalog, models):
    prices = {}
    for model in models:
        matches = [item for item in catalog.get("data", []) if item.get("id") == model]
        if not matches:
            raise ValueError(f"Model not in catalog: {model}")
        rates = []
        for item in matches:
            pricing = item.get("pricing", {})
            input_rate = pricing.get("input_cost_per_token", pricing.get("prompt"))
            output_rate = pricing.get("output_cost_per_token", pricing.get("completion"))
            rates.append((float(input_rate), float(output_rate)))
        input_rate = max(rate[0] for rate in rates)
        output_rate = max(rate[1] for rate in rates)
        if not all(math.isfinite(rate) and rate > 0 for rate in (input_rate, output_rate)):
            raise ValueError(f"Invalid catalog prices: {model}")
        prices[model] = {"input": input_rate, "output": output_rate}
    return prices


def read_events(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_event(file, event):
    file.write(json.dumps(event) + "\n")
    file.flush()
    os.fsync(file.fileno())
