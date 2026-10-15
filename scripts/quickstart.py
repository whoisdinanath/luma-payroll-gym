import argparse
import json
import urllib.request


parser = argparse.ArgumentParser()
parser.add_argument("--url", default="https://luma-payroll-sandbox.whoisdinanath.workers.dev")
args = parser.parse_args()


def call(path, data=None):
    request = urllib.request.Request(args.url + path,
                                     data=json.dumps(data).encode() if data is not None else None,
                                     headers={"User-Agent": "Mozilla/5.0 LumaQuickstart", "Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


batch = call("/tasks?n=3")
result = call("/submit", {"run_id": batch["run_id"], "answers": [
    {"id": task["id"], "answer": {"decision": "invalid"}} for task in batch["tasks"]
]})
print(json.dumps(result, indent=2))
