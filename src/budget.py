import fcntl
import json
import math
from pathlib import Path


LIMIT = 25.0


class Budget:
    def __init__(self, path="results/spending.json", limit=LIMIT):
        self.path = Path(path)
        self.limit = min(limit, LIMIT)
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError(f"Budget must be positive and at most ${LIMIT:g}")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self):
        if self.path.exists():
            return
        total = 0.0
        sources = {}
        for directory in self.path.parent.iterdir():
            if not directory.is_dir():
                continue
            transcript = directory / "transcript.jsonl"
            events = []
            if transcript.exists():
                events = [json.loads(line) for line in transcript.read_text().splitlines() if line.strip()]
            responses = [row for row in events if row.get("event") == "response"]
            if responses:
                costs = [row.get("cost_usd") for row in responses]
            else:
                progress = directory / "progress.jsonl"
                if not progress.exists():
                    continue
                rows = [json.loads(line) for line in progress.read_text().splitlines() if line.strip()]
                costs = [row.get("charged_usd", row.get("cost_usd")) for row in rows
                         if row.get("event") == "finished"]
            if any(cost is None for cost in costs):
                raise ValueError("Reconcile historical unknown costs before initializing budget")
            if costs:
                sources[directory.name] = sum(costs)
                total += sum(costs)
        with self.path.open("x") as file:
            json.dump({"spent_usd": total, "reserved_usd": 0.0, "historical_runs": sources}, file, indent=2)

    def change(self, amount=0, reserved=0):
        self.initialize()
        with self.path.open("r+") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            data = json.load(file)
            if data.get("accounting_error"):
                raise ValueError("Budget accounting requires review before more paid work")
            spent = data["spent_usd"] + amount
            pending = data["reserved_usd"] + reserved
            if spent + pending > self.limit + 1e-9:
                raise ValueError(f"OpenRouter budget stop: request would exceed the shared ${self.limit:g} limit")
            data["spent_usd"] = spent
            data["reserved_usd"] = max(0, pending)
            file.seek(0)
            json.dump(data, file, indent=2)
            file.truncate()
            return data

    def reserve(self, amount):
        return self.change(reserved=amount)

    def settle(self, reserved, actual):
        if actual > reserved:
            with self.path.open("r+") as file:
                fcntl.flock(file, fcntl.LOCK_EX)
                data = json.load(file)
                data["spent_usd"] += actual
                data["reserved_usd"] = max(0, data["reserved_usd"] - reserved)
                data["accounting_error"] = "Actual charge exceeded reservation; paid work must stop"
                file.seek(0)
                json.dump(data, file, indent=2)
                file.truncate()
            raise ValueError("Provider cost exceeded reservation; stop and reconcile")
        return self.change(amount=actual, reserved=-reserved)
