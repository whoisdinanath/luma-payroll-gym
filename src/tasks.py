import hashlib
import json
import random
from decimal import Decimal

from luma_gym import RULESET_VERSION, SCHEMA_VERSION
from luma_gym.reference import (INITIAL_VERSION, RETRIEVAL_VALUES, calculate_payroll, reference_name, reference_table,
                                retrieve_rule, validate_record)


GENERATOR_VERSION = "3"


class Task:
    def __init__(self, id, slice, difficulty, prompt, record, ground_truth, verifier,
                 tags, schema_version, ruleset_version, seed, retrieval_field=None):
        self.id = id
        self.slice = slice
        self.difficulty = difficulty
        self.prompt = prompt
        self.record = record
        self.ground_truth = ground_truth
        self.verifier = verifier
        self.tags = tags
        self.schema_version = schema_version
        self.ruleset_version = ruleset_version
        self.seed = seed
        self.retrieval_field = retrieval_field
        self.validate()

    @classmethod
    def from_dict(cls, data):
        validate_task(data)
        return cls(**data)

    def to_dict(self):
        return vars(self).copy()

    def validate(self):
        validate_task(self.to_dict())


def validate_task(task):
    required = {
        "id", "slice", "difficulty", "prompt", "record", "ground_truth",
        "verifier", "tags", "schema_version", "ruleset_version", "seed",
    }
    if not isinstance(task, dict) or not required.issubset(task):
        raise ValueError("Missing task fields")
    if set(task) - required - {"retrieval_field"}:
        raise ValueError("Unknown task field")
    version = task["ruleset_version"]
    if task["schema_version"] != SCHEMA_VERSION or version not in (INITIAL_VERSION, "2", RULESET_VERSION):
        raise ValueError("Unsupported version")
    for field in ("id", "prompt"):
        if not isinstance(task[field], str) or not task[field].strip():
            raise ValueError(f"Invalid {field}")
    if task["slice"] != "monthly_payroll" or task["verifier"] not in ("luma_payroll_v1", "payroll"):
        raise ValueError("Unsupported slice or verifier")
    tier = task["difficulty"]
    if type(tier) is not int or tier not in (1, 2, 3):
        raise ValueError("Invalid difficulty")
    if not isinstance(task["tags"], list):
        raise ValueError("Tags must be a list")
    for tag in task["tags"]:
        if not isinstance(tag, str):
            raise ValueError("Tags must contain strings")
    if task["seed"] is not None and type(task["seed"]) is not int:
        raise ValueError("Invalid seed")
    validate_record(task["record"], version)
    field = task.get("retrieval_field")
    if tier == 1:
        if not isinstance(field, str):
            raise ValueError("Missing retrieval field")
        expected = retrieve_rule(field, task["record"], version)
    else:
        if field is not None:
            raise ValueError("Unexpected retrieval field")
        expected = calculate_payroll(task["record"], version)
        correct_tier = 3 if expected["decision"] == "clarify" else 2
        if tier != correct_tier:
            raise ValueError("Difficulty does not match the expected decision")
    if task["ground_truth"] != expected:
        raise ValueError(f"Incorrect ground truth: {task['id']}")


def validate_taskset(tasks):
    if not tasks:
        raise ValueError("Taskset must not be empty")
    ids = set()
    signatures = set()
    for task in tasks:
        task.validate()
        if task.id in ids:
            raise ValueError(f"Duplicate task id: {task.id}")
        signature = json.dumps([task.difficulty, task.record, task.retrieval_field], sort_keys=True)
        if signature in signatures:
            raise ValueError(f"Duplicate task inputs: {task.id}")
        ids.add(task.id)
        signatures.add(signature)


def taskset_hash(tasks):
    content = json.dumps([task.to_dict() for task in tasks], sort_keys=True).encode()
    return hashlib.sha256(content).hexdigest()


def semantic_signature(task):
    if task.difficulty == 1:
        content = [task.ruleset_version, "retrieve", task.retrieval_field,
                   reference_name(task.record.get("payroll_month"), task.ruleset_version),
                   task.record.get("reference_tables")]
    else:
        record = task.record.copy()
        month = record.get("payroll_month")
        amendment = record.pop("amendment", None)
        if amendment and record.get("gross_pay") is not None and record.get("contract_gross_pay") != record.get("gross_pay"):
            resolves = bool(amendment["approved"] and month
                            and amendment["effective_month"] <= month
                            and Decimal(amendment["gross_pay"]) == Decimal(record["gross_pay"]))
            if resolves:
                record.pop("contract_gross_pay", None)
        if month is not None:
            record["payroll_month"] = reference_name(month, task.ruleset_version) or "uncovered"
        for field in ("gross_pay", "contract_gross_pay", "allowed_deduction"):
            if record.get(field) is not None:
                record[field] = format(Decimal(record[field]), ".2f")
        record.setdefault("allowed_deduction", "0.00")
        if record.get("contract_gross_pay") == record.get("gross_pay"):
            record.pop("contract_gross_pay", None)
        for field in ("gross_pay", "payroll_month"):
            if record.get(field) is None:
                record.pop(field, None)
        content = [task.ruleset_version, record]
    return json.dumps(content, sort_keys=True)


def generate_tasks(n=100, seed=7, tiers=(1, 2, 3), boundary_fraction=0.5,
                   deduction_fraction=0.7, contract_fraction=0.5):
    if type(n) is not int or not 1 <= n <= 10000:
        raise ValueError("Task count must be between 1 and 10000")
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    if not tiers or any(type(tier) is not int or tier not in (1, 2, 3) for tier in tiers):
        raise ValueError("Invalid tiers")
    if len(set(tiers)) != len(tiers):
        raise ValueError("Tiers must not repeat")
    for fraction in (boundary_fraction, deduction_fraction, contract_fraction):
        if not 0 <= fraction <= 1:
            raise ValueError("Generator fractions must be between zero and one")
    if set(tiers) == {1} and n > len(RETRIEVAL_VALUES) * 2:
        raise ValueError("At most six retrieval tasks are available")
    rng = random.Random(seed)
    tasks = []
    signatures = set()
    retrieval_fields = [(field, month) for field in RETRIEVAL_VALUES for month in ("2026-06", "2026-07")]
    rng.shuffle(retrieval_fields)
    attempts = 0
    while len(tasks) < n:
        attempts += 1
        if attempts > n * 200:
            raise ValueError("Could not generate enough distinct tasks")
        index = len(tasks)
        tier = tiers[index % len(tiers)]
        if tier == 1 and not retrieval_fields:
            tier = rng.choice([value for value in tiers if value != 1])
        near_boundary = rng.random() < boundary_fraction
        month = f"2026-{rng.randint(1, 12):02d}"
        table = reference_table(month)
        if near_boundary:
            boundaries = [0, int(table["free"] / (1 - table["rate"]) * 100),
                          int(table["upper"] / (1 - table["rate"]) * 100), int(table["cap"] * 100)]
            gross_cents = rng.choice(boundaries) + rng.choice([-2, -1, 0, 1, 2])
            gross_cents = max(0, gross_cents)
        else:
            gross_cents = rng.randint(0, 1000000)
        gross = Decimal(gross_cents) / 100
        deduction = Decimal(0)
        if rng.random() < deduction_fraction:
            deduction = Decimal(rng.randint(0, min(int(table["deduction"] * 100), int(gross * 90)))) / 100
        record = {
            "gross_pay": format(gross, ".2f"),
            "payroll_month": month,
        }
        if deduction or rng.random() < 0.5:
            record["allowed_deduction"] = format(deduction, ".2f")
        if rng.random() < contract_fraction:
            record["contract_gross_pay"] = record["gross_pay"]
            if tier != 1 and rng.random() < 0.5:
                record["contract_gross_pay"] = format(gross + Decimal("10.00"), ".2f")
                record["amendment"] = {"gross_pay": record["gross_pay"], "effective_month": month, "approved": True}
        field = None
        tags = [GENERATOR_VERSION, "boundary" if near_boundary else "general"]
        if tier == 1:
            field, retrieval_month = retrieval_fields.pop()
            record = {"payroll_month": retrieval_month}
            answer = retrieve_rule(field, record)
            prompt = f"Retrieve {field} from the supplied rulebook. Return the R8 JSON answer."
            tags.append("retrieval")
        else:
            prompt = "Process this monthly payroll record under the supplied rulebook. Return the R6 or R7 JSON answer."
            if tier == 3:
                trap = rng.choice(["missing_gross", "missing_month", "conflict", "uncovered_month", "draft_amendment", "future_amendment"])
                tags.append(trap)
                if trap == "missing_gross":
                    del record["gross_pay"]
                    record.pop("contract_gross_pay", None)
                    record.pop("amendment", None)
                elif trap == "missing_month":
                    del record["payroll_month"]
                elif trap == "uncovered_month":
                    record["payroll_month"] = "2027-01"
                else:
                    record["contract_gross_pay"] = format(gross + Decimal("1.00"), ".2f")
                    record.pop("amendment", None)
                    if trap in ("draft_amendment", "future_amendment"):
                        record["amendment"] = {"gross_pay": record["gross_pay"],
                                               "effective_month": "2027-01" if trap == "future_amendment" else month,
                                               "approved": trap != "draft_amendment"}
            answer = calculate_payroll(record)
        task = {
            "id": f"generated-{seed}-{index:05d}",
            "slice": "monthly_payroll",
            "difficulty": tier,
            "prompt": prompt,
            "record": record,
            "ground_truth": answer,
            "verifier": "payroll",
            "tags": tags,
            "schema_version": SCHEMA_VERSION,
            "ruleset_version": RULESET_VERSION,
            "seed": seed,
            "retrieval_field": field,
        }
        task = Task.from_dict(task)
        signature = semantic_signature(task)
        if signature in signatures:
            continue
        signatures.add(signature)
        tasks.append(task)
    return tasks
