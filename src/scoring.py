import json
import re
from decimal import Decimal

from luma_gym.reference import money


RUBRIC_VERSION = "8"
REJECTED = -0.5
PARTIAL = 0.2
MONEY_FIELDS = (
    "gross_pay", "employee_contribution", "allowed_deduction",
    "taxable_pay", "income_tax", "net_pay",
)
DERIVED = ("employee_contribution", "taxable_pay", "income_tax", "net_pay", "value")
CLAUSES = {
    "currency": ["R1", "R6"],
    "gross_pay": ["R1", "R6"],
    "employee_contribution": ["R2", "R3"],
    "allowed_deduction": ["R4"],
    "taxable_pay": ["R3", "R4"],
    "income_tax": ["R2", "R5"],
    "net_pay": ["R6"],
    "reason_code": ["R7"],
}


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Invalid JSON constant")


def parse_answer(text):
    if not isinstance(text, str) or len(text) > 16000:
        raise ValueError("Invalid answer text")
    answer = json.loads(text, object_pairs_hook=unique_keys, parse_constant=reject_constant)
    if not isinstance(answer, dict):
        raise ValueError("Expected an object")
    decision = answer.get("decision")
    if decision == "calculate":
        fields = {"decision", "currency", "clauses", *MONEY_FIELDS}
    elif decision == "retrieve":
        fields = {"decision", "field", "value", "clauses"}
    elif decision == "clarify":
        fields = {"decision", "reason_code", "clauses"}
    else:
        raise ValueError("Invalid decision")
    if not fields.issubset(answer) or set(answer) - fields - {"explanation"}:
        raise ValueError("Invalid answer fields")
    for field in fields - {"clauses"}:
        if not isinstance(answer[field], str):
            raise ValueError("Answer fields must be strings")
    if "explanation" in answer and not isinstance(answer["explanation"], str):
        raise ValueError("Explanation must be a string")
    clauses = answer["clauses"]
    if not isinstance(clauses, list):
        raise ValueError("Invalid clauses")
    for clause in clauses:
        if not isinstance(clause, str) or not re.fullmatch(r"R[1-9]", clause):
            raise ValueError("Invalid clause")
    if len(clauses) != len(set(clauses)):
        raise ValueError("Repeated clause")
    if decision == "calculate":
        for field in MONEY_FIELDS:
            if not re.fullmatch(r"[0-9]+\.[0-9]{2}", answer[field]):
                raise ValueError("Invalid money format")
    if decision == "retrieve":
        money(answer["value"])
    return answer


def score_answer(task, text):
    task.validate()
    expected = task.ground_truth
    result = {
        "task_id": task.id,
        "ruleset_version": task.ruleset_version,
        "rubric_version": RUBRIC_VERSION,
        "score": REJECTED,
        "passed": False,
        "status": "invalid_format",
        "breakdown": {
            "decision": {"score": 0.0, "clauses": expected["clauses"]},
            "exact": {"score": 0.0, "clauses": expected["clauses"]},
            "field_accuracy": {"score": 0.0, "clauses": expected["clauses"]},
            "citations": {"score": 0.0, "clauses": expected["clauses"]},
        },
        "diagnostics": {"valid_format": False, "correct_decision": False},
        "fields": {},
    }
    try:
        answer = parse_answer(text)
    except (ValueError, RecursionError):
        return result
    result["diagnostics"]["valid_format"] = True
    if answer["decision"] != expected["decision"]:
        result["status"] = "wrong_decision"
        return result
    result["diagnostics"]["correct_decision"] = True
    result["breakdown"]["decision"]["score"] = 1.0
    fields = [field for field in expected if field not in ("decision", "clauses")]
    for field in fields:
        if field in MONEY_FIELDS or field == "value":
            correct = Decimal(answer[field]) == Decimal(expected[field])
        else:
            correct = answer[field] == expected[field]
        result["fields"][field] = {
            "correct": correct,
            "clauses": CLAUSES.get(field, expected["clauses"]),
        }
    matches = sum(item["correct"] for item in result["fields"].values())
    exact = matches == len(fields)
    citations = set(answer["clauses"]) == set(expected["clauses"])
    if task.difficulty == 3 and not exact:
        result["status"] = "wrong_clarification"
        return result
    derived = [field for field in fields if field in DERIVED]
    accuracy = 1.0 if exact else sum(result["fields"][field]["correct"] for field in derived) / max(len(derived), 1)
    result["breakdown"]["exact"]["score"] = float(exact)
    result["breakdown"]["field_accuracy"]["score"] = accuracy
    result["breakdown"]["citations"]["score"] = float(citations)
    if exact:
        reward = 0.70 + 0.25 + 0.05 * citations
    else:
        reward = REJECTED + PARTIAL * accuracy
    result["score"] = round(reward, 6)
    result["passed"] = exact and citations
    result["status"] = "scored"
    return result
