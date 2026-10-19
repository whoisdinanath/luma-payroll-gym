import json

from luma_gym.scoring import MONEY_FIELDS, RUBRIC_VERSION, score_answer
from luma_gym.tasks import taskset_hash, validate_taskset


def copy_example():
    from luma_gym.local import DECISION_EXAMPLES
    from luma_gym.reference import calculate_payroll

    return calculate_payroll(DECISION_EXAMPLES[0])


def baseline_answers(task, example=None):
    format_only = {"decision": "calculate", "currency": "LCR", "clauses": []}
    for field in MONEY_FIELDS:
        format_only[field] = "0.00"
    truth = task.ground_truth
    if truth["decision"] == "clarify":
        decision_only = truth
    elif truth["decision"] == "retrieve":
        decision_only = {**truth, "value": "0.00"}
    else:
        decision_only = {**format_only, "clauses": truth["clauses"]}
    refuse = {"decision": "clarify", "reason_code": "MISSING_GROSS_PAY", "clauses": ["R1", "R7"]}
    return {
        "right_decision_no_arithmetic": decision_only,
        "refuse_payslips_perfect_traps": truth if truth["decision"] == "clarify" else refuse,
        "reference_sanity_check": task.ground_truth,
        "format_only": format_only,
        "always_clarify": {
            "decision": "clarify", "reason_code": "MISSING_GROSS_PAY", "clauses": ["R1", "R7"],
        },
        "always_clarify_conflict": {
            "decision": "clarify", "reason_code": "CONFLICTING_GROSS_PAY", "clauses": ["R7", "R9"],
        },
        "always_clarify_month": {
            "decision": "clarify", "reason_code": "MISSING_PAYROLL_MONTH", "clauses": ["R1", "R7"],
        },
        "always_clarify_reference": {
            "decision": "clarify", "reason_code": "NO_APPLICABLE_REFERENCE", "clauses": ["R1", "R7"],
        },
        "copy_example": example or copy_example(),
        "copied_record": {
            **format_only, "gross_pay": task.record.get("gross_pay", "0.00"),
            "net_pay": task.record.get("gross_pay", "0.00"),
        },
    }


def audit_baselines(tasks):
    validate_taskset(tasks)
    results = {}
    for task in tasks:
        for name, answer in baseline_answers(task).items():
            result = score_answer(task, json.dumps(answer))
            results.setdefault(name, []).append({"tier": task.difficulty, **result})
    summary = {}
    for name, rows in results.items():
        tiers = {}
        for tier in (1, 2, 3):
            scores = [row["score"] for row in rows if row["tier"] == tier]
            if scores:
                tiers[str(tier)] = round(sum(scores) / len(scores), 6)
        summary[name] = {
            "mean_reward": round(sum(row["score"] for row in rows) / len(rows), 6),
            "pass_rate": sum(row["passed"] for row in rows) / len(rows),
            "tiers": tiers,
        }
    return {
        "rubric_version": RUBRIC_VERSION,
        "taskset_sha256": taskset_hash(tasks),
        "tasks": len(tasks),
        "summary": summary,
        "results": results,
    }
