import json
import random
import statistics

from luma_gym.scoring import unwrap_answer


SKILLS = ["exact_calculation", "zero_pay_calculation", "correct_clarification", "false_calculation",
          "false_clarification", "invalid"]


def chosen_decision(text):
    try:
        value = json.loads(unwrap_answer(text))
    except (TypeError, ValueError):
        return None
    return value.get("decision") if isinstance(value, dict) else None


def used_other_row(task, text):
    from luma_gym.reference import calculate_payroll

    tables = task.record["reference_tables"]
    swapped = calculate_payroll(dict(task.record, reference_tables={"first-half": tables["second-half"],
                                                                    "second-half": tables["first-half"]}))
    contribution = swapped["employee_contribution"]
    try:
        answer = json.loads(unwrap_answer(text))
    except (TypeError, ValueError):
        return False
    return (isinstance(answer, dict) and answer.get("decision") == "calculate"
            and contribution != task.ground_truth["employee_contribution"]
            and answer.get("employee_contribution") == contribution)


def skill_flags(truth, score, text):
    calculation = truth["decision"] == "calculate"
    zero_pay = calculation and truth["gross_pay"] == "0.00"
    exact = score["breakdown"]["exact"]["score"] == 1
    decision = chosen_decision(text)
    return {
        "exact_calculation": exact if calculation and not zero_pay else None,
        "zero_pay_calculation": exact if zero_pay else None,
        "correct_clarification": score["status"] == "scored" if truth["decision"] == "clarify" else None,
        "false_calculation": decision == "calculate" if truth["decision"] == "clarify" else None,
        "false_clarification": decision == "clarify" if calculation else None,
        "invalid": score["status"] == "invalid_format",
    }


def bootstrap(values, samples=1000, seed=7):
    generator = random.Random(seed)
    means = sorted(statistics.mean(generator.choice(values) for _ in values) for _ in range(samples))
    return means[int(samples * 0.025)], means[int(samples * 0.975) - 1]


def paired(before, after, samples=1000):
    tasks = sorted(set(before) & set(after))
    if not tasks:
        return None
    differences = [after[task] - before[task] for task in tasks]
    low, high = bootstrap(differences, samples)
    return {"tasks": len(tasks), "difference": statistics.mean(differences), "interval_95": [low, high]}


def measure_evaluation(directory, tasks_path):
    if not (directory / "scores.json").exists():
        return None
    truths = {item["id"]: item["ground_truth"] for item in json.loads(tasks_path.read_text())}
    texts = {}
    for line in (directory / "generations.jsonl").read_text().splitlines():
        row = json.loads(line)
        texts[row["id"]] = row["final_answer"]
    per_task = {}
    for row in json.loads((directory / "scores.json").read_text()):
        task = row["id"].rsplit(":", 1)[0]
        flags = skill_flags(truths[task], row["score"], texts[row["id"]])
        flags["reward"] = row["score"]["score"]
        per_task.setdefault(task, []).append(flags)
    summary = {"answers": sum(len(rows) for rows in per_task.values())}
    means = {}
    for metric in SKILLS + ["reward"]:
        values = [flags[metric] for rows in per_task.values() for flags in rows if flags[metric] is not None]
        summary[metric] = statistics.mean(values) if values else None
        summary[metric + "_answers"] = len(values)
        means[metric] = {task: statistics.mean(flags[metric] for flags in rows)
                         for task, rows in per_task.items() if rows[0][metric] is not None}
    return summary, means
