import json
import sys
from collections import Counter
from pathlib import Path

from luma_gym.hosted import read_events
from luma_gym.reference import calculate_payroll, reference_name
from luma_gym.rollout import program_for
from luma_gym.scoring import score_answer
from luma_gym.skills import chosen_decision, used_other_row
from luma_gym.tasks import Task


ROOT = Path(__file__).resolve().parents[1]
COMPUTED = ["employee_contribution", "taxable_pay", "income_tax", "net_pay"]


def lines(program):
    return {name.strip(): value.replace(" ", "") for name, _, value in
            (line.partition("=") for line in program.splitlines() if "=" in line)}


def classify(task, row):
    if score_answer(task, row["final_answer"])["breakdown"]["exact"]["score"] == 1:
        return "exact"
    decision = chosen_decision(row["final_answer"])
    if decision == "clarify":
        return "refused"
    calls = [turn for turn in row.get("turns", []) if turn.get("call") == "calculate"]
    if decision != "calculate":
        return "malformed or other decision"
    if not calls:
        return "calculated without the tool"
    program = calls[-1]["arguments"].get("program", "")
    wanted, written = lines(program_for(task.record)), lines(program)
    if all(written.get(name) == value for name, value in wanted.items()):
        return "right program, results copied wrong"
    if used_other_row(task, row["final_answer"]):
        return "used the other half-year row"
    wrong = sorted(name for name, value in wanted.items() if written.get(name) != value)
    return "wrong program line: " + ", ".join(wrong)


def other_row(record):
    tables = record["reference_tables"]
    return calculate_payroll(dict(record, reference_tables={"first-half": tables["second-half"], "second-half": tables["first-half"]}))


def training_drift(run, tasks):
    blocks = {}
    for row in read_events(ROOT / "results/v4" / run / "samples.jsonl"):
        if row["tier"] != 2 or row["stage"] != "first":
            continue
        task = tasks[row["task_id"]]
        start = (row["step"] - 1) // 20 * 20 + 1
        counts = blocks.setdefault(f"steps {start}-{start + 19}, {reference_name(task.record['payroll_month'])}", Counter())
        counts["answers"] += 1
        counts["prompts"] += row["index"] == 0
        if row["passed"]:
            counts["exact"] += 1
        elif used_other_row(task, row["text"]):
            counts["other half-year row"] += 1
        elif chosen_decision(row["text"]) == "clarify":
            counts["refused"] += 1
    return {key: dict(value) for key, value in sorted(blocks.items())}


def checks(rows, tasks, demos):
    from luma_gym.rollout import EXPLANATIONS

    found = Counter()
    demo_values = {demo.ground_truth[field] for demo in demos if demo.difficulty == 2 for field in COMPUTED}
    for row in rows:
        task = tasks[row["task_id"]]
        try:
            answer = json.loads(row["final_answer"])
        except (TypeError, ValueError):
            answer = None
        answer = answer if isinstance(answer, dict) else {}
        found["truncated"] += bool(row.get("truncated"))
        if task.difficulty == 3:
            found["trap answers"] += 1
            found["calculator called on a trap"] += row.get("calls", 0) > 0
            explanation = answer.get("explanation")
            if explanation in EXPLANATIONS.values():
                found["template explanation"] += 1
                found["template from another reason"] += explanation != EXPLANATIONS.get(answer.get("reason_code"))
        else:
            found["payslip answers"] += 1
            found["wrong value copied from a demonstration"] += not used_other_row(task, row["final_answer"]) and any(
                answer.get(field) in demo_values and answer.get(field) != task.ground_truth[field] for field in COMPUTED)
    return dict(found)


def row_choice(conditions):
    found = {}
    for condition in conditions:
        halves, months = {"first-half": [0, 0], "second-half": [0, 0]}, {}
        for split, name in (("dev", "development"), ("test", "test")):
            path = ROOT / "results/v4" / f"{split}-{condition}" / "generations.jsonl"
            if not path.exists():
                break
            tasks = {item["id"]: Task.from_dict(item) for item in json.loads((ROOT / f"data/v4/{name}.json").read_text())}
            for row in read_events(path):
                task = tasks[row["task_id"]]
                if task.difficulty != 2 or chosen_decision(row["final_answer"]) != "calculate":
                    continue
                wrong = used_other_row(task, row["final_answer"])
                for cell in (halves[reference_name(task.record["payroll_month"])],
                             months.setdefault(task.record["payroll_month"][5:], [0, 0])):
                    cell[0] += wrong
                    cell[1] += 1
        else:
            first = halves["first-half"][0] / halves["first-half"][1]
            second = halves["second-half"][0] / halves["second-half"][1]
            attempts = halves["first-half"][1] + halves["second-half"][1]
            found[condition] = {
                "wrong_row_share": (halves["first-half"][0] + halves["second-half"][0]) / attempts, "attempts": attempts,
                "wrong_row_january_june": first, "wrong_row_july_december": second,
                "month_use": 1 - first - second, "lean_toward_july_december_row": first - second,
                "wrong_row_by_month": {month: wrong / total for month, (wrong, total) in sorted(months.items())}}
    return found


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    name = {"dev": "development", "test": "test"}[split]
    tasks = {item["id"]: Task.from_dict(item) for item in json.loads((ROOT / f"data/v4/{name}.json").read_text())}
    report, by_half, audit = {}, {}, {}
    demos = [Task.from_dict(item) for item in json.loads((ROOT / "data/v4/demos.json").read_text())]
    for condition in ("raw-calculator", "sft-calculator", "rl-0.001-calculator", "rl-0.04-calculator",
                      "rl-10-calculator", "rl-100-calculator", "rl-0.04-balanced-calculator", "rl-0.04-signal-calculator"):
        path = ROOT / "results/v4" / f"{split}-{condition}" / "generations.jsonl"
        if not path.exists():
            continue
        rows = [row for row in read_events(path) if row["tier"] == 2]
        kinds = Counter(classify(tasks[row["task_id"]], row) for row in rows)
        report[condition] = dict(kinds.most_common())
        halves = {}
        for row in rows:
            half = halves.setdefault(reference_name(tasks[row["task_id"]].record["payroll_month"]), [0, 0])
            half[0] += classify(tasks[row["task_id"]], row) == "used the other half-year row"
            half[1] += 1
        by_half[condition] = dict(sorted(halves.items()))
        audit[condition] = checks(read_events(path), tasks, demos)
        print(condition, sum(kinds.values()), dict(kinds.most_common()), "other row by half:", by_half[condition])
    target = ROOT / "reports/protocol4-errors.json"
    saved = json.loads(target.read_text()) if target.exists() else {}
    saved[split] = report
    saved.setdefault("other_row_by_half", {})[split] = by_half
    saved.setdefault("checks", {})[split] = audit
    train = {item["id"]: Task.from_dict(item) for item in json.loads((ROOT / "data/v4/train.json").read_text())}
    same = {}
    for task in train.values():
        if task.difficulty == 2:
            counts = same.setdefault(reference_name(task.record["payroll_month"]), [0, 0])
            swapped = other_row(task.record)
            counts[0] += all(swapped[field] == task.ground_truth[field] for field in COMPUTED)
            counts[1] += 1
    saved["train_payslips_other_row_same_answer"] = dict(sorted(same.items()))
    saved["row_choice"] = row_choice(["sft-calculator", "rl-0.001-calculator", "rl-0.04-calculator", "rl-0.04-balanced-calculator",
                                      "rl-10-calculator", "rl-100-calculator", "rl-0.04-signal-calculator"])
    saved["training"] = {run: training_drift(run, train) for run in ("rl-0.001", "rl-0.04", "rl-10", "rl-100", "rl-0.04-balanced", "rl-0.04-signal")
                         if (ROOT / "results/v4" / run / "samples.jsonl").exists()}
    target.write_text(json.dumps(saved, indent=2) + "\n")


if __name__ == "__main__":
    main()
