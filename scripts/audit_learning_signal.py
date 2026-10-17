import hashlib
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from luma_gym.reference import calculate_payroll, reference_table
from luma_gym.scoring import DERIVED, score_answer
from luma_gym.skills import measure_evaluation
from luma_gym.tasks import Task, semantic_signature


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SPLITS = ROOT / "data/training-splits"


def lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def dataset_audit():
    output, signatures = {}, {}
    for name in ("train", "development", "held_out"):
        path = SPLITS / f"{name}.json"
        tasks = [Task.from_dict(row) for row in json.loads(path.read_text())]
        calculations = [task for task in tasks if task.difficulty == 2]
        boundary_tasks = [task for task in calculations if "boundary" in task.tags]
        actual_boundaries = 0
        for task in boundary_tasks:
            table = reference_table(task.record["payroll_month"], tables=task.record.get("reference_tables"))
            if table is None:
                raise ValueError("Calculation task has no applicable reference")
            gross = Decimal(task.ground_truth["gross_pay"])
            taxable = Decimal(task.ground_truth["taxable_pay"])
            if min(abs(gross), abs(gross - table["cap"]), abs(taxable - table["free"]),
                   abs(taxable - table["upper"])) <= Decimal("0.05"):
                actual_boundaries += 1
        output[name] = {
            "tasks": len(tasks),
            "tiers": dict(Counter(task.difficulty for task in tasks)),
            "trap_reasons": dict(Counter(task.ground_truth.get("reason_code") for task in tasks if task.difficulty == 3)),
            "trap_tags": dict(Counter(tag for task in tasks for tag in task.tags
                                      if tag in ("conflict", "draft_amendment", "future_amendment", "missing_gross", "missing_month", "uncovered_month"))),
            "zero_gross_calculations": sum(task.ground_truth["gross_pay"] == "0.00" for task in calculations),
            "zero_tax_calculations": sum(task.ground_truth["income_tax"] == "0.00" for task in calculations),
            "boundary_tagged_calculations": sum("boundary" in task.tags for task in calculations),
            "boundary_tagged_within_5_cents_of_actual_zero_cap_or_tax_threshold": actual_boundaries,
            "reference_replay_disagreements": sum(calculate_payroll(task.record) != task.ground_truth for task in tasks),
            "unique_signatures": len({semantic_signature(task) for task in tasks}),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        signatures[name] = {semantic_signature(task) for task in tasks}
    output["cross_split_overlap"] = {
        f"{a}/{b}": len(signatures[a] & signatures[b])
        for a, b in (("train", "development"), ("train", "held_out"), ("development", "held_out"))
    }
    return output


def training_audit(directory, tasks):
    config = json.loads((directory / "config.json").read_text())
    samples = lines(directory / "samples.jsonl")
    steps = lines(directory / "steps.jsonl")
    groups = defaultdict(list)
    for row in samples:
        groups[(row["step"], row["task_id"])].append(row)
    output = {
        "steps": len(steps), "planned": config["steps"], "rubric": config.get("rubric_version"),
        "complete_training": len(steps) >= config["steps"], "superseded": config.get("superseded"),
        "sampled_completions": len(samples), "unique_tasks": len({row["task_id"] for row in samples}),
        "hours": sum(sum(row["seconds"].values()) for row in steps) / 3600,
        "last_10": {key: statistics.mean(row[key] for row in steps[-10:])
                    for key in ("reward", "kl", "entropy", "false_clarification", "correct_clarification")
                    if steps and key in steps[0]},
        "by_tier": {},
    }
    for tier in (2, 3):
        selected = [row for row in samples if row["tier"] == tier]
        tier_groups = [rows for rows in groups.values() if rows[0]["tier"] == tier]
        unique_tasks = {row["task_id"] for row in selected}
        info = {
            "completions": len(selected), "unique_tasks": len(unique_tasks), "groups": len(tier_groups),
            "logged_reward_counts": dict(Counter(str(row["reward"]) for row in selected)),
            "groups_with_reward_variance": sum(len({row["reward"] for row in rows if row["reward"] is not None}) > 1
                                              for rows in tier_groups),
            "groups_with_positive_reward": sum(any(row["reward"] is not None and row["reward"] > 0 for row in rows)
                                                for rows in tier_groups),
            "decision_counts": dict(Counter(row.get("decision") for row in selected)),
        }
        if tier == 2:
            rescored = [score_answer(tasks[row["task_id"]], row["text"]) for row in selected]
            valid_calc = [score for score in rescored if score["diagnostics"]["correct_decision"]]
            info["exact_under_current_rubric"] = sum(score["breakdown"]["exact"]["score"] == 1 for score in rescored)
            info["exact_nonzero_under_current_rubric"] = sum(
                score["breakdown"]["exact"]["score"] == 1 and tasks[row["task_id"]].ground_truth["gross_pay"] != "0.00"
                for row, score in zip(selected, rescored))
            info["derived_correct_counts_conditional_on_valid_calculate"] = {
                field: {"correct": sum(score["fields"].get(field, {}).get("correct", False) for score in valid_calc),
                        "denominator": len(valid_calc)} for field in DERIVED if field != "value"
            }
            info["counterfactual_field_signal_not_used_in_training"] = sum(
                len({score_answer(tasks[row["task_id"]], row["text"])["breakdown"]["field_accuracy"]["score"]
                     for row in rows if score_answer(tasks[row["task_id"]], row["text"])["diagnostics"]["correct_decision"]}) > 1
                for rows in tier_groups)
        output["by_tier"][str(tier)] = info
    return output


def evaluation_audit(directory, split):
    measured = measure_evaluation(directory, SPLITS / f"{split}.json")
    if not measured:
        return {"generated": len(lines(directory / "generations.jsonl")), "scored": False}
    summary = {**measured[0], "scored": True, "complete_400_answers": measured[0]["answers"] == 400}
    rows = json.loads((directory / "scores.json").read_text())
    tasks = {task["id"]: Task.from_dict(task) for task in json.loads((SPLITS / f"{split}.json").read_text())}
    calculations = [row for row in rows if tasks[row["id"].rsplit(":", 1)[0]].difficulty == 2]
    summary["calculation_status_counts"] = dict(Counter(row["score"]["status"] for row in calculations))
    valid = [row["score"] for row in calculations if row["score"]["diagnostics"]["correct_decision"]]
    nonzero = [row["score"] for row in calculations
               if tasks[row["id"].rsplit(":", 1)[0]].ground_truth["gross_pay"] != "0.00"]
    summary["nonzero_calculation_derived_fields_unconditional"] = {
        field: {"correct": sum(score["fields"].get(field, {}).get("correct", False) for score in nonzero),
                "denominator": len(nonzero)} for field in DERIVED if field != "value"
    }
    generations = lines(directory / "generations.jsonl")
    summary["generation_wall_seconds_sum"] = sum(row.get("seconds", 0) for row in generations)
    summary["mean_input_tokens"] = statistics.mean(row["input_tokens"] for row in generations)
    summary["mean_output_tokens"] = statistics.mean(row["output_tokens"] for row in generations)
    summary["derived_fields_conditional_on_valid_calculate"] = {
        field: {"correct": sum(score["fields"].get(field, {}).get("correct", False) for score in valid),
                "denominator": len(valid)} for field in DERIVED if field != "value"
    }
    return summary


def main():
    tasks = {row["id"]: Task.from_dict(row) for row in json.loads((SPLITS / "train.json").read_text())}
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": dataset_audit(), "training": {}, "evaluations": {},
    }
    for directory in sorted(RESULTS.glob("rl-*")):
        if (directory / "config.json").exists():
            result["training"][directory.name] = training_audit(directory, tasks)
    for prefix, split in (("development", "development"), ("heldout", "held_out")):
        for tag in ("control", "rubric-7-beta-0.04", "rubric-7-beta-0.001"):
            directory = RESULTS / f"minicpm-{prefix}-{tag}"
            if directory.exists():
                result["evaluations"][directory.name] = evaluation_audit(directory, split)
    target = ROOT / "reports/learning-signal-audit.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
