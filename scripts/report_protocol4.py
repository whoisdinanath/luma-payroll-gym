import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from luma_gym.hosted import read_events
from luma_gym.reference import reference_name
from luma_gym.skills import bootstrap, chosen_decision, paired, used_other_row
from luma_gym.tasks import Task


ROOT = Path(__file__).resolve().parents[1]
SPLITS = {"dev": "development", "test": "test"}
SWEEP = ["rl-0.001", "rl-0.04", "rl-10", "rl-100"]
CONDITIONS = (["raw-single", "raw-calculator", "sft-calculator"] + [run + "-calculator" for run in SWEEP]
              + ["rl-0.04-balanced-calculator", "rl-0.04-signal-calculator"])
METRICS = ["exact_calculation", "correct_clarification", "false_calculation", "false_clarification", "invalid",
           "tool_use_on_payslips", "tool_use_on_traps", "other_row_first_half", "other_row_second_half", "reward"]


def flags(task, score, row):
    truth = task.ground_truth
    calculation = truth["decision"] == "calculate"
    decision = chosen_decision(row["final_answer"])
    half = reference_name(task.record["payroll_month"]) if calculation else None
    other = used_other_row(task, row["final_answer"]) if calculation else None
    return {
        "exact_calculation": score["breakdown"]["exact"]["score"] == 1 if calculation else None,
        "correct_clarification": score["status"] == "scored" if not calculation else None,
        "false_calculation": decision == "calculate" if not calculation else None,
        "false_clarification": decision == "clarify" if calculation else None,
        "invalid": score["status"] == "invalid_format",
        "tool_use_on_payslips": row.get("calls", 0) > 0 if calculation else None,
        "tool_use_on_traps": row.get("calls", 0) > 0 if not calculation else None,
        "other_row_first_half": other if half == "first-half" else None,
        "other_row_second_half": other if half == "second-half" else None,
        "reward": score["score"],
    }


def measure(directory, tasks):
    scores = {row["id"]: row["score"] for row in json.loads((directory / "scores.json").read_text())}
    per_task, strata, reasons, contracts, errors = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), 0
    halves = defaultdict(list)
    for row in read_events(directory / "generations.jsonl"):
        task = tasks[row["task_id"]]
        found = flags(task, scores[row["id"]], row)
        per_task[task.id].append(found)
        errors += row.get("calculator_errors", 0)
        if task.difficulty == 2:
            stratum = next(tag[8:] for tag in task.tags if tag.startswith("stratum:"))
            contract = next(tag[9:] for tag in task.tags if tag.startswith("contract:"))
            strata[stratum].append(found["exact_calculation"])
            halves[reference_name(task.record["payroll_month"])].append(found["exact_calculation"])
            contracts[contract].append(found["false_clarification"])
        else:
            reasons[task.ground_truth["reason_code"]].append(found["correct_clarification"])
    summary, means = {"answers": sum(len(rows) for rows in per_task.values()), "calculator_errors": errors}, {}
    for metric in METRICS:
        values = [row[metric] for rows in per_task.values() for row in rows if row[metric] is not None]
        means[metric] = {task: statistics.mean(row[metric] for row in rows) for task, rows in per_task.items()
                         if rows[0][metric] is not None}
        summary[metric] = statistics.mean(values) if values else None
        if values and metric != "reward":
            low, high = bootstrap(list(means[metric].values()))
            summary[metric + "_task_interval"] = [low, high]
    summary["exact_by_stratum"] = {name: statistics.mean(values) for name, values in sorted(strata.items())}
    summary["exact_by_half"] = {name: statistics.mean(values) for name, values in sorted(halves.items())}
    summary["clarification_by_reason"] = {name: statistics.mean(values) for name, values in sorted(reasons.items())}
    summary["refusal_by_contract"] = {name: statistics.mean(values) for name, values in sorted(contracts.items())}
    return summary, means


def training(run):
    rows = read_events(ROOT / "results/v4" / run / "steps.jsonl")
    if not rows:
        return None
    epochs = [epoch for row in rows for epoch in row["epochs"]]
    summary = {"steps": len(rows), "hours": sum(sum(row["seconds"].values()) for row in rows) / 3600}
    for key in ("reward", "exact_calculation", "false_clarification", "correct_clarification", "kl", "entropy",
                "completion_tokens"):
        for name, part in (("first_10", rows[:10]), ("last_10", rows[-10:])):
            values = [row[key] for row in part if row.get(key) is not None]
            summary[f"{key}_{name}"] = statistics.mean(values) if values else None
    summary.update({
        "kl_max": max(row["kl"] for row in rows),
        "informative_groups": {tier: sum(row["informative"][tier] for row in rows) for tier in ("2", "3")},
        "groups_per_tier": 2 * len(rows),
        "steps_with_update": sum(any(epoch["applied"] for epoch in row["epochs"]) for row in rows),
        "clip_fraction": statistics.mean(epoch["clip_fraction"] for epoch in epochs) if epochs else None,
        "drift_max": max((epoch["drift"] for epoch in epochs), default=None),
        "epochs_stopped_by_drift": sum(not epoch["applied"] for epoch in epochs),
        "judge_calls": sum(row["judge_calls"] for row in rows),
        "judge_cost_usd": sum(row["judge_cost_usd"] for row in rows),
        "unscored": sum(row["unscored"] for row in rows),
        "calculator_errors": sum(row["calculator_errors"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
    })
    return summary


def tool_mode(directory, tasks):
    scores = {row["id"]: row["score"] for row in json.loads((directory / "scores.json").read_text())}
    rows = read_events(directory / "generations.jsonl")
    tiers = defaultdict(list)
    for row in rows:
        tiers[tasks[row["task_id"]].difficulty].append(scores[row["id"]])
    used = defaultdict(int)
    for row in rows:
        for name in set(row.get("tool_names", [])):
            used[name] += 1
    return {"answers": len(rows), "reward": statistics.mean(score["score"] for score in scores.values()),
            "reward_by_tier": {str(tier): statistics.mean(score["score"] for score in values) for tier, values in sorted(tiers.items())},
            "exact_by_tier": {str(tier): statistics.mean(score["breakdown"]["exact"]["score"] == 1 for score in values)
                              for tier, values in sorted(tiers.items())},
            "malformed": statistics.mean(score["status"] == "invalid_format" for score in scores.values()),
            "tool_calls_per_answer": statistics.mean(row.get("calls", 0) for row in rows),
            "answers_using_tool": {name: count / len(rows) for name, count in sorted(used.items())}}


def spread(values):
    return {"runs": values, "mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else None}


def main():
    seeds = sorted(path.name for path in (ROOT / "results/v4").glob("rl-*-seed*") if path.is_dir())
    conditions = CONDITIONS + [run + "-calculator" for run in seeds]
    report = {"protocol": "4", "conditions": conditions, "splits": {}}
    for short, name in SPLITS.items():
        tasks = {item["id"]: Task.from_dict(item) for item in json.loads((ROOT / f"data/v4/{name}.json").read_text())}
        found = {}
        for condition in conditions:
            directory = ROOT / "results/v4" / f"{short}-{condition}"
            scores, generations = directory / "scores.json", directory / "generations.jsonl"
            if scores.exists() and len(json.loads(scores.read_text())) == len(read_events(generations)) > 0:
                found[condition] = measure(directory, tasks)
        split = {"summaries": {condition: value[0] for condition, value in found.items()}, "changes": {}}
        pairs = [("raw-single", "raw-calculator"), ("raw-calculator", "sft-calculator"),
                 ("raw-single", "rl-0.04-calculator"), ("raw-single", "rl-0.001-calculator"),
                 ("sft-calculator", "rl-0.04-balanced-calculator"), ("rl-0.04-calculator", "rl-0.04-balanced-calculator"),
                 ("sft-calculator", "rl-0.04-signal-calculator"), ("rl-0.04-calculator", "rl-0.04-signal-calculator")]
        pairs += [("sft-calculator", run + "-calculator") for run in SWEEP + seeds]
        for before, after in pairs:
            if before in found and after in found:
                split["changes"][f"{before} -> {after}"] = {
                    metric: paired(found[before][1][metric], found[after][1][metric]) for metric in METRICS
                    if found[before][1][metric] and found[after][1][metric]}
        report["splits"][short] = split
    report["training"] = {run: training(run) for run in SWEEP + ["rl-0.04-balanced", "rl-0.04-signal"] + seeds
                          if (ROOT / "results/v4" / run / "steps.jsonl").exists()}
    report["seed_runs"] = {}
    for base in sorted({run.rsplit("-seed", 1)[0] for run in seeds}):
        runs = [base] + [run for run in seeds if run.startswith(base + "-seed")]
        for short, split in report["splits"].items():
            found = [split["summaries"][run + "-calculator"] for run in runs if run + "-calculator" in split["summaries"]]
            if len(found) > 1:
                report["seed_runs"].setdefault(base, {})[short] = {
                    "runs": runs[:len(found)], "exact_calculation": spread([value["exact_calculation"] for value in found]),
                    "reward": spread([value["reward"] for value in found])}
    development = {item["id"]: Task.from_dict(item) for item in json.loads((ROOT / "data/v4/development.json").read_text())}
    report["checkpoints"] = {}
    for run in ("rl-0.04", "rl-10", "rl-0.001"):
        points = []
        for step in (0, 20, 40, 60):
            name = "dev-sft-calculator" if step == 0 else f"dev-{run}-calculator" if step == 60 else f"dev-{run}-step{step}-calculator"
            directory = ROOT / "results/v4" / name
            scores, generations = directory / "scores.json", directory / "generations.jsonl"
            if scores.exists() and len(json.loads(scores.read_text())) == len(read_events(generations)) > 0:
                summary = measure(directory, development)[0]
                points.append({"step": step, **{key: summary[key] for key in ("exact_calculation", "false_clarification", "reward")}})
        if len(points) > 2:
            report["checkpoints"][run] = points
    eval_tasks = {item["id"]: Task.from_dict(item) for item in json.loads((ROOT / "data/v4/eval.json").read_text())}
    report["tool_mode"] = {}
    for directory in sorted((ROOT / "results/v4").glob("eval-*-tools")):
        scores, generations = directory / "scores.json", directory / "generations.jsonl"
        if scores.exists() and len(json.loads(scores.read_text())) == len(read_events(generations)) > 0:
            report["tool_mode"][directory.name] = tool_mode(directory, eval_tasks)
    (ROOT / "reports/protocol4-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for short, split in report["splits"].items():
        print(f"== {short}")
        for condition, summary in split["summaries"].items():
            print(f"  {condition:22s}", " ".join(f"{metric}={summary[metric]:.3f}" for metric in METRICS
                                                  if summary.get(metric) is not None))
    if "--verbose" in sys.argv:
        print(json.dumps(report, indent=2)[:6000])


if __name__ == "__main__":
    main()
