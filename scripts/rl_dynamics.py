import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from luma_gym.hosted import read_events
from luma_gym.reference import reference_name
from luma_gym.skills import chosen_decision, used_other_row
from luma_gym.tasks import Task


ROOT = Path(__file__).resolve().parents[1]
RUNS = ["rl-0.04", "rl-0.001", "rl-0.04-balanced", "rl-10", "rl-100", "rl-0.04-signal"]


def kind(task, answer):
    if answer["passed"]:
        return "exact"
    if chosen_decision(answer["text"]) == "clarify":
        return "refused" if task.difficulty == 2 else "wrong reason"
    if task.difficulty == 2 and used_other_row(task, answer["text"]):
        return "wrong row"
    return "other wrong"


def dynamics(run, tasks):
    steps = read_events(ROOT / "results/v4" / run / "steps.jsonl")
    groups = defaultdict(list)
    for answer in read_events(ROOT / "results/v4" / run / "samples.jsonl"):
        groups[(answer["step"], answer["task_id"])].append(answer)
    signal = {tier: Counter() for tier in ("2", "3")}
    tokens = {"all": 0, "informative": 0}
    lengths = defaultdict(list)
    row_push = defaultdict(Counter)
    for (step, task_id), answers in groups.items():
        task = tasks[task_id]
        tier = str(task.difficulty)
        kinds = Counter(kind(task, answer) for answer in answers)
        informative = answers[0]["informative"]
        if informative:
            signal[tier]["informative"] += 1
        elif set(kinds) == {"exact"}:
            signal[tier]["tied, every answer right"] += 1
        else:
            signal[tier]["tied, every answer wrong: " + "/".join(sorted(kinds))] += 1
        for answer in answers:
            tokens["all"] += answer["tokens"]
            tokens["informative"] += answer["tokens"] if informative else 0
            lengths["payslip answer" if chosen_decision(answer["text"]) == "calculate" else "clarification"].append(answer["tokens"])
        if task.difficulty == 2 and informative:
            block = f"steps {(step - 1) // 20 * 20 + 1}-{(step - 1) // 20 * 20 + 20}"
            half = reference_name(task.record["payroll_month"])
            if kinds["exact"] and kinds["wrong row"]:
                row_push[block]["toward the first-half row" if half == "first-half" else "toward the second-half row"] += 1
            if kinds["exact"] and kinds["refused"]:
                row_push[block]["toward attempting (refusal contrast)"] += 1
    epochs = [epoch for step in steps for epoch in step["epochs"]]
    applied = [epoch for epoch in epochs if epoch["applied"]]
    return {
        "steps": len(steps),
        "groups": {tier: dict(counts) for tier, counts in signal.items()},
        "answers": sum(len(answers) for answers in groups.values()),
        "share_of_generated_tokens_in_informative_groups": tokens["informative"] / tokens["all"],
        "steps_without_update": sum(not any(epoch["applied"] for epoch in step["epochs"]) for step in steps),
        "mean_tokens": {name: statistics.mean(values) for name, values in lengths.items()},
        "update_size": {
            "learning_rate": steps[0]["lr"], "updates": len(applied),
            "drift_per_update_mean": statistics.mean(epoch["drift"] for epoch in applied),
            "drift_per_update_max": max(epoch["drift"] for epoch in applied),
            "drift_stop": 0.02,
            "clip_fraction_mean": statistics.mean(epoch["clip_fraction"] for epoch in applied),
            "clip_fraction_max": max(epoch["clip_fraction"] for epoch in applied),
            "ratio_range": [min(epoch["ratio_min"] for epoch in applied), max(epoch["ratio_max"] for epoch in applied)],
            "grad_norm_mean": statistics.mean(epoch["grad_norm"] for epoch in applied),
            "share_of_updates_with_grad_norm_clipped": statistics.mean(epoch["grad_norm"] > 1.0 for epoch in applied),
        },
        "kl_to_reference": {"mean": statistics.mean(step["kl"] for step in steps), "max": max(step["kl"] for step in steps),
                            "last_10": statistics.mean(step["kl"] for step in steps[-10:])},
        "entropy_per_token_mean": statistics.mean(step["entropy"] for step in steps),
        "payslip_contrasts_by_block": {block: dict(counts) for block, counts in sorted(row_push.items())},
    }


def main():
    tasks = {item["id"]: Task.from_dict(item) for item in json.loads((ROOT / "data/v4/train.json").read_text())}
    report = {run: dynamics(run, tasks) for run in RUNS if (ROOT / "results/v4" / run / "samples.jsonl").exists()}
    (ROOT / "reports/rl-dynamics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1)[:12000] if "--print" in sys.argv else f"wrote reports/rl-dynamics.json for {list(report)}")


if __name__ == "__main__":
    main()
