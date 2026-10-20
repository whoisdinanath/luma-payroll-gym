import argparse
import hashlib
import json
import statistics
from pathlib import Path

from luma_gym.audit import audit_tasks
from luma_gym.cases import generate_cases
from luma_gym.hosted import read_events
from luma_gym.scoring import RUBRIC_VERSION, score_answer
from luma_gym.skills import bootstrap, paired
from luma_gym.tasks import Task, semantic_signature


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "data/v4/confirmation.json"
REPORT = ROOT / "reports/fresh-confirmation.json"
CONDITIONS = ("sft", "rl-0.04", "rl-0.04-signal")


def prepare():
    tasks = generate_cases(51021, calculations=48, traps=48, prefix="confirmation")
    audit = audit_tasks(tasks)
    if audit["semantic_duplicate_instances"] or audit["prompt_leak_candidates"]:
        raise ValueError("Fresh tasks failed the duplicate or leakage audit")
    signatures = {semantic_signature(task) for task in tasks}
    checked = []
    for path in sorted((ROOT / "data").rglob("*.json")):
        if path == TASKS:
            continue
        rows = json.loads(path.read_text())
        if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict) or "ground_truth" not in rows[0]:
            continue
        previous = [Task.from_dict(row) for row in rows]
        if any(task.seed == 51021 or semantic_signature(task) in signatures for task in previous):
            raise ValueError(f"Fresh seed or semantic overlap with {path}")
        checked.append(str(path.relative_to(ROOT)))
    text = json.dumps([task.to_dict() for task in tasks], indent=2) + "\n"
    if TASKS.exists() and TASKS.read_text() != text:
        raise ValueError("Refusing to replace a frozen confirmation set")
    TASKS.write_text(text)
    report = {
        "status": "prepared", "seed": 51021, "tasks": 96, "payslips": 48, "traps": 48,
        "rollouts": 4, "temperature": 0.7, "sampling_seed": 17,
        "task_file": str(TASKS.relative_to(ROOT)),
        "task_file_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "demos_sha256": hashlib.sha256((ROOT / "data/v4/demos.json").read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256((ROOT / "rules/payroll.md").read_bytes()).hexdigest(),
        "conditions": list(CONDITIONS), "overlap_checked_files": checked, "audit": audit,
        "budget_usd": 5, "compute_reservation_usd": 4, "judge_reservation_usd": 1,
        "selection": "Fixed after original test inspection; no tuning on this fresh set.",
    }
    if REPORT.exists():
        previous = json.loads(REPORT.read_text())
        if previous["task_file_sha256"] != report["task_file_sha256"]:
            raise ValueError("Confirmation manifest does not match frozen tasks")
        for key in ("demos_sha256", "rules_sha256"):
            if key in previous and previous[key] != report[key]:
                raise ValueError(f"Confirmation inputs changed: {key}")
            previous[key] = report[key]
        REPORT.write_text(json.dumps(previous, indent=2) + "\n")
        return previous
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    return report


def summarize():
    report = prepare()
    tasks = {row["id"]: Task.from_dict(row) for row in json.loads(TASKS.read_text())}
    expected = {f"{task_id}:{rollout}" for task_id in tasks for rollout in range(4)}
    summaries, task_means = {}, {}
    for condition in CONDITIONS:
        directory = ROOT / "results/v4" / f"confirmation-{condition}-calculator"
        config = json.loads((directory / "config.json").read_text())
        required = {
            "taskset_sha256": report["task_file_sha256"], "demos_sha256": report["demos_sha256"],
            "rules_sha256": report["rules_sha256"], "temperature": 0.7, "seed": 17,
            "rollouts": 4, "mode": "calculator", "max_new_tokens_per_turn": 384,
        }
        if any(config.get(key) != value for key, value in required.items()):
            raise ValueError(f"{condition}: evaluation configuration differs from the frozen comparison")
        provenance = json.loads((ROOT / "results/v4/sft/merged/provenance.json").read_text())
        if config.get("base") != provenance:
            raise ValueError(f"{condition}: SFT base provenance differs from the local checkpoint")
        if condition != "sft":
            step = "step-40" if condition.endswith("-signal") else "step-60"
            adapter = ROOT / "results/v4" / condition / "checkpoints" / step / "adapter_model.safetensors"
            if config.get("adapter_sha256") != hashlib.sha256(adapter.read_bytes()).hexdigest():
                raise ValueError(f"{condition}: remote adapter differs from the local checkpoint")
        rows = read_events(directory / "generations.jsonl")
        if len(rows) != len(expected) or {row["id"] for row in rows} != expected:
            raise ValueError(f"{condition}: incomplete or duplicate generations, {len(rows)}/{len(expected)}")
        per_task, payslips, traps, rewards = {}, [], [], []
        scores = []
        for row in rows:
            task = tasks[row["task_id"]]
            score = score_answer(task, row["final_answer"])
            exact = score["breakdown"]["exact"]["score"] == 1
            (payslips if task.difficulty == 2 else traps).append(exact)
            per_task.setdefault(task.id, []).append(exact)
            rewards.append(score["score"])
            scores.append({"id": row["id"], "tier": task.difficulty, "score": score})
        (directory / "correctness.json").write_text(json.dumps(scores, indent=2) + "\n")
        calculation_means = {task_id: statistics.mean(values) for task_id, values in per_task.items()
                             if tasks[task_id].difficulty == 2}
        trap_means = [statistics.mean(values) for task_id, values in per_task.items()
                      if tasks[task_id].difficulty == 3]
        summaries[condition] = {
            "answers": len(rows), "exact_payslips": statistics.mean(payslips),
            "exact_payslips_count": sum(payslips), "payslip_answers": len(payslips),
            "correct_trap_decisions": statistics.mean(traps),
            "correct_trap_decisions_count": sum(traps), "trap_answers": len(traps),
            "malformed": statistics.mean(row["score"]["status"] == "invalid_format" for row in scores),
            "exact_payslips_interval_95": bootstrap(list(calculation_means.values())),
            "correct_trap_decisions_interval_95": bootstrap(trap_means),
            "programmatic_reward_mean": statistics.mean(rewards),
            "programmatic_reward_sd": statistics.stdev(rewards), "rubric_version": RUBRIC_VERSION,
            "reward_note": "Programmatic score only; explanation judge not included. Not the full rubric reward.",
            "seconds": sum(row["seconds"] for row in rows),
            "modal": json.loads((directory / "modal.json").read_text()),
        }
        task_means[condition] = calculation_means
    report.update({
        "status": "complete", "summaries": summaries,
        "paired_exact_payslips": {
            "original_rl_minus_sft": paired(task_means["sft"], task_means["rl-0.04"]),
            "focused_rl_minus_original_rl": paired(task_means["rl-0.04"], task_means["rl-0.04-signal"]),
        },
        "judge_cost_usd": 0,
        "uncertainty": "95% task-level bootstrap intervals, 1000 resamples, seed 7; no training-seed replication.",
        "scorer_sha256": hashlib.sha256((ROOT / "src/scoring.py").read_bytes()).hexdigest(),
        "modal_runtime_seconds": sum(row["modal"]["seconds"] for row in summaries.values()),
        "modal_runtime_cost_upper_estimate_usd": sum(row["modal"]["seconds"] for row in summaries.values())
        * (0.000222 + 2 * 0.0000131 + 8 * 0.00000222),
        "modal_pricing_source": "https://modal.com/pricing",
        "modal_pricing_checked": "2026-10-20",
        "modal_estimate_note": "L4 runtime plus maximum 2 CPU cores and 8 GiB; excludes build, startup, storage and transfer. Not an invoice.",
        "cost_note": "Deterministic correctness evaluation. Modal invoice cost not yet available; reservations are not charges.",
    })
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    result = summarize() if args.summarize else prepare()
    print(json.dumps(result, indent=2))
