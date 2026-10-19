import argparse
import json
from pathlib import Path

from luma_gym.audit import check_splits
from luma_gym.tasks import GENERATOR_VERSION, generate_tasks, semantic_signature, taskset_hash
from luma_gym.cases import prepare


def build_splits(seeds, n=100, case_schedules=False):
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Provide three different seeds for train, development, and held-out")
    splits = {}
    used = set()
    rejected = {}
    for name, seed in zip(("train", "development", "held_out"), seeds):
        if case_schedules:
            candidates = [task for task in prepare(min(n * 4, 10000), seed) if task.difficulty != 1]
        else:
            candidates = generate_tasks(min(n * 3, 10000), seed, tiers=(2, 3))
        tasks = []
        rejected[name] = 0
        for task in candidates:
            signature = semantic_signature(task)
            if signature in used:
                rejected[name] += 1
                continue
            used.add(signature)
            tasks.append(task)
            if len(tasks) == n:
                break
        if len(tasks) != n:
            raise ValueError("Not enough unique split candidates; choose another seed or smaller split")
        splits[name] = tasks
    check_splits(splits)
    return splits, rejected


def main():
    parser = argparse.ArgumentParser(description="Create semantically separated task splits")
    parser.add_argument("--seeds", nargs=3, type=int, default=[101, 202, 303])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-schedules", action="store_true")
    args = parser.parse_args()
    splits, rejected = build_splits(args.seeds, args.n, args.case_schedules)
    args.output.mkdir(exist_ok=False)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "case_schedules": args.case_schedules,
        "protocol": "Calculation and clarification only; finite retrieval questions excluded from held-out novelty claims",
        "controls": {"tiers": [2, 3], "boundary_fraction": 0.5,
                     "deduction_fraction": 0.7, "contract_fraction": 0.5},
        "splits": {},
    }
    for name, tasks in splits.items():
        (args.output / f"{name}.json").write_text(json.dumps([task.to_dict() for task in tasks], indent=2) + "\n")
        manifest["splits"][name] = {
            "seed": tasks[0].seed, "tasks": len(tasks), "sha256": taskset_hash(tasks),
            "cross_split_duplicates_rejected": rejected[name],
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
