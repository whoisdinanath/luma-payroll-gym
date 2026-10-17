import hashlib
import json
from collections import Counter
from pathlib import Path

from luma_gym.audit import audit_tasks, check_splits
from luma_gym.cases import CASE_GENERATOR, features, generate_cases, matches
from luma_gym.tasks import Task, semantic_signature, taskset_hash


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/v4"
SETS = {
    "train": (4001, 192, 192, 0),
    "development": (4002, 48, 48, 0),
    "test": (4003, 96, 96, 0),
    "eval": (4041, 48, 48, 12),
    "sft": (4004, 128, 128, 0),
}
DEMO_SEED = 4099


def coverage(tasks):
    calculations = [task for task in tasks if task.difficulty == 2]
    return {
        "tiers": dict(Counter(str(task.difficulty) for task in tasks)),
        "strata": dict(Counter(tag[8:] for task in calculations for tag in task.tags if tag.startswith("stratum:"))),
        "brackets": dict(Counter(tag[8:] for task in calculations for tag in task.tags if tag.startswith("bracket:"))),
        "cap": dict(Counter(tag[4:] for task in calculations for tag in task.tags if tag.startswith("cap:"))),
        "boundaries": dict(Counter(tag[9:] for task in calculations for tag in task.tags if tag.startswith("boundary:"))),
        "contracts": dict(Counter(tag[9:] for task in calculations for tag in task.tags if tag.startswith("contract:"))),
        "rounding_ties": sum("rounding-tie" in task.tags for task in calculations),
        "with_deduction": sum("deduction" in task.tags for task in calculations),
        "halves": dict(Counter("first" if task.record["payroll_month"] <= "2026-06" else "second" for task in calculations)),
        "trap_reasons": dict(Counter(task.ground_truth["reason_code"] for task in tasks if task.difficulty == 3)),
        "conflict_variants": dict(Counter(tag[9:] for task in tasks for tag in task.tags if tag.startswith("conflict:"))),
        "retrieval_fields": dict(Counter(task.retrieval_field for task in tasks if task.difficulty == 1)),
    }


def pick_demos():
    pool = generate_cases(DEMO_SEED, calculations=64, traps=16, prefix="demo4")
    chosen = []
    wanted = [
        lambda t: "stratum:bracket_middle" in t.tags and "contract:none" in t.tags and "deduction" in t.tags
        and t.record["payroll_month"] <= "2026-06",
        lambda t: "stratum:cap_binding" in t.tags and "contract:amendment" in t.tags and t.record["payroll_month"] >= "2026-07",
        lambda t: t.ground_truth.get("reason_code") == "MISSING_GROSS_PAY",
        lambda t: t.ground_truth.get("reason_code") == "MISSING_PAYROLL_MONTH",
        lambda t: t.ground_truth.get("reason_code") == "NO_APPLICABLE_REFERENCE",
        lambda t: t.ground_truth.get("reason_code") == "CONFLICTING_GROSS_PAY" and "conflict:plain" in t.tags,
    ]
    for test in wanted:
        chosen.append(next(task for task in pool if test(task)))
    # Interleave the examples: with every clarification last, the frozen model refused most payslips.
    calculation, amendment, gross, month, reference, conflict = chosen
    return [calculation, gross, amendment, conflict, month, reference]


def older_signatures():
    seen = set()
    for path in [*sorted((ROOT / "data/training-splits").glob("*.json")), ROOT / "data/cases.json"]:
        if path.name == "manifest.json":
            continue
        seen |= {semantic_signature(Task.from_dict(item)) for item in json.loads(path.read_text())}
    return seen


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sets = {name: generate_cases(seed, calculations, traps, retrievals)
            for name, (seed, calculations, traps, retrievals) in SETS.items()}
    demos = pick_demos()
    check_splits({**sets, "demos": demos})
    older = older_signatures()
    manifest = {
        "case_generator": CASE_GENERATOR,
        "sets": {}, "demos": [task.id for task in demos],
    }
    for name, tasks in [*sets.items(), ("demos", demos)]:
        for task in tasks:
            if task.difficulty == 2:
                stratum = next(tag[8:] for tag in task.tags if tag.startswith("stratum:"))
                if not matches(stratum, features(task.record, task.ground_truth)):
                    raise SystemExit(f"Stratum mismatch: {task.id}")
        path = OUTPUT / f"{name}.json"
        path.write_text(json.dumps([task.to_dict() for task in tasks], indent=2) + "\n")
        audit = audit_tasks(tasks)
        overlap = sum(semantic_signature(task) in older for task in tasks)
        if audit["semantic_duplicate_instances"] or audit["prompt_leak_candidates"] or overlap:
            raise SystemExit(f"{name}: duplicates, leaks, or overlap with earlier datasets")
        manifest["sets"][name] = {
            "seed": tasks[0].seed, "tasks": len(tasks), "sha256": taskset_hash(tasks),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "semantic_duplicates": 0, "prompt_leaks": 0, "overlap_with_earlier_datasets": 0,
            "coverage": coverage(tasks),
        }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({name: {key: value for key, value in item.items() if key != "coverage"}
                      for name, item in manifest["sets"].items()}, indent=2))
    print(json.dumps(manifest["sets"]["train"]["coverage"], indent=2))


if __name__ == "__main__":
    main()
