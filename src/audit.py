from collections import Counter

from luma_gym.tasks import semantic_signature, validate_taskset


def audit_tasks(tasks):
    validate_taskset(tasks)
    signatures = Counter(semantic_signature(task) for task in tasks)
    leaks = []
    for task in tasks:
        prompt = task.prompt.lower()
        reason = task.ground_truth.get("reason_code")
        if reason and reason.lower() in prompt:
            leaks.append({"task_id": task.id, "issue": "reason_code_in_prompt"})
        for field in ("net_pay", "income_tax", "taxable_pay"):
            value = task.ground_truth.get(field)
            if value and len(value) >= 5 and value in prompt:
                leaks.append({"task_id": task.id, "issue": "computed_value_in_prompt"})
    calculation_prompts = {task.prompt for task in tasks if task.difficulty == 2}
    trap_prompts = {task.prompt for task in tasks if task.difficulty == 3}
    retrieval = sum(task.difficulty == 1 for task in tasks)
    return {
        "tasks": len(tasks),
        "semantic_unique": len(signatures),
        "semantic_duplicate_instances": sum(count - 1 for count in signatures.values()),
        "retrieval_instances": retrieval,
        "retrieval_skills": len({task.retrieval_field for task in tasks if task.difficulty == 1}),
        "prompt_leak_candidates": leaks,
        "calculation_and_trap_share_prompt": bool(calculation_prompts & trap_prompts),
    }


def check_splits(splits):
    seen_seeds = set()
    seen_signatures = set()
    for name, tasks in splits.items():
        validate_taskset(tasks)
        seeds = {task.seed for task in tasks}
        if None in seeds or seeds & seen_seeds:
            raise ValueError(f"Missing or overlapping generator seeds: {name}")
        signatures = {semantic_signature(task) for task in tasks}
        if len(signatures) != len(tasks):
            raise ValueError(f"Semantic duplicates inside split: {name}")
        if signatures & seen_signatures:
            raise ValueError(f"Semantic task overlap across splits: {name}")
        seen_seeds.update(seeds)
        seen_signatures.update(signatures)
