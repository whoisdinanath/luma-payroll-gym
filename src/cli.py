import argparse
import json
from collections import Counter
from pathlib import Path

from luma_gym.baselines import audit_baselines
from luma_gym.audit import audit_tasks
from luma_gym.cases import CALC_STRATA, CONFLICTS, CONTRACTS, TRAP_REASONS, generate_cases
from luma_gym.tasks import Task, taskset_hash, validate_taskset


def main():
    parser = argparse.ArgumentParser(description="Generate and validate fictional payroll tasks")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate seeded tasks with reference answers")
    generate.add_argument("--seed", type=int, default=7)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--calculations", type=int, default=48, help="Tier 2 payslips")
    generate.add_argument("--traps", type=int, default=48, help="Tier 3 traps")
    generate.add_argument("--retrievals", type=int, default=12, help="Tier 1 lookups")
    generate.add_argument("--strata", nargs="+", default=CALC_STRATA, choices=CALC_STRATA,
                          help="Payslip strata, cycled evenly")
    generate.add_argument("--reasons", nargs="+", default=TRAP_REASONS, choices=TRAP_REASONS, help="Trap reasons, cycled evenly")
    generate.add_argument("--conflicts", nargs="+", default=CONFLICTS, choices=CONFLICTS,
                          help="Conflict variants for conflicting-pay traps")
    generate.add_argument("--contracts", nargs="+", default=CONTRACTS, choices=sorted(set(CONTRACTS)),
                          help="Contract variants on payslips")
    validate = commands.add_parser("validate")
    validate.add_argument("path", type=Path)
    audit = commands.add_parser("audit-baselines")
    audit.add_argument("path", type=Path)
    audit.add_argument("--output", type=Path)
    task_audit = commands.add_parser("audit-tasks")
    task_audit.add_argument("path", type=Path)
    task_audit.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "generate":
            tasks = generate_cases(args.seed, args.calculations, args.traps, args.retrievals, strata=args.strata,
                                   reasons=args.reasons, conflicts=args.conflicts, contracts=args.contracts)
            validate_taskset(tasks)
            with args.output.open("x") as file:
                json.dump([task.to_dict() for task in tasks], file, indent=2)
                file.write("\n")
        else:
            data = json.loads(args.path.read_text())
            if not isinstance(data, list):
                raise ValueError("Taskset must be a JSON array")
            tasks = [Task.from_dict(item) for item in data]
            validate_taskset(tasks)
            if args.command in ("audit-baselines", "audit-tasks"):
                report = audit_baselines(tasks) if args.command == "audit-baselines" else audit_tasks(tasks)
                if args.output:
                    with args.output.open("x") as file:
                        json.dump(report, file, indent=2)
                        file.write("\n")
                print(json.dumps(report.get("summary", report), indent=2))
                return
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")
    print(json.dumps({
        "tasks": len(tasks),
        "tiers": dict(sorted(Counter(task.difficulty for task in tasks).items())),
        "sha256": taskset_hash(tasks),
    }, indent=2))


if __name__ == "__main__":
    main()
