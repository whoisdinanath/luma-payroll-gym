import json
import random
from decimal import Decimal
from pathlib import Path

from luma_gym import RULESET_VERSION, SCHEMA_VERSION
from luma_gym.reference import RETRIEVAL_VALUES, TABLES, calculate_payroll, reference_table, retrieve_rule, rounded
from luma_gym.tasks import Task, generate_tasks, validate_taskset


CASE_GENERATOR = "4"
CENT = Decimal("0.01")
CALC_STRATA = ["bracket_zero", "bracket_middle", "bracket_upper", "cap_binding",
               "boundary_cap", "boundary_free", "boundary_upper", "rounding_tie"]
TRAP_REASONS = ["MISSING_GROSS_PAY", "MISSING_PAYROLL_MONTH", "NO_APPLICABLE_REFERENCE", "CONFLICTING_GROSS_PAY"]
CONFLICTS = ["plain", "draft_amendment", "future_amendment", "amount_mismatch"]
CONTRACTS = ["none", "matching", "amendment", "none"]
CALCULATION_PROMPT = "Process this monthly payroll record under the supplied rulebook. Return the R6 or R7 JSON answer."


def prepare(n, seed):
    # Generator 3, kept as is so old datasets reproduce. Its boundary tags are unreliable because it picks
    # boundary inputs from the default schedule and then replaces the schedule.
    rng = random.Random(seed + 10000)
    tasks = generate_tasks(n, seed)
    for task in tasks:
        tables = {name: row.copy() for name, row in TABLES.items()}
        for row in tables.values():
            row["rate"] = rng.choice(["0.04", "0.05", "0.06", "0.07", "0.08"])
            row["cap"] = rng.choice(["3500.00", "4000.00", "4500.00", "5000.00"])
            row["deduction"] = "400.00"
            row["free"] = rng.choice(["900.00", "1000.00", "1100.00", "1200.00"])
            row["upper"] = rng.choice(["2800.00", "3000.00", "3300.00", "3600.00"])
            row["middle_rate"] = rng.choice(["0.08", "0.10", "0.12", "0.14"])
            row["upper_rate"] = rng.choice(["0.18", "0.20", "0.22", "0.24"])
        task.record["reference_tables"] = tables
        task.ruleset_version = RULESET_VERSION
        task.tags.append("case-reference-schedule")
        task.ground_truth = (retrieve_rule(task.retrieval_field, task.record)
                             if task.difficulty == 1 else calculate_payroll(task.record))
        task.validate()
    validate_taskset(tasks)
    return tasks


def draw_schedule(rng):
    tables = {}
    for name in ("first-half", "second-half"):
        tables[name] = {
            "rate": rng.choice(["0.04", "0.05", "0.06", "0.07", "0.08"]),
            "cap": rng.choice(["3500.00", "4000.00", "4500.00", "5000.00"]),
            "deduction": "400.00",
            "free": rng.choice(["900.00", "1000.00", "1100.00", "1200.00"]),
            "upper": rng.choice(["2800.00", "3000.00", "3300.00", "3600.00"]),
            "middle_rate": rng.choice(["0.08", "0.10", "0.12", "0.14"]),
            "upper_rate": rng.choice(["0.18", "0.20", "0.22", "0.24"]),
        }
    return tables


def row_for(month, tables):
    row = reference_table(month, tables=tables)
    if row is None:
        raise ValueError("Month has no reference row")
    return row


def money(value):
    return format(value.quantize(CENT), ".2f")


def payslip(gross, deduction, row):
    contribution = rounded(min(gross, row["cap"]) * row["rate"])
    return contribution, gross - contribution - deduction


def features(record, truth):
    # Tags are measured from the reference answer, not taken from the requested stratum.
    row = row_for(record["payroll_month"], record["reference_tables"])
    gross = Decimal(truth["gross_pay"])
    taxable = Decimal(truth["taxable_pay"])
    contribution = min(gross, row["cap"]) * row["rate"]
    middle = min(max(taxable - row["free"], Decimal(0)), row["upper"] - row["free"])
    tax = middle * row["middle_rate"] + max(taxable - row["upper"], Decimal(0)) * row["upper_rate"]
    boundary = [name for name, distance in (("cap", gross - row["cap"]), ("free", taxable - row["free"]),
                                            ("upper", taxable - row["upper"])) if abs(distance) <= Decimal("0.02")]
    return {
        "bracket": "zero" if taxable <= row["free"] else "middle" if taxable <= row["upper"] else "upper",
        "cap_binding": gross > row["cap"],
        "boundary": boundary,
        "rounding_tie": any((value * 100) % 1 == Decimal("0.5") for value in (contribution, tax)),
        "deduction": Decimal(truth["allowed_deduction"]) > 0,
        "amendment_resolved": "R9" in truth["clauses"],
    }


def matches(stratum, found):
    return {
        "bracket_zero": found["bracket"] == "zero" and not found["cap_binding"] and not found["boundary"],
        "bracket_middle": found["bracket"] == "middle" and not found["cap_binding"] and not found["boundary"],
        "bracket_upper": found["bracket"] == "upper" and not found["cap_binding"] and not found["boundary"],
        "cap_binding": found["cap_binding"] and not found["boundary"],
        "boundary_cap": "cap" in found["boundary"],
        "boundary_free": "free" in found["boundary"],
        "boundary_upper": "upper" in found["boundary"],
        "rounding_tie": found["rounding_tie"] and not found["boundary"],
    }[stratum]


def gross_for_taxable(target, deduction, row):
    # Solve taxable(gross) == target to the cent; contribution rounding can make a target unreachable.
    estimate = (target + deduction) / (1 - row["rate"])
    for step in range(-12, 13):
        gross = (estimate + step * CENT).quantize(CENT)
        if gross >= 0 and payslip(gross, deduction, row)[1] == target:
            return gross
    return None


def calculation_record(rng, stratum, tables, month, contract="none"):
    row = row_for(month, tables)
    deduction = Decimal(rng.randint(1, 40000)) / 100 if rng.random() < 0.7 else Decimal(0)
    target = None
    if stratum == "bracket_zero":
        target = Decimal(rng.randint(1, int((row["free"] - Decimal("0.10")) * 100))) / 100
    elif stratum == "bracket_middle":
        target = Decimal(rng.randint(int((row["free"] + Decimal("0.10")) * 100), int((row["upper"] - Decimal("0.10")) * 100))) / 100
    elif stratum == "bracket_upper":
        high = row["cap"] * (1 - row["rate"]) - deduction - Decimal("0.10")
        low = row["upper"] + Decimal("0.10")
        if high <= low:
            return None
        target = Decimal(rng.randint(int(low * 100), int(high * 100))) / 100
    elif stratum == "boundary_free":
        target = row["free"] + rng.choice([-2, -1, 0, 1, 2]) * CENT
    elif stratum == "boundary_upper":
        target = row["upper"] + rng.choice([-2, -1, 0, 1, 2]) * CENT
    if target is not None:
        gross = gross_for_taxable(target, deduction, row)
    elif stratum == "cap_binding":
        gross = Decimal(rng.randint(int(row["cap"] * 100) + 5000, 1000000)) / 100
    elif stratum == "boundary_cap":
        gross = row["cap"] + rng.choice([-2, -1, 0, 1, 2]) * CENT
    else:
        gross = None
        for _ in range(400):
            candidate = Decimal(rng.randint(100000, int(row["cap"] * 100))) / 100
            if (candidate * row["rate"] * 100) % 1 == Decimal("0.5"):
                gross = candidate
                break
    if gross is None or deduction > row["deduction"] or deduction > gross - payslip(gross, Decimal(0), row)[0]:
        return None
    record = {"gross_pay": money(gross), "payroll_month": month}
    if deduction or rng.random() < 0.5:
        record["allowed_deduction"] = money(deduction)
    if contract == "matching":
        record["contract_gross_pay"] = record["gross_pay"]
    elif contract == "amendment":
        record["contract_gross_pay"] = money(gross + Decimal(rng.randint(1, 50000)) / 100)
        record["amendment"] = {"gross_pay": record["gross_pay"], "approved": True,
                               "effective_month": f"2026-{rng.randint(1, int(month[5:])):02d}"}
    record["reference_tables"] = tables
    return record


def trap_record(rng, reason, variant, tables, month):
    record = None
    while record is None:
        record = calculation_record(rng, rng.choice(["bracket_middle", "bracket_upper", "cap_binding"]), tables, month)
    record.pop("amendment", None)
    record.pop("contract_gross_pay", None)
    if reason == "MISSING_GROSS_PAY":
        del record["gross_pay"]
    elif reason == "MISSING_PAYROLL_MONTH":
        del record["payroll_month"]
    elif reason == "NO_APPLICABLE_REFERENCE":
        record["payroll_month"] = f"{rng.choice(['2025', '2027'])}-{rng.randint(1, 12):02d}"
    else:
        gross = Decimal(record["gross_pay"])
        shift = Decimal(rng.choice([1, 100, 2500, 10000, 50000])) / 100
        record["contract_gross_pay"] = money(gross + shift if rng.random() < 0.5 or gross <= shift else gross - shift)
        effective = f"2026-{rng.randint(1, int(month[5:])):02d}"
        if variant == "draft_amendment":
            record["amendment"] = {"gross_pay": record["gross_pay"], "effective_month": effective, "approved": False}
        elif variant == "future_amendment":
            later = int(month[5:]) + rng.randint(1, 3)
            record["amendment"] = {"gross_pay": record["gross_pay"], "approved": True,
                                   "effective_month": f"2026-{later:02d}" if later <= 12 else f"2027-{later - 12:02d}"}
        elif variant == "amount_mismatch":
            record["amendment"] = {"gross_pay": money(gross + shift + Decimal("5.00")), "effective_month": effective,
                                   "approved": True}
    return record


def build_task(task_id, seed, difficulty, record, tags, field=None):
    truth = retrieve_rule(field, record) if difficulty == 1 else calculate_payroll(record)
    prompt = f"Retrieve {field} from the supplied rulebook. Return the R8 JSON answer." if difficulty == 1 else CALCULATION_PROMPT
    return Task(id=task_id, slice="monthly_payroll", difficulty=difficulty, prompt=prompt, record=record,
                ground_truth=truth, verifier="payroll", tags=tags, schema_version=SCHEMA_VERSION,
                ruleset_version=RULESET_VERSION, seed=seed, retrieval_field=field)


def generate_cases(seed, calculations=0, traps=0, retrievals=0, prefix="case4", strata=CALC_STRATA, reasons=TRAP_REASONS,
                   conflicts=CONFLICTS, contracts=CONTRACTS):
    unknown = (set(strata) - set(CALC_STRATA)) | (set(reasons) - set(TRAP_REASONS)) | (set(conflicts) - set(CONFLICTS)) \
        | (set(contracts) - set(CONTRACTS))
    if unknown or not (strata and reasons and conflicts and contracts):
        raise ValueError(f"Unknown or empty generator option: {sorted(unknown)}")
    rng = random.Random(seed)
    plan = [("calculate", strata[index % len(strata)], contracts[(index // len(strata)) % len(contracts)])
            for index in range(calculations)]
    plan += [("clarify", reasons[index % len(reasons)], conflicts[(index // len(reasons)) % len(conflicts)])
             for index in range(traps)]
    fields = [(field, half) for field in RETRIEVAL_VALUES for half in ("first", "second")]
    plan += [("retrieve", *fields[index % len(fields)]) for index in range(retrievals)]
    rng.shuffle(plan)
    tasks = []
    for index, item in enumerate(plan):
        task_id = f"{prefix}-{seed}-{index:05d}"
        while True:
            tables = draw_schedule(rng)
            month = f"2026-{rng.randint(1, 12):02d}"
            tags = [f"generator-{CASE_GENERATOR}"]
            if item[0] == "retrieve":
                month = f"2026-{rng.randint(1, 6) if item[2] == 'first' else rng.randint(7, 12):02d}"
                task = build_task(task_id, seed, 1, {"payroll_month": month, "reference_tables": tables},
                                  tags + ["retrieval"], item[1])
                break
            if item[0] == "clarify":
                record = trap_record(rng, item[1], item[2], tables, month)
                tags += ["trap", f"reason:{item[1]}"]
                if item[1] == "CONFLICTING_GROSS_PAY":
                    tags.append(f"conflict:{item[2]}")
                try:
                    task = build_task(task_id, seed, 3, record, tags)
                except ValueError:
                    continue
                if task.ground_truth.get("reason_code") == item[1]:
                    break
                continue
            record = calculation_record(rng, item[1], tables, month, item[2])
            if record is None:
                continue
            try:
                task = build_task(task_id, seed, 2, record, tags)
            except ValueError:
                continue
            found = features(record, task.ground_truth)
            if task.ground_truth["decision"] != "calculate" or not matches(item[1], found):
                continue
            task.tags += [f"stratum:{item[1]}", f"bracket:{found['bracket']}",
                          "cap:binding" if found["cap_binding"] else "cap:open"]
            task.tags += [f"boundary:{name}" for name in found["boundary"]]
            task.tags += [tag for tag, present in (("rounding-tie", found["rounding_tie"]), ("deduction", found["deduction"]),
                                                   ("amendment-resolved", found["amendment_resolved"])) if present]
            task.tags.append(f"contract:{item[2]}")
            break
        task.validate()
        tasks.append(task)
    validate_taskset(tasks)
    return tasks


def main():
    root = Path.cwd()
    for name, n, seed in [("cases", 100, 41), ("case-pilot", 6, 43)]:
        tasks = prepare(n, seed)
        with (root / f"data/{name}.json").open("x") as file:
            json.dump([task.to_dict() for task in tasks], file, indent=2)
            file.write("\n")
        print(f"Prepared {n} tasks: {name}")


if __name__ == "__main__":
    main()
