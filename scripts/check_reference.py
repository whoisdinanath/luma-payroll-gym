import json
from pathlib import Path

from luma_gym.reference import calculate_payroll


def cents(value):
    whole, fraction = value.split(".")
    return int(whole) * 100 + int(fraction)


def percentage(value, rate):
    basis_points = int(rate.replace(".", ""))
    return (value * basis_points + 50) // 100


root = Path(__file__).resolve().parents[1]
tasks = json.loads((root / "data/cases.json").read_text())
checked = []
for task in tasks:
    if task["difficulty"] != 2:
        continue
    record = task["record"]
    row = record["reference_tables"]["first-half" if record["payroll_month"] <= "2026-06" else "second-half"]
    gross = cents(record["gross_pay"])
    contribution = percentage(min(gross, cents(row["cap"])), row["rate"])
    deduction = cents(record.get("allowed_deduction", "0.00"))
    taxable = gross - contribution - deduction
    middle = min(max(0, taxable - cents(row["free"])), cents(row["upper"]) - cents(row["free"]))
    upper = max(0, taxable - cents(row["upper"]))
    middle_rate = int(row["middle_rate"].replace(".", ""))
    upper_rate = int(row["upper_rate"].replace(".", ""))
    tax = (middle * middle_rate + upper * upper_rate + 50) // 100
    actual = {"employee_contribution": contribution, "allowed_deduction": deduction,
              "taxable_pay": taxable, "income_tax": tax, "net_pay": taxable - tax}
    mismatches = [field for field, value in actual.items() if cents(task["ground_truth"][field]) != value]
    checked.append({"task_id": task["id"], "clauses": ["R1", "R2", "R3", "R4", "R5", "R6"],
                    "integer_cent_calculation": actual, "mismatches": mismatches})
    if len(checked) == 10:
        break
report = {"method": "integer-cent arithmetic",
          "checked": len(checked), "divergence_rate": sum(bool(row["mismatches"]) for row in checked) / len(checked),
          "results": checked}
(root / "reports/reference-crosscheck.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "results"}, indent=2))
