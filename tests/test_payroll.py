import asyncio
import json
import unittest
from collections import Counter
from pathlib import Path

from luma_gym.audit import audit_tasks, check_splits
from luma_gym.calculator import calculate
from luma_gym.cases import CALC_STRATA, features, generate_cases, matches
from luma_gym.judge import grade_answer
from luma_gym.reference import calculate_payroll
from luma_gym.scoring import parse_answer, score_answer
from luma_gym.splits import build_splits
from luma_gym.tasks import Task, generate_tasks, taskset_hash, validate_taskset


class PayrollTests(unittest.TestCase):
    def test_calculation_and_rounding(self):
        answer = calculate_payroll({
            "gross_pay": "5000.00",
            "allowed_deduction": "300.00",
            "payroll_month": "2026-01",
        })
        self.assertEqual(answer["employee_contribution"], "200.00")
        self.assertEqual(answer["income_tax"], "500.00")
        self.assertEqual(answer["net_pay"], "4000.00")
        answer = calculate_payroll({"gross_pay": "1000.10", "payroll_month": "2026-01"})
        self.assertEqual(answer["employee_contribution"], "50.01")

    def test_missing_and_invalid_inputs(self):
        self.assertEqual(calculate_payroll({})["reason_code"], "MISSING_GROSS_PAY")
        with self.assertRaises(ValueError):
            calculate_payroll({"gross_pay": "1.001", "payroll_month": "2026-01"})

    def test_seed_answers_and_reproducibility(self):
        path = Path(__file__).resolve().parents[1] / "data/seed_tasks.json"
        validate_taskset([Task.from_dict(item) for item in json.loads(path.read_text())])
        tasks = generate_tasks(100, 7)
        validate_taskset(tasks)
        self.assertEqual(taskset_hash(tasks), taskset_hash(generate_tasks(100, 7)))
        self.assertNotEqual(taskset_hash(tasks), taskset_hash(generate_tasks(100, 8)))

    def test_reward_gates_and_parser(self):
        tasks = generate_tasks(3, 7)
        task = tasks[1]
        answer = task.ground_truth.copy()
        self.assertEqual(score_answer(task, json.dumps(answer))["score"], 1.0)
        answer["net_pay"] = "999999.00"
        self.assertEqual(score_answer(task, json.dumps(answer))["score"], -0.35)
        self.assertEqual(score_answer(tasks[2], json.dumps(answer))["score"], -0.5)
        refusal = {"decision": "clarify", "reason_code": "MISSING_GROSS_PAY", "clauses": ["R7"]}
        self.assertEqual(score_answer(task, json.dumps(refusal))["score"], -0.5)
        nothing_right = {**task.ground_truth, **{field: "999999.00" for field in
                                                 ("employee_contribution", "taxable_pay", "income_tax", "net_pay")}}
        self.assertEqual(score_answer(task, json.dumps(nothing_right))["score"], -0.5)
        wrong_reason = {**tasks[2].ground_truth, "reason_code": "MISSING_PAYROLL_MONTH"
                        if tasks[2].ground_truth["reason_code"] != "MISSING_PAYROLL_MONTH" else "MISSING_GROSS_PAY"}
        self.assertEqual(score_answer(tasks[2], json.dumps(wrong_reason))["score"], -0.5)
        self.assertEqual(score_answer(task, "not json")["score"], -0.5)
        for text in ['{"decision":"clarify","decision":"calculate"}', '```json\n{}\n```']:
            with self.assertRaises(ValueError):
                parse_answer(text)
        trap = tasks[2]
        empty_explanation = asyncio.run(grade_answer(trap, json.dumps(trap.ground_truth)))
        self.assertEqual(empty_explanation["score"], 0.9)
        with self.assertRaises(ValueError):
            asyncio.run(grade_answer(trap, json.dumps({**trap.ground_truth, "explanation": "Please clarify."})))

    def test_generator_controls_and_split_leakage(self):
        tasks = generate_tasks(100, 7)
        self.assertEqual(audit_tasks(tasks)["semantic_duplicate_instances"], 0)
        self.assertEqual(sum(task.difficulty == 1 for task in tasks), 6)
        splits, rejected = build_splits([101, 202, 303], 30)
        check_splits(splits)
        with self.assertRaises(ValueError):
            build_splits([101, 101, 303], 10)
        other = Task.from_dict(splits["train"][0].to_dict())
        other.seed = 999
        other.id = "disguised-duplicate"
        with self.assertRaises(ValueError):
            check_splits({"train": splits["train"], "held_out": [other]})

    def test_calculator_matches_reference_and_rejects_code(self):
        record = {"gross_pay": "8191.57", "allowed_deduction": "275.02", "payroll_month": "2026-07"}
        truth = calculate_payroll(record)
        program = ("contribution = round(min(8191.57, 4500.00) * 0.06, 2)\n"
                   "taxable = 8191.57 - contribution - 275.02\n"
                   "tax = round(min(max(taxable - 1100.00, 0), 2200.00) * 0.12 + max(taxable - 3300.00, 0) * 0.22, 2)\n"
                   "net = 8191.57 - contribution - 275.02 - tax")
        results = calculate(program)["results"]
        self.assertEqual([results[name] for name in ("contribution", "taxable", "tax", "net")],
                         [truth[field] for field in ("employee_contribution", "taxable_pay", "income_tax", "net_pay")])
        self.assertEqual(calculate("x = round(1234.50 * 0.05, 2)")["results"]["x"], "61.73")
        for code in ("import os", "x = __import__('os')", "x = 2 ** 9999", "x = (1).real", "x = 1 / 0", "x = y"):
            self.assertIn("error", calculate(code))

    def test_case_generator_is_balanced_and_deterministic(self):
        tasks = generate_cases(77, calculations=16, traps=8, retrievals=6)
        self.assertEqual(taskset_hash(tasks), taskset_hash(generate_cases(77, calculations=16, traps=8, retrievals=6)))
        strata = Counter(tag[8:] for task in tasks for tag in task.tags if tag.startswith("stratum:"))
        self.assertEqual(strata, Counter({name: 2 for name in CALC_STRATA}))
        self.assertEqual(set(Counter(task.ground_truth.get("reason_code") for task in tasks if task.difficulty == 3).values()), {2})
        for task in tasks:
            if task.difficulty == 2:
                stratum = next(tag[8:] for tag in task.tags if tag.startswith("stratum:"))
                self.assertTrue(matches(stratum, features(task.record, task.ground_truth)), task.id)
        self.assertEqual(audit_tasks(tasks)["semantic_duplicate_instances"], 0)

    def test_dated_rates_and_amendment_precedence(self):
        record = {"gross_pay": "5000.00", "allowed_deduction": "300.00", "payroll_month": "2026-06"}
        self.assertEqual(calculate_payroll(record)["net_pay"], "4000.00")
        record["payroll_month"] = "2026-07"
        answer = calculate_payroll(record)
        self.assertEqual(answer["employee_contribution"], "270.00")
        self.assertEqual(answer["income_tax"], "512.60")
        self.assertEqual(answer["net_pay"], "3917.40")
        record["contract_gross_pay"] = "4800.00"
        record["amendment"] = {"gross_pay": "5000.00", "effective_month": "2026-07", "approved": True}
        self.assertEqual(calculate_payroll(record)["net_pay"], "3917.40")
        record["amendment"]["approved"] = False
        self.assertEqual(calculate_payroll(record)["reason_code"], "CONFLICTING_GROSS_PAY")
        record["amendment"]["approved"] = True
        record["amendment"]["effective_month"] = "2026-08"
        self.assertEqual(calculate_payroll(record)["reason_code"], "CONFLICTING_GROSS_PAY")
        record["payroll_month"] = "2027-01"
        self.assertEqual(calculate_payroll(record)["reason_code"], "NO_APPLICABLE_REFERENCE")


if __name__ == "__main__":
    unittest.main()
