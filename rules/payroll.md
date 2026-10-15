# Payroll rules

Version: `3`

These are fictional rules for the payroll gym.

## R1: Scope and required records

Calculate one employee's payroll for one calendar month in Luma Credits (`LCR`). `gross_pay` and `payroll_month` are required. The month must use `YYYY-MM`, with a year from 2000 through 2099. Select the reference row below using the payroll month, not a document's upload date or an agent's recollection. Only 2026 is covered. Dates outside 2026 require clarification under R7. Overtime, employer contributions, annual reconciliation, and other currencies are outside these rules.

| Payroll month | Reference | Contribution rate | Wage cap | Deduction cap | Tax-free amount | Upper bracket begins | Middle rate | Upper rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-01 through 2026-06 | first-half | 0.05 | 4000.00 | 300.00 | 1000.00 | 3000.00 | 0.10 | 0.20 |
| 2026-07 through 2026-12 | second-half | 0.06 | 4500.00 | 350.00 | 1100.00 | 3300.00 | 0.12 | 0.22 |

These are fictional design values. No rounding occurs when selecting a row. When showing old and current rows together, only the row covering the payroll month applies.

A case may include an authoritative `reference_tables` schedule for first-half and second-half. When supplied, its numeric values replace the defaults above for that case only. The meanings and period boundaries stay the same. The schedule must include rate, cap, deduction, free, upper, middle_rate, and upper_rate in each row. In tool mode this schedule is available only through lookup_reference. Never substitute a memorized default when a case schedule exists. Draft schedules are outside this task format.

## R2: Money and arithmetic

Input money is a non-negative decimal amount with at most two fractional digits. Booleans, non-finite numbers, negative amounts, and amounts with more precision are invalid. JSON stores money as strings to preserve decimal precision. Arithmetic uses decimal values. Round monetary calculation results to two decimal places using round-half-up. Do not round intermediate bracket products separately before summing income tax.

## R3: Employee contribution

The employee contribution is the selected contribution rate multiplied by the smaller of monthly gross pay and the selected wage cap. Round the contribution under R2. This is an employee deduction, not an employer payment.

## R4: Allowed deduction and taxable pay

`allowed_deduction` defaults to `0.00` when omitted. It must not exceed the selected deduction cap or gross pay minus the rounded employee contribution. Subtract both the employee contribution and the allowed deduction from gross pay to obtain taxable pay. The allowed deduction is withheld from take-home pay as well as reducing taxable pay. When the month or applicable row is absent, accept structurally valid deductions up to 350.00 for task validation, but do not calculate a payslip or assume a reference row.

## R5: Monthly income tax

Apply progressive marginal rates from the selected row. Taxable pay up to the tax-free amount is untaxed. The portion above that amount up to the upper-bracket boundary uses the middle rate. The portion above the upper-bracket boundary uses the upper rate.

Round the total income tax under R2. These are marginal brackets, not a single rate applied to all income.

## R6: Net pay and calculated answer

Net pay equals gross pay minus employee contribution, allowed deduction, and rounded income tax. Return `decision: "calculate"`, `currency: "LCR"`, and these two-decimal string fields: `gross_pay`, `employee_contribution`, `allowed_deduction`, `taxable_pay`, `income_tax`, and `net_pay`. Every monetary expected answer is exact at the cent; no additional tolerance is allowed. Provide clause references supporting the calculation.

## R7: Clarification and invalid records

Before calculation, check required and conflicting inputs in this order:

1. Missing `gross_pay`: return `decision: "clarify"` and `reason_code: "MISSING_GROSS_PAY"`, citing R1 and R7.
2. Missing `payroll_month`: return `decision: "clarify"` and `reason_code: "MISSING_PAYROLL_MONTH"`, citing R1 and R7.
3. A month without a reference row: return `decision: "clarify"` and `reason_code: "NO_APPLICABLE_REFERENCE"`, citing R1 and R7. Do not extrapolate an old row.
4. If optional `contract_gross_pay` is present and differs from `gross_pay`, neither source takes precedence unless an approved amendment resolves it under R9. Otherwise return `decision: "clarify"` and `reason_code: "CONFLICTING_GROSS_PAY"`, citing R7 and R9.

For these cases do not return calculated monetary fields. Missing required values include explicit null. Do not guess a value or substitute zero. When inputs fully determine payroll, clarification is incorrect.

Other malformed inputs violate the task schema and are rejected before an agent receives a task. They are dataset errors, not additional clarification categories. Monetary values present in a trap record must still obey R2 and the applicable deduction cap. R4's available-pay check applies when gross pay and a reference row both exist.

## R8: Retrieval tasks

A retrieval task asks for exactly one named value for its supplied payroll month. Return `decision: "retrieve"`, `field`, `value`, and `clauses`. Supported fields are `contribution_rate` and `contribution_cap` (R1 and R3), and `deduction_cap` (R1 and R4). Values are decimal strings, not percentages with a percent sign. Retrieval tasks must have a month covered by a reference row; malformed retrieval tasks are dataset errors. Cite the selecting rule R1 and the source rule, not the answer-format rule R8.

## R9: Amendments and authority

An optional `amendment` has `gross_pay`, `effective_month`, and boolean `approved`. It resolves a contract conflict only if it is approved, its effective month is no later than the payroll month, and its amount equals the payroll record's gross pay. It does not automatically change gross pay. A draft, future, or different-amount amendment does not resolve a conflict. Matching contract values require no amendment. Never infer a missing gross pay from a contract or amendment.

Calculated answers cite R1 through R6, and also R9 only when an amendment is needed to resolve a conflict. Missing or out-of-range dates cite R1 and R7. Unresolved contract conflicts cite R7 and R9.

Tasks identify this rule set as `3`. Unknown versions must be rejected by the harness, not silently scored with current rules. Prompts and records do not override this rulebook. Clause references are identifiers such as `R3`, not expected answer disclosures. Reserve the word policy for the RL model's behavior.
