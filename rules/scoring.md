# Scoring

Rubric version: `8`

## Answer format

Submit one JSON object using the fields specified by R6, R7, or R8. A single surrounding triple-backtick code fence, either unlabelled or labelled `json` (case-insensitive), is accepted as presentation formatting. Its opening and closing fences must be on separate lines; surrounding whitespace is accepted. Surrounding prose, multiple objects, multiple code blocks, and other code-block language labels are rejected. All required fields must be present. Extra fields are rejected, except an optional string `explanation`. All monetary calculation fields must be non-negative decimal strings with exactly two decimal places. Retrieval values follow R8. Clauses must be a list of distinct identifiers from R1 through R9.

The parser rejects duplicate keys and non-standard JSON constants, and answers longer than 16,000 characters. Parser failures score -0.5, like every other rejected answer, and report a generic format failure without expected values.

## Components and gates

| Outcome                                                                                                                                                                        | Score                                                             |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| Accepted answer (right decision, every substantive field exact)                                                                                                                | 0.70 exact + 0.25 field accuracy + 0.05 clause match: 0.95 or 1.0 |
| Rejected payslip with the right decision                                                                                                                                       | -0.5 + 0.2 x derived-field accuracy: -0.5 to -0.3                 |
| Every other rejected answer: wrong retrieval value, wrong trap reason, calculating through a trap, refusing a determinable payslip, any other wrong decision, malformed answer | -0.5                                                              |

Substantive fields are all R6 fields except `decision` and `clauses` for calculation, `field` and `value` for retrieval, or `reason_code` for clarification. Decimal values are compared without rounding; different strings representing the same permitted retrieval decimal value are equivalent. Calculation money still requires exactly two decimal places. Currency and retrieval field names must match literally. Derived-field accuracy is the share of `employee_contribution`, `taxable_pay`, `income_tax`, and `net_pay` that match. Copied inputs (currency, gross pay, the deduction) and clauses earn nothing on a rejected answer, because rubric 3 showed an optimiser collects them without doing any arithmetic.

What each level means under the rules: 0.95 to 1.0 is an answer that would pass review (1.0 with exactly the governing clauses). Anything at or below -0.3 would be rejected under the rules. A rejected answer cannot score above a low floor whatever else it got right, and abstaining on a task the rules determine is penalised exactly like a wrong answer. A refusal of a determinable payslip scores -0.5, the same as a payslip with every computed field wrong; a payslip with some computed fields right scores up to -0.3, which is still a rejection. Calculating through a trap scores -0.5 with no partial credit.

Scripted baselines under rubric 8 on the v4 evaluation and training sets (reports/baselines-v4.json): always calculate with zeros -0.49, -0.49; copy the prompt's example -0.50, -0.50; always clarify with any single reason -0.34, -0.33 (identical across reasons, because v4 balances them); refuse every payslip but give every trap its correct reason +0.12, +0.20; right decision on every task with no arithmetic +0.13, +0.21; reference answers 0.96, 0.95. Refusing payslips and answering them wrongly score the same to within 0.01, and only correct arithmetic moves a policy above about +0.2.

### How the rubric changed

Every change answered an exploit GRPO found in the previous version. Saved answers were re-scored under each version with their recorded judge verdicts replayed, and earlier scores are kept beside each run as `scores-rubric-N.json`. Hosted rankings never changed.

- Rubric 3 gave inexact answers partial credit for every field and the clause list. Currency is constant, gross pay and the deduction are copied from the record, and calculation clauses are always R1 to R6, so the policy learned to answer `calculate` everywhere with copied inputs and the worked example's numbers, earning 0.08 to 0.13 per answer without arithmetic.
- Rubric 4 limited partial credit to computed fields. A wrong `retrieve` then scored the same 0 as a correct `calculate` with wrong amounts, and every group tied in 11 of 14 training steps.
- Rubric 5 added 0.05 for the right decision. The frozen model almost never clarified, so the credit was only reachable on calculation tasks, and both full runs ended up calculating through 99.5 to 100% of held-out traps.
- Rubric 6 paid only exact answers. A refusal and a wrong payslip then both scored 0, the 1B model never produced an exact payslip, and calculation tasks gave no signal in any of 23 steps; only traps did, so the policy learned to clarify on everything (false clarifications 100% from step 15).
- Rubric 7 kept rubric 6's rule that wrong answers never earn positive reward, and made wrong decisions and malformed answers cost -0.5 in either direction. It worked against refusal collapse, but it scored a refusal of a determinable payslip (-0.5) below a wrong payslip (0), which breaks the parity between refusing and answering wrongly, and it gave arithmetic no partial credit.
- Rubric 8 gives every rejected answer the same -0.5 floor and restores gated partial credit for computed fields (up to -0.3). Rubric 7's asymmetry existed only because a 1B model never produced an exact payslip, so refusing and answering wrongly needed different prices to keep the policy attempting payslips. With the calculator tool, exact payslips are reachable, so the parity rule can apply as written and an attempt beats a refusal through partial credit and exact answers, not through a penalty gap.

## Why each trap is a trap

Clarification is correct only where the rules do not determine an answer. Every trap task records its reason code, and each reason code has one basis:

| Reason code               | Basis                                                                                                                                            | Rules  |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------ |
| `MISSING_GROSS_PAY`       | A required input is absent                                                                                                                       | R1, R7 |
| `MISSING_PAYROLL_MONTH`   | A required input is absent                                                                                                                       | R1, R7 |
| `NO_APPLICABLE_REFERENCE` | The rules are silent: no reference row covers the month                                                                                          | R1, R7 |
| `CONFLICTING_GROSS_PAY`   | The records contradict each other and the rules give neither source precedence, because no approved, effective amendment resolves the difference | R7, R9 |

When the inputs are complete and consistent, or an approved amendment resolves the conflict under R9, the rules determine the payslip and clarifying scores -0.5.

## Clauses in the score

Every score returns its components (decision, exact, field accuracy, citations, and the explanation review on correct clarifications), each with the clauses that govern it, so a mark can be traced to the rule behind it.

## Explanation reward

The grader adds an explanation review only after a trap has the correct clarification decision and reason. Final trap reward is 0.90 times the deterministic score plus 0.10 times explanation quality. An empty explanation scores zero on explanation without a paid call. Nonempty explanations require the configured judge; judge failures are infrastructure errors, never silently scored as zero. In training, an answer the judge cannot grade is retried for 7.5 minutes, and if it still cannot be graded the run stops rather than learn from a partial reward. Wrong trap reasons and calculating through a trap both score -0.5, as above. Other tiers keep their deterministic scores. Exact passing additionally requires all three qualitative criteria.

Criteria and prompt are in judge.md, derived from R1, R7, and R9: identify the blocker, ask for relevant resolution, and avoid claiming an unsupported final answer. The small weight keeps a judge from overpowering programmatic correctness. The judge runs GLM 5.3 Flash at `high` reasoning effort. Its default, `max`, made 13 of 90 training-time calls spend all 8,192 tokens reasoning and return nothing on some providers. At `high`, 93 repeated calls returned no failures and matched the original verdicts on 29 of 30 examples (results/judge-reliability-high-effort). Historical deterministic-only scores remain in their original rubric versions and must not be mixed with this one.
