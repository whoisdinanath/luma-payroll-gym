import re
from decimal import Decimal, ROUND_HALF_UP


CENT = Decimal("0.01")
RETRIEVAL_VALUES = {
    "contribution_rate": ("0.05", "R3"),
    "contribution_cap": ("4000.00", "R3"),
    "deduction_cap": ("300.00", "R4"),
}
INITIAL_VERSION = "luma-payroll-v1"
TABLES = {
    "first-half": {"rate": "0.05", "cap": "4000.00", "deduction": "300.00",
                   "free": "1000.00", "upper": "3000.00", "middle_rate": "0.10", "upper_rate": "0.20"},
    "second-half": {"rate": "0.06", "cap": "4500.00", "deduction": "350.00",
                    "free": "1100.00", "upper": "3300.00", "middle_rate": "0.12", "upper_rate": "0.22"},
}


def reference_name(month, version="3"):
    # R1
    if version == INITIAL_VERSION:
        return "first-half"
    if version not in ("2", "3"):
        raise ValueError("Unsupported rule set")
    if month and month.startswith("2026-"):
        return "first-half" if month <= "2026-06" else "second-half"
    return None


def reference_table(month, version="3", tables=None):
    name = reference_name(month, version)
    if name is None:
        return None
    return {field: Decimal(value) for field, value in (tables or TABLES)[name].items()}


def valid_month(value):
    return isinstance(value, str) and bool(re.fullmatch(r"20[0-9]{2}-(0[1-9]|1[0-2])", value))


def money(value):
    # R2
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]{1,2})?", value):
        raise ValueError("Money must be a non-negative decimal string with at most two decimal places")
    return Decimal(value)


def rounded(value):
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def validate_record(record, version="3"):
    if not isinstance(record, dict):
        raise ValueError("Payroll record must be an object")
    allowed = {"gross_pay", "payroll_month", "allowed_deduction", "contract_gross_pay"}
    if version in ("2", "3"):
        allowed.add("amendment")
    if version == "3":
        allowed.add("reference_tables")
    if set(record) - allowed:
        raise ValueError("Unknown payroll record field")
    for field in ("gross_pay", "allowed_deduction", "contract_gross_pay"):
        if field in record and record[field] is not None:
            money(record[field])
    if "allowed_deduction" in record and record["allowed_deduction"] is None:
        raise ValueError("Allowed deduction cannot be null")
    if "contract_gross_pay" in record and record["contract_gross_pay"] is None:
        raise ValueError("Contract gross pay cannot be null")
    month = record.get("payroll_month")
    if month is not None and not valid_month(month):
        raise ValueError("Payroll month must be YYYY-MM between 2000 and 2099")
    if "amendment" in record:
        amendment = record["amendment"]
        if not isinstance(amendment, dict) or set(amendment) != {"gross_pay", "effective_month", "approved"}:
            raise ValueError("Invalid amendment fields")
        money(amendment["gross_pay"])
        if not valid_month(amendment["effective_month"]) or type(amendment["approved"]) is not bool:
            raise ValueError("Invalid amendment date or approval")
    tables = record.get("reference_tables")
    if tables is not None:
        if not isinstance(tables, dict) or set(tables) != set(TABLES):
            raise ValueError("Invalid reference schedule")
        for row in tables.values():
            if not isinstance(row, dict) or set(row) != set(TABLES["first-half"]):
                raise ValueError("Invalid reference row")
            values = {field: money(value) for field, value in row.items()}
            if any(values[field] >= 1 for field in ("rate", "middle_rate", "upper_rate")):
                raise ValueError("Rates must be below one")
            if not 0 < values["free"] < values["upper"] or values["cap"] <= 0:
                raise ValueError("Invalid reference boundaries")
    table = reference_table(month, version, tables)
    deduction = money(record.get("allowed_deduction", "0.00"))
    cap = table["deduction"] if table else max(Decimal(row["deduction"]) for row in (tables or TABLES).values())
    if deduction > cap:
        raise ValueError("Allowed deduction exceeds R4 cap")
    if record.get("gross_pay") is not None and table:
        gross = money(record["gross_pay"])
        contribution = rounded(min(gross, table["cap"]) * table["rate"])
        if deduction > gross - contribution:
            raise ValueError("Allowed deduction exceeds available pay under R4")


def calculate_payroll(record, version="3"):
    validate_record(record, version)
    # R7
    for field, reason in (
        ("gross_pay", "MISSING_GROSS_PAY"),
        ("payroll_month", "MISSING_PAYROLL_MONTH"),
    ):
        if record.get(field) is None:
            return {"decision": "clarify", "reason_code": reason, "clauses": ["R1", "R7"]}
    gross = money(record["gross_pay"])
    table = reference_table(record["payroll_month"], version, record.get("reference_tables"))
    if table is None:
        return {"decision": "clarify", "reason_code": "NO_APPLICABLE_REFERENCE", "clauses": ["R1", "R7"]}
    contract_gross = record.get("contract_gross_pay")
    resolved = False
    if contract_gross is not None and money(contract_gross) != gross:
        # R9
        amendment = record.get("amendment", {})
        resolved = bool(
            version in ("2", "3") and amendment.get("approved")
            and amendment["effective_month"] <= record["payroll_month"]
            and money(amendment["gross_pay"]) == gross
        )
        if not resolved:
            clauses = ["R7"] if version == INITIAL_VERSION else ["R7", "R9"]
            return {"decision": "clarify", "reason_code": "CONFLICTING_GROSS_PAY", "clauses": clauses}

    contribution = rounded(min(gross, table["cap"]) * table["rate"])  # R1, R3
    deduction = money(record.get("allowed_deduction", "0.00"))
    taxable = gross - contribution - deduction  # R4
    middle = min(max(taxable - table["free"], Decimal(0)), table["upper"] - table["free"])
    upper = max(taxable - table["upper"], Decimal(0))
    tax = rounded(middle * table["middle_rate"] + upper * table["upper_rate"])  # R1, R5
    amounts = {
        "gross_pay": gross,
        "employee_contribution": contribution,
        "allowed_deduction": deduction,
        "taxable_pay": taxable,
        "income_tax": tax,
        "net_pay": gross - contribution - deduction - tax,  # R6
    }
    clauses = ["R2", "R3", "R4", "R5", "R6"]
    if version in ("2", "3"):
        clauses.insert(0, "R1")
        if resolved:
            clauses.append("R9")
    return {
        "decision": "calculate",
        "currency": "LCR",
        **{field: format(rounded(value), ".2f") for field, value in amounts.items()},
        "clauses": clauses,
    }


def retrieve_rule(field, record=None, version="3"):
    if field not in RETRIEVAL_VALUES:
        raise ValueError("Unknown retrieval field")
    value, clause = RETRIEVAL_VALUES[field]
    if version == INITIAL_VERSION:
        return {"decision": "retrieve", "field": field, "value": value, "clauses": [clause]}
    record = record or {}
    validate_record(record, version)
    table = reference_table(record.get("payroll_month"), version, record.get("reference_tables"))
    if table is None:
        raise ValueError("Retrieval needs a covered payroll month")
    key = {"contribution_rate": "rate", "contribution_cap": "cap", "deduction_cap": "deduction"}[field]
    return {"decision": "retrieve", "field": field, "value": format(table[key], ".2f"), "clauses": ["R1", clause]}
