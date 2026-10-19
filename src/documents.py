import hashlib
import json

from luma_gym.calculator import TOOL as CALCULATOR, calculate
from luma_gym.reference import TABLES


class CaseDocuments:
    def __init__(self, task):
        record = task.record.copy()
        self.tables = record.pop("reference_tables", TABLES)
        amendment = record.pop("amendment", None)
        contract = record.pop("contract_gross_pay", None)
        token = hashlib.sha256(json.dumps([task.id, record], sort_keys=True).encode()).hexdigest()[:16]
        self.documents = {f"{token}-payroll": {"kind": "payroll", "record": record}}
        if contract is not None:
            self.documents[f"{token}-contract"] = {"kind": "contract", "gross_pay": contract}
        if amendment is not None:
            self.documents[f"{token}-amendment"] = {"kind": "amendment", **amendment}

    def list_documents(self):
        return [{"id": key, "kind": value["kind"]} for key, value in self.documents.items()]

    def read_document(self, document_id):
        if not isinstance(document_id, str) or document_id not in self.documents:
            raise ValueError("Document not found in this case")
        return self.documents[document_id]

    def lookup_reference(self, reference):
        if not isinstance(reference, str) or reference not in TABLES:
            raise ValueError("Unknown reference")
        months = {"first-half": ["2026-01", "2026-06"], "second-half": ["2026-07", "2026-12"]}
        return {"reference": reference, "months": months[reference], **self.tables[reference]}

    def calculate(self, program):
        return calculate(program)


def tool_prompt(task, rules, calculator=True):
    from luma_gym.hosted import messages_for

    visible_rules = "\n".join(line for line in rules.splitlines() if not line.startswith("|"))
    visible_rules = visible_rules.replace("up to 350.00", "up to the largest reference deduction cap")
    prompt = messages_for(task, visible_rules)
    prompt[1]["content"] = (task.prompt + "\nRetrieve the case inputs with list_documents and read_document. "
                            "Reference rows are available only through lookup_reference."
                            + (" Use calculate for exact decimal arithmetic." if calculator else ""))
    return prompt


def tool_demo_messages(demos):
    # Each demonstration as the tool-call trajectory the reference would take, one call per turn.
    from luma_gym.reference import reference_name
    from luma_gym.rollout import EXPLANATIONS, program_for

    messages = []
    for task in demos:
        case = CaseDocuments(task)
        messages.append({"role": "user", "content": tool_prompt(task, "")[1]["content"]})
        calls = [("list_documents", {})] + [("read_document", {"document_id": key}) for key in case.documents]
        answer = dict(task.ground_truth)
        if answer["decision"] == "calculate":
            calls += [("lookup_reference", {"reference": reference_name(task.record["payroll_month"])}),
                      ("calculate", {"program": program_for(task.record)})]
        else:
            answer["explanation"] = EXPLANATIONS[answer["reason_code"]]
        for name, arguments in calls:
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": name, "arguments": arguments}}]})
            messages.append({"role": "tool", "content": tool_reply(case, name, arguments)})
        messages.append({"role": "assistant", "content": json.dumps(answer)})
    return messages


def tool_reply(case, name, arguments):
    methods = {"list_documents": case.list_documents, "read_document": case.read_document,
               "lookup_reference": case.lookup_reference, "calculate": case.calculate}
    try:
        if not isinstance(arguments, dict) or name not in methods:
            raise ValueError("Invalid tool request")
        return json.dumps(methods[name](**arguments))
    except (ValueError, TypeError):
        return json.dumps({"error": "Invalid tool request"})


def tool_definitions():
    return [
        {"name": "list_documents", "description": "List source document IDs for this payroll case.",
         "parameters": {"type": "object", "properties": {}, "additionalProperties": False}},
        {"name": "read_document", "description": "Read one source document belonging to this case.",
         "parameters": {"type": "object", "properties": {"document_id": {"type": "string"}},
                        "required": ["document_id"], "additionalProperties": False}},
        {"name": "lookup_reference", "description": "Read first-half or second-half payroll reference rates and effective months.",
         "parameters": {"type": "object", "properties": {"reference": {
             "type": "string", "enum": ["first-half", "second-half"]}},
                        "required": ["reference"], "additionalProperties": False}},
        CALCULATOR,
    ]
