import json
import re

from luma_gym.calculator import TOOL, response_text
from luma_gym.documents import tool_reply
from luma_gym.hosted import messages_for
from luma_gym.reference import reference_name


CALL = re.compile(r'<function name="([^"]*)">(.*?)</function>', re.S)
PARAM = re.compile(r'<param name="([^"]*)">(.*?)</param>', re.S)
CDATA = re.compile(r"\s*<!\[CDATA\[(.*?)\]\]>\s*", re.S)
TOOLS = [{"type": "function", "function": TOOL}]
NEXT_TURN = ("\n<|im_start|>user\n<tool_response>\n{reply}\n</tool_response><|im_end|>\n"
             "<|im_start|>assistant\n<think>\n\n</think>\n\n")
CALCULATOR_NOTE = (
    "\n\nA calculate tool does exact decimal arithmetic. For a payslip, call it once with the R3 to R6 computations "
    "using this case's reference row, then copy its results into the JSON answer. For a clarification, answer "
    "directly without the tool."
)
EXPLANATIONS = {
    "MISSING_GROSS_PAY": "Gross pay is missing. Please provide the monthly gross pay before calculation.",
    "MISSING_PAYROLL_MONTH": "The payroll month is missing, so no reference row can be chosen. Please provide the payroll month.",
    "NO_APPLICABLE_REFERENCE": "No reference row covers this payroll month. Please provide an authoritative schedule that covers it.",
    "CONFLICTING_GROSS_PAY": "Payroll and contract gross pay differ and no approved amendment resolves it. Please reconcile the records or provide an approved amendment.",
}


def program_for(record):
    row = record["reference_tables"][reference_name(record["payroll_month"])]
    deduction = record.get("allowed_deduction", "0.00")
    return "\n".join([
        f"gross = {record['gross_pay']}",
        f"contribution = round(min(gross, {row['cap']}) * {row['rate']}, 2)",
        f"taxable = gross - contribution - {deduction}",
        f"tax = round(min(max(taxable - {row['free']}, 0), {row['upper']} - {row['free']}) * {row['middle_rate']}"
        f" + max(taxable - {row['upper']}, 0) * {row['upper_rate']}, 2)",
        "net = taxable - tax",
    ])


def demo_messages(demos, calculator):
    messages = []
    for task in demos:
        messages.append({"role": "user", "content": task.prompt + "\nRecord: " + json.dumps(task.record)})
        answer = dict(task.ground_truth)
        if answer["decision"] == "clarify":
            answer["explanation"] = EXPLANATIONS[answer["reason_code"]]
        if calculator and answer["decision"] == "calculate":
            program = program_for(task.record)
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "calculate", "arguments": {"program": program}}}]})
            messages.append({"role": "tool", "content": response_text(program)})
        messages.append({"role": "assistant", "content": json.dumps(answer)})
    return messages


def task_messages(task, rules, demos, calculator):
    messages = messages_for(task, rules)
    if calculator:
        messages[0]["content"] += CALCULATOR_NOTE
    messages[1:1] = demo_messages(demos, calculator)
    return messages


def prompt_ids(tokenizer, messages, calculator, tools=None):
    return tokenizer.apply_chat_template(messages, tools=tools or (TOOLS if calculator else None), enable_thinking=False,
                                         add_generation_prompt=True, return_dict=True,
                                         return_tensors="pt").input_ids[0].tolist()


def parse_call(text):
    match = CALL.search(text)
    if not match:
        return None
    arguments = {}
    for name, value in PARAM.findall(match.group(2)):
        inner = CDATA.fullmatch(value)
        arguments[name] = inner.group(1) if inner else value
    return match.group(1), arguments


def generate(model, rows, settings, pad, eos, seed):
    import torch
    from transformers import set_seed

    width = max(len(row) for row in rows)
    ids = torch.tensor([[pad] * (width - len(row)) + row for row in rows], device="cuda")
    mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in rows], device="cuda")
    set_seed(seed)
    with torch.inference_mode():
        output = model.generate(input_ids=ids, attention_mask=mask, do_sample=True, temperature=settings["temperature"],
                                top_p=1.0, max_new_tokens=settings["max_new_tokens"], use_cache=True, pad_token_id=pad)
    results = []
    for row in output[:, width:].tolist():
        if eos in row:
            row = row[:row.index(eos) + 1]
        results.append(row)
    return results


def run_rollouts(model, tokenizer, prompts, settings, seed):
    # Mask 1 marks model tokens. Tool replies get mask 0, so they are never trained on.
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    states = [{"tokens": [], "mask": [], "turns": [], "done": False} for _ in prompts]
    for turn in range(settings["max_turns"]):
        active = [index for index, state in enumerate(states) if not state["done"]]
        for start in range(0, len(active), settings["batch"]):
            chunk = active[start:start + settings["batch"]]
            outputs = generate(model, [prompts[index] + states[index]["tokens"] for index in chunk], settings, pad, eos,
                               seed + 7919 * turn + start)
            for index, output in zip(chunk, outputs):
                state = states[index]
                state["tokens"] += output
                state["mask"] += [1] * len(output)
                # <function>, <param> and </function> are added special tokens: parse the raw text, answer the clean one.
                raw = tokenizer.decode(output, skip_special_tokens=False).replace(tokenizer.eos_token, "").strip()
                text = tokenizer.decode(output, skip_special_tokens=True).strip()
                ended = bool(output) and output[-1] == eos
                call = parse_call(raw) if settings["calculator"] else None
                if call is None or not ended or turn == settings["max_turns"] - 1:
                    state.update(done=True, final=text, truncated=not ended)
                    state["turns"].append({"text": raw})
                    continue
                name, arguments = call
                if settings.get("cases"):
                    reply = tool_reply(settings["cases"][index], name, arguments)
                else:
                    reply = (response_text(arguments.get("program")) if name == "calculate" and set(arguments) == {"program"}
                             else json.dumps({"error": "Unknown tool or arguments; use calculate(program)"}))
                environment = tokenizer.encode(NEXT_TURN.format(reply=reply), add_special_tokens=False)
                state["tokens"] += environment
                state["mask"] += [0] * len(environment)
                state["turns"].append({"text": raw, "call": name, "arguments": arguments, "reply": reply})
    for state in states:
        calls = [turn for turn in state["turns"] if "call" in turn]
        state["calls"] = len(calls)
        state["calculator_errors"] = sum('"error"' in turn["reply"] for turn in calls)
        state["tool_names"] = [turn["call"] for turn in calls]
    return states
