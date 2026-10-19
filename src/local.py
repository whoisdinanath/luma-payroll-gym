import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from luma_gym.hosted import append_event, messages_for, read_events
from luma_gym.tasks import Task, validate_taskset


MODEL = "openbmb/MiniCPM5-1B"
REVISION = "87179e5c1f455ef22e6223592d2d61351b525bfc"
FORMAT_EXAMPLE = ({"gross_pay": "1000.00", "payroll_month": "2026-01"}, {
    "decision": "calculate", "currency": "LCR", "gross_pay": "1000.00",
    "employee_contribution": "50.00", "allowed_deduction": "0.00",
    "taxable_pay": "950.00", "income_tax": "0.00", "net_pay": "950.00",
    "clauses": ["R1", "R2", "R3", "R4", "R5", "R6"],
})
BALANCED_EXAMPLES = [
    ({"gross_pay": "1000.00", "payroll_month": "2026-01"}, {
        "decision": "calculate", "currency": "LCR", "gross_pay": "1000.00",
        "employee_contribution": "50.00", "allowed_deduction": "0.00", "taxable_pay": "950.00",
        "income_tax": "0.00", "net_pay": "950.00", "clauses": ["R1", "R2", "R3", "R4", "R5", "R6"],
    }),
    ({"payroll_month": "2026-01", "request": "contribution_rate"}, {
        "decision": "retrieve", "field": "contribution_rate", "value": "0.05", "clauses": ["R1", "R3"],
    }),
    ({"payroll_month": "2026-01"}, {
        "decision": "clarify", "reason_code": "MISSING_GROSS_PAY", "clauses": ["R1", "R7"],
        "explanation": "Gross pay is absent. Please provide the monthly gross pay before calculation.",
    }),
]


DECISION_EXAMPLES = [
    {"gross_pay": "2400.00", "payroll_month": "2026-03", "allowed_deduction": "100.00"},
    {"payroll_month": "2026-05"},
    {"gross_pay": "1500.00", "contract_gross_pay": "1400.00", "payroll_month": "2026-04",
     "amendment": {"gross_pay": "1500.00", "effective_month": "2026-02", "approved": True}},
    {"gross_pay": "1500.00", "contract_gross_pay": "1400.00", "payroll_month": "2026-04"},
    {"gross_pay": "1800.00"},
    {"gross_pay": "1800.00", "payroll_month": "2027-02"},
]
EXPLANATIONS = {
    "MISSING_GROSS_PAY": "Gross pay is missing. Please provide the monthly gross pay before calculation.",
    "MISSING_PAYROLL_MONTH": "The payroll month is missing, so no reference row can be chosen. Please provide the payroll month.",
    "NO_APPLICABLE_REFERENCE": "No reference row covers this payroll month. Please provide an authoritative schedule that covers it.",
    "CONFLICTING_GROSS_PAY": "Payroll and contract gross pay differ and no approved amendment resolves it. Please reconcile the records or provide an approved amendment.",
}


def decision_demonstrations():
    from luma_gym.reference import calculate_payroll

    messages = []
    for record in DECISION_EXAMPLES:
        answer = calculate_payroll(record)
        if answer["decision"] == "clarify":
            answer["explanation"] = EXPLANATIONS[answer["reason_code"]]
        messages.extend([
            {"role": "user", "content": "Example with the default 2026 reference rows: " + json.dumps(record)},
            {"role": "assistant", "content": json.dumps(answer)},
        ])
    return messages


def build_messages(task, rules, format_example=False, balanced_examples=False, decision_examples=False):
    messages = messages_for(task, rules)
    if decision_examples:
        messages[1:1] = decision_demonstrations()
    if format_example:
        record, answer = FORMAT_EXAMPLE
        messages[1:1] = [
            {"role": "user", "content": "Formatting example only. Calculate using default January rates: " + json.dumps(record)},
            {"role": "assistant", "content": json.dumps(answer)},
        ]
    if balanced_examples:
        demonstrations = []
        for record, answer in BALANCED_EXAMPLES:
            demonstrations.extend([
                {"role": "user", "content": "Example with default January rates: " + json.dumps(record)},
                {"role": "assistant", "content": json.dumps(answer)},
            ])
        messages[1:1] = demonstrations
    return messages


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    tokenizer.eos_token = "<|im_end|>"
    return tokenizer


def load_model(tokenizer, base=None):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    source = {"pretrained_model_name_or_path": str(base)} if base else {"pretrained_model_name_or_path": MODEL,
                                                                           "revision": REVISION}
    model = AutoModelForCausalLM.from_pretrained(
        **source, local_files_only=True, device_map={"": 0},
        dtype=torch.bfloat16, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                              bnb_4bit_use_double_quant=True,
                                              bnb_4bit_compute_dtype=torch.bfloat16),
    )
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model


def generate_with_demos(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers

    from luma_gym.rollout import TOOLS, prompt_ids, run_rollouts, task_messages

    torch.set_num_threads(4)
    tasks = [Task.from_dict(item) for item in json.loads(args.tasks.read_text())][:args.limit]
    demos = [Task.from_dict(item) for item in json.loads(args.demos.read_text())]
    validate_taskset(tasks)
    rules = Path("rules/payroll.md").read_text()
    tokenizer = load_tokenizer()
    tool_mode = args.mode == "tools"
    calculator = args.mode in ("calculator", "tools")
    tools = TOOLS if calculator else []
    if tool_mode:
        from luma_gym.documents import CaseDocuments, tool_definitions, tool_prompt

        tools = [{"type": "function", "function": definition} for definition in tool_definitions()]
    config = {
        "protocol": "4", "mode": args.mode, "model": MODEL, "revision": REVISION,
        "quantization": "NF4 double quantization, bf16 compute", "thinking": False,
        "temperature": 0.7, "top_p": 1.0, "seed": args.seed, "rollouts": args.rollouts,
        "max_new_tokens_per_turn": args.max_new_tokens, "max_turns": args.max_turns if calculator else 1,
        "tools": tools, "task_ids": [task.id for task in tasks],
        "taskset_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "demos_sha256": hashlib.sha256(args.demos.read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256(rules.encode()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__, "python_executable": sys.executable,
        "gpu": torch.cuda.get_device_name(),
    }
    if args.adapter:
        config["adapter"] = str(args.adapter)
        config["adapter_sha256"] = hashlib.sha256((args.adapter / "adapter_model.safetensors").read_bytes()).hexdigest()
    if args.base:
        config["base"] = json.loads((args.base / "provenance.json").read_text())
    if tool_mode:
        config["prompt"] = "the hosted tool-mode prompt and tools from the environment, with no demonstrations"
        del config["demos_sha256"]
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Protocol changed; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    done = {row["task_id"] for row in read_events(args.output / "generations.jsonl")}
    model = load_model(tokenizer, args.base)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    settings = {"temperature": 0.7, "max_new_tokens": args.max_new_tokens, "max_turns": args.max_turns,
                "batch": args.rollouts, "calculator": calculator}
    with (args.output / "generations.jsonl").open("a") as file:
        for task_index, task in enumerate(tasks):
            if task.id in done:
                continue
            messages = tool_prompt(task, rules) if tool_mode else task_messages(task, rules, demos, calculator)
            prompt = prompt_ids(tokenizer, messages, calculator, tools if tool_mode else None)
            if tool_mode:
                settings["cases"] = [CaseDocuments(task)] * args.rollouts
            seed = args.seed + task_index * 1000
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            states = run_rollouts(model, tokenizer, [prompt] * args.rollouts, settings, seed)
            seconds = time.monotonic() - start
            for rollout, state in enumerate(states):
                event = {
                    "id": f"{task.id}:{rollout}", "task_id": task.id, "tier": task.difficulty, "rollout": rollout,
                    "seed": seed, "final_answer": state["final"], "turns": state["turns"], "calls": state["calls"],
                    "calculator_errors": state["calculator_errors"], "truncated": state["truncated"],
                    "input_tokens": len(prompt), "output_tokens": sum(state["mask"]),
                    "environment_tokens": len(state["mask"]) - sum(state["mask"]),
                    "seconds": seconds / args.rollouts,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "prompt_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
                }
                if tool_mode:
                    event["tool_names"] = state["tool_names"]
                if rollout == 0:
                    event["messages"] = messages
                append_event(file, event)
            print(json.dumps({"task": task.id, "tier": task.difficulty, "calls": [state["calls"] for state in states],
                              "seconds": round(seconds, 1)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate MiniCPM answers for payroll tasks")
    parser.add_argument("--tasks", type=Path, default=Path("data/case-pilot.json"))
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--format-example", action="store_true")
    parser.add_argument("--balanced-examples", action="store_true")
    parser.add_argument("--decision-examples", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--demos", type=Path, help="Demonstration tasks; enables --mode, --max-turns, and --base")
    parser.add_argument("--mode", choices=["single", "calculator", "tools"], default="single")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--base", type=Path, help="Merged supervised checkpoint to start from")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.demos:
        return generate_with_demos(args)
    if min(args.limit, args.rollouts, args.max_new_tokens) <= 0:
        parser.error("--limit, --rollouts, and --max-new-tokens must be positive")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    from transformers import set_seed

    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU required")
    tasks = [Task.from_dict(item) for item in json.loads(args.tasks.read_text())][:args.limit]
    validate_taskset(tasks)
    rules = Path("rules/payroll.md").read_text()
    tokenizer = load_tokenizer()
    config = {
        "model": MODEL, "revision": REVISION, "quantization": "NF4 double quantization, bf16 compute",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "python_executable": sys.executable,
        "gpu": torch.cuda.get_device_name(), "temperature": 0.7, "top_p": 1.0,
        "seed": args.seed, "thinking": args.thinking, "max_new_tokens": args.max_new_tokens,
        "format_example": args.format_example,
        "balanced_examples": args.balanced_examples,
        "rollouts": args.rollouts, "task_ids": [task.id for task in tasks],
        "taskset_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256(rules.encode()).hexdigest(),
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "eos_token": tokenizer.eos_token, "eos_token_id": tokenizer.eos_token_id,
        "protocol": "single-turn generation only; raw output retained, shared rubric applied separately",
    }
    if args.decision_examples:
        config["decision_examples"] = True
    if args.adapter:
        weights = (args.adapter / "adapter_model.safetensors").read_bytes()
        config["adapter"] = str(args.adapter)
        config["adapter_sha256"] = hashlib.sha256(weights).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Protocol changed; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    existing = read_events(args.output / "generations.jsonl")
    done = {row["id"] for row in existing}
    if done >= {f"{task.id}:{rollout}" for task in tasks for rollout in range(args.rollouts)}:
        print(json.dumps({"already_complete": str(args.output), "answers": len(done)}), flush=True)
        return
    model = load_model(tokenizer)
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(json.dumps({"loaded": MODEL, "revision": REVISION, "gpu": config["gpu"],
                      "allocated_mib": torch.cuda.memory_allocated() / 2**20}), flush=True)
    with (args.output / "generations.jsonl").open("a") as file:
        for task_index, task in enumerate(tasks):
            messages = build_messages(task, rules, args.format_example, args.balanced_examples, args.decision_examples)
            inputs = tokenizer.apply_chat_template(messages, enable_thinking=args.thinking,
                                                   add_generation_prompt=True, return_dict=True,
                                                   return_tensors="pt").to("cuda")
            for rollout in range(args.rollouts):
                identity = f"{task.id}:{rollout}"
                if identity in done:
                    continue
                seed = args.seed + task_index * args.rollouts + rollout
                set_seed(seed)
                torch.cuda.reset_peak_memory_stats()
                start = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                            do_sample=True, temperature=0.7, top_p=1.0,
                                            use_cache=True, pad_token_id=tokenizer.pad_token_id)
                torch.cuda.synchronize()
                tokens = output[0, inputs.input_ids.shape[1]:]
                raw = tokenizer.decode(tokens, skip_special_tokens=False)
                text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
                if args.thinking:
                    text = text.split("</think>", 1)[1].strip() if "</think>" in text else ""
                event = {
                    "id": identity, "task_id": task.id, "tier": task.difficulty,
                    "rollout": rollout, "seed": seed, "messages": messages,
                    "raw_output": raw, "final_answer": text,
                    "input_tokens": inputs.input_ids.shape[1], "output_tokens": len(tokens),
                    "truncated": len(tokens) == args.max_new_tokens and int(tokens[-1]) != tokenizer.eos_token_id,
                    "seconds": time.monotonic() - start,
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    "prompt_sha256": hashlib.sha256(json.dumps(messages).encode()).hexdigest(),
                }
                append_event(file, event)
                print(json.dumps({key: event[key] for key in ("id", "output_tokens", "truncated", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
