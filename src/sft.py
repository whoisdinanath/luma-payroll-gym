import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

from luma_gym.calculator import response_text
from luma_gym.hosted import append_event
from luma_gym.local import MODEL, REVISION, load_model, load_tokenizer
from luma_gym.rollout import EXPLANATIONS, NEXT_TURN, program_for, prompt_ids, task_messages
from luma_gym.tasks import Task, validate_taskset


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def example(task, tokenizer, rules, demos):
    eos = tokenizer.eos_token_id
    prompt = prompt_ids(tokenizer, task_messages(task, rules, demos, True), True)
    answer = dict(task.ground_truth)
    tokens, mask = [], []
    if answer["decision"] == "calculate":
        program = program_for(task.record)
        call = f'<function name="calculate"><param name="program"><![CDATA[{program}]]></param></function>'
        segment = tokenizer.encode(call, add_special_tokens=False) + [eos]
        reply = tokenizer.encode(NEXT_TURN.format(reply=response_text(program)), add_special_tokens=False)
        tokens += segment + reply
        mask += [1] * len(segment) + [0] * len(reply)
    else:
        answer["explanation"] = EXPLANATIONS[answer["reason_code"]]
    final = tokenizer.encode(json.dumps(answer), add_special_tokens=False) + [eos]
    return prompt, tokens + final, mask + [1] * len(final)


def loss_for(model, prompt, completion, mask):
    import torch

    ids = torch.tensor([prompt + completion], device="cuda")
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=len(completion) + 1,
                   use_cache=False).logits[0, :-1].float()
    target = ids[0, len(prompt):]
    keep = torch.tensor(mask, device="cuda", dtype=torch.bool)
    losses = torch.nn.functional.cross_entropy(logits, target, reduction="none")[keep]
    correct = (logits.argmax(-1) == target)[keep].float().mean()
    return losses.mean(), correct


def train(args):
    import peft
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model

    torch.set_num_threads(4)
    tasks = [Task.from_dict(item) for item in json.loads(args.tasks.read_text())][:args.limit]
    demos = [Task.from_dict(item) for item in json.loads(args.demos.read_text())]
    validate_taskset(tasks)
    rules = Path("rules/payroll.md").read_text()
    tokenizer = load_tokenizer()
    examples = [example(task, tokenizer, rules, demos) for task in tasks]
    config = {
        "stage": "supervised warm start for calculator tool use", "model": MODEL, "revision": REVISION,
        "quantization": "NF4 double quantization, bf16 compute", "labels": "Python reference and calculator only",
        "loss": "mean cross-entropy over the model's own turns; prompt and calculator replies masked",
        "lora": {"rank": args.rank, "alpha": 2 * args.rank, "dropout": 0.0, "targets": TARGETS},
        "optimizer": "AdamW, betas 0.9 and 0.99, no weight decay, gradient norm clipped to 1.0",
        "lr": args.lr, "warmup_steps": args.warmup, "epochs": args.epochs, "accumulation": args.accumulation,
        "tasks": str(args.tasks), "task_count": len(tasks),
        "taskset_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "demos_sha256": hashlib.sha256(args.demos.read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256(rules.encode()).hexdigest(), "seed": args.seed,
        "target_tokens": sum(sum(mask) for _, _, mask in examples),
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "python_executable": sys.executable,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    model = load_model(tokenizer)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    order = random.Random(args.seed)
    step = 0
    with (args.output / "steps.jsonl").open("a") as log:
        for epoch in range(args.epochs):
            indices = list(range(len(examples)))
            order.shuffle(indices)
            for start in range(0, len(indices), args.accumulation):
                began = time.monotonic()
                batch = indices[start:start + args.accumulation]
                model.train()
                losses, accuracies = [], []
                for index in batch:
                    loss, accuracy = loss_for(model, *examples[index])
                    (loss / len(batch)).backward()
                    losses.append(loss.item())
                    accuracies.append(accuracy.item())
                step += 1
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * min(1.0, step / args.warmup)
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0).item()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                record = {"step": step, "epoch": epoch + 1, "loss": sum(losses) / len(losses),
                          "token_accuracy": sum(accuracies) / len(accuracies), "grad_norm": norm,
                          "lr": optimizer.param_groups[0]["lr"], "seconds": round(time.monotonic() - began, 1),
                          "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)}
                append_event(log, record)
                print(json.dumps(record), flush=True)
    model.save_pretrained(args.output / "adapter", save_embedding_layers=False)


def merge(args):
    # RL starts from this merged model, so disabling the RL adapter recovers the SFT model as the KL reference.
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.output / "adapter").merge_and_unload()
    target = args.output / "merged"
    model.save_pretrained(target)
    weights = (args.output / "adapter" / "adapter_model.safetensors").read_bytes()
    (target / "provenance.json").write_text(json.dumps({
        "base": MODEL, "revision": REVISION, "adapter": str(args.output / "adapter"),
        "adapter_sha256": hashlib.sha256(weights).hexdigest(), "merge": "bf16 base plus LoRA, merge_and_unload",
    }, indent=2) + "\n")
    print(json.dumps({"merged": str(target)}))


def main():
    parser = argparse.ArgumentParser(description="Supervised warm start for MiniCPM calculator tool use")
    parser.add_argument("--tasks", type=Path, default=Path("data/v4/sft.json"))
    parser.add_argument("--demos", type=Path, default=Path("data/v4/demos.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--merge", action="store_true")
    args = parser.parse_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if args.merge:
        merge(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
