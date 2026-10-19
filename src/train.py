import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from luma_gym.budget import Budget
from luma_gym.hosted import append_event, load_key
from luma_gym.judge import HostedJudge, JudgeError, grade_answer
from luma_gym.local import MODEL, REVISION, build_messages, load_model, load_tokenizer
from luma_gym.reference import reference_name
from luma_gym.scoring import MONEY_FIELDS, RUBRIC_VERSION
from luma_gym.tasks import Task, validate_taskset


TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
OBJECTIVE = ("GRPO with PPO clipping: maximise the mean over informative groups of the mean over that group's answers of "
             "(1/|o|) sum_t [min(rho_t A, clip(rho_t, 1-e, 1+e) A) - beta k3(pi_theta, pi_ref)] for several epochs, "
             "with rho_t = pi_theta / pi_old using fixed old log-probabilities, "
             "A = (r - mean(r)) / (std(r) + 1e-4) over the group, and k3 = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1")


def pick(pool, position, seed):
    epoch, index = divmod(position, len(pool))
    order = list(range(len(pool)))
    random.Random(seed * 1000 + epoch).shuffle(order)
    return pool[order[index]]


def batch_for(step, calculations, traps, args):
    if args.balance_halves:
        pools = [[task for task in calculations if reference_name(task.record["payroll_month"]) == half]
                 for half in ("first-half", "second-half")]
        chosen = [pick(pools[index % 2], step * (args.calc_prompts // 2) + index // 2, args.seed + 10 + index % 2)
                  for index in range(args.calc_prompts)]
    else:
        chosen = [pick(calculations, step * args.calc_prompts + index, args.seed) for index in range(args.calc_prompts)]
    reasons = sorted(traps)
    for index in range(args.trap_prompts):
        position = step * args.trap_prompts + index
        reason = reasons[position % len(reasons)]
        chosen.append(pick(traps[reason], position // len(reasons), args.seed + 1 + reasons.index(reason)))
    return chosen


def sample(model, prompts, counts, args, eos, pad, seed):
    import torch
    from transformers import set_seed

    rows = [prompt for prompt, count in zip(prompts, counts) for _ in range(count)]
    completions = []
    model.eval()
    for start in range(0, len(rows), args.sample_batch):
        chunk = rows[start:start + args.sample_batch]
        width = max(len(row) for row in chunk)
        ids = torch.tensor([[pad] * (width - len(row)) + row for row in chunk], device="cuda")
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in chunk], device="cuda")
        set_seed(seed + start)
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=mask, do_sample=True, temperature=args.temperature,
                                    top_p=1.0, max_new_tokens=args.max_new_tokens, use_cache=True, pad_token_id=pad)
        for row in output[:, width:].tolist():
            if eos in row:
                row = row[:row.index(eos) + 1]
            completions.append(row)
    grouped, start = [], 0
    for count in counts:
        grouped.append(completions[start:start + count])
        start += count
    return grouped


def draw(model, tokenizer, prompts, counts, args, seed):
    eos, pad = tokenizer.eos_token_id, tokenizer.pad_token_id
    if not args.demos:
        return [[{"tokens": completion, "mask": [1] * len(completion), "turns": [], "calls": 0, "errors": 0,
                  "text": tokenizer.decode(completion, skip_special_tokens=True).strip(),
                  "truncated": not completion or completion[-1] != eos} for completion in group]
                for group in sample(model, prompts, counts, args, eos, pad, seed)]
    from luma_gym.rollout import run_rollouts

    model.eval()
    settings = {"temperature": args.temperature, "max_new_tokens": args.max_new_tokens, "max_turns": args.max_turns,
                "batch": args.sample_batch, "calculator": args.mode == "calculator"}
    states = run_rollouts(model, tokenizer, [prompt for prompt, count in zip(prompts, counts) for _ in range(count)],
                          settings, seed)
    grouped, start = [], 0
    for count in counts:
        grouped.append([{"tokens": state["tokens"], "mask": state["mask"], "text": state["final"],
                         "turns": state["turns"], "calls": state["calls"], "errors": state["calculator_errors"],
                         "truncated": state["truncated"]} for state in states[start:start + count]])
        start += count
    return grouped


def token_logps(model, prompt, completion, temperature):
    import torch

    tokens, mask = completion["tokens"], completion["mask"]
    ids = torch.tensor([prompt + tokens], device="cuda")
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=len(tokens) + 1,
                   use_cache=False).logits[0, :-1].float() / temperature
    keep = torch.tensor(mask, device="cuda", dtype=torch.bool)
    logits = logits[keep]
    target = ids[0, len(prompt):][keep]
    chosen = logits.gather(-1, target[:, None])[:, 0] - torch.logsumexp(logits, dim=-1)
    with torch.no_grad():
        logp = torch.log_softmax(logits.detach(), dim=-1)
        entropy = -(logp.exp() * logp).sum(-1)
    return chosen, entropy


def k3(log_ratio):
    import torch

    return torch.exp(log_ratio) - log_ratio - 1


def measure(model, sequences, args):
    import torch

    kl = entropy = tokens = 0.0
    model.eval()
    with torch.no_grad():
        for prompt, completion in sequences:
            logps, token_entropy = token_logps(model, prompt, completion, args.temperature)
            with model.disable_adapter():
                reference, _ = token_logps(model, prompt, completion, args.temperature)
            kl += k3(reference - logps).sum().item()
            entropy += token_entropy.sum().item()
            tokens += sum(completion["mask"])
    return kl / tokens, entropy / tokens


def update(model, optimizer, sequences, args):
    import torch

    model.eval()
    fixed = []
    with torch.no_grad():
        for prompt, completion, _, _ in sequences:
            old, _ = token_logps(model, prompt, completion, args.temperature)
            with model.disable_adapter():
                reference, _ = token_logps(model, prompt, completion, args.temperature)
            fixed.append((old, reference))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    epochs = []
    for epoch in range(args.epochs):
        model.train()
        totals = Counter()
        highest, lowest = 0.0, math.inf
        for (prompt, completion, advantage, weight), (old, reference) in zip(sequences, fixed):
            logps, _ = token_logps(model, prompt, completion, args.temperature)
            ratio = torch.exp(logps - old)
            clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
            surrogate = torch.minimum(ratio * advantage, clipped * advantage)
            kl = k3(reference - logps)
            loss = -(surrogate - args.beta * kl).mean() * weight
            loss.backward()
            with torch.no_grad():
                totals["tokens"] += sum(completion["mask"])
                totals["kl"] += kl.sum().item()
                totals["drift"] += k3(logps - old).sum().item()
                totals["ratio"] += ratio.sum().item()
                totals["clipped"] += (clipped * advantage < ratio * advantage).sum().item()
                totals["loss"] += loss.item()
                highest, lowest = max(highest, ratio.max().item()), min(lowest, ratio.min().item())
        stats = {"epoch": epoch + 1, "kl": totals["kl"] / totals["tokens"], "drift": totals["drift"] / totals["tokens"],
                 "ratio_mean": totals["ratio"] / totals["tokens"], "ratio_max": highest, "ratio_min": lowest,
                 "clip_fraction": totals["clipped"] / totals["tokens"], "loss": totals["loss"]}
        if epoch and stats["drift"] > args.target_drift:
            optimizer.zero_grad(set_to_none=True)
            epochs.append({**stats, "applied": False})
            break
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0).item()
        if not math.isfinite(grad_norm):
            raise ValueError("Non-finite gradient; checkpoint left unchanged")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        epochs.append({**stats, "grad_norm": grad_norm, "applied": True})
    return epochs


async def score(task, texts, log, prompt):
    from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI

    async with AsyncOpenAI(api_key=load_key(Path.home() / ".env", "OPENROUTER_API_KEY"),
                           base_url="https://openrouter.ai/api/v1", timeout=90, max_retries=0) as native:
        judge = HostedJudge(native, log, Budget())
        results = await asyncio.gather(*(grade_answer(task, text, judge, prompt=prompt) for text in texts),
                                       return_exceptions=True)
        for index, result in enumerate(results):
            for wait in (30, 120, 300):
                if not isinstance(result, Exception):
                    break
                append_event(log, {"event": "retry", "error": str(result)[:300], "wait_seconds": wait})
                await asyncio.sleep(wait)
                try:
                    result = await grade_answer(task, texts[index], judge, prompt=prompt)
                except (JudgeError, APIConnectionError, APIStatusError, APITimeoutError) as error:
                    result = error
            if isinstance(result, Exception):
                append_event(log, {"event": "unscored", "error": str(result)[:300], "explanation_text": texts[index][:600]})
                raise RuntimeError("The judge could not score an answer; stopping so no step trains on a partial reward") from result
            results[index] = result
        return results, judge.calls, judge.cost


def needs_more(group, symmetric=False):
    rewards = [result["score"] for result in group["results"] if result]
    if symmetric and rewards and all(result["passed"] for result in group["results"] if result):
        return False
    if len(set(rewards)) <= 1:
        return True
    return (symmetric or group["task"].difficulty == 3) and not any(reward > 0 for reward in rewards)


def decision(text):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return "invalid"
    return value.get("decision", "other") if isinstance(value, dict) else "invalid"


def save_checkpoint(model, optimizer, output, step, keep):
    import torch

    folder = output / "checkpoints"
    directory = folder / f"step-{step}"
    model.save_pretrained(directory, save_embedding_layers=False)
    torch.save(optimizer.state_dict(), directory / "optimizer.pt")
    (folder / "latest.tmp").write_text(str(step))
    os.replace(folder / "latest.tmp", folder / "latest")
    for old in folder.glob("step-*"):
        number = int(old.name.split("-")[1])
        if number != step and number % keep:
            shutil.rmtree(old)


def rate(rows, test):
    return statistics.mean(test(row) for row in rows) if rows else None


def main():
    parser = argparse.ArgumentParser(description="GRPO with PPO epochs and LoRA for MiniCPM5-1B on the payroll rubric")
    parser.add_argument("--tasks", type=Path, default=Path("data/training-splits/train.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--calc-prompts", type=int, default=2)
    parser.add_argument("--trap-prompts", type=int, default=2)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--max-group", type=int, default=16)
    parser.add_argument("--calc-max-group", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--target-drift", type=float, default=0.02)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--sample-batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--keep-every", type=int, default=20)
    parser.add_argument("--demos", type=Path, help="Demonstration tasks; enables --mode and --max-turns")
    parser.add_argument("--mode", choices=["single", "calculator"], default="single")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--base", type=Path, help="Merged SFT checkpoint to start from; also the KL reference")
    parser.add_argument("--balance-halves", action="store_true",
                        help="Draw calculation prompts evenly from first-half and second-half payroll months")
    args = parser.parse_args()
    if args.group < 4 or args.max_group < args.group or args.calc_max_group < args.group:
        parser.error("--group must be at least 4 and at most --max-group and --calc-max-group")
    if args.balance_halves and args.calc_prompts % 2:
        parser.error("--balance-halves needs an even --calc-prompts")
    if args.beta <= 0:
        parser.error("--beta must be positive")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import peft
    import torch
    import transformers
    from peft import LoraConfig, PeftModel, get_peft_model

    torch.set_num_threads(4)
    tasks = [Task.from_dict(item) for item in json.loads(args.tasks.read_text())]
    validate_taskset(tasks)
    trivial = sorted(task.id for task in tasks if task.ground_truth["decision"] == "calculate"
                     and all(task.ground_truth[field] == "0.00" for field in MONEY_FIELDS))
    calculations = [task for task in tasks if task.difficulty == 2 and task.id not in trivial]
    traps = {}
    for task in tasks:
        if task.difficulty == 3:
            traps.setdefault(task.ground_truth["reason_code"], []).append(task)
    rules = Path("rules/payroll.md").read_text()
    judge_prompt = Path("rules/judge.md").read_text()
    demos = [Task.from_dict(item) for item in json.loads(args.demos.read_text())] if args.demos else []
    calculator = args.mode == "calculator"
    tokenizer = load_tokenizer()
    config = {
        "algorithm": "GRPO with PPO epochs", "objective": OBJECTIVE, "model": MODEL, "revision": REVISION,
        "quantization": "NF4 double quantization, bf16 compute", "rubric_version": RUBRIC_VERSION,
        "reference": "frozen initial policy: the same NF4 weights with the LoRA adapter disabled",
        "kl_estimator": "k3", "beta": args.beta, "clip": args.clip, "epochs": args.epochs,
        "target_drift": args.target_drift, "old_policy": "log-probabilities fixed before the first epoch",
        "lora": {"rank": args.rank, "alpha": 2 * args.rank, "dropout": 0.0, "targets": TARGETS},
        "optimizer": "AdamW, betas 0.9 and 0.99, no weight decay, gradient norm clipped to 1.0, one step per epoch",
        "lr": args.lr, "steps": args.steps, "calc_prompts": args.calc_prompts, "trap_prompts": args.trap_prompts,
        "group": args.group, "max_group": args.max_group, "calc_max_group": args.calc_max_group,
        "sampling": "trap reasons rotate evenly; a group whose first answers all score the same gets more answers, "
                    "and so does a trap group without a correct answer; only groups with different rewards are trained on",
        "group_weighting": "every informative group carries the same weight, shared equally by its answers",
        "excluded_tasks": trivial, "temperature": args.temperature, "top_p": 1.0,
        "max_new_tokens": args.max_new_tokens, "prompt": "decision examples, thinking disabled",
        "tasks": str(args.tasks), "taskset_sha256": hashlib.sha256(args.tasks.read_bytes()).hexdigest(),
        "rules_sha256": hashlib.sha256(rules.encode()).hexdigest(),
        "judge_prompt_sha256": hashlib.sha256(judge_prompt.encode()).hexdigest(),
        "seed": args.seed, "torch": torch.__version__, "transformers": transformers.__version__,
        **({"protocol": "4", "mode": args.mode, "max_turns": args.max_turns,
            "prompt": "v4 interleaved demonstrations" + (", calculate tool" if calculator else ""),
            "demos_sha256": hashlib.sha256(args.demos.read_bytes()).hexdigest(),
            "expansion": ("on either tier, a group that ties below a full pass or has no positive reward gets more "
                          "answers; a group where every answer passes is skipped, as it has no gradient"),
            "base": json.loads((args.base / "provenance.json").read_text()) if args.base else MODEL,
            "reference": "frozen initial policy: the starting checkpoint with the RL adapter disabled"}
           if args.demos else {}),
        **({"calc_sampling": "each step draws calculation prompts evenly from first-half and second-half months"}
           if args.balance_halves else {}),
        "peft": peft.__version__, "python_executable": sys.executable, "gpu": torch.cuda.get_device_name(),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if {**previous, "steps": 0} != {**config, "steps": 0}:
            raise ValueError("Training protocol changed; use a new output directory")
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    model = load_model(tokenizer, args.base)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    latest = args.output / "checkpoints" / "latest"
    start = int(latest.read_text()) if latest.exists() else 0
    if start:
        model = PeftModel.from_pretrained(model, args.output / "checkpoints" / f"step-{start}", is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.0,
                                                 target_modules=TARGETS, task_type="CAUSAL_LM"))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, betas=(0.9, 0.99), weight_decay=0.0)
    if start:
        optimizer.load_state_dict(torch.load(args.output / "checkpoints" / f"step-{start}" / "optimizer.pt"))
    print(json.dumps({"start_step": start, "calculation_tasks": len(calculations),
                      "trap_tasks": {reason: len(pool) for reason, pool in sorted(traps.items())},
                      "excluded": trivial, "allocated_mib": round(torch.cuda.memory_allocated() / 2**20)}), flush=True)

    with (args.output / "steps.jsonl").open("a") as step_log, \
            (args.output / "samples.jsonl").open("a") as sample_log, \
            (args.output / "judge-transcript.jsonl").open("a") as judge_log:
        for step in range(start, args.steps):
            torch.cuda.reset_peak_memory_stats()
            began = time.monotonic()
            groups, judge_calls, judge_cost = [], 0, 0.0
            batch = batch_for(step, calculations, traps, args)
            if args.demos:
                from luma_gym.rollout import prompt_ids, task_messages

                prompts = [prompt_ids(tokenizer, task_messages(task, rules, demos, calculator), calculator) for task in batch]
            else:
                prompts = [tokenizer.apply_chat_template(build_messages(task, rules, decision_examples=True),
                                                         enable_thinking=False, add_generation_prompt=True, return_dict=True,
                                                         return_tensors="pt").input_ids[0].tolist() for task in batch]
            seed = args.seed * 100000 + step * 100
            for task, prompt, completions in zip(batch, prompts, draw(model, tokenizer, prompts, [args.group] * len(batch),
                                                                       args, seed)):
                texts = [completion["text"] for completion in completions]
                results, calls, cost = asyncio.run(score(task, texts, judge_log, judge_prompt))
                judge_calls, judge_cost = judge_calls + calls, judge_cost + cost
                groups.append({"task": task, "prompt": prompt, "completions": completions, "texts": texts,
                               "results": results})
            failed = [group for group in groups if needs_more(group, bool(args.demos))
                      and (args.max_group if group["task"].difficulty == 3 else args.calc_max_group) > args.group]
            if failed:
                extras = draw(model, tokenizer, [group["prompt"] for group in failed],
                              [(args.max_group if group["task"].difficulty == 3 else args.calc_max_group) - args.group
                               for group in failed], args, seed + 50)
                for group, completions in zip(failed, extras):
                    texts = [completion["text"] for completion in completions]
                    results, calls, cost = asyncio.run(score(group["task"], texts, judge_log, judge_prompt))
                    judge_calls, judge_cost = judge_calls + calls, judge_cost + cost
                    group["completions"] += completions
                    group["texts"] += texts
                    group["results"] += results
            for group in groups:
                rewards = [result["score"] for result in group["results"] if result]
                spread = statistics.pstdev(rewards) if len(rewards) > 1 else 0.0
                mean = statistics.mean(rewards) if rewards else 0.0
                group["informative"] = spread > 0
                group["advantages"] = [(result["score"] - mean) / (spread + 1e-4) if result and spread else 0.0
                                       for result in group["results"]]
            generated = time.monotonic()
            kl, entropy = measure(model, [(group["prompt"], completion) for group in groups
                                          for completion in group["completions"][:2]], args)
            measured = time.monotonic()
            trained_groups = [group for group in groups if group["informative"]]
            sequences = []
            for group in trained_groups:
                answers = [(completion, advantage) for completion, advantage, result
                           in zip(group["completions"], group["advantages"], group["results"]) if result]
                sequences += [(group["prompt"], completion, advantage, 1 / (len(trained_groups) * len(answers)))
                              for completion, advantage in answers]
            epochs = update(model, optimizer, sequences, args) if sequences else []
            trained = time.monotonic()

            first = [(group, index) for group in groups for index in range(args.group) if group["results"][index]]
            calc_rows = [row for row in first if row[0]["task"].difficulty == 2]
            trap_rows = [row for row in first if row[0]["task"].difficulty == 3]
            record = {
                "step": step + 1, "beta": args.beta, "lr": args.lr,
                "reward": rate(first, lambda row: row[0]["results"][row[1]]["score"]),
                "pass_rate": rate(first, lambda row: row[0]["results"][row[1]]["passed"]),
                "pass_rate_by_tier": {"2": rate(calc_rows, lambda row: row[0]["results"][row[1]]["passed"]),
                                      "3": rate(trap_rows, lambda row: row[0]["results"][row[1]]["passed"])},
                "exact_calculation": rate(calc_rows, lambda row: row[0]["results"][row[1]]["breakdown"]["exact"]["score"] == 1),
                "correct_clarification": rate(trap_rows, lambda row: row[0]["results"][row[1]]["status"] == "scored"),
                "false_calculation": rate(trap_rows, lambda row: decision(row[0]["texts"][row[1]]) == "calculate"),
                "false_clarification": rate(calc_rows, lambda row: decision(row[0]["texts"][row[1]]) == "clarify"),
                "decisions": {tier: dict(Counter(decision(row[0]["texts"][row[1]]) for row in rows))
                              for tier, rows in (("2", calc_rows), ("3", trap_rows))},
                "informative": {tier: sum(group["informative"] for group in groups if group["task"].difficulty == level)
                                for tier, level in (("2", 2), ("3", 3))},
                "tied_groups": statistics.mean(not group["informative"] for group in groups),
                "samples": sum(len(group["completions"]) for group in groups),
                "expanded": sum(len(group["completions"]) > args.group for group in groups),
                "trained_sequences": len(sequences), "kl": kl, "entropy": entropy,
                "completion_tokens": rate(first, lambda row: sum(row[0]["completions"][row[1]]["mask"])),
                "truncated": sum(completion["truncated"] for group in groups for completion in group["completions"]),
                "tool_use": {"2": rate(calc_rows, lambda row: row[0]["completions"][row[1]]["calls"] > 0),
                             "3": rate(trap_rows, lambda row: row[0]["completions"][row[1]]["calls"] > 0)},
                "calculator_errors": sum(completion["errors"] for group in groups for completion in group["completions"]),
                "epochs": epochs, "judge_calls": judge_calls, "judge_cost_usd": judge_cost,
                "unscored": sum(result is None for group in groups for result in group["results"]),
                "seconds": {"generate": round(generated - began, 1), "measure": round(measured - generated, 1),
                            "train": round(trained - measured, 1)},
                "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20),
            }
            for group in groups:
                for index, completion in enumerate(group["completions"]):
                    result = group["results"][index] or {"score": None, "status": "unscored", "passed": None}
                    append_event(sample_log, {
                        "step": step + 1, "task_id": group["task"].id, "tier": group["task"].difficulty,
                        "reason": group["task"].ground_truth.get("reason_code"), "index": index,
                        "stage": "first" if index < args.group else "extra", "text": group["texts"][index],
                        "tokens": sum(completion["mask"]), "truncated": completion["truncated"], "reward": result["score"],
                        "calls": completion["calls"], "turns": completion["turns"] if args.demos else None,
                        "status": result["status"], "passed": result["passed"], "decision": decision(group["texts"][index]),
                        "advantage": group["advantages"][index], "informative": group["informative"],
                    })
            append_event(step_log, record)
            save_checkpoint(model, optimizer, args.output, step + 1, args.keep_every)
            last = epochs[-1] if epochs else {}
            print(json.dumps({"step": record["step"], "reward": round(record["reward"], 4),
                              "informative": record["informative"], "samples": record["samples"],
                              "kl": round(kl, 5), "entropy": round(entropy, 4),
                              "clip": round(last.get("clip_fraction", 0), 4), "drift": round(last.get("drift", 0), 5),
                              "epochs": len(epochs), "seconds": record["seconds"]}), flush=True)


if __name__ == "__main__":
    main()
