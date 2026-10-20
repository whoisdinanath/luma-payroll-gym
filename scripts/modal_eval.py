import json
import time
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parents[1]
BASE, REVISION = "openbmb/MiniCPM5-1B", "87179e5c1f455ef22e6223592d2d61351b525bfc"
ADAPTERS = "whoisdinanath/MiniCPM5-1B-Luma-Payroll-GRPO"
WORK = "/root/work"
MERGED_SHA256 = "e26ca657f9c9edaea8796743e900fc595a8dfd861a9cc9a1080e762748d93531"


def prepare():
    # The base model, the published adapters, and the merged SFT model built the way src/sft.py builds it.
    import hashlib
    import shutil

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    snapshot_download(BASE, revision=REVISION)
    AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    published = Path(snapshot_download(ADAPTERS, allow_patterns=["sft/adapter/*", "rl/rl-0.04/adapter/*",
                                                                 "rl/rl-0.001/adapter/*"]))
    results = Path(WORK) / "results/v4"
    shutil.copytree(published / "sft/adapter", results / "sft/adapter")
    for run in ("rl-0.04", "rl-0.001"):
        shutil.copytree(published / f"rl/{run}/adapter", results / run / "checkpoints/step-60")
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, results / "sft/adapter").merge_and_unload()
    merged = results / "sft/merged"
    model.save_pretrained(merged)
    adapter_sha = hashlib.sha256((results / "sft/adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    provenance = {"base": BASE, "revision": REVISION, "adapter": "results/v4/sft/adapter", "adapter_sha256": adapter_sha,
                  "merge": "bf16 base plus LoRA, merge_and_unload"}
    (merged / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("merged SFT model sha256 matches the local one:",
          hashlib.sha256((merged / "model.safetensors").read_bytes()).hexdigest() == MERGED_SHA256)


image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.14.0", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("transformers==5.19.0", "peft==0.21.0", "bitsandbytes==0.50.2", "accelerate==1.15.0",
                 "safetensors==0.8.0", "tokenizers==0.23.3", "huggingface_hub==1.33.0", "numpy==2.5.2")
    .env({"HF_HOME": "/root/hf", "TOKENIZERS_PARALLELISM": "false"})
    .run_function(prepare, cpu=8, memory=16384)
    .add_local_dir(ROOT / "src", "/root/luma_gym", ignore=["__pycache__"])
    .add_local_dir(ROOT / "rules", f"{WORK}/rules")
    .add_local_dir(ROOT / "data", f"{WORK}/data")
)
SIGNAL = ROOT / "results/v4/rl-0.04-signal/checkpoints/step-40"
if SIGNAL.exists():
    image = image.add_local_dir(SIGNAL, f"{WORK}/results/v4/rl-0.04-signal/checkpoints/step-40")
app = modal.App("luma-payroll-evals", image=image)
progress = modal.Volume.from_name("luma-payroll-confirmation", create_if_missing=True)


@app.function(gpu="L4", timeout=3 * 3600)
def evaluate(name, arguments):
    import os
    import subprocess

    output = Path("/tmp/out") / name
    command = ["python", "-u", "-m", "luma_gym.local", "--demos", "data/v4/demos.json", "--max-new-tokens", "384",
               "--limit", "1000", *arguments, "--output", str(output)]
    environment = {**os.environ, "PYTHONPATH": "/root", "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(command, cwd=WORK, env=environment, capture_output=True, text=True)
    files = {path.name: path.read_text() for path in output.glob("*")} if output.exists() else {}
    return name, result.returncode, (result.stdout + result.stderr)[-2000:], files


@app.function(gpu="L4", cpu=(1, 2), memory=(4096, 8192), timeout=3000, retries=0,
              volumes={"/progress": progress})
def confirm(name, arguments):
    import os
    import subprocess

    started = time.monotonic()
    output = Path("/progress") / name
    output.mkdir(parents=True, exist_ok=True)
    command = ["python", "-u", "-m", "luma_gym.local", "--demos", "data/v4/demos.json",
               "--max-new-tokens", "384", "--limit", "1000", *arguments, "--output", str(output)]
    environment = {**os.environ, "PYTHONPATH": "/root", "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "2"}
    with (output / "generation.log").open("a") as log:
        process = subprocess.Popen(command, cwd=WORK, env=environment, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None and time.monotonic() - started < 2800:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                progress.commit()
        timed_out = process.poll() is None
        if timed_out:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    metadata = {
        "seconds": time.monotonic() - started, "gpu": "L4", "exit_code": process.returncode,
        "timed_out": timed_out, "timeout_seconds": 3000, "automatic_retries": 0,
        "compute_reservation_usd": 4 / 3,
        "cost_note": "Reservation, not an invoice-confirmed charge; CPU capped at 2 cores and memory at 8 GiB.",
    }
    (output / "modal.json").write_text(json.dumps(metadata, indent=2) + "\n")
    progress.commit()
    files = {path.name: path.read_text() for path in output.iterdir() if path.is_file()}
    code = 124 if timed_out else process.returncode
    return name, code, files["generation.log"][-2000:], files


def model(run):
    if run == "raw":
        return []
    sft = ["--base", "results/v4/sft/merged"]
    return sft if run == "sft" else sft + ["--adapter", f"results/v4/{run}/checkpoints/step-60"]


def jobs():
    calculator = ["--mode", "calculator", "--rollouts", "4"]
    planned = []
    for run in ("raw", "sft", "rl-0.04", "rl-0.001"):
        planned.append((f"transfer-fixed-line-{run}-calculator",
                        calculator + ["--tasks", "data/transfer-fixed-line.json"] + model(run)))
        planned.append((f"test-{run}-calculator-noexamples",
                        calculator + ["--zero-shot", "--tasks", "data/v4/test.json"] + model(run)))
    for run in ("sft", "rl-0.04"):
        for tasks, prefix in (("data/transfer-human.json", "transfer-local"), ("data/transfer-fixed-line.json", "transfer-fixed-line")):
            planned.append((f"{prefix}-{run}-calculator-noexamples", calculator + ["--zero-shot", "--tasks", tasks] + model(run)))
        planned.append((f"eval-{run}-tools-examples", ["--mode", "tools", "--tool-demos", "--rollouts", "3", "--max-turns", "8",
                                                       "--tasks", "data/v4/eval.json"] + model(run)))
    signal = ["--base", "results/v4/sft/merged", "--adapter", "results/v4/rl-0.04-signal/checkpoints/step-40"]
    planned.append(("test-rl-0.04-signal-calculator", calculator + ["--tasks", "data/v4/test.json"] + signal))
    planned.append(("transfer-local-rl-0.04-signal-calculator", calculator + ["--tasks", "data/transfer-human.json"] + signal))
    planned.append(("transfer-fixed-line-rl-0.04-signal-calculator", calculator + ["--tasks", "data/transfer-fixed-line.json"] + signal))
    planned.append(("eval-rl-0.04-signal-tools-examples", ["--mode", "tools", "--tool-demos", "--rollouts", "3", "--max-turns", "8",
                                                            "--tasks", "data/v4/eval.json"] + signal))
    planned.append(("test-rl-0.04-signal-calculator-noexamples", calculator + ["--zero-shot", "--tasks", "data/v4/test.json"] + signal))
    planned.append(("transfer-local-rl-0.04-signal-calculator-noexamples",
                    calculator + ["--zero-shot", "--tasks", "data/transfer-human.json"] + signal))
    planned.append(("transfer-fixed-line-rl-0.04-signal-calculator-noexamples",
                    calculator + ["--zero-shot", "--tasks", "data/transfer-fixed-line.json"] + signal))
    planned.append(("eval-rl-0.04-signal-tools", ["--mode", "tools", "--rollouts", "3", "--max-turns", "8",
                                                  "--tasks", "data/v4/eval.json"] + signal))
    return planned


@app.local_entrypoint()
def main(only: str = "", fresh: bool = False):
    if fresh:
        if only:
            raise ValueError("Fresh confirmation runs all three fixed conditions; do not select by results")
        if not (ROOT / "data/v4/confirmation.json").exists():
            raise ValueError("Prepare the fixed fresh tasks with scripts/confirm_training.py first")
        planned = []
        for run in ("sft", "rl-0.04", "rl-0.04-signal"):
            arguments = (["--base", "results/v4/sft/merged", "--adapter",
                          "results/v4/rl-0.04-signal/checkpoints/step-40"] if run.endswith("-signal") else model(run))
            name = f"confirmation-{run}-calculator"
            if (ROOT / "results/v4" / name / "modal.json").exists():
                raise ValueError(f"{name} already ran; inspect saved progress before spending again")
            planned.append((name, ["--mode", "calculator", "--rollouts", "4", "--seed", "17",
                                   "--tasks", "data/v4/confirmation.json"] + arguments))
        runner = confirm
    else:
        planned = [job for job in jobs() if not only or job[0] in only.split(",")]
        runner = evaluate
    print(f"{len(planned)} evaluations")
    failed = []
    for name, code, log, files in runner.starmap(planned, order_outputs=False):
        target = ROOT / "results/v4" / name
        if files.get("generations.jsonl") or fresh:
            target.mkdir(parents=True, exist_ok=True)
            for file_name, text in files.items():
                (target / file_name).write_text(text)
        print(f"{name}: exit {code}, {len((files.get('generations.jsonl') or '').splitlines())} generations", flush=True)
        if code:
            print(log, flush=True)
            failed.append(name)
    if failed:
        raise RuntimeError("Incomplete evaluations: " + ", ".join(failed))
