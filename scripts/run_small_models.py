import concurrent.futures
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "llama-1b": "meta-llama/llama-3.2-1b-instruct",
    "llama-3b": "meta-llama/llama-3.2-3b-instruct",
    "gemma-4b": "google/gemma-3-4b-it",
    "qwen-7b": "qwen/qwen-2.5-7b-instruct",
}


def run_model(item):
    name, model = item
    command = [sys.executable, "-u", "-m", "luma_gym.eval", "--model", model, "--tasks", "data/v4/eval.json",
               "--mode", "single", "--output", f"results/v4/hosted-{name}-single", "--execute"]
    with (ROOT / "results" / "v4" / f"hosted-{name}.log").open("a") as log:
        for attempt in range(3):
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log)
            if result.returncode == 0:
                break
    print(f"{name} {'completed' if result.returncode == 0 else 'stopped for review'}", flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_model, MODELS.items()))
