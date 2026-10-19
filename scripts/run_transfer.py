import concurrent.futures
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "deepseek": "deepseek/deepseek-v4.1-flash",
    "mimo": "xiaomi/mimo-v2.6-pro",
    "gemini": "google/gemini-3.1-flash-lite-preview",
    "glm": "z-ai/glm-5.3-flash",
    "luna-default": "openai/gpt-5.6-luna",
    "llama-1b": "meta-llama/llama-3.2-1b-instruct",
    "llama-3b": "meta-llama/llama-3.2-3b-instruct",
    "gemma-4b": "google/gemma-3-4b-it",
    "qwen-7b": "qwen/qwen-2.5-7b-instruct",
}


def run_model(item):
    name, model = item
    command = [sys.executable, "-u", "-m", "luma_gym.eval", "--model", model, "--tasks", "data/transfer-human.json",
               "--mode", "single", "--output", f"results/v4/transfer-human-{name}", "--execute"]
    if name == "luna-default":
        command.append("--provider-default-temperature")
    code = 1
    with (ROOT / "results/v4" / f"transfer-human-{name}.log").open("a") as log:
        for _ in range(3):
            code = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log).returncode
            if code == 0:
                break
    print(f"{name} {'completed' if code == 0 else 'stopped for review'}", flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(run_model, MODELS.items()))
