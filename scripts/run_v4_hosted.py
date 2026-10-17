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
}


def run_model(item):
    name, model = item
    with (ROOT / "results" / "v4" / f"hosted-{name}.log").open("a") as log:
        for mode in ("single", "tools"):
            command = [sys.executable, "-u", "-m", "luma_gym.eval", "--model", model, "--tasks", "data/v4/eval.json",
                       "--mode", mode, "--output", f"results/v4/hosted-{name}-{mode}", "--execute"]
            if name == "luna-default":
                command.append("--provider-default-temperature")
            for attempt in range(3):
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=log)
                if result.returncode == 0:
                    break
            print(f"{name} {mode} {'completed' if result.returncode == 0 else 'stopped for review'}", flush=True)


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run_model, MODELS.items()))
