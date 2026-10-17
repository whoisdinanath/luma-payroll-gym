import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path

from run_v4_hosted import MODELS, ROOT


def run_model(item):
    name, model = item
    with (ROOT / "results" / "v4" / f"ablation-{name}.log").open("a") as log:
        command = [sys.executable, "-u", "-m", "luma_gym.eval", "--model", model, "--tasks", "data/v4/eval.json",
                   "--mode", "tools", "--no-calculator", "--output", f"results/v4/hosted-{name}-tools-nocalc", "--execute"]
        if name == "luna-default":
            command.append("--provider-default-temperature")
        for attempt in range(3):
            if subprocess.run(command, cwd=ROOT, stdout=log, stderr=log).returncode == 0:
                break
        print(f"{name} tools without calculator done", flush=True)


if __name__ == "__main__":
    while subprocess.run(["pgrep", "-f", "run_v4_hosted.py"], capture_output=True).returncode == 0:
        time.sleep(60)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(run_model, MODELS.items()))
