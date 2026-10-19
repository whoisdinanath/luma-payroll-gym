import json
import statistics
from pathlib import Path


root = Path(__file__).resolve().parents[1]
MODELS = ("deepseek", "mimo", "gemini", "glm", "luna-default", "llama-1b", "llama-3b", "gemma-4b", "qwen-7b")
TIE = 0.01

rows = []
for name in MODELS:
    scores_path = root / "results/v4" / f"transfer-human-{name}" / "scores.json"
    generated_path = root / "results/v4" / f"hosted-{name}-single" / "metrics.json"
    if not scores_path.exists() or not generated_path.exists():
        continue
    values = [row["score"]["score"] for row in json.loads(scores_path.read_text()) if row.get("score")]
    rows.append({"model": name, "answers": len(values), "mean": statistics.mean(values),
                 "sd": statistics.stdev(values) if len(values) > 1 else 0,
                 "generated_mean": json.loads(generated_path.read_text())["mean_reward"]})
pairs = []
for index, left in enumerate(rows):
    for right in rows[index + 1:]:
        a = left["generated_mean"] - right["generated_mean"]
        b = left["mean"] - right["mean"]
        pairs.append({"models": [left["model"], right["model"]],
                      "result": "tie" if abs(a) < TIE or abs(b) < TIE else "concordant" if a * b > 0 else "discordant"})
tasks = json.loads((root / "data/transfer-human.json").read_text())
report = {"hand_written": {"tasks": len(tasks), "tie_threshold": TIE, "results": rows, "ranking_pairs": pairs}}
(root / "reports/transfer.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"{len(rows)} models, {len(pairs)} pairs")
