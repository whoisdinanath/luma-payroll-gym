import json
import statistics
from pathlib import Path


root = Path(__file__).resolve().parents[1]
results = []
for directory in sorted((root / "results").glob("transfer-*")):
    if not (directory / "scores.json").exists():
        continue
    scores = json.loads((directory / "scores.json").read_text())
    values = [row["score"]["score"] for row in scores if row.get("score")]
    if values:
        results.append({"model": json.loads((directory / "config.json").read_text())["model"],
                        "rollouts": len(values), "mean": statistics.mean(values),
                        "sd": statistics.stdev(values) if len(values) > 1 else 0})
matched_file = root / "reports/matched-comparison.json"
matched = json.loads(matched_file.read_text()) if matched_file.exists() else []
generated = {row["model"]: row["mean"] for row in matched}
pairs = []
complete = [row for row in results if row["rollouts"] == 15 and row["model"] in generated]
for index, left in enumerate(complete):
    for right in complete[index + 1:]:
        a = generated[left["model"]] - generated[right["model"]]
        b = left["mean"] - right["mean"]
        pairs.append({"models": [left["model"], right["model"]],
                      "result": "tie" if abs(a) < 1e-6 or abs(b) < 1e-6 else "concordant" if a * b > 0 else "discordant"})
report = {"tasks": 5, "results": results, "ranking_pairs": pairs}


def compare(rows):
    found = []
    for index, left in enumerate(rows):
        for right in rows[index + 1:]:
            a = left["generated_mean"] - right["generated_mean"]
            b = left["mean"] - right["mean"]
            found.append({"models": [left["model"], right["model"]],
                          "result": "tie" if abs(a) < 0.01 or abs(b) < 0.01 else "concordant" if a * b > 0 else "discordant"})
    return found


hand_written = []
for name in ("deepseek", "mimo", "gemini", "glm", "luna-default", "llama-1b", "llama-3b", "gemma-4b", "qwen-7b"):
    scores_path = root / "results/v4" / f"transfer-human-{name}" / "scores.json"
    generated_path = root / "results/v4" / f"hosted-{name}-single" / "metrics.json"
    if not scores_path.exists() or not generated_path.exists():
        continue
    values = [row["score"]["score"] for row in json.loads(scores_path.read_text()) if row.get("score")]
    if values:
        hand_written.append({"model": name, "answers": len(values), "mean": statistics.mean(values),
                             "sd": statistics.stdev(values) if len(values) > 1 else 0,
                             "generated_mean": json.loads(generated_path.read_text())["mean_reward"]})
if hand_written:
    tasks = json.loads((root / "data/transfer-human.json").read_text())
    report["hand_written"] = {"tasks": len(tasks), "tie_threshold": 0.01, "results": hand_written,
                              "ranking_pairs": compare(hand_written)}
(root / "reports/transfer.json").write_text(json.dumps(report, indent=2) + "\n")
lines = ["# Transfer test", "",
         "Five cases: a colleague's stale-rate suggestion, an approved amendment, pressure to use a draft amendment, a future amendment, and carrying last year's rates forward.", "",
         "| Model | Completed rollouts / 15 | Mean +/- SD |", "| --- | --- | --- |"]
for row in results:
    lines.append(f"| {row['model']} | {row['rollouts']} | {row['mean']:.4f} +/- {row['sd']:.4f} |")
lines += ["", "Pairwise ranking checks against the generated-task leaderboard are in transfer.json."]
(root / "reports/transfer.md").write_text("\n".join(lines) + "\n")
print(f"Transfer reports: {len(results)} models")
