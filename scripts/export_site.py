import json
import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from luma_gym.dashboard import RESULTS, load_training
from luma_gym.server import app


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public"


def save(client, url, name):
    response = client.get(url)
    if response.status_code != 200:
        raise SystemExit(f"{url} returned {response.status_code}")
    target = OUTPUT / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(response.text)


if OUTPUT.exists():
    shutil.rmtree(OUTPUT)
client = TestClient(app)
save(client, "/", "index.html")
save(client, "/report", "report.html")
save(client, "/rules", "rules.html")
runs = load_training()[0]
for run in runs:
    save(client, f"/run/{run['id']}", f"run/{run['id']}.html")
folders = [(directory, directory.name) for directory in sorted(RESULTS.glob("comparison-*-*"))]
folders += [(directory, f"v4-{directory.name}") for pattern in ("hosted-*-*", "transfer-human-*")
            for directory in sorted((RESULTS / "v4").glob(pattern)) if directory.is_dir()]
for directory, run in folders:
    if directory.is_dir() and (directory / "scores.json").exists():
        for task in sorted({row["task_id"] for row in json.loads((directory / "scores.json").read_text())}):
            save(client, f"/transcript/{run}/{task}", f"transcript/{run}/{task}.html")
print(sum(1 for _ in OUTPUT.rglob("*.html")), "pages written to", OUTPUT)
