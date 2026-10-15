import json
import os
import subprocess
from pathlib import Path

from luma_gym.budget import Budget
from luma_gym.hosted import load_key


root = Path(__file__).resolve().parents[1]
marker = root / "results/remote-budget.json"
if marker.exists():
    raise SystemExit("Judge budget already provisioned")
Budget().reserve(0.50)
marker.write_text(json.dumps({"reserved_usd": 0.50, "purpose": "Cloudflare sandbox judge ceiling"}, indent=2))
environment = dict(os.environ)
for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
    environment[name] = load_key(Path.home() / ".env", name)
secret = json.dumps({"OPENROUTER_API_KEY": load_key(Path.home() / ".env", "OPENROUTER_API_KEY")})
subprocess.run(["uv", "run", "pywrangler", "secret", "bulk"], input=secret, text=True,
               cwd=root / "deploy", env=environment, check=True)
subprocess.run(["uv", "run", "pywrangler", "d1", "execute", "luma-payroll-runs", "--remote", "--command",
                "UPDATE budget SET ceiling=0.50 WHERE id=1 AND ceiling=0"],
               cwd=root / "deploy", env=environment, check=True)
print("Sandbox judge enabled with a $0.50 ceiling")
