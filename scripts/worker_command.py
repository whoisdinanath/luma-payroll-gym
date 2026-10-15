import os
import shutil
import subprocess
import sys
from pathlib import Path

from luma_gym.hosted import load_key


root = Path(__file__).resolve().parents[1]
deploy = root / "deploy"
shutil.copytree(root / "src/luma_gym", deploy / "luma_gym", dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
for name in ("payroll.md", "judge.md"):
    shutil.copyfile(root / "rules" / name, deploy / name)
texts = "RULES = " + repr((root / "rules/payroll.md").read_text()) + "\nJUDGE = " + repr((root / "rules/judge.md").read_text()) + "\n"
(deploy / "rule_text.py").write_text(texts)
env = dict(os.environ)
for name in ("CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"):
    env[name] = load_key(Path.home() / ".env", name)
result = subprocess.run(["uv", "run", "pywrangler", *sys.argv[1:]], cwd=deploy, env=env)
raise SystemExit(result.returncode)
