import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from luma_gym.hosted import load_key
from luma_gym.local import MODEL, REVISION


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".hf-cache"
FILES = ["config.json", "generation_config.json", "model-00000-of-00001.safetensors", "model.safetensors.index.json",
         "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]

token = load_key(Path.home() / ".env", "HF_TOKEN")
folder = Path(snapshot_download(MODEL, revision=REVISION, cache_dir=CACHE, allow_patterns=FILES, token=token))
info = HfApi(token=token).model_info(MODEL, revision=REVISION, files_metadata=True)
expected = {sibling.rfilename: sibling.lfs.sha256 for sibling in info.siblings if sibling.lfs}
checks = {}
for name, digest in expected.items():
    sha = hashlib.sha256()
    with (folder / name).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            sha.update(block)
    checks[name] = sha.hexdigest() == digest
print(json.dumps({"snapshot": str(folder), "revision": REVISION, "sha256_matches": checks}, indent=2))
if not all(checks.values()):
    raise SystemExit("Downloaded weights do not match the published hash")
