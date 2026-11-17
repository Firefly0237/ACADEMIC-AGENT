"""Build a small explicit parser context without source documents or credentials."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_METADATA = ROOT / "assets/tokenizer/model-download.json"
MODEL_IMAGE_PATH = Path("reports/stages/v2/S02/parser/models-initial/model-download.json")
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--name", default="context-r2")
args = parser.parse_args()
if not args.name.startswith("context-") or "/" in args.name or "\\" in args.name or ".." in args.name:
    raise ValueError("Owned context basename required")
target = ROOT / "runtime/s04-linux/.state" / args.name
if target.exists():
    raise ValueError("New context required")
files = [
    *sorted((ROOT / "src/citequest").rglob("*.py")),
    *sorted((ROOT / "contracts").rglob("*.json")),
    *sorted((ROOT / "runtime/s04").glob("*.json")),
    *(
        ROOT / "runtime/s04-linux" / name
        for name in [
            "Dockerfile",
            "pyproject.toml",
            "uv.lock",
            "requirements.txt",
            "worker-profile.json",
            "chunk-worker-profile.json",
        ]
    ),
    *sorted((ROOT / "scripts/s04_linux").glob("*.py")),
    ROOT / "runtime/v2/uv.lock",
    ROOT / "runtime/s04-app/uv.lock",
    MODEL_METADATA,
    ROOT / "data/runtime-s04/tokenizer/tokenizer.json",
]
manifest = {}
for path in files:
    if not path.is_file() or path.is_symlink() or path.name == ".gitignore":
        continue
    raw = path.read_bytes()
    relative = MODEL_IMAGE_PATH if path == MODEL_METADATA else path.relative_to(ROOT)
    dest = target / relative
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    manifest[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
(target / "Dockerfile").write_bytes((ROOT / "runtime/s04-linux/Dockerfile").read_bytes())
(target / "source-manifest.json").write_text(json.dumps(manifest, indent=2))
print(  # noqa: T201
    json.dumps(
        {
            "files": len(manifest),
            "bytes": sum((target / p).stat().st_size for p in manifest),
            "credentials_included": False,
        }
    )
)
