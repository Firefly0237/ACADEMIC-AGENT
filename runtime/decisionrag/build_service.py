"""Build only from exact existing local image IDs, with no package download."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tag", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
path = Path(__file__).with_name("service-runtime.lock.json")
manifest = json.loads(path.read_bytes())
for name in ("CPU_base", "ASGI_source"):
    record = manifest[name]
    actual = subprocess.check_output(["docker", "image", "inspect", record["tag"], "--format", "{{.Id}}"], text=True).strip()
    if actual != record["image_id"]:
        raise RuntimeError("existing_local_base_digest_mismatch:" + name)
lock = root / manifest["CPU_base"]["lock"]
if hashlib.sha256(lock.read_bytes()).hexdigest() != manifest["CPU_base"]["lock_sha256"]:
    raise RuntimeError("CPU_role_lock_changed")
subprocess.run(["docker", "build", "--pull=false", "--network=none",
    "--build-arg", "CPU_IMAGE=" + manifest["CPU_base"]["image_id"],
    "--build-arg", "ASGI_IMAGE=" + manifest["ASGI_source"]["image_id"], "--file", str(path.with_name("Dockerfile")),
    "--tag", args.tag, str(root)], check=True)
image_id = subprocess.check_output(["docker", "image", "inspect", args.tag, "--format", "{{.Id}}"], text=True).strip()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"status": "built_from_verified_local_bases", "image_id": image_id,
    "manifest": manifest, "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "Dockerfile_sha256": hashlib.sha256(path.with_name("Dockerfile").read_bytes()).hexdigest()}, indent=2) + "\n")
