"""Explicit CI build/certify commands; importing or --help performs no resource IO."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from citequest.exports_v3.release_build import (  # noqa: E402
    certify_release, file_sha256, sha256, source_files, write_json,
)


def execute(command: list[str], *, timeout: int, output: Path | None = None) -> bytes:
    try:
        result = subprocess.run(command, check=False, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        if output is not None:
            output.write_bytes((error.stdout or b"") + b"\n" + (error.stderr or b""))
        raise
    if output is not None:
        output.write_bytes(result.stdout + b"\n" + result.stderr)
    result.check_returncode()
    return result.stdout


def build_candidate(workspace: Path, output: Path, docker: str) -> dict:
    workspace = workspace.resolve(strict=True)
    if output.exists():
        raise FileExistsError("fresh_candidate_output_required")
    output.mkdir(parents=True)
    output = output.resolve()
    for path in (workspace, output):
        if "," in str(path):
            raise ValueError("docker_bind_path_contains_comma")
    sources = source_files(workspace)
    iidfile = output / "image-id.txt"
    execute([docker, "build", "--platform", "linux/amd64", "--iidfile", str(iidfile),
        "--file", str(workspace / "runtime/s13/Dockerfile"), str(workspace)],
        timeout=2400, output=output / "build.log")
    image_id = iidfile.read_text(encoding="utf8").strip()
    if re.fullmatch(r"sha256:[a-f0-9]{64}", image_id) is None:
        raise ValueError("docker_build_actual_image_id_required")
    actual = execute([docker, "image", "inspect", image_id, "--format", "{{.Id}}"], timeout=15).decode().strip()
    if actual != image_id:
        raise ValueError("actual_built_image_id_mismatch")
    isolated = [docker, "run", "--rm", "--network", "none", "--read-only", "--user", "10001:10001",
        "--cpus", "2", "--memory", "2g", "--pids-limit", "128", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m"]
    described = json.loads(execute([*isolated, "--entrypoint", "python", image_id,
        "scripts/s13/bundle_entry.py", "--describe-toolchain"], timeout=30))
    identity = described["identity"]
    if (described["sha256"] != sha256(identity) or identity.get("source_files_sha256") != sources
        or identity.get("source_sha256") != sha256(sources)):
        raise ValueError("actual_image_source_toolchain_fingerprint_mismatch")
    write_json(output / "toolchain-identity.json", identity)
    layout = output / "layout"
    layout.mkdir(mode=0o777)
    if os.name != "nt":
        layout.chmod(0o777)
    execute([*isolated, "--mount", "type=bind,source=" + str(layout) + ",target=/outputs",
        "--entrypoint", "python", image_id, "scripts/s13/capacity_entry.py", "--image-id", image_id],
        timeout=240, output=output / "capacity.log")
    execute([docker, "image", "save", "--output", str(output / "renderer-image.tar"), image_id], timeout=300)
    if source_files(workspace) != sources:
        raise ValueError("export_source_changed_during_build")
    candidate = {"schema_version": "citequest-export-release-candidate/1", "status": "awaiting_certification",
        "image_id": image_id, "toolchain_sha256": described["sha256"],
        "source_sha256": sha256(sources), "source_files_sha256": sources,
        "artifacts": {path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(output.rglob("*")) if path.is_file()}, "actual_model_requests": 0}
    write_json(output / "candidate.json", candidate)
    return {"status": candidate["status"], "image_id": image_id, "toolchain_sha256": described["sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[2])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Explicitly build, render the physical fixture and archive a candidate")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--docker", default="docker")
    certify = commands.add_parser("certify", help="Assemble a release only from matching actual certification")
    certify.add_argument("--candidate", type=Path, required=True)
    certify.add_argument("--certification", type=Path, required=True)
    certify.add_argument("--certification-sha256", required=True)
    certify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build_candidate(args.workspace, args.output, args.docker)
    else:
        result = certify_release(args.workspace.resolve(strict=True), args.candidate,
            args.certification, args.certification_sha256, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
