"""Trusted developer queue orchestrator, never a web/API service."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID

from sqlalchemy.engine import make_url

from citequest.materials_v2.processor import MaterialProcessor
from citequest.materials_v3.broker_client import BrokerParser
from citequest.v2.db import engine, sessions
from citequest.v2.storage_config import app_storage

ROOT = Path("/app")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-id", type=UUID, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--parser-profile", type=Path, default=ROOT / "runtime/s04-linux/worker-profile.json")
    args = parser.parse_args()
    if args.receipt.exists() or not args.receipt.resolve().is_relative_to(Path("/evidence")):
        raise ValueError("New owned receipt required")
    raw = args.profile.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.profile_sha256:
        raise ValueError("Exact approved execution profile required")
    profile = json.loads(raw)
    if profile["schema_version"] != "cq.linux-parser-execution/1" or profile["product_api_docker_access"] is not False:
        raise ValueError("Trusted non-API orchestrator profile required")
    if str(args.file_id) != profile["assigned_file_id"] or profile["broker_mode"] != "fixed-host-cli-no-socket":
        raise ValueError("Exact assigned File and restricted host broker required")
    parser_profile = args.parser_profile.resolve()
    expected_relative = profile.get("parser_profile_relative", "runtime/s04-linux/worker-profile.json")
    if not parser_profile.is_relative_to(ROOT) or parser_profile.relative_to(ROOT).as_posix() != expected_relative:
        raise ValueError("Exact trusted parser release profile required")
    for relative, expected in profile["source_bindings"].items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Captured parser image source drift")
    dependency_files = profile.get("dependency_lock_files", ["runtime/s04-linux/uv.lock"])
    if not isinstance(dependency_files, list) or not dependency_files:
        raise ValueError("Explicit dependency lock list required")
    for relative in dependency_files:
        path = (ROOT / relative).resolve()
        if (
            not path.is_relative_to(ROOT / "runtime")
            or path.name not in {"uv.lock", "requirements.txt", "pyproject.toml"}
            or relative not in profile["source_bindings"]
        ):
            raise ValueError("Trusted image dependency locks required")
    dsn = Path("/run/secrets/material_app_dsn").read_text().strip()
    url = make_url(dsn)
    if url.username != "cq_s03_app" or url.host != "citequest-v2-s03-postgres-1" or url.database != "citequest_s03":
        raise ValueError("Trusted main app secretref required")
    database = engine(dsn)
    try:
        artifact_root = ROOT / "data/runtime-s04-linux/artifacts"
        runner = BrokerParser(root=ROOT, exchange=ROOT / profile["exchange_relative"], nonce=profile["nonce"])
        processor = MaterialProcessor(
            sessions(database),
            app_storage().cas(),
            root=ROOT,
            parser_python=Path("/usr/local/bin/python"),
            artifact_root=artifact_root,
            runner=runner,
            parser_profile=parser_profile,
            chunk_profile=ROOT / "runtime/s04-linux/chunk-worker-profile.json",
            extra_configuration_files=(
                *dependency_files,
                "src/citequest/materials_v3/broker_client.py",
                "src/citequest/materials_v3/seccomp.py",
                "scripts/s04_linux/worker_entry.py",
                args.profile.resolve().relative_to(ROOT).as_posix(),
            ),
        )
        result = processor.run_next(file_id=args.file_id)
        args.receipt.write_text(
            json.dumps(
                {
                    "status": "observed",
                    "result": result,
                    "profile_sha256": args.profile_sha256,
                    "image_id": profile["image_id"],
                    "scope": "Nonroot Linux PG queue parent without Docker socket; fixed host CLI broker and isolated no-network child; no cloud SLA",
                },
                indent=2,
            )
        )
    finally:
        database.dispose()
        exchange = ROOT / profile["exchange_relative"]
        (exchange / "parent-done.json").write_text(
            json.dumps({"file_id": str(args.file_id), "profile_sha256": args.profile_sha256})
        )


if __name__ == "__main__":
    main()
