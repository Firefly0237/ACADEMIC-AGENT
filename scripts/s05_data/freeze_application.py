"""Internal trusted-source qualification CLI; no upstream/model calls or endpoint."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from citequest.data_v2 import models as m
from citequest.data_v2.application_contract import EVALUATION, POLICY, SNAPSHOT, ApplicationFreezeInput
from citequest.data_v2.application_freeze import freeze_application, qualify_application
from citequest.data_v2.landing import Landing
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor
from citequest.v2.services import Catalog, digest
from citequest.v2.storage_config import app_storage

ROOT = Path(__file__).resolve().parents[2]


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("qualify", "freeze"))
    parser.add_argument("--request", type=Path)
    args = parser.parse_args()
    if (args.action == "freeze") != bool(args.request):
        parser.error("freeze requires strict --request JSON; qualification accepts no request")
    request = ApplicationFreezeInput.model_validate_json(args.request.read_bytes()) if args.request else None
    run = (
        ROOT
        / "reports/stages/v2/S05/application-freeze"
        / (args.action + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    )
    files = []
    paths = sorted(
        {
            *ROOT.glob("src/citequest/data_v2/*.py"),
            *ROOT.glob("src/citequest/v2/*.py"),
            *ROOT.glob("src/citequest/updates_v2/*.py"),
            ROOT / "src/citequest/source_refs_v2.py",
            Path(__file__),
            ROOT / "contracts/v3/application-freeze-profile.json",
            ROOT / "docs/decisions/ADR-0015-application-corpus-stop-rule-adoption.md",
            ROOT / "migrations/app/versions/s05_data_0003_application_freeze.py",
            ROOT / "migrations/app/env.py",
        }
    )
    for path in paths:
        target = run / "source" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        files.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    save(run / "source-manifest.json", {"files": files, "pinned_before_product_reads": True})
    save(
        run / "admission.json",
        {
            "action": args.action,
            "read_only_PG": args.action == "qualify",
            "source_manifest_sha256": sha(run / "source-manifest.json"),
            "network_or_model_calls": 0,
            "runtime": {
                name: importlib.metadata.version(name) for name in ("sqlalchemy", "alembic", "psycopg", "pydantic")
            },
        },
    )
    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    db_engine = engine(dsn("cq_s03_app"))
    started = time.perf_counter()
    try:
        with sessions(db_engine).begin() as db:
            if args.action == "qualify":
                db.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
            head = db.execute(text("SELECT version_num FROM cq_app.alembic_version")).scalar_one()
            allowed_heads = {"s05_index_0002", "s05_data_0003"} if request is None else {"s05_data_0003"}
            if head not in allowed_heads:
                raise RuntimeError("Application freeze CLI database head differs from reviewed admission")
            snapshot = db.get(m.SourceSnapshot, SNAPSHOT)
            before = {
                "status": snapshot.status,
                "adopted": snapshot.profile_adopted,
                "seal_sha256": snapshot.seal_sha256,
                "policy_sha256": snapshot.policy_sha256,
                "importer_owner_id": str(snapshot.importer_owner_id),
            }
            landing = Landing(
                Catalog(db, Actor(snapshot.importer_owner_id, publisher=True)),
                policy_sha256=POLICY,
                evaluation_sha256=EVALUATION,
            )
            store = app_storage(ROOT).cas()
            result = (
                qualify_application(landing, store, ROOT)
                if request is None
                else freeze_application(landing, store, ROOT, request)
            )
            # Mutation aborts before commit if any reviewed source drifted.
            if any(sha(ROOT / row["path"]) != row["sha256"] for row in files):
                raise RuntimeError("Application qualification implementation changed during execution")
            save(run / "result.json", result)
            save(
                run / "verification.json",
                {
                    "state": "readonly_qualification_complete" if request is None else "atomic_application_adoption",
                    "database_head": head,
                    "before": before,
                    "result_sha256": sha(run / "result.json"),
                    "qualification_sha256": digest(result)
                    if request is None
                    else request.expected_qualification_sha256,
                    "source_manifest_sha256": sha(run / "source-manifest.json"),
                    "elapsed_seconds": time.perf_counter() - started,
                    "read_only_PG": request is None,
                    "network_or_model_calls": 0,
                    "S05_or_S09_gate_passed": False,
                },
            )
        save(run / "transaction.json", {"committed": True, "read_only_PG": request is None})
        print(  # noqa: T201
            json.dumps(
                {
                    "run": run.relative_to(ROOT).as_posix(),
                    "qualification_sha256": digest(result)
                    if request is None
                    else request.expected_qualification_sha256,
                    "qualified": result["coverage"]["qualified"] if request is None else result["qualified"],
                }
            )
        )
    finally:
        db_engine.dispose()


if __name__ == "__main__":
    main()
