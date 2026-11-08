"""Trusted supervisor's fenced failure CAS; never adopts a parser result.

Not an HTTP tool. The host must first prove exact local physical ownership and
stop, then mount that immutable proof into this narrow business-role helper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.engine import make_url

from citequest.business_storage_v3.contracts import StorageBinding
from citequest.materials_v2 import models as m
from citequest.materials_v2.processor import MaterialProcessor
from citequest.materials_v2.services import MaterialService
from citequest.platform_v3.blob_config import BlobStoreConfig
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor
from citequest.v2.models import File
from citequest.v2.services import digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--proof-sha256", required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--dsn", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert not args.out.exists()
    raw = args.proof.read_bytes()
    assert len(raw) <= 65536 and hashlib.sha256(raw).hexdigest() == args.proof_sha256
    proof = json.loads(raw)
    profile_raw = args.profile.read_bytes()
    profile = json.loads(profile_raw)
    storage_raw = args.storage.read_bytes()
    assert hashlib.sha256(storage_raw).hexdigest() == profile["storage_config_sha256"]
    binding = StorageBinding.configured(BlobStoreConfig.model_validate_json(storage_raw))
    # Reuse the actual fixed image's configuration fingerprint implementation;
    # no duplicated business hash algorithm, SDK construction or object read.
    configuration = SimpleNamespace(root=Path("/app"), parser_profile_key=profile["parser_profile_relative"],
        chunk_profile_key="runtime/s04-linux/chunk-worker-profile.json", business_storage=SimpleNamespace(binding=binding),
        extra_configuration_files=("runtime/material-parent/requirements.txt", "runtime/material-parent/dependency-provenance.json",
            "scripts/s04_linux/consume_private.py", "runtime/material-parent/execution-active.json"))
    expected_configuration = digest(MaterialProcessor._config(configuration))
    assert proof["version"] == "local-private-parent-stop/1" and proof["physical_stop_confirmed"] is True
    assert proof["profile_sha256"] == hashlib.sha256(profile_raw).hexdigest()
    assert proof["execution_id"] == profile["assigned_execution_id"] and proof["source_sha256"] == profile["assigned_source_sha256"]
    assert profile["schema_version"] == "cq.linux-material-parent-execution/3"
    assert proof["parent"]["running"] is False and all(child["running"] is False for child in proof["children"])
    dsn = args.dsn.read_text().strip()
    url = make_url(dsn)
    assert url.database == profile["database_name"] == "cq_s15_blob_20261106_r2" and url.username == profile["database_role"] == "cq_s08_business"
    database = engine(dsn, connect_timeout=3, pool_timeout=3)
    factory = sessions(database)
    result = {"status": "FAIL", "model_calls": 0, "main_database": False, "publication_permitted": False}
    try:
        with factory.begin() as db:
            db.execute(text("SET LOCAL statement_timeout='5s'"))
            db.execute(text("SET LOCAL lock_timeout='5s'"))
            actor = Actor(UUID(profile["assigned_owner_id"]))
            service = MaterialService(db, actor, None)
            service.catalog.lock_owner()
            row = db.scalar(select(m.MaterialIngestion).where(m.MaterialIngestion.id == UUID(profile["assigned_ingestion_id"])).with_for_update())
            file = db.get(File, UUID(profile["assigned_file_id"]))
            assert row and file and row.owner_id == file.owner_id == actor.owner_id
            assert row.file_id == file.id and file.sha256 == profile["assigned_source_sha256"]
            assert row.execution_id == UUID(profile["assigned_execution_id"]), "execution_fence_changed_no_reap"
            assert row.profile_sha256 == expected_configuration, "claimed_profile_fence_changed_no_reap"
            assert row.state not in {"ready", "partial"}, "published_result_requires_separate_ACK_reconciliation"
            assert row.lease_id in {None, row.execution_id}, "lease_fence_changed_no_reap"
            before = {"state": row.state, "execution_outcome": row.execution_outcome,
                      "error_code": row.error_code, "revision": row.revision, "budget_seconds": row.execution_budget_seconds}
            if row.execution_outcome in {"failed", "cancelled"} and row.state in {"failed", "cancelled"}:
                result["status"] = "already_failed_or_cancelled_no_new_dispatch"
            else:
                assert row.execution_outcome in {"running", "uncertain"}
                code = row.error_code or "PARENT_TERMINATED_BEFORE_PUBLICATION"
                if row.state not in {"failed", "cancelled"}:
                    service._advance(row, file, "failed", code)
                else:
                    row.revision += 1
                    db.add(m.MaterialEvent(ingestion_id=row.id, seq=row.revision, owner_id=row.owner_id,
                        state=row.state, occurred_at=db.execute(text("SELECT clock_timestamp()")).scalar_one(),
                        error_code="PARSER_STOP_CONFIRMED_AFTER_PARENT_EXIT"))
                row.execution_outcome = "cancelled" if row.state == "cancelled" else "failed"
                result["status"] = "failed_execution_closed_after_physical_stop"
            db.flush()
            assert row.execution_id == UUID(profile["assigned_execution_id"])
            assert row.execution_budget_seconds == before["budget_seconds"] and row.representation_id is None
            result.update(before=before, after={"state":row.state,"execution_outcome":row.execution_outcome,
                "error_code":row.error_code,"revision":row.revision,"budget_seconds":row.execution_budget_seconds},
                execution_id=str(row.execution_id), source_sha256=file.sha256, proof_sha256=args.proof_sha256,
                late_outputs_adopted=False, task_or_execution_window_reset=False)
        args.out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    finally:
        database.dispose()


if __name__ == "__main__":
    main()
