"""Progressive actual-PG clean-exact landing and source adoption, never index publication."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from citequest.data_v2 import models as m
from citequest.data_v2.landing import Landing
from citequest.data_v2.schemas import BatchInput, ProjectionInput, SnapshotFreezeInput, SnapshotInput
from citequest.data_v2.standard_seal import CANONICAL_CLEAN, QUERIES
from citequest.v2 import models as core
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor
from citequest.v2.services import Catalog, digest

ROOT = Path(__file__).resolve().parents[2]
EXPORT = ROOT / "data/s05/metadata-export/20261104T222945293337Z"
RAW = ROOT / "data/s05/litsearch/raw"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def source_rows(manifest: dict) -> Iterator[ProjectionInput]:
    for item in manifest["files"]:
        blob = (EXPORT / item["path"]).read_bytes()
        if len(blob) != item["bytes"] or hashlib.sha256(blob).hexdigest() != item["sha256"]:
            raise ValueError("Real source batch integrity mismatch")
        for line in blob.splitlines():
            yield ProjectionInput.model_validate_json(line)


def memory() -> dict:
    result = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            f"Get-Process -Id {os.getpid()} | Select-Object WorkingSet64,PeakWorkingSet64 | ConvertTo-Json -Compress",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode:
        return {"state": "OS_process_memory_measurement_failed", "exit_code": result.returncode}
    return {"state": "Windows_GetProcess_actual_controller_only", **json.loads(result.stdout)}


def verify_prefix(db: Session, snapshot_id: UUID, manifest: dict, count: int) -> dict:
    actual_count = db.scalar(
        select(func.count()).select_from(m.SourceRow).where(m.SourceRow.snapshot_id == snapshot_id)
    )
    if actual_count < count or actual_count > 64183 or (count == 64183 and actual_count != count):
        raise ValueError("Actual PG source count differs from milestone")
    statement = (
        select(m.SourceRow, m.SourceProjection)
        .join(m.SourceProjection, m.SourceProjection.source_row_id == m.SourceRow.id)
        .where(m.SourceRow.snapshot_id == snapshot_id)
        .order_by(m.SourceRow.artifact_path, m.SourceRow.row_ordinal)
        .limit(count)
    )
    rows = db.execute(statement).yield_per(500)
    canonical, empty, checked = hashlib.sha256(), 0, 0
    for (source, projection), expected in zip(rows, itertools.islice(source_rows(manifest), count), strict=True):
        pointer = digest(
            {
                "snapshot_manifest_sha256": manifest["source_snapshot_manifest_sha256"],
                "native_id": expected.native_id,
                "native_id_source_type": expected.native_id_source_type,
                **expected.locator.model_dump(mode="json"),
            }
        )
        expected_projection = digest(
            {
                "source_pointer_sha256": pointer,
                "representation_profile": "litsearch-clean-exact/1",
                "title": expected.title,
                "abstract": expected.abstract,
                "field_provenance": expected.field_provenance,
            }
        )
        if (
            source.native_id != expected.native_id
            or source.source_pointer_sha256 != pointer
            or any(getattr(source, key) != value for key, value in expected.locator.model_dump().items())
            or projection.title != expected.title
            or projection.abstract != expected.abstract
            or projection.field_provenance != expected.field_provenance
            or projection.projection_sha256 != expected_projection
        ):
            raise ValueError("Real SQL source/projection readback differs from original Arrow values/pointer")
        canonical.update(
            (
                json.dumps(
                    {"corpusid": int(source.native_id), "title": projection.title, "abstract": projection.abstract},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        )
        empty += int(not ((projection.title or "").strip() or (projection.abstract or "").strip()))
        checked += 1
    if checked != count or (count == 64183 and canonical.hexdigest() != CANONICAL_CLEAN):
        raise ValueError("Native source rows missing or full standard canonical hash changed")
    return {
        "rows": checked,
        "actual_snapshot_rows": actual_count,
        "actual_pointer_and_projection_hashes_matched": checked,
        "canonical_prefix_sha256": canonical.hexdigest(),
        "empty_representations_preserved": empty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Explicit authorized product import required")
    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    manifest = json.loads((EXPORT / "verification.json").read_bytes())
    if manifest["rows"] != 64183 or manifest["standard_canonical_sha256"] != CANONICAL_CLEAN:
        raise ValueError("Fixed original clean source required")
    policy, evaluation = sha(ROOT / "contracts/v2/policy.json"), sha(ROOT / "contracts/v2/evaluation.json")
    actor = Actor(uuid5(NAMESPACE_URL, "CiteQuest S05 pinned source importer"), publisher=True)
    request = SnapshotInput(
        source_namespace=manifest["source_namespace"],
        source_revision=manifest["source_revision"],
        source_manifest_sha256=manifest["source_snapshot_manifest_sha256"],
        policy_sha256=policy,
        evaluation_sha256=evaluation,
        representation_profile="litsearch-clean-exact/1",
        purpose="standard",
        artifacts=manifest["artifacts"],
        client_operation_id=uuid5(
            actor.owner_id, "register-clean-exact:" + manifest["source_snapshot_manifest_sha256"]
        ),
    )
    db_engine = engine(dsn("cq_s03_app"))
    factory = sessions(db_engine)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run = ROOT / "reports/stages/v2/S05/data-execution" / ("standard-import-" + stamp)
    save(
        run / "admission.json",
        {
            "state": "authorized_progressive_real_PG_import",
            "S04_dependency_manifest_sha256": "d7711475bdd9826769ffeb9d2f21682c66da3b1bb8b3697ca171a510f5b28493",
            "milestones": [100, 5000, 64183],
            "export_manifest_sha256": sha(EXPORT / "verification.json"),
            "policy_sha256": policy,
            "evaluation_sha256": evaluation,
            "actor_id": str(actor.owner_id),
            "application_frozen": False,
            "index_published": False,
            "model_calls": 0,
            "measurement_scope": "PG source landing, not Milvus/TEI/search SLA; index services may concurrently start",
            "script_sha256": sha(Path(__file__)),
        },
    )
    started = time.perf_counter()
    try:
        with factory.begin() as db:
            owner = db.get(core.Owner, actor.owner_id)
            if owner is None:
                db.add(core.Owner(id=actor.owner_id, name="S05 pinned LitSearch source importer", can_publish=True))
            elif not owner.can_publish:
                raise ValueError("Previously registered source importer lacks trusted publisher permission")
        with factory.begin() as db:
            landed = Landing(Catalog(db, actor), policy_sha256=policy, evaluation_sha256=evaluation).register_snapshot(
                request, RAW, public=True
            )
            snapshot_id = UUID(landed["snapshot_id"])
            initial_rows = db.scalar(
                select(func.count()).select_from(m.SourceRow).where(m.SourceRow.snapshot_id == snapshot_id)
            )
        save(
            run / "snapshot.json",
            {**landed, "importer_owner_id": str(actor.owner_id), "existing_rows_at_start": initial_rows},
        )
        stream, processed, milestones = iter(source_rows(manifest)), 0, []
        for target in (100, 5000, 64183):
            phase_start, phase_batches, phase_seconds = time.perf_counter(), 0, []
            while processed < target:
                take = min(500, target - processed)
                rows = list(itertools.islice(stream, take))
                if len(rows) != take:
                    raise ValueError("Native source stream ended early")
                batch = BatchInput(
                    rows=rows, client_operation_id=uuid5(snapshot_id, f"rows:{processed}:{processed + take}")
                )
                call_start = time.perf_counter()
                with factory.begin() as db:
                    Landing(Catalog(db, actor), policy_sha256=policy, evaluation_sha256=evaluation).ingest_batch(
                        snapshot_id, batch
                    )
                phase_seconds.append(time.perf_counter() - call_start)
                phase_batches += 1
                processed += take
            with factory.begin() as db:
                proof = verify_prefix(db, snapshot_id, manifest, target)
                database_bytes = db.scalar(select(func.pg_database_size(func.current_database())))
            result = {
                **proof,
                "phase_requested_rows": target - (milestones[-1]["rows"] if milestones else 0),
                "phase_expected_new_rows": max(
                    0, target - max(initial_rows, milestones[-1]["rows"] if milestones else 0)
                ),
                "phase_batches": phase_batches,
                "phase_wall_seconds_including_readback": time.perf_counter() - phase_start,
                "individual_batch_transaction_seconds": phase_seconds,
                "elapsed_seconds": time.perf_counter() - started,
                "database_bytes_total_including_other_stage_data": database_bytes,
                "controller_memory": memory(),
            }
            milestones.append(result)
            save(run / f"milestone-{target}.json", result)
            print(  # noqa: T201
                json.dumps(
                    {"milestone": target, "elapsed_seconds": result["elapsed_seconds"], "snapshot_id": str(snapshot_id)}
                ),
                flush=True,
            )
        with factory.begin() as db:
            sealed = Landing(
                Catalog(db, actor), policy_sha256=policy, evaluation_sha256=evaluation
            ).freeze_standard_snapshot(
                snapshot_id,
                SnapshotFreezeInput(
                    client_operation_id=uuid5(snapshot_id, "freeze-clean-exact"),
                    expected_export_manifest_sha256=sha(EXPORT / "verification.json"),
                    expected_queries_sha256=QUERIES,
                ),
                verified_export_dir=EXPORT,
                qrels_path=ROOT / "reports/stages/v2/S05/preparation/reconciliation-20261105-r1/queries-native.json",
            )
        save(run / "standard-seal.json", sealed)
        save(
            run / "verification.json",
            {
                "state": "standard_source_landed_and_sealed",
                "snapshot_id": str(snapshot_id),
                "milestones": milestones,
                "seal_sha256": sealed["seal_sha256"],
                "standard_rows": 64183,
                "original_queries": 597,
                "original_qrels_edges": 639,
                "source_ID_reachability": 1,
                "application_qualified_or_frozen": False,
                "index_published": False,
                "S05_gate_passed": False,
                "model_calls": 0,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
    except BaseException as exc:
        save(
            run / "stop.json",
            {
                "type": type(exc).__name__,
                "reason": str(exc)[:1000],
                "elapsed_seconds": time.perf_counter() - started,
                "partial_milestones_not_full_success": True,
            },
        )
        raise
    finally:
        db_engine.dispose()


if __name__ == "__main__":
    main()
