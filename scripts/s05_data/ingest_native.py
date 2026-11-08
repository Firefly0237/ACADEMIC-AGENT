"""Actual 3-to-692 audited native paper chains; source adoption remains false."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import func, select

from citequest.data_v2 import models as m
from citequest.data_v2.landing import Landing
from citequest.data_v2.native_index import read_for_index
from citequest.data_v2.native_registration import NativePaperRegistrar
from citequest.data_v2.schemas import SnapshotInput
from citequest.v2 import models as core
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor
from citequest.v2.services import Catalog
from citequest.v2.storage_config import app_storage

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "data/s05/native_artifacts-export/20261104T225527390394Z/verification.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main() -> None:
    if sys.argv[1:] != ["--execute"]:
        raise ValueError("Explicit authorized native-chain import required")
    sys.path.insert(0, str(ROOT / "scripts/s03"))
    from bootstrap import dsn

    if sha(ARTIFACTS) != "5e5fc1f06e5e0f24def8ce52efa2cdcc8f56c2ffa605035d169bc9c0b4320735":
        raise ValueError("Official native artifact metadata changed")
    artifacts = json.loads(ARTIFACTS.read_bytes())
    actor = Actor(uuid5(NAMESPACE_URL, "CiteQuest S05 pinned source importer"), publisher=True)
    policy, evaluation = sha(ROOT / "contracts/v2/policy.json"), sha(ROOT / "contracts/v2/evaluation.json")
    store = app_storage(ROOT).cas()
    db_engine = engine(dsn("cq_s03_app"))
    factory = sessions(db_engine)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run = ROOT / "reports/stages/v2/S05/data-execution" / ("native-import-" + stamp)
    save(
        run / "admission.json",
        {
            "state": "audited_source_identity_registration_not_application_freeze",
            "source_component_candidates": 692,
            "milestones": [3, 692],
            "original_candidate_order_preserved": True,
            "source_artefacts_sha256": sha(ARTIFACTS),
            "policy_sha256": policy,
            "evaluation_sha256": evaluation,
            "script_sha256": sha(Path(__file__)),
            "network_or_model_calls": 0,
            "no_synthetic_File_or_PDF_coordinates": True,
            "application_profile_adopted": False,
        },
    )
    started = time.perf_counter()
    try:
        with factory.begin() as db:
            request = SnapshotInput(
                source_namespace="litsearch_s2orc",
                source_revision="9573fb284a1026c998df47024b888a163f0f0e25",
                source_manifest_sha256=artifacts["source_snapshot_manifest_sha256"],
                policy_sha256=policy,
                evaluation_sha256=evaluation,
                representation_profile="litsearch-s2orc-native-text/1",
                purpose="application",
                artifacts=artifacts["artifacts"],
                client_operation_id=uuid5(
                    actor.owner_id, "register-native-source:" + artifacts["source_snapshot_manifest_sha256"]
                ),
            )
            result = Landing(Catalog(db, actor), policy_sha256=policy, evaluation_sha256=evaluation).register_snapshot(
                request, ROOT / "data/s05/litsearch/raw", public=True
            )
            snapshot_id = UUID(result["snapshot_id"])
        save(run / "snapshot.json", {**result, "importer_owner_id": str(actor.owner_id)})
        with factory() as db:
            landing = Landing(Catalog(db, actor), policy_sha256=policy, evaluation_sha256=evaluation)
            registrar = NativePaperRegistrar(landing, store, ROOT)
            candidates = sorted(registrar.candidates)
            outcomes = []
            for number, rank in enumerate(candidates, 1):
                with db.begin():
                    value = registrar.register(snapshot_id, rank)
                    source = read_for_index(landing, store, value["representation_id"])
                    if (
                        source.profile_adopted
                        or source.snapshot_state != "landing"
                        or str(source.work_version_id) != value["work_version_id"]
                    ):
                        raise ValueError("Source identity accidentally changed adoption/lineage")
                    outcomes.append(
                        {
                            "rank": rank,
                            "native_id": source.native_id,
                            "work_id": str(source.work_id),
                            "work_version_id": str(source.work_version_id),
                            "representation_id": source.representation_id,
                            "projection_id": str(source.projection_id),
                            "source_text_sha256": source.source_text_sha256,
                            "source_pointer_sha256": source.source_pointer_sha256,
                            "state": value["state"],
                            "profile_adopted": source.profile_adopted,
                        }
                    )
                if number in {3, 692}:
                    with db.begin():
                        native_count = db.scalar(
                            select(func.count())
                            .select_from(m.NativeRepresentation)
                            .join(m.SourceRow, m.SourceRow.id == m.NativeRepresentation.source_row_id)
                            .where(m.SourceRow.snapshot_id == snapshot_id)
                        )
                        resolved_count = db.scalar(
                            select(func.count())
                            .select_from(m.SourceResolution)
                            .join(m.SourceRow, m.SourceRow.id == m.SourceResolution.source_row_id)
                            .where(
                                m.SourceRow.snapshot_id == snapshot_id,
                                m.SourceResolution.state == "paper_identity_verified",
                            )
                        )
                        importer_files = db.scalar(
                            select(func.count()).select_from(core.File).where(core.File.owner_id == actor.owner_id)
                        )
                        if (
                            native_count < number
                            or native_count > 692
                            or resolved_count != native_count
                            or importer_files != 0
                        ):
                            raise ValueError("Actual native-chain counts or no-File guarantee differ")
                    save(
                        run / f"milestone-{number}.json",
                        {
                            "source_chains": number,
                            "native_representations": native_count,
                            "paper_identity_verified_resolutions": resolved_count,
                            "importer_File_count": importer_files,
                            "elapsed_seconds": time.perf_counter() - started,
                            "application_frozen": False,
                        },
                    )
                    print(  # noqa: T201
                        json.dumps(
                            {
                                "source_chains": number,
                                "snapshot_id": str(snapshot_id),
                                "elapsed_seconds": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
                elif number % 100 == 0:
                    save(
                        run / f"checkpoint-{number}.json",
                        {"processed_candidates": number, "elapsed_seconds": time.perf_counter() - started},
                    )
            save(run / "outcomes.json", outcomes)
            save(
                run / "verification.json",
                {
                    "state": "real_native_paper_identity_chains_registered",
                    "source_chains": len(outcomes),
                    "distinct_works": len({row["work_id"] for row in outcomes}),
                    "distinct_versions": len({row["work_version_id"] for row in outcomes}),
                    "snapshot_id": str(snapshot_id),
                    "source_profile_adopted": False,
                    "application_frozen": False,
                    "application_qualified_fulltexts": None,
                    "index_published": False,
                    "S05_gate_passed": False,
                    "network_or_model_calls": 0,
                    "elapsed_seconds": time.perf_counter() - started,
                    "outcomes_sha256": sha(run / "outcomes.json"),
                    "remaining": ["application stop-rule/profile decision and final import/index acceptance"],
                },
            )
    except BaseException as exc:
        save(
            run / "stop.json",
            {
                "type": type(exc).__name__,
                "reason": str(exc)[:1000],
                "elapsed_seconds": time.perf_counter() - started,
                "partial_chains_not_full_pass": True,
            },
        )
        raise
    finally:
        db_engine.dispose()


if __name__ == "__main__":
    main()
