"""Fixed adopted snapshots only: preflight or existing coordinator publication.

No migration, encoding, update-seal publication or arbitrary generation input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from citequest.index_v2 import models as index
from citequest.index_v2.native import RPCAudit
from citequest.index_v2.publication import AliasTransport, PublicationCoordinator, publication_alias
from citequest.index_v3.models import BackendAdoption
from citequest.index_v3.publication import FencedAliasCalls, PublicationSessionFence
from citequest.index_v3.repository import FullBuildRepository
from citequest.search_v3.native import SearchNative
from citequest.v2 import models as core
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor
from citequest.v2.services import Catalog, digest
from citequest.v2.storage_config import app_storage

ROOT = Path("/app")
ACTOR = UUID("eeee445c-f7d2-5e86-9ac0-322731ac6fd1")
BACKEND = "1f22bfeb2283e657265f8a6c5529418c3f525e7e8b5246ea33b74f88eb9dd409"
GENERATIONS = [UUID("06e9f1d2-f3d3-4c23-854a-af629d05d72b"), UUID("98f269a4-8dbd-47aa-8935-99140350cc38")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--profile-sha256")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(Path("/evidence")):
        raise ValueError("new_owned_publication_report_required")
    args.output.mkdir(parents=True)
    policy_raw = (ROOT / "contracts/v2/policy.json").read_bytes()
    evaluation_raw = (ROOT / "contracts/v2/evaluation.json").read_bytes()
    policy, evaluation = hashlib.sha256(policy_raw).hexdigest(), hashlib.sha256(evaluation_raw).hexdigest()
    dsn = Path("/run/secrets/index_migrator_dsn" if args.execute else "/run/secrets/index_app_dsn").read_text().strip()
    parsed = make_url(dsn)
    expected_role = "cq_s03_migrator" if args.execute else "cq_s03_app"
    if parsed.username != expected_role or parsed.database != "citequest_s03" or parsed.host != "citequest-v2-s03-postgres-1" or parsed.port != 5432:
        raise ValueError("trusted_main_development_role_required")
    database, store = engine(dsn), app_storage(ROOT).cas()
    factory, actor = sessions(database), Actor(ACTOR, publisher=True)
    audit = RPCAudit(args.output / "rpc.jsonl")
    native = SearchNative(uri="http://citequest-s05-index-milvus-1:19530", tei_info_url="http://tei-strict-gpu:8080/info",
                          function_endpoint="http://tei-strict-gpu:8080", backend_profile_sha256=BACKEND, audit=audit)

    def repository(db: Session) -> FullBuildRepository:
        return FullBuildRepository(Catalog(db, actor), store=store, policy_sha256=policy, evaluation_sha256=evaluation)

    def tx(operation: Callable) -> Any:
        with factory.begin() as db:
            return operation(repository(db))

    report = {"status": "started", "execute": args.execute, "fixed_generations": list(map(str, GENERATIONS)),
              "policy_sha256": policy, "evaluation_sha256": evaluation, "provider_calls": 0,
              "update_publication_authorized": False, "DDL_calls": 0, "generations": []}
    destination = args.output / "verification.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    try:
        with factory.begin() as db:
            if not args.execute:
                db.execute(text("SET TRANSACTION READ ONLY"))
            revision = db.execute(text("SELECT version_num FROM cq_app.alembic_version")).scalar_one()
            owner = db.get(core.Owner, ACTOR, populate_existing=True)
            if owner is None or not owner.can_publish or revision != "s08_agent_0001":
                raise ValueError("exact_main_schema_and_existing_publisher_required")
            report["main_revision"] = revision
            for identifier in GENERATIONS:
                repo = repository(db)
                generation = repo.generation(identifier)
                repo._publication_ready(generation)
                adoption = db.get(BackendAdoption, identifier, populate_existing=True)
                if adoption is None or adoption.backend_profile_sha256 != BACKEND or adoption.quality_report_sha256 != "7456de497dbb784b57c416351c36f043382f3a28cbadc776eea79d7f6a41d682":
                    raise ValueError("actual_adopted_fixed_backend_required")
                catalog = db.get(index.IndexCatalog, (generation.catalog_name, "public"), populate_existing=True)
                row = {"generation_id": str(identifier), "catalog": generation.catalog_name,
                       "collection": generation.collection_name, "state": generation.state,
                       "units": generation.unit_count, "chunks": generation.chunk_count,
                       "manifest_sha256": generation.expected_manifest_sha256,
                       "catalog_before": None if catalog is None else {"active_generation_id": str(catalog.active_generation_id) if catalog.active_generation_id else None,
                                                                        "epoch": catalog.epoch, "pending_intent_id": str(catalog.pending_intent_id) if catalog.pending_intent_id else None},
                       "backend": adoption.backend_profile_sha256,
                       "readback_sha256": digest(generation.readback_proof), "source_seal_sha256": generation.source_seal_sha256}
                if generation.state != "verified" or (catalog and (catalog.active_generation_id is not None or catalog.epoch != 0 or catalog.pending_intent_id is not None)):
                    raise ValueError("first_publication_verified_generation_and_empty_catalog_required")
                if db.scalar(select(index.IndexPublishIntent.id).where(index.IndexPublishIntent.target_generation_id == identifier).limit(1)):
                    raise ValueError("existing_intent_requires_readonly_reconciliation_not_new_publication")
                native.verify(generation.collection_name, generation.server_proof, dense=True)
                alias = publication_alias(generation.catalog_name, "public")
                row["alias"] = alias
                row["alias_before"] = AliasTransport(native).target(alias)
                if row["alias_before"] is not None:
                    raise ValueError("first_publication_alias_must_be_absent")
                report["generations"].append(row)
        if args.execute:
            if args.profile is None or args.profile_sha256 is None:
                raise ValueError("exact_approved_publication_profile_required")
            raw = args.profile.read_bytes()
            if hashlib.sha256(raw).hexdigest() != args.profile_sha256:
                raise ValueError("publication_profile_hash_mismatch")
            profile = json.loads(raw)
            required = {"schema_version": "cq.fixed-publication/1", "scope": "two_frozen_snapshots_only",
                        "policy_sha256": policy, "evaluation_sha256": evaluation, "generations": report["generations"],
                        "controller_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "fence_sha256": hashlib.sha256((ROOT / "src/citequest/index_v3/publication.py").read_bytes()).hexdigest(),
                        "new_paid_calls": 0, "update_publication": False}
            if any(profile.get(key) != value for key, value in required.items()):
                raise ValueError("approved_publication_scope_changed")
            report["approved_profile_sha256"] = args.profile_sha256
            (args.output / "approved-profile.json").write_bytes(raw)
            report["publications"] = []
            for row, intent in zip(report["generations"], profile["intent_ids"], strict=True):
                generation_id, intent_id = UUID(row["generation_id"]), UUID(intent)
                with psycopg.connect(dsn.replace("postgresql+psycopg://", "postgresql://", 1), autocommit=True, row_factory=dict_row) as connection:
                    fence = PublicationSessionFence(connection, row["catalog"], "public")
                    with fence.held():
                        def authority(fence: PublicationSessionFence = fence, intent_id: UUID = intent_id,
                                      generation_id: UUID = generation_id, catalog_name: str = row["catalog"]) -> str:
                            fence.check()
                            with factory.begin() as db:
                                repo = repository(db)
                                plan = repo.intent(intent_id)
                                current = db.get(index.IndexCatalog, (catalog_name, "public"), populate_existing=True)
                                if current is None or plan["target_generation_id"] != str(generation_id) or plan["expected_epoch"] != 0:
                                    raise RuntimeError("publication_intent_scope_changed")
                                if plan["state"] in {"prepared", "alias_confirmed", "reconcile_required"}:
                                    if current.epoch != 0 or current.active_generation_id is not None or current.pending_intent_id != intent_id:
                                        raise RuntimeError("pending_publication_fence_changed")
                                elif plan["state"] == "committed":
                                    if current.epoch != 1 or current.active_generation_id != generation_id or current.pending_intent_id is not None:
                                        raise RuntimeError("committed_publication_fence_changed")
                                else:
                                    raise RuntimeError("closed_publication_fence")
                                return plan["target_collection"]

                        aliases = AliasTransport(FencedAliasCalls(native, fence, authority))
                        coordinator = PublicationCoordinator(tx, aliases)
                        result = coordinator.publish(generation_id, intent_id, expected_epoch=0)
                        report["publications"].append(result)
                        destination.write_text(json.dumps(report, indent=2) + "\n")
        report["status"] = "two_fixed_snapshots_published" if args.execute else "readonly_publication_preflight_pass"
    except Exception as error:
        report.update(status="failed_preserved_readonly_reconciliation_required", error_type=type(error).__name__, error=str(error)[:240])
        raise
    finally:
        native.close()
        database.dispose()
        destination.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "provider_calls": 0}))  # noqa: T201


if __name__ == "__main__":
    main()
