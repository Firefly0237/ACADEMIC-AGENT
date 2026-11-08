"""Official SQLAlchemy/PostgreSQL batch landing; no Work creation or publication."""

from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from citequest.v2 import models as core
from citequest.v2.domain import DomainError
from citequest.v2.services import Catalog, digest

from . import models as m
from .schemas import (
    BatchInput,
    IndexProjection,
    ProjectionReceipt,
    SnapshotFreezeInput,
    SnapshotInput,
    unresolved_reasons,
)


def verify_artifacts(root: Path, request: SnapshotInput) -> None:
    """Verify supplied immutable source bytes once at snapshot admission.

    Readers must recheck the bound artifact before a batch read. A successful
    hash is an integrity proof, not a licence or scientific quality decision.
    """
    base = root.resolve(strict=True)
    for artifact in request.artifacts:
        current = base
        for component in artifact.path.split("/"):
            current = current / component
            if current.is_symlink():
                raise DomainError("source_artifact_symlink", 422)
        if not current.resolve(strict=True).is_relative_to(base):
            raise DomainError("source_artifact_escape", 422)
        sha, size = hashlib.sha256(), 0
        with current.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                sha.update(block)
                size += len(block)
        if sha.hexdigest() != artifact.sha256 or size != artifact.size_bytes:
            raise DomainError("source_artifact_integrity_failed", 422)


class Landing:
    def __init__(self, catalog: Catalog, *, policy_sha256: str, evaluation_sha256: str) -> None:
        self.catalog, self.db = catalog, catalog.db
        self.policy_sha256, self.evaluation_sha256 = policy_sha256, evaluation_sha256

    def freeze_standard_snapshot(
        self,
        snapshot_id: UUID,
        request: SnapshotFreezeInput,
        *,
        verified_export_dir: Path,
        qrels_path: Path,
    ) -> dict:
        """Internal trusted-reader/publisher boundary, never an LLM or upload tool."""
        from .standard_seal import freeze_standard

        return freeze_standard(self, snapshot_id, request, verified_export_dir, qrels_path)

    def snapshot(self, snapshot_id: UUID, *, writing: bool = False) -> m.SourceSnapshot:
        if writing:
            self.catalog.lock_owner()
        row = self.db.get(m.SourceSnapshot, snapshot_id, populate_existing=True, with_for_update=writing)
        if row is None or row.security_domain not in {"public", self.catalog.actor.security_domain}:
            raise DomainError("not_found", 404)
        if writing:
            if row.security_domain == "public":
                self.catalog.require_publisher()
            if row.status != "landing":
                raise DomainError("source_snapshot_not_writable", 409)
        return row

    def index_projection(self, projection_id: UUID) -> IndexProjection:
        """Read through the same current snapshot domain guard; no publication."""
        projection = self.db.get(m.SourceProjection, projection_id)
        source = self.db.get(m.SourceRow, projection.source_row_id) if projection is not None else None
        if source is None:
            raise DomainError("not_found", 404)
        snapshot = self.snapshot(source.snapshot_id)
        if snapshot.purpose != "standard" or snapshot.representation_profile == "litsearch-s2orc-native-text/1":
            raise DomainError("not_a_standard_metadata_projection", 409)
        resolution = self.db.scalar(
            select(m.SourceResolution)
            .where(m.SourceResolution.source_row_id == source.id)
            .order_by(m.SourceResolution.revision.desc())
            .limit(1)
        )
        work_id = resolution.work_id if resolution is not None else None
        if work_id is not None:
            self.catalog.work(work_id)  # A source identity never overrides current Work ACL.
        return IndexProjection(
            projection_id=projection.id,
            source_row_id=source.id,
            snapshot_id=snapshot.id,
            native_id=source.native_id,
            source_namespace=snapshot.source_namespace,
            source_revision=snapshot.source_revision,
            security_domain=snapshot.security_domain,
            representation_profile=snapshot.representation_profile,
            source_pointer_sha256=source.source_pointer_sha256,
            projection_sha256=projection.projection_sha256,
            title=projection.title,
            abstract=projection.abstract,
            field_provenance=projection.field_provenance,
            year=None,
            work_id=work_id,
            searchable_text_present=projection.searchable_text_present,
            profile_adopted=snapshot.profile_adopted,
            snapshot_state=snapshot.status,
        )

    def register_snapshot(self, request: SnapshotInput, artifact_root: Path, *, public: bool = False) -> dict:
        self.catalog.lock_owner()
        if public:
            self.catalog.require_publisher()
        domain = "public" if public else self.catalog.actor.security_domain
        if request.policy_sha256 != self.policy_sha256 or request.evaluation_sha256 != self.evaluation_sha256:
            raise DomainError("ingest_contract_binding_mismatch", 409)
        verify_artifacts(artifact_root, request)
        payload = {
            "operation": "register_source_snapshot",
            "security_domain": domain,
            "request": request.model_dump(mode="json"),
        }
        snapshot_id = uuid5(
            NAMESPACE_URL,
            digest(
                [
                    domain,
                    request.source_namespace,
                    request.source_revision,
                    request.representation_profile,
                    request.source_manifest_sha256,
                ]
            ),
        )

        def effect() -> dict:
            values = request.model_dump(mode="json", exclude={"client_operation_id", "artifacts"})
            row = m.SourceSnapshot(
                id=snapshot_id, security_domain=domain, importer_owner_id=self.catalog.actor.owner_id, **values
            )
            result = self.db.execute(
                insert(m.SourceSnapshot)
                .values(id=snapshot_id, security_domain=domain, importer_owner_id=self.catalog.actor.owner_id, **values)
                .on_conflict_do_nothing(index_elements=["id"])
            )
            if result.rowcount == 0:
                prior = self.snapshot(snapshot_id, writing=True)
                if any(getattr(prior, key) != value for key, value in values.items()):
                    raise DomainError("source_snapshot_identity_conflict", 409)
            else:
                self.db.add_all(
                    [m.SourceArtifact(snapshot_id=snapshot_id, **a.model_dump(mode="json")) for a in request.artifacts]
                )
                self.db.flush()
            actual = list(self.db.scalars(select(m.SourceArtifact).where(m.SourceArtifact.snapshot_id == snapshot_id)))
            expected = {a.path: a.model_dump(mode="json") for a in request.artifacts}
            for artifact in actual:
                current = {key: getattr(artifact, key) for key in expected.get(artifact.path, {})}
                if artifact.path not in expected or current != expected[artifact.path]:
                    raise DomainError("source_snapshot_artifact_conflict", 409)
            if len(actual) != len(expected):
                raise DomainError("source_snapshot_artifact_conflict", 409)
            return {
                "snapshot_id": str(row.id),
                "representation_profile": request.representation_profile,
                "source_manifest_sha256": request.source_manifest_sha256,
                "profile_adopted": False,
                "status": "landing",
            }

        return self.catalog.operation(request.client_operation_id, payload, effect)

    def ingest_batch(self, snapshot_id: UUID, request: BatchInput) -> dict:
        snapshot = self.snapshot(snapshot_id, writing=True)  # Always before receipt replay.
        if snapshot.representation_profile == "litsearch-s2orc-native-text/1":
            raise DomainError("native_fulltext_requires_native_projection_contract", 409)
        payload = {
            "operation": "raw_ingest_batch",
            "snapshot_id": str(snapshot_id),
            "request": request.model_dump(mode="json"),
        }

        def effect() -> dict:
            artifacts = {
                a.path: a
                for a in self.db.scalars(select(m.SourceArtifact).where(m.SourceArtifact.snapshot_id == snapshot_id))
            }
            source_rows, projections, events = [], [], []
            for record in request.rows:
                locator = record.locator
                artifact = artifacts.get(locator.artifact_path)
                if artifact is None or artifact.format != "parquet" or artifact.sha256 != locator.artifact_sha256:
                    raise DomainError("source_locator_artifact_mismatch", 422)
                counts = artifact.row_group_sizes
                if (
                    counts is None
                    or locator.row_group >= len(counts)
                    or locator.row_in_group >= counts[locator.row_group]
                    or locator.row_ordinal != sum(counts[: locator.row_group]) + locator.row_in_group
                ):
                    raise DomainError("source_locator_row_mismatch", 422)
                pointer_sha = digest(
                    {
                        "snapshot_manifest_sha256": snapshot.source_manifest_sha256,
                        "native_id": record.native_id,
                        "native_id_source_type": record.native_id_source_type,
                        **locator.model_dump(mode="json"),
                    }
                )
                row_id = uuid5(snapshot_id, "native:" + record.native_id)
                projection_id = uuid5(row_id, snapshot.representation_profile)
                source_rows.append(
                    {
                        "id": row_id,
                        "snapshot_id": snapshot_id,
                        "security_domain": snapshot.security_domain,
                        "native_id": record.native_id,
                        "native_id_source_type": record.native_id_source_type,
                        **locator.model_dump(mode="json"),
                        "source_pointer_sha256": pointer_sha,
                    }
                )
                projection_sha = digest(
                    {
                        "source_pointer_sha256": pointer_sha,
                        "representation_profile": snapshot.representation_profile,
                        "title": record.title,
                        "abstract": record.abstract,
                        "field_provenance": record.field_provenance,
                    }
                )
                reasons = unresolved_reasons(record.title, record.abstract)
                receipt = ProjectionReceipt(
                    projection_id=projection_id,
                    source_row_id=row_id,
                    snapshot_id=snapshot_id,
                    native_id=record.native_id,
                    representation_profile=snapshot.representation_profile,
                    projection_sha256=projection_sha,
                    source_pointer_sha256=pointer_sha,
                    title=record.title,
                    abstract=record.abstract,
                    field_provenance=record.field_provenance,
                    work_id=None,
                    resolution="unresolved",
                    unresolved_reasons=reasons,
                    searchable_text_present=bool(
                        (record.title and record.title.strip()) or (record.abstract and record.abstract.strip())
                    ),
                )
                projections.append(
                    {
                        key: value
                        for key, value in receipt.model_dump().items()
                        if key
                        not in {
                            "projection_id",
                            "snapshot_id",
                            "native_id",
                            "source_pointer_sha256",
                            "work_id",
                            "resolution",
                        }
                    }
                    | {"id": projection_id}
                )
                envelope = {
                    "schema_version": "cq.outbox.payload/1",
                    "entity_type": "SourceProjection",
                    "entity_schema": "contracts/v2-data/ingest.schema.json#/$defs/ProjectionReceipt",
                    "entity_id": str(projection_id),
                    "entity_revision": 1,
                    "security_domain": snapshot.security_domain,
                    "operation": "stage",
                    "ingestion_state": "landing",
                    "profile_adopted": False,
                    "data": receipt.model_dump(mode="json"),
                }
                events.append(
                    {
                        "idempotency_key": f"SourceProjection:{projection_id}:1:stage",
                        "entity_id": str(projection_id),
                        "entity_revision": 1,
                        "security_domain": snapshot.security_domain,
                        "entity_type": "SourceProjection",
                        "operation": "stage",
                        "schema_version": "cq.outbox.payload/1",
                        "payload": envelope,
                        "payload_sha256": digest(envelope),
                    }
                )
            self.db.execute(
                insert(m.SourceRow)
                .values(source_rows)
                .on_conflict_do_nothing(index_elements=["snapshot_id", "native_id"])
            )
            actual_rows = {
                r.id: r
                for r in self.db.scalars(select(m.SourceRow).where(m.SourceRow.id.in_([r["id"] for r in source_rows])))
            }
            if any(
                actual_rows.get(r["id"]) is None
                or actual_rows[r["id"]].source_pointer_sha256 != r["source_pointer_sha256"]
                for r in source_rows
            ):
                raise DomainError("source_row_changed_within_snapshot", 409)
            self.db.execute(
                insert(m.SourceProjection)
                .values(projections)
                .on_conflict_do_nothing(index_elements=["source_row_id", "representation_profile"])
            )
            actual_projections = {
                r.id: r.projection_sha256
                for r in self.db.scalars(
                    select(m.SourceProjection).where(m.SourceProjection.id.in_([r["id"] for r in projections]))
                )
            }
            if any(actual_projections.get(r["id"]) != r["projection_sha256"] for r in projections):
                raise DomainError("projection_changed_within_snapshot", 409)
            self.db.execute(
                insert(core.Outbox).values(events).on_conflict_do_nothing(index_elements=["idempotency_key"])
            )
            actual_events = dict(
                self.db.execute(
                    select(core.Outbox.idempotency_key, core.Outbox.payload_sha256).where(
                        core.Outbox.idempotency_key.in_([e["idempotency_key"] for e in events])
                    )
                ).all()
            )
            if any(actual_events.get(e["idempotency_key"]) != e["payload_sha256"] for e in events):
                raise DomainError("source_outbox_payload_conflict", 409)
            return {
                "snapshot_id": str(snapshot_id),
                "rows": len(source_rows),
                "projection_ids": [str(row["id"]) for row in projections],
                "work_created": 0,
                "index_published": False,
                "profile_adopted": False,
            }

        return self.catalog.operation(request.client_operation_id, payload, effect)
