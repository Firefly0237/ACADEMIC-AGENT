"""Append-only canonical resolution; source projections and their hashes never change."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid5

from sqlalchemy import select

from citequest.v2 import models as core
from citequest.v2.domain import DomainError, IdentifierInput, WorkInput
from citequest.v2.presentation import work as present_work
from citequest.v2.services import digest

from . import models as m
from .canonical import CanonicalResolver
from .schemas import unresolved_reasons

if TYPE_CHECKING:
    from .landing import Landing


class Resolver:
    def __init__(self, landing: Landing) -> None:
        self.landing, self.catalog, self.db = landing, landing.catalog, landing.db

    def source(self, source_row_id: UUID) -> tuple[m.SourceRow, m.SourceSnapshot]:
        self.catalog.lock_owner()
        row = self.db.get(m.SourceRow, source_row_id, with_for_update=True, populate_existing=True)
        if row is None:
            raise DomainError("not_found", 404)
        snapshot = self.landing.snapshot(row.snapshot_id)
        if snapshot.status == "disabled":
            raise DomainError("source_snapshot_disabled", 409)
        if snapshot.security_domain == "public":
            self.catalog.require_publisher()
        return row, snapshot

    def latest(self, source_row_id: UUID) -> m.SourceResolution | None:
        return self.db.scalar(
            select(m.SourceResolution)
            .where(m.SourceResolution.source_row_id == source_row_id)
            .order_by(m.SourceResolution.revision.desc())
            .limit(1)
        )

    def append(
        self,
        row: m.SourceRow,
        state: str,
        proof: dict,
        *,
        work_id: UUID | None = None,
        version_id: UUID | None = None,
        representation_id: str | None = None,
    ) -> dict:
        # Re-fetch the row through the current source/domain/publisher guard.
        # A caller holding an ORM row is not granted publication authority.
        row, snapshot = self.source(row.id)
        if state == "paper_identity_verified":
            from .native_identity import verify_native_identity

            verify_native_identity(self.landing, row, snapshot, work_id=work_id,
                                   version_id=version_id, representation_id=representation_id, proof=proof)
        elif work_id is not None:
            from .native_identity import common_work

            common_work(self.landing, row, snapshot, work_id)
        previous = self.latest(row.id)
        proof_sha = digest(proof)
        if previous is not None and (
            previous.proof_sha256 == proof_sha
            and previous.state == state
            and previous.work_id == work_id
            and previous.work_version_id == version_id
            and previous.representation_id == representation_id
        ):
            return self.present(previous)
        if snapshot.purpose == "application" and snapshot.profile_adopted:
            raise DomainError("adopted_application_resolution_is_immutable", 409)
        if previous is not None and previous.state == "paper_identity_verified" and state != "paper_identity_verified":
            raise DomainError("verified_paper_resolution_cannot_silently_downgrade", 409)
        revision = previous.revision + 1 if previous is not None else 1
        resolution = m.SourceResolution(
            source_row_id=row.id,
            security_domain=row.security_domain,
            revision=revision,
            state=state,
            work_id=work_id,
            work_version_id=version_id,
            representation_id=representation_id,
            proof=proof,
            proof_sha256=proof_sha,
        )
        self.db.add(resolution)
        self.db.flush()
        result = self.present(resolution)
        envelope = {
            "schema_version": "cq.outbox.payload/1",
            "entity_type": "SourceResolution",
            "entity_id": str(row.id),
            "entity_revision": revision,
            "security_domain": row.security_domain,
            "operation": "upsert",
            "entity_schema": "cq.source-resolution/1",
            "data": result,
        }
        self.db.add(
            core.Outbox(
                idempotency_key=f"SourceResolution:{row.id}:{revision}:upsert",
                entity_id=str(row.id),
                entity_revision=revision,
                operation="upsert",
                payload=envelope,
                payload_sha256=digest(envelope),
                security_domain=row.security_domain,
                schema_version="cq.outbox.payload/1",
                entity_type="SourceResolution",
            )
        )
        self.db.flush()
        return result

    @staticmethod
    def present(value: m.SourceResolution) -> dict:
        return {
            "source_row_id": str(value.source_row_id),
            "revision": value.revision,
            "state": value.state,
            "work_id": str(value.work_id) if value.work_id else None,
            "work_version_id": str(value.work_version_id) if value.work_version_id else None,
            "representation_id": value.representation_id,
            "proof": value.proof,
            "proof_sha256": value.proof_sha256,
            "work_existence_grants_fulltext_rights": False,
            "application_profile_adopted": False,
        }

    def resolve_metadata(self, projection_id: UUID, client_operation_id: UUID) -> dict:
        projection = self.db.get(m.SourceProjection, projection_id)
        if projection is None:
            raise DomainError("not_found", 404)
        row, snapshot = self.source(projection.source_row_id)
        if (
            snapshot.source_namespace not in {"litsearch_clean", "litsearch_s2orc"}
            or row.native_id_source_type != "integer"
            or snapshot.representation_profile
            not in {
                "litsearch-clean-exact/1",
                "litsearch-native-annotation-decimal-offsets/1",
            }
        ):
            raise DomainError("not_a_metadata_projection", 409)
        current = self.latest(row.id)
        if current is not None and current.work_id is not None:
            self.catalog.work(current.work_id)  # Current revocation precedes receipt replay.

        def effect() -> dict:
            previous = self.latest(row.id)
            if previous is not None and previous.work_id is not None:
                self.catalog.work(previous.work_id)
                return self.present(previous)
            proof = {
                "schema_version": "cq.source-resolution-proof/1",
                "source_snapshot_id": str(snapshot.id),
                "source_manifest_sha256": snapshot.source_manifest_sha256,
                "source_pointer_sha256": row.source_pointer_sha256,
                "projection_id": str(projection.id),
                "projection_sha256": projection.projection_sha256,
                "source_namespace": snapshot.source_namespace,
                "common_identity_namespace": "litsearch",
                "native_id": row.native_id,
                "representation_profile": snapshot.representation_profile,
                "fulltext_identity_verified": False,
                "license_status": "unknown",
                "outbound_permission_granted": False,
            }
            reasons = unresolved_reasons(projection.title, projection.abstract)
            if reasons:
                return self.append(row, "unresolved", {**proof, "reasons": reasons})
            request = WorkInput(
                title=projection.title,
                authors=None,
                year=None,
                identifiers=[IdentifierInput(namespace="litsearch_corpus_id", raw=row.native_id)],
                # Same operation identity for all source profiles, never one Work per profile.
                client_operation_id=uuid5(
                    snapshot.importer_owner_id,
                    "resolve-litsearch-native:" + snapshot.security_domain + ":" + row.native_id,
                ),
            )
            obj = CanonicalResolver(self.catalog).resolve(
                "litsearch", row.native_id, request, public=snapshot.security_domain == "public"
            )
            work_id = obj.id
            # Source provenance belongs to the append-only resolution. Existing
            # canonical publication metadata is preserved; no title-similarity merge.
            return self.append(
                row,
                "metadata_resolved",
                {
                    **proof,
                    "work_metadata_revision": obj.metadata_revision,
                    "canonical_metadata_snapshot": present_work(
                        self.catalog, obj, public_only=snapshot.security_domain == "public"
                    ),
                    "source_title_raw": projection.title,
                    "canonical_title_normalization": "WorkInput strips outer whitespace",
                    "authors_unknown": obj.authors is None,
                    "year_unknown": obj.year is None,
                },
                work_id=work_id,
            )

        return self.catalog.operation(
            client_operation_id,
            {
                "operation": "resolve_source_metadata",
                "projection_id": str(projection_id),
                "source_projection_sha256": projection.projection_sha256,
            },
            effect,
        )
