"""Current source-lineage barriers and immutable update-seal index inputs."""

from __future__ import annotations

from uuid import UUID
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from citequest.updates_v2 import models as m
from citequest.updates_v2.protocols import digest
from citequest.updates_v2.schemas import ProtocolLocator, UpdateIndexProjection
from citequest.v2 import models as core
from citequest.v2.domain import Actor, DomainError
from citequest.v2.services import Catalog
from citequest.v2.storage import CAS, StoredBlob
from citequest.v2.storage_config import app_storage


class UpdateReader:
    def __init__(self, db: Session, actor: Actor, store: CAS) -> None:
        app_storage(Path(__file__).resolve().parents[3]).assert_store(store)
        self.db, self.actor, self.store = db, actor, store
        self.catalog = Catalog(db, actor)

    def _domain(self, domain: str) -> None:
        if domain not in {"public", self.actor.security_domain}:
            raise DomainError("source_scope_not_found", 404)

    def current_denials(self, source_namespace: str, items: list[tuple[str, str | None]], security_domain: str = "public") -> list[dict]:
        """Only this source/native/revision lineage; never whole-Work inference."""
        self._domain(security_domain)
        if not items:
            return []
        ids = {identifier for identifier, _ in items}
        wanted = set(items)
        rows = self.db.scalars(select(m.SourceDenial).where(m.SourceDenial.security_domain == security_domain,
                              m.SourceDenial.source_namespace == source_namespace, m.SourceDenial.native_id.in_(ids), m.SourceDenial.active.is_(True))).all()
        return [{"source_namespace": row.source_namespace, "native_id": row.native_id, "source_revision": row.source_revision,
                 "reason": row.reason, "scope": "this_source_all_versions" if row.source_revision == "*" else "this_source_version",
                 "observation_id": str(row.observation_id), "blocks_current_default_evidence": True}
                for row in rows if row.source_revision == "*" or (row.native_id, row.source_revision) in wanted]

    def seal_index_projection(self, seal_id: UUID, observation_id: UUID) -> UpdateIndexProjection:
        seal = self.db.get(m.UpdateSeal, seal_id)
        if seal is None:
            raise DomainError("update_seal_not_found", 404)
        self._domain(seal.security_domain)
        members = self.db.scalars(select(m.SealMember).where(m.SealMember.seal_id == seal_id).order_by(m.SealMember.source_namespace, m.SealMember.native_id)).all()
        membership = [{"source_namespace": row.source_namespace, "native_id": row.native_id, "observation_id": str(row.observation_id), "projection_sha256": row.projection_sha256} for row in members]
        if digest(membership) != seal.membership_sha256 or digest(seal.payload) != seal.seal_sha256 or seal.payload.get("membership_sha256") != seal.membership_sha256:
            raise DomainError("update_seal_or_membership_hash_mismatch", 409)
        observation = self.db.get(m.UpdateObservation, observation_id)
        if observation is None or observation.security_domain != seal.security_domain:
            raise DomainError("update_observation_not_found", 404)
        member = self.db.get(m.SealMember, (seal_id, observation.source_namespace, observation.native_id))
        if member is None or member.observation_id != observation_id or member.projection_sha256 != observation.projection_sha256:
            raise DomainError("observation_not_in_frozen_update_seal", 404)
        if self.current_denials(observation.source_namespace, [(observation.native_id, observation.source_revision)], seal.security_domain):
            raise DomainError("source_lineage_quarantined", 409)
        if observation.work_id:
            self.catalog.work(observation.work_id)
        locator = ProtocolLocator.model_validate(observation.locator)
        page = self.db.get(m.UpdatePage, observation.page_id)
        if page is None or not page.body_complete or page.parsed is None or locator.raw_response_sha256 != page.raw_sha256 or locator.record_ordinal != observation.record_ordinal:
            raise DomainError("protocol_origin_page_not_verified", 409)
        blob = self.db.get(core.Blob, (page.raw_security_domain, page.raw_sha256))
        if blob is None:
            raise DomainError("protocol_raw_blob_missing", 409)
        raw = self.store.read(StoredBlob(blob.security_domain, blob.sha256, blob.size_bytes))
        # Re-read the actual protocol record, rather than trusting a DB hash
        # that could have been computed over a mismapped field/ordinal.
        from citequest.updates_v2.service import UpdateService
        decoded = UpdateService._decode(raw, page.request, observation.source_namespace)
        records = decoded.get('records', [])
        ordinal = locator.record_ordinal
        if ordinal is None or ordinal >= len(records):
            raise DomainError('protocol_record_ordinal_not_present', 409)
        origin = records[ordinal]
        expected_metadata = {**origin, 'title': origin.get('title'), 'abstract': origin.get('abstract'), 'year': origin.get('year')}
        if (origin['native_id'] != observation.native_id or origin.get('source_revision') != observation.source_revision
                or origin['record_sha256'] != observation.record_sha256 or expected_metadata != observation.metadata_payload):
            raise DomainError('protocol_record_or_field_mapping_differs_from_actual_raw', 409)
        return UpdateIndexProjection(observation_id=observation.id, unit_id=observation.id, seal_id=seal.id,
                    security_domain=seal.security_domain, source_namespace=observation.source_namespace,
                    native_id=observation.native_id, source_revision=observation.source_revision,
                    title=observation.metadata_payload.get("title"), abstract=observation.metadata_payload.get("abstract"),
                    year=observation.metadata_payload.get("year"), work_id=observation.work_id,
                    record_sha256=observation.record_sha256, projection_sha256=observation.projection_sha256,
                    source_pointer_sha256=digest(locator.model_dump(mode="json")), locator=locator,
                    metadata_payload=observation.metadata_payload, seal_sha256=seal.seal_sha256,
                    membership_sha256=seal.membership_sha256, publication_eligible=seal.publication_eligible)
