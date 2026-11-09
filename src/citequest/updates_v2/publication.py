"""Explicit immutable publication adoption over unchanged observation receipts."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select

from citequest.agents_v3.contracts import canonical, digest
from citequest.v2.domain import DomainError
from . import models as m
from .publication_contracts import PROFILE, merge_members


class SnapshotService:
    def __init__(self, repository):
        self.repository, self.db, self.catalog = repository, repository.db, repository.catalog

    def snapshot(self, identifier: UUID):
        value = self.db.get(m.PublicationSnapshot, identifier, populate_existing=True)
        if (value is None or value.security_domain not in {"public", self.catalog.actor.security_domain}
            or value.profile_id != PROFILE or digest(value.payload_canonical.encode()) != value.seal_sha256
            or json.loads(value.payload_canonical) != value.payload
            or value.payload.get("membership_sha256") != value.membership_sha256
            or value.payload.get("member_count") != value.member_count):
            raise DomainError("bounded_publication_snapshot_invalid", 409)
        return value

    @staticmethod
    def member_record(member) -> dict:
        return {"source_unit_id": str(member.source_unit_id), "source_namespace": member.source_namespace,
            "native_id": member.native_id, "source_revision": member.source_revision,
            "origin_generation_id": str(member.origin_generation_id) if member.origin_generation_id else None,
            "origin_seal_id": str(member.origin_seal_id) if member.origin_seal_id else None,
            "reference": member.reference, "projection_sha256": member.projection_sha256,
            "source_pointer_sha256": member.source_pointer_sha256}

    def members(self, identifier: UUID) -> list:
        snapshot = self.snapshot(identifier)
        rows = list(self.db.scalars(select(m.PublicationMember).where(m.PublicationMember.snapshot_id == identifier)
            .order_by(m.PublicationMember.source_unit_id)))
        records = [self.member_record(item) for item in rows]
        if len(rows) != snapshot.member_count or digest(records) != snapshot.membership_sha256:
            raise DomainError("bounded_publication_membership_changed", 409)
        return rows

    def create(self, base_generation_id: UUID, seal_ids: list[UUID], identifier: UUID) -> dict:
        from citequest.index_v2 import models as index
        self.catalog.lock_owner()
        self.catalog.require_publisher()
        base = self.repository.generation(base_generation_id)
        self.repository._read_ready(base)
        if base.build_scope != "full" or base.catalog_name != "standard_metadata" or not seal_ids or len(set(seal_ids)) != len(seal_ids) or len(seal_ids) > 100:
            raise DomainError("complete_metadata_base_and_bounded_seals_required", 409)
        existing = self.db.get(m.PublicationSnapshot, identifier)
        if existing is not None:
            value = self.snapshot(identifier)
            if value.base_generation_id != base_generation_id or [item["seal_id"] for item in value.payload["observations"]] != [str(key) for key in seal_ids]:
                raise DomainError("publication_snapshot_idempotency_conflict", 409)
            return value.payload
        base_denied = set()
        if base.source_root_kind == "publication_snapshot":
            base_members = [self.member_record(item) for item in self.members(base.publication_snapshot_id)]
        else:
            base_units = list(self.db.scalars(select(index.IndexUnit).where(index.IndexUnit.generation_id == base.id)))
            base_members = [{"source_unit_id": str(unit.projection_id), "source_namespace": unit.source_namespace,
                "native_id": unit.native_id, "source_revision": unit.source_revision, "origin_generation_id": str(base.id),
                "origin_seal_id": None, "reference": unit.projection["source_reference"],
                "projection_sha256": unit.projection_sha256, "source_pointer_sha256": unit.source_pointer_sha256} for unit in base_units]
            if len(base_members) != base.unit_count:
                raise DomainError("complete_base_generation_membership_required", 409)
            base_denied = {(unit.source_namespace, unit.native_id) for unit in base_units if unit.source_denied}
        additions, observations = [], []
        previous_upper = None
        for seal_id in seal_ids:
            seal = self.db.get(m.UpdateSeal, seal_id)
            window = self.db.get(m.UpdateWindow, seal.window_id) if seal else None
            if (seal is None or window is None or seal.security_domain != base.security_domain or window.state != "observed"
                or window.config.get("mode") != "production" or digest(seal.payload) != seal.seal_sha256
                or seal.publication_eligible or (previous_upper is not None and window.upper_bound < previous_upper)):
                raise DomainError("ordered_complete_production_observations_required", 409)
            previous_upper = window.upper_bound
            for member in self.db.scalars(select(m.SealMember).where(m.SealMember.seal_id == seal_id)):
                projection = self.repository.readers.updates.seal_index_projection(seal_id, member.observation_id)
                additions.append({"source_unit_id": str(projection.observation_id), "source_namespace": projection.source_namespace,
                    "native_id": projection.native_id, "source_revision": projection.source_revision,
                    "origin_generation_id": None, "origin_seal_id": str(seal_id),
                    "reference": {"kind": "update_observation", "update_observation_id": str(projection.observation_id)},
                    "projection_sha256": projection.projection_sha256, "source_pointer_sha256": projection.source_pointer_sha256})
            observations.append({"seal_id": str(seal_id), "seal_sha256": seal.seal_sha256,
                "source_namespace": seal.source_namespace, "scope_sha256": seal.scope_sha256,
                "lower_bound": window.lower_bound.isoformat(), "observed_watermark": window.upper_bound.isoformat(),
                "observed_at": seal.created_at.isoformat()})
        denied_rows = list(self.db.scalars(select(m.SourceDenial).where(m.SourceDenial.security_domain == base.security_domain,
            m.SourceDenial.active.is_(True))))
        # Version-specific denials are applied only to matching concrete members;
        # wildcard denials cover this source lineage, never a whole Work.
        candidates = [*base_members, *additions]
        denied = base_denied | {(item["source_namespace"], item["native_id"]) for item in candidates for denial in denied_rows
            if denial.source_namespace == item["source_namespace"] and denial.native_id == item["native_id"]
            and denial.source_revision in {"*", item["source_revision"]}}
        records = merge_members(base_members, additions, denied)
        payload = {"schema_version": PROFILE, "snapshot_id": str(identifier), "base_generation_id": str(base.id),
            "base_source_manifest_sha256": base.source_manifest_sha256, "catalog_name": base.catalog_name,
            "security_domain": base.security_domain, "member_count": len(records), "membership_sha256": digest(records),
            "observations": observations, "created_at": datetime.now(UTC).isoformat(),
            "current_status_confirmed": False, "complete_global_or_latest_sync_claimed": False,
            "publication_scope": "complete_base_plus_admitted_bounded_observations_minus_source_quarantine",
            "quarantine": [{"source_namespace": row.source_namespace, "native_id": row.native_id,
                "source_revision": row.source_revision, "observation_id": str(row.observation_id), "reason": row.reason} for row in denied_rows]}
        self.db.add(m.PublicationSnapshot(id=identifier, base_generation_id=base.id, base_snapshot_id=base.snapshot_id,
            security_domain=base.security_domain, catalog_name=base.catalog_name, profile_id=PROFILE,
            policy_sha256=base.policy_sha256, evaluation_sha256=base.evaluation_sha256,
            membership_sha256=payload["membership_sha256"], member_count=len(records), payload=payload,
            payload_canonical=canonical(payload).decode(), seal_sha256=digest(payload)))
        self.db.flush()
        for record in records:
            fields = {**record, "source_unit_id": UUID(record["source_unit_id"]),
                "origin_generation_id": UUID(record["origin_generation_id"]) if record["origin_generation_id"] else None,
                "origin_seal_id": UUID(record["origin_seal_id"]) if record["origin_seal_id"] else None}
            self.db.add(m.PublicationMember(snapshot_id=identifier, **fields, record_canonical=canonical(record).decode()))
        self.db.flush()
        return payload
