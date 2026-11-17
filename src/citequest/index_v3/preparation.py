"""Bounded SQL selection and preparation without any embedding/network call."""

from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import exists, func, insert, select

from citequest.data_v2 import models as data
from citequest.data_v2.application_members import ApplicationMember
from citequest.index_v2 import models as m
from citequest.index_v2.repository_v2 import VariantIndexRepository
from citequest.index_v2.rows import canonical_sha
from citequest.index_v2.source_variants import MetadataReference, NativeReference
from citequest.v2.domain import DomainError


class FullPreparationRepository(VariantIndexRepository):
    """Same authoritative source reads/FKs, with SQL anti-join admission."""

    def pending_units(self, identifier: UUID, *, limit: int = 100) -> list:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise DomainError("bounded_full_preparation_batch_required", 400)
        value = self.generation(identifier)
        if value.state != "preparing":
            return []
        root = self.readers.root(value)
        if root.kind == "publication_snapshot":
            from pydantic import TypeAdapter
            from citequest.updates_v2.models import PublicationMember
            from citequest.index_v2.source_variants import SourceReference
            import json
            members = self.db.scalars(select(PublicationMember).where(
                PublicationMember.snapshot_id == root.source_root_id,
                ~exists(select(1).where(m.IndexUnit.generation_id == identifier,
                    m.IndexUnit.projection_id == PublicationMember.source_unit_id)))
                .order_by(PublicationMember.source_unit_id).limit(limit))
            return [self.readers.read(root, TypeAdapter(SourceReference).validate_json(json.dumps(member.reference))) for member in members]
        if value.build_scope != "full" or root.kind not in {"metadata_snapshot", "native_snapshot"}:
            raise DomainError("adopted_fixed_full_source_required", 409)
        if not root.profile_adopted or not root.publication_eligible or root.state != "staged":
            raise DomainError("adopted_full_source_required", 409)
        if root.kind == "metadata_snapshot":
            p, row = data.SourceProjection, data.SourceRow
            query = (
                select(p.id)
                .join(row, p.source_row_id == row.id)
                .where(
                    row.snapshot_id == root.source_root_id,
                    row.security_domain == root.security_domain,
                    p.representation_profile == root.representation_profile,
                    ~exists(
                        select(1).where(m.IndexUnit.generation_id == identifier, m.IndexUnit.projection_id == p.id)
                    ),
                )
                .order_by(row.artifact_path, row.row_ordinal, p.id)
                .limit(limit)
            )
            refs = [MetadataReference(metadata_projection_id=key) for key in self.db.scalars(query)]
        else:
            # The existing reader verifies compact header + immutable member
            # table count/digest. 692 references are bounded; no text is loaded.
            self.readers.references(root)
            n, row, member = data.NativeRepresentation, data.SourceRow, ApplicationMember
            query = (
                select(n.id)
                .join(row, n.source_row_id == row.id)
                .join(
                    member,
                    (member.snapshot_id == root.source_root_id)
                    & (member.source_row_id == row.id)
                    & (member.representation_id == n.id),
                )
                .where(
                    row.snapshot_id == root.source_root_id,
                    row.security_domain == root.security_domain,
                    ~exists(
                        select(1).where(
                            m.IndexUnit.generation_id == identifier, m.IndexUnit.projection_id == n.index_projection_id
                        )
                    ),
                )
                .order_by(member.rank, n.id)
                .limit(limit)
            )
            refs = [NativeReference(native_representation_id=key) for key in self.db.scalars(query)]
        # No hand-made DTO: retain actual source hashes, current Work ACL,
        # source denials, trusted native identity and canonical CAS verification.
        return [self.readers.read(root, reference) for reference in refs]

    def seal_preparation(self, identifier: UUID, *, transport_batch_size: int = 4) -> dict:
        value = self.generation(identifier, writing=True)
        if value.state != "preparing":
            return self.describe(identifier)
        if type(transport_batch_size) is not int or not 1 <= transport_batch_size <= 16:
            raise DomainError("bounded_full_transport_batch_required", 400)
        root = self.readers.root(value)
        refs = self.readers.references(root)
        wanted = {self.readers.unit_id(reference) for reference in refs}
        # Select scalar identity/count fields, not 64183 full compilation JSONs.
        units = list(
            self.db.execute(
                select(
                    m.IndexUnit.projection_id,
                    m.IndexUnit.native_id,
                    m.IndexUnit.source_reference_kind,
                    m.IndexUnit.projection_sha256,
                    m.IndexUnit.source_pointer_sha256,
                    m.IndexUnit.searchable,
                    m.IndexUnit.source_denied,
                    m.IndexUnit.chunk_count,
                ).where(m.IndexUnit.generation_id == identifier)
            )
        )
        if root.kind == "publication_snapshot":
            from citequest.updates_v2.models import PublicationSnapshot
            expected_count = self.db.get(PublicationSnapshot, root.source_root_id).member_count
        else:
            expected_count = 64183 if root.kind == "metadata_snapshot" else 692
        if (
            value.build_scope != "full"
            or len(wanted) != expected_count
            or {row.projection_id for row in units} != wanted
        ):
            raise DomainError("all_fixed_full_source_members_required", 409)
        chunks = list(
            self.db.execute(
                select(m.IndexChunk.chunk_id, m.IndexChunk.row_sha256)
                .where(m.IndexChunk.generation_id == identifier)
                .order_by(m.IndexChunk.chunk_id)
            )
        )
        if sum(row.chunk_count for row in units) != len(chunks) or not chunks:
            raise DomainError("complete_prepared_chunks_required", 409)
        if self.db.scalar(
            select(func.count()).select_from(m.IndexBatch).where(m.IndexBatch.generation_id == identifier)
        ):
            raise DomainError("preparation_must_not_duplicate_existing_batches", 409)
        # Core executemany keeps memory bounded; SQLAlchemy supplies real UUID
        # defaults. This one transaction also sets the immutable generation seal.
        for offset in range(0, len(chunks), transport_batch_size * 500):
            payloads = []
            for inner in range(offset, min(offset + transport_batch_size * 500, len(chunks)), transport_batch_size):
                rows = {row.chunk_id: row.row_sha256 for row in chunks[inner : inner + transport_batch_size]}
                payloads.append(
                    {
                        "id": uuid4(),
                        "generation_id": identifier,
                        "sequence": inner // transport_batch_size + 1,
                        "chunk_ids": list(rows),
                        "batch_manifest": {
                            "coverage_scope": "batch_only",
                            "rows": rows,
                            "count": len(rows),
                            "sha256": canonical_sha(rows),
                        },
                        "state": "ready",
                        "logical_attempts": 0,
                    }
                )
            self.db.execute(insert(m.IndexBatch), payloads)
        value.unit_count = len(units)
        value.empty_unit_count = sum(not row.searchable and not row.source_denied for row in units)
        value.denied_unit_count = sum(row.source_denied for row in units)
        value.chunk_count = len(chunks)
        value.membership_sha256 = canonical_sha(
            {
                str(row.projection_id): {
                    "kind": row.source_reference_kind,
                    "native_id": row.native_id,
                    "projection_sha256": row.projection_sha256,
                    "pointer_sha256": row.source_pointer_sha256,
                }
                for row in units
            }
        )
        value.expected_manifest_sha256 = canonical_sha({row.chunk_id: row.row_sha256 for row in chunks})
        value.state = "building"
        return self.describe(identifier)
