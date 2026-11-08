"""Same delivery/lease guards with a SQL-bounded single-batch selector.

The inherited selector used scalar() without LIMIT, materializing/locking every
pending ORM row. Only the first row was ever used. No skip-locked, state filter,
manifest, source, unknown-write or lease policy changes accompany this fix.
"""

from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import select

from citequest.index_v2 import models as m
from citequest.index_v2.rows import canonical_sha
from citequest.v2.domain import DomainError

from .repository import FullBuildRepository


class BoundedFullBuildRepository(FullBuildRepository):
    def batch_claim(self, identifier: UUID) -> dict | None:
        value = self.generation(identifier, writing=True)
        if value.state != "building" or value.server_proof is None:
            raise DomainError("verified_server_required_before_delivery", 409)
        row = self.db.scalar(
            select(m.IndexBatch)
            .where(m.IndexBatch.generation_id == identifier, m.IndexBatch.state != "verified")
            .order_by(m.IndexBatch.sequence)
            .limit(1)
            .with_for_update()
        )
        if row is None:
            return None
        now = datetime.now(UTC)
        if row.lease_id is not None and row.lease_expires_at and row.lease_expires_at > now:
            return {"status": "busy", "batch_id": str(row.id)}
        was_uncertain = row.state in {"sending", "uncertain"}
        row.state, row.lease_id, row.lease_expires_at = "sending", uuid4(), now + timedelta(seconds=240)
        row.logical_attempts += 1
        chunks = {
            chunk.chunk_id: chunk
            for chunk in self.db.scalars(
                select(m.IndexChunk).where(
                    m.IndexChunk.generation_id == identifier, m.IndexChunk.chunk_id.in_(row.chunk_ids)
                )
            )
        }
        if set(chunks) != set(row.chunk_ids) or any(
            chunks[key].row_sha256 != row.batch_manifest["rows"][key]
            or canonical_sha(chunks[key].payload) != chunks[key].row_sha256
            for key in chunks
        ):
            raise DomainError("persisted_expected_batch_corrupt", 409)
        return {
            "status": "claimed",
            "batch_id": str(row.id),
            "lease_id": str(row.lease_id),
            "generation_id": str(identifier),
            "collection_name": value.collection_name,
            "read_before_write": was_uncertain,
            "rows": [chunks[key].payload for key in row.chunk_ids],
            "batch_manifest": row.batch_manifest,
            "logical_attempt": row.logical_attempts,
        }
