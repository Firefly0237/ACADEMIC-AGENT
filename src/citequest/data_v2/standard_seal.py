"""Fixed clean-exact adoption after real source/SQL/qrels checks, separate from index publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID, uuid5

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from citequest.v2 import models as core
from citequest.v2.domain import DomainError
from citequest.v2.services import digest

from . import models as m
from .schemas import ProjectionInput, SnapshotFreezeInput

if TYPE_CHECKING:
    from .landing import Landing

REVISION = "9573fb284a1026c998df47024b888a163f0f0e25"
SOURCE_MANIFEST = "91a42512b833e35937a453690261559afc05d4cf85e504ca990195d73849cbb0"
CANONICAL_CLEAN = "dbc1ead59600e4f219c0de62a23960138491f18052cd5e0cc8762ec9ce9b1b97"
QUERIES = "2e6cd1e16a5ca64455b210a69763b3f53737a76be239f63e1c4f4a73da325382"
PROFILE = "litsearch-clean-exact/1"


def _read(path: Path, expected: str, *, maximum: int) -> bytes:
    if path.is_symlink():
        raise DomainError("source_seal_symlink", 422)
    with path.open("rb") as stream:
        value = stream.read(maximum + 1)
    if len(value) > maximum or hashlib.sha256(value).hexdigest() != expected:
        raise DomainError("source_seal_artifact_changed", 422)
    return value


def freeze_standard(
    landing: Landing,
    snapshot_id: UUID,
    request: SnapshotFreezeInput,
    export_dir: Path,
    qrels_path: Path,
) -> dict:
    # Callers supply the trusted Arrow export path, not a user's upload path.
    # The two canonical dataset hashes also prevent declaring different source
    # text/queries standard merely by recalculating a local manifest.
    manifest_bytes = _read(export_dir / "verification.json", request.expected_export_manifest_sha256, maximum=1024**2)
    exported = json.loads(manifest_bytes)
    queries = json.loads(_read(qrels_path, QUERIES, maximum=2 * 1024**2))
    if (
        request.expected_queries_sha256 != QUERIES
        or exported["state"] != "exact_metadata_source_export_not_ingestion"
        or exported["representation_profile"] != PROFILE
        or exported["source_revision"] != REVISION
        or exported["source_snapshot_manifest_sha256"] != SOURCE_MANIFEST
        or exported["standard_canonical_sha256"] != CANONICAL_CLEAN
        or exported["rows"] != 64183
        or not 1 <= len(exported["files"]) <= 129
    ):
        raise DomainError("not_fixed_standard_corpus", 409)
    edges = [target for query in queries for target in query["corpusids"]]
    if len(queries) != 597 or len(edges) != 639 or any(type(target) is not int for target in edges):
        raise DomainError("native_standard_qrels_mismatch", 422)
    landing.catalog.lock_owner()
    snapshot = landing.db.get(m.SourceSnapshot, snapshot_id, with_for_update=True, populate_existing=True)
    if snapshot is None or snapshot.security_domain not in {"public", landing.catalog.actor.security_domain}:
        raise DomainError("not_found", 404)
    if snapshot.security_domain == "public":
        landing.catalog.require_publisher()
    if (
        snapshot.status == "disabled"
        or snapshot.source_namespace != "litsearch_clean"
        or snapshot.source_revision != REVISION
        or snapshot.source_manifest_sha256 != SOURCE_MANIFEST
        or snapshot.representation_profile != PROFILE
        or snapshot.purpose != "standard"
        or snapshot.policy_sha256 != landing.policy_sha256
        or snapshot.evaluation_sha256 != landing.evaluation_sha256
    ):
        raise DomainError("source_snapshot_not_adoptable", 409)
    payload = {
        "operation": "freeze_fixed_standard_snapshot",
        "snapshot_id": str(snapshot_id),
        "request": request.model_dump(mode="json"),
    }

    def effect() -> dict:
        # A different operation ID cannot skip the full proof. Same-ID replays
        # are already protected by Catalog.operation and current ACL checks.
        artifacts = list(
            landing.db.scalars(select(m.SourceArtifact).where(m.SourceArtifact.snapshot_id == snapshot_id))
        )
        actual_artifacts = [
            {key: getattr(item, key) for key in ("path", "sha256", "size_bytes", "format", "row_group_sizes")}
            for item in artifacts
        ]
        if sorted(actual_artifacts, key=lambda item: item["path"]) != sorted(
            exported["artifacts"], key=lambda item: item["path"]
        ):
            raise DomainError("source_snapshot_export_artifacts_differ", 409)
        source_count = landing.db.scalar(
            select(func.count()).select_from(m.SourceRow).where(m.SourceRow.snapshot_id == snapshot_id)
        )
        projection_count = landing.db.scalar(
            select(func.count())
            .select_from(m.SourceProjection)
            .join(m.SourceRow, m.SourceRow.id == m.SourceProjection.source_row_id)
            .where(m.SourceRow.snapshot_id == snapshot_id)
        )
        if source_count != 64183 or projection_count != 64183:
            raise DomainError("standard_snapshot_count_mismatch", 409)
        canonical, identities, seen_positions = hashlib.sha256(), set(), set()
        empty = 0
        for index, item in enumerate(exported["files"]):
            if item["path"] != f"batch-{index:04d}.jsonl" or not 1 <= item["rows"] <= 500:
                raise DomainError("source_export_batch_identity_mismatch", 422)
            blob = _read(export_dir / item["path"], item["sha256"], maximum=16 * 1024**2)
            if len(blob) != item["bytes"]:
                raise DomainError("source_export_batch_size_mismatch", 422)
            rows = [ProjectionInput.model_validate_json(line) for line in blob.splitlines()]
            if len(rows) != item["rows"]:
                raise DomainError("source_export_batch_rows_mismatch", 422)
            ids = [uuid5(snapshot_id, "native:" + row.native_id) for row in rows]
            actual = {
                row.id: (row, projection)
                for row, projection in landing.db.execute(
                    select(m.SourceRow, m.SourceProjection)
                    .join(m.SourceProjection, m.SourceProjection.source_row_id == m.SourceRow.id)
                    .where(m.SourceRow.id.in_(ids), m.SourceProjection.representation_profile == PROFILE)
                    .execution_options(populate_existing=True)
                )
            }
            for source_id, expected in zip(ids, rows, strict=True):
                position = (expected.locator.artifact_path, expected.locator.row_ordinal)
                if (
                    expected.native_id in identities
                    or position in seen_positions
                    or expected.native_id_source_type != "integer"
                ):
                    raise DomainError("source_export_duplicate_identity_or_position", 422)
                identities.add(expected.native_id)
                seen_positions.add(position)
                source, projection = actual.get(source_id, (None, None))
                pointer = digest(
                    {
                        "snapshot_manifest_sha256": SOURCE_MANIFEST,
                        "native_id": expected.native_id,
                        "native_id_source_type": "integer",
                        **expected.locator.model_dump(mode="json"),
                    }
                )
                expected_sha = digest(
                    {
                        "source_pointer_sha256": pointer,
                        "representation_profile": PROFILE,
                        "title": expected.title,
                        "abstract": expected.abstract,
                        "field_provenance": expected.field_provenance,
                    }
                )
                if (
                    source is None
                    or projection is None
                    or source.source_pointer_sha256 != pointer
                    or source.security_domain != snapshot.security_domain
                    or source.native_id != expected.native_id
                    or any(getattr(source, key) != value for key, value in expected.locator.model_dump().items())
                    or projection.projection_sha256 != expected_sha
                    or projection.title != expected.title
                    or projection.abstract != expected.abstract
                    or projection.field_provenance != expected.field_provenance
                ):
                    raise DomainError("stored_standard_projection_differs_from_source", 409)
                canonical.update(
                    (
                        json.dumps(
                            {
                                "corpusid": int(expected.native_id),
                                "title": expected.title,
                                "abstract": expected.abstract,
                            },
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode()
                )
                empty += int(not ((expected.title or "").strip() or (expected.abstract or "").strip()))
        if (
            len(identities) != 64183
            or canonical.hexdigest() != CANONICAL_CLEAN
            or any(str(target) not in identities for target in edges)
        ):
            raise DomainError("standard_source_or_qrels_not_complete", 409)
        seal = {
            "schema_version": "cq.standard-source-seal/1",
            "snapshot_id": str(snapshot_id),
            "source_manifest_sha256": SOURCE_MANIFEST,
            "export_manifest_sha256": request.expected_export_manifest_sha256,
            "standard_canonical_sha256": CANONICAL_CLEAN,
            "queries_sha256": QUERIES,
            "record_count": 64183,
            "query_count": 597,
            "qrels_edges": 639,
            "qrels_ID_reachable_edges": 639,
            "native_ID_set_sha256": digest(sorted(identities)),
            "empty_representations_preserved": empty,
            "policy_sha256": landing.policy_sha256,
            "evaluation_sha256": landing.evaluation_sha256,
            "no_nonempty_gold_hard_gate_added": True,
            "source_immutable": True,
        }
        seal_sha = digest(seal)
        if snapshot.status == "staged":
            if not snapshot.profile_adopted or snapshot.seal_sha256 != seal_sha or snapshot.seal != seal:
                raise DomainError("standard_snapshot_seal_conflict", 409)
        elif snapshot.status == "landing" and not snapshot.profile_adopted and snapshot.seal is None:
            snapshot.status, snapshot.profile_adopted = "staged", True
            snapshot.seal, snapshot.seal_sha256 = seal, seal_sha
        else:
            raise DomainError("source_snapshot_not_adoptable", 409)
        envelope = {
            "schema_version": "cq.outbox.payload/1",
            "entity_type": "SourceSnapshot",
            "entity_revision": 1,
            "entity_id": str(snapshot_id),
            "security_domain": snapshot.security_domain,
            "operation": "stage",
            "entity_schema": "cq.standard-source-seal/1",
            "data": {"status": "staged", "profile_adopted": True, "seal_sha256": seal_sha, "seal": seal},
        }
        event_key = f"SourceSnapshot:{snapshot_id}:1:stage"
        landing.db.flush()
        landing.db.execute(
            insert(core.Outbox)
            .values(
                idempotency_key=event_key,
                entity_id=str(snapshot_id),
                entity_revision=1,
                security_domain=snapshot.security_domain,
                entity_type="SourceSnapshot",
                operation="stage",
                schema_version="cq.outbox.payload/1",
                payload=envelope,
                payload_sha256=digest(envelope),
            )
            .on_conflict_do_nothing(index_elements=["idempotency_key"])
        )
        event = landing.db.scalar(select(core.Outbox).where(core.Outbox.idempotency_key == event_key))
        if event is None or event.payload_sha256 != digest(envelope):
            raise DomainError("source_snapshot_outbox_seal_conflict", 409)
        return {
            "snapshot_id": str(snapshot_id),
            "status": "staged",
            "profile_adopted": True,
            "seal_sha256": seal_sha,
            "seal": seal,
            "index_published": False,
        }

    return landing.catalog.operation(request.client_operation_id, payload, effect)
