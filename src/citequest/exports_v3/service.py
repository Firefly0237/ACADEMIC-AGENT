"""Durable export queue, pointer idempotency and current-authority downloads."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from citequest.agents_v3.contracts import AgentError, ReviewSource, canonical, digest
from citequest.platform_v3.blob_store import BlobIdentity, BlobReference, BlobStore
from citequest.research_v3.school_contracts import SchoolDocument
from citequest.research_v3.school_export import validate_export_document
from citequest.v2.domain import Actor, DomainError
from citequest.v2.pagination import Pager

from .contracts import ExportCancel, ExportCreate, ExportDTO, Format

TABLE = "cq_app.agent_document_exports"
MEDIA = {
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "html": "text/html; charset=utf-8",
    "pdf": "application/pdf",
}


@dataclass(frozen=True)
class ExportProfile:
    id: str
    font_profile: str
    toolchain_sha256: str
    deadline_seconds: int = 90
    lease_seconds: int = 120
    owner_active_limit: int = 2
    global_active_limit: int = 100

    def __post_init__(self) -> None:
        if not self.id or len(self.id) > 120 or not self.font_profile or len(self.font_profile) > 120:
            raise ValueError("explicit_export_toolchain_and_font_profiles_required")
        if len(self.toolchain_sha256) != 64 or any(c not in "abcdef0123456789" for c in self.toolchain_sha256):
            raise ValueError("trusted_toolchain_sha256_required")
        if self.deadline_seconds != 90 or self.lease_seconds != 120:
            raise ValueError("reviewed_export_deadline_and_fencing_lease_required")
        if self.owner_active_limit != 2 or self.global_active_limit != 100:
            raise ValueError("reviewed_export_queue_resource_bounds_required")


class ExportService:
    def __init__(
        self,
        factory: sessionmaker[Session],
        authorize_task_result: Callable[[Actor, UUID], dict],
        store: BlobStore,
        profile: ExportProfile,
        pager: Pager,
        *,
        authorize_version: Callable[[Actor, UUID, ReviewSource], dict] | None = None,
    ) -> None:
        self.factory, self.authorize, self.store, self.profile, self.pager = (
            factory,
            authorize_task_result,
            store,
            profile,
            pager,
        )
        self.authorize_version = authorize_version

    def ready(self) -> None:
        with self.factory() as db:
            if not db.execute(
                text(
                    """SELECT to_regclass('cq_app.agent_document_exports') IS NOT NULL
                    AND to_regclass('cq_app.agent_export_operations') IS NOT NULL
                    AND to_regclass('cq_app.agent_export_events') IS NOT NULL
                    AND EXISTS(SELECT 1 FROM information_schema.columns WHERE table_schema='cq_app'
                        AND table_name='agent_document_exports' AND column_name='source_binding')"""
                )
            ).scalar_one():
                raise DomainError("export_migration_required", 503)

    @staticmethod
    def _row(db: Session, owner: UUID, identifier: UUID, *, lock: bool = False) -> dict:
        row = (
            db.execute(
                text(f"SELECT * FROM {TABLE} WHERE id=:id AND owner_id=:owner" + (" FOR UPDATE" if lock else "")),
                {"id": identifier, "owner": owner},
            )
            .mappings()
            .first()
        )
        if row is None:
            raise DomainError("not_found", 404)
        return dict(row)

    def document(self, actor: Actor, task_id: UUID, expected: dict | None = None, *,
                 source_binding: ReviewSource | dict | None = None) -> tuple[dict, SchoolDocument]:
        # The injected parent closure owns Brief/memory/Skill/evidence projection.
        # Do not reimplement its guard or accept a cached result as a fallback.
        saved_binding = expected.get("source_binding") if expected is not None else source_binding
        binding = (saved_binding if isinstance(saved_binding, ReviewSource)
            else ReviewSource.model_validate_json(canonical(saved_binding), strict=True) if saved_binding is not None else None)
        if expected is not None and source_binding is not None:
            explicit = source_binding if isinstance(source_binding, ReviewSource) else ReviewSource.model_validate_json(canonical(source_binding), strict=True)
            if explicit != binding:
                raise DomainError("export_source_version_conflict", 409)
        if binding is not None:
            if binding.source_task_id != task_id or self.authorize_version is None:
                raise DomainError("export_exact_source_authorizer_required", 409)
            task = self.authorize_version(actor, task_id, binding)
        else:
            task = self.authorize(actor, task_id)
        if str(task.get("id")) != str(task_id) or task.get("kind") != "research" or task.get("state") != "succeeded":
            raise DomainError("reviewed_school_task_required", 409)
        result = task.get("result")
        if (
            task.get("result_availability") != "available"
            or not isinstance(result, dict)
            or not result.get("school_document")
        ):
            raise DomainError("school_result_currently_unavailable", 409)
        if result.get("publication_status") != "validated_ai_review" or result.get("status") != "school_document_ready":
            raise DomainError("reviewed_school_document_required", 409)
        document = SchoolDocument.model_validate_json(json.dumps(result["school_document"]), strict=True)
        if document.reference_admission.provenance != "current_authorized_metadata":
            raise DomainError("fixture_document_cannot_be_exported", 409)
        validate_export_document(document)
        if binding is not None and (digest(document.model_dump(mode="json")) != binding.document_sha256
                or digest(document.context_manifest) != binding.context_sha256):
            raise DomainError("export_exact_source_binding_changed", 409)
        if expected is not None and (
            digest(document.model_dump(mode="json")) != expected["document_sha256"]
            or digest(document.context_manifest) != expected["context_sha256"]
            or str(task["project_id"]) != str(expected["project_id"])
        ):
            raise DomainError("export_task_document_or_context_changed", 409)
        return task, document

    @staticmethod
    def _owner_lock(db: Session, actor: Actor) -> None:
        if (
            db.execute(
                text("SELECT id FROM cq_app.owners WHERE id=:id FOR UPDATE"), {"id": actor.owner_id}
            ).scalar_one_or_none()
            is None
        ):
            raise DomainError("not_found", 404)

    @staticmethod
    def _event(db: Session, row: dict, kind: str, payload: dict | None = None) -> None:
        db.execute(
            text(
                "INSERT INTO cq_app.agent_export_events(export_id,seq,event_type,payload) VALUES(:id,:seq,:kind,CAST(:payload AS jsonb))"
            ),
            {"id": row["id"], "seq": row["revision"], "kind": kind, "payload": json.dumps(payload or {})},
        )

    @staticmethod
    def _replay(db: Session, actor: Actor, operation: UUID, fingerprint: str) -> UUID | None:
        row = (
            db.execute(
                text(
                    "SELECT payload_sha256,export_id FROM cq_app.agent_export_operations WHERE owner_id=:owner AND operation_id=:operation"
                ),
                {"owner": actor.owner_id, "operation": operation},
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        if row["payload_sha256"] != fingerprint:
            raise DomainError("idempotency_key_conflict", 409)
        return row["export_id"]

    @staticmethod
    def _record(db: Session, actor: Actor, operation: UUID, fingerprint: str, identifier: UUID) -> None:
        db.execute(
            text(
                "INSERT INTO cq_app.agent_export_operations(owner_id,operation_id,payload_sha256,export_id) VALUES(:owner,:operation,:hash,:id)"
            ),
            {"owner": actor.owner_id, "operation": operation, "hash": fingerprint, "id": identifier},
        )

    def create(self, actor: Actor, task_id: UUID, body: ExportCreate) -> dict:
        formats = sorted(body.formats)
        identity = {"kind": "create", "task_id": str(task_id), "formats": formats}
        binding = body.source_binding.model_dump(mode="json") if body.source_binding is not None else None
        if binding is not None:
            identity["source_binding"] = binding
        fingerprint = digest(identity)
        with self.factory() as db:
            prior = self._replay(db, actor, body.operation_id, fingerprint)
            prior_row = self._row(db, actor.owner_id, prior) if prior is not None else None
        if prior_row is not None:
            return self.present(actor, prior_row)
        # Current source authorization may perform private blob IO. Keep it
        # outside the short global/owner admission locks and reauthorize again
        # at dispatch/publication/download rather than holding SQL over IO.
        task, document = self.document(actor, task_id, source_binding=body.source_binding)
        self.expire()
        with self.factory.begin() as db:
            # One short transaction-wide queue admission lock; no network/office
            # operation occurs under it. It is not a second model budget ledger.
            db.execute(text("SELECT pg_advisory_xact_lock(hashtextextended('citequest:export-queue-admission:v1',0))"))
            self._owner_lock(db, actor)
            replay = self._replay(db, actor, body.operation_id, fingerprint)
            if replay is not None:
                row = self._row(db, actor.owner_id, replay)
            else:
                counts = (
                    db.execute(
                        text(
                            f"SELECT count(*) AS total,count(*) FILTER(WHERE owner_id=:owner) AS owned FROM {TABLE} WHERE state IN ('queued','running')"
                        ),
                        {"owner": actor.owner_id},
                    )
                    .mappings()
                    .one()
                )
                if counts["owned"] >= self.profile.owner_active_limit:
                    raise DomainError("export_owner_active_limit", 429)
                if counts["total"] >= self.profile.global_active_limit:
                    raise DomainError("export_queue_capacity_limit", 429)
                row = dict(
                    db.execute(
                        text(f"""INSERT INTO {TABLE}
                    (id,owner_id,project_id,task_id,document_sha256,context_sha256,requested_formats,toolchain_profile,toolchain_sha256,font_profile,source_binding)
                    VALUES(:id,:owner,:project,:task,:document,:context,CAST(:formats AS jsonb),:toolchain,:toolchain_sha,:font,CAST(:source_binding AS jsonb)) RETURNING *"""),
                        {
                            "id": uuid4(),
                            "owner": actor.owner_id,
                            "project": task["project_id"],
                            "task": task_id,
                            "document": digest(document.model_dump(mode="json")),
                            "context": digest(document.context_manifest),
                            "formats": json.dumps(formats),
                            "toolchain": self.profile.id,
                            "toolchain_sha": self.profile.toolchain_sha256,
                            "font": self.profile.font_profile,
                            "source_binding": json.dumps(binding) if binding is not None else None,
                        },
                    )
                    .mappings()
                    .one()
                )
                self._record(db, actor, body.operation_id, fingerprint, row["id"])
                self._event(db, row, "queued")
        return self.present(actor, row)

    def get(self, actor: Actor, identifier: UUID) -> dict:
        with self.factory() as db:
            row = self._row(db, actor.owner_id, identifier)
        return self.present(actor, row)

    def page(self, actor: Actor, task_id: UUID, *, limit: int = 10, cursor: str | None = None) -> dict:
        task = self.authorize(actor, task_id)
        if str(task.get("id")) != str(task_id) or task.get("kind") != "research":
            raise DomainError("reviewed_school_task_required", 409)
        if not 1 <= limit <= 20:
            raise DomainError("invalid_export_page", 400)
        identity = {"owner": str(actor.owner_id), "task_id": str(task_id), "kind": "school_exports", "limit": limit}
        parameters = {"owner": actor.owner_id, "task": task_id, "limit": limit + 1}
        continuation = ""
        if cursor:
            last = self.pager.decode(cursor, identity)
            continuation = " AND (created_at,id)<(:created,:id)"
            parameters.update(created=datetime.fromisoformat(last["created"]), id=UUID(last["id"]))
        with self.factory() as db:
            rows = [
                dict(row)
                for row in db.execute(
                    text(
                        f"SELECT * FROM {TABLE} WHERE owner_id=:owner AND task_id=:task{continuation} ORDER BY created_at DESC,id DESC LIMIT :limit"
                    ),
                    parameters,
                ).mappings()
            ]
        more = None
        if len(rows) > limit:
            last = rows[limit - 1]
            more = self.pager.encode(
                {
                    "identity": identity,
                    "id": str(last["id"]),
                    "created": last["created_at"].isoformat(),
                    "expires": datetime.now(UTC).timestamp() + 900,
                }
            )
        return {"items": [self.present(actor, row) for row in rows[:limit]], "next_cursor": more}

    def present(self, actor: Actor, row: dict) -> dict:
        available = True
        try:
            self.document(actor, row["task_id"], row)
        except (AgentError, DomainError, ValueError):
            available = False
        files = []
        if available and row["state"] == "succeeded":
            files = [
                {
                    "format": kind,
                    "size_bytes": value["identity"]["size_bytes"],
                    "sha256": value["identity"]["sha256"],
                    "content_url": f"/api/agent/exports/{row['id']}/artifacts/{kind}",
                }
                for kind, value in sorted(row["artifacts"].items())
            ]
        availability = (
            "source_unavailable"
            if not available
            else "available"
            if files
            else "pending"
            if row["state"] in {"queued", "running"}
            else "not_available"
        )
        return ExportDTO(
            id=row["id"],
            task_id=row["task_id"],
            source_binding=row.get("source_binding"),
            state=row["state"],
            requested_formats=row["requested_formats"],
            availability=availability,
            revision=row["revision"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            font_profile=row["font_profile"],
            error_code=row["error_code"],
            artifacts=files,
        ).model_dump(mode="json")

    def cancel(self, actor: Actor, identifier: UUID, body: ExportCancel) -> dict:
        fingerprint = digest({"kind": "cancel", "export_id": str(identifier)})
        with self.factory.begin() as db:
            self._owner_lock(db, actor)
            replay = self._replay(db, actor, body.operation_id, fingerprint)
            row = self._row(db, actor.owner_id, identifier, lock=True)
            if replay is None:
                if row["state"] in {"queued", "running"}:
                    row = dict(
                        db.execute(
                            text(
                                f"UPDATE {TABLE} SET cancellation_requested=true,state='cancelled',revision=revision+1,updated_at=clock_timestamp() WHERE id=:id RETURNING *"
                            ),
                            {"id": identifier},
                        )
                        .mappings()
                        .one()
                    )
                    self._event(db, row, "cancelled")
                self._record(db, actor, body.operation_id, fingerprint, identifier)
            elif replay != identifier:
                raise DomainError("idempotency_key_conflict", 409)
        return self.present(actor, row)

    def expire(self) -> int:
        # A lost/expired worker is a retained failure, never an implicit retry.
        with self.factory.begin() as db:
            rows = (
                db.execute(
                    text(
                        f"UPDATE {TABLE} SET state='failed',error_code='export_deadline_or_lease_expired',revision=revision+1,updated_at=clock_timestamp() WHERE state='running' AND (lease_until<=clock_timestamp() OR deadline_at<=clock_timestamp()) RETURNING *"
                    )
                )
                .mappings()
                .all()
            )
            for row in rows:
                self._event(db, dict(row), "failed", {"code": "export_deadline_or_lease_expired"})
        return len(rows)

    def claim(self, worker_id: str) -> dict | None:
        if not worker_id or len(worker_id) > 100:
            raise ValueError("bounded_export_worker_id_required")
        self.expire()
        with self.factory.begin() as db:
            row = (
                db.execute(
                    text(f"""WITH candidate AS (SELECT id FROM {TABLE}
                 WHERE state='queued' AND cancellation_requested=false AND toolchain_profile=:profile AND toolchain_sha256=:sha AND font_profile=:font
                 ORDER BY created_at,id FOR UPDATE SKIP LOCKED LIMIT 1)
                UPDATE {TABLE} e SET state='running',epoch=epoch+1,worker_id=:worker,
                 lease_until=clock_timestamp()+interval '120 seconds',deadline_at=clock_timestamp()+interval '90 seconds',
                 revision=revision+1,updated_at=clock_timestamp() FROM candidate c WHERE e.id=c.id RETURNING e.*"""),
                    {
                        "profile": self.profile.id,
                        "sha": self.profile.toolchain_sha256,
                        "font": self.profile.font_profile,
                        "worker": worker_id,
                    },
                )
                .mappings()
                .first()
            )
            if row is None:
                return None
            result = dict(row)
            self._event(db, result, "running", {"epoch": result["epoch"]})
        return result

    def fence(self, row: dict, *, db: Session | None = None) -> None:
        def check(session: Session) -> None:
            valid = session.execute(
                text(
                    f"SELECT 1 FROM {TABLE} WHERE id=:id AND owner_id=:owner AND epoch=:epoch AND worker_id=:worker AND state='running' AND cancellation_requested=false AND lease_until>clock_timestamp() AND deadline_at>clock_timestamp()"
                ),
                {"id": row["id"], "owner": row["owner_id"], "epoch": row["epoch"], "worker": row["worker_id"]},
            ).scalar_one_or_none()
            if valid is None:
                raise DomainError("export_execution_fence_lost", 409)

        if db is not None:
            check(db)
        else:
            with self.factory() as session:
                check(session)

    def finish(self, row: dict, artifacts: dict, receipt: dict) -> None:
        self.document(Actor(row["owner_id"]), row["task_id"], row)
        with self.factory.begin() as db:
            current = self._row(db, row["owner_id"], row["id"], lock=True)
            self.fence(row, db=db)
            if set(artifacts) != set(current["requested_formats"]):
                raise ValueError("all_requested_artifact_formats_required")
            finished = dict(
                db.execute(
                    text(
                        f"UPDATE {TABLE} SET state='succeeded',artifacts=CAST(:artifacts AS jsonb),receipt=CAST(:receipt AS jsonb),revision=revision+1,updated_at=clock_timestamp() WHERE id=:id RETURNING *"
                    ),
                    {"id": row["id"], "artifacts": json.dumps(artifacts), "receipt": json.dumps(receipt)},
                )
                .mappings()
                .one()
            )
            self._event(db, finished, "succeeded", {"formats": sorted(artifacts), "actual_model_requests": 0})

    def fail(self, row: dict, code: str, *, audit: dict | None = None) -> dict:
        with self.factory.begin() as db:
            value = (
                db.execute(
                    text(
                        f"UPDATE {TABLE} SET state='failed',error_code=:code,receipt=CAST(:audit AS jsonb),revision=revision+1,updated_at=clock_timestamp() WHERE id=:id AND owner_id=:owner AND epoch=:epoch AND worker_id=:worker AND state='running' RETURNING *"
                    ),
                    {
                        "id": row["id"],
                        "owner": row["owner_id"],
                        "epoch": row["epoch"],
                        "worker": row["worker_id"],
                        "code": code,
                        "audit": json.dumps(audit),
                    },
                )
                .mappings()
                .first()
            )
            if value:
                self._event(db, dict(value), "failed", {"code": code})
            current = self._row(db, row["owner_id"], row["id"])
        return {
            "id": str(current["id"]),
            "state": current["state"],
            "error_code": current["error_code"],
            "actual_model_requests": 0,
        }

    def store_bytes(self, actor: Actor, data: bytes) -> dict:
        if not 0 < len(data) <= 32 * 1024**2:
            raise ValueError("bounded_nonempty_export_artifact_required")
        identity = BlobIdentity(actor.security_domain, digest(data), len(data))
        return asdict(self.store.put(identity, io.BytesIO(data)))

    def download(self, actor: Actor, identifier: UUID, kind: Format) -> tuple[bytes, str, str]:
        with self.factory() as db:
            row = self._row(db, actor.owner_id, identifier)
        self.document(actor, row["task_id"], row)
        if row["state"] != "succeeded" or kind not in row["artifacts"]:
            raise DomainError("export_artifact_not_ready", 409)
        value = row["artifacts"][kind]
        reference = BlobReference(
            BlobIdentity(**value["identity"]),
            value["backend"],
            value["store_id"],
            value["object_key"],
            value.get("bucket"),
            value.get("generation"),
        )
        if reference.identity.security_domain != actor.security_domain:
            raise DomainError("export_artifact_security_domain_mismatch", 403)
        with self.store.open_verified(reference) as stream:
            data = stream.read(32 * 1024**2 + 1)
        if len(data) > 32 * 1024**2:
            raise DomainError("export_artifact_size_exceeded", 413)
        self.document(actor, row["task_id"], row)
        return data, MEDIA[kind], "research-proposal." + kind
