"""Short product transactions; network/native parsing never occurs in a UOW."""
from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from jsonschema import Draft202012Validator, FormatChecker
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from citequest.v2 import models as core
from citequest.v2.domain import Actor, DomainError
from citequest.v2.presentation import file as base_file
from citequest.v2.services import Catalog, digest
from citequest.v2.storage import CAS, StoredBlob

from . import models as m
from .text import MaterialError, load_limits, prepare_text

if TYPE_CHECKING:
    from citequest.business_storage_v3.contracts import PreparedBlobReceipt
    from citequest.business_storage_v3.service import BusinessStorage

ROOT = Path(__file__).resolve().parents[3]


@lru_cache(maxsize=1)
def _contract() -> dict:
    return json.loads((ROOT / "contracts/v2/domain.schema.json").read_text(encoding="utf-8"))


def checked(name: str, value: dict, *, response: bool = False) -> dict:
    schema = _contract()
    valid = Draft202012Validator({"$ref": "#/$defs/" + name, "$defs": schema["$defs"]}, format_checker=FormatChecker()).is_valid(value)
    if not valid:
        raise DomainError("material_response_contract_failed" if response else "invalid_input", 500 if response else 400)
    return value


@dataclass(frozen=True)
class UploadAdmission:
    upload_id: UUID
    owner_id: UUID
    file_name: str
    media_type: str
    size_bytes: int
    expected_sha256: str
    project_id: UUID | None
    expires_at: datetime
    max_bytes: int
    file_id: UUID | None


def material_blob_operation(kind: str, key: UUID) -> UUID:
    """Server-derived storage operation; never supplied by the HTTP caller."""
    if kind not in {"upload", "text"}:
        raise ValueError("unsupported_material_blob_operation")
    return uuid5(NAMESPACE_URL, f"citequest:material-blob:{kind}:{key}")


class MaterialService:
    def __init__(self, db: Session, actor: Actor, store: CAS | None) -> None:
        self.db, self.actor, self.store = db, actor, store
        self.catalog = Catalog(db, actor)

    def create_upload(self, body: dict, *, key: UUID) -> dict:
        checked("UploadInput", body)
        name = body["file_name"]
        if any(c in name for c in ("/", "\\", "\x00", "\r", "\n")) or name in {".", ".."}:
            raise DomainError("invalid_file_name", 400)
        limits = load_limits(ROOT / "contracts/v2/policy.json")
        maximum = {"application/pdf": limits.pdf_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document": limits.docx_bytes,
                   "text/plain": limits.text_bytes, "text/markdown": limits.text_bytes}[body["media_type"]]
        if body["size_bytes"] > maximum:
            raise DomainError("file_too_large", 413)
        project = UUID(body["project_id"]) if body["project_id"] else None
        if project:
            self.catalog.project(project)

        def effect() -> dict:
            upload = m.MaterialUpload(id=uuid4(), owner_id=self.actor.owner_id, project_id=project, file_name=name, display_title=name,
                                      media_type=body["media_type"], size_bytes=body["size_bytes"], expected_sha256=body["expected_sha256"],
                                      expires_at=datetime.now(UTC) + timedelta(hours=1), file_receipt=None)
            self.db.add(upload)
            self.db.flush()
            return {"upload_id": str(upload.id), "state": "queued", "expires_at": upload.expires_at.isoformat(),
                    "max_bytes": upload.size_bytes, "upload_url": f"/api/v2/uploads/{upload.id}/content"}

        result = self.catalog.operation(key, {"operation": "create_upload", **body}, effect)
        self.upload(UUID(result["upload_id"]))
        return checked("UploadTicket", result, response=True)

    def upload(self, upload_id: UUID) -> UploadAdmission:
        row = self.db.scalar(select(m.MaterialUpload).where(m.MaterialUpload.id == upload_id, m.MaterialUpload.owner_id == self.actor.owner_id))
        if row is None or (row.file_id is None and row.expires_at <= datetime.now(UTC)):
            raise DomainError("not_found", 404)
        if row.project_id:
            self.catalog.project(row.project_id)
        if row.file_id:
            self.catalog.file(row.file_id)
        return UploadAdmission(row.id, row.owner_id, row.file_name, row.media_type, row.size_bytes, row.expected_sha256, row.project_id, row.expires_at, row.size_bytes, row.file_id)

    def put_upload_content(self, upload_id: UUID, blob: StoredBlob) -> dict:
        if self.store is None:
            raise DomainError("material_local_storage_not_configured", 503)
        self.catalog.lock_owner()
        admission = self.upload(upload_id)
        self._check_upload_blob(admission, blob)
        self.store.verify(blob)
        return self._commit_upload(admission, blob, lambda: self.catalog.register_file(
            self.store, blob, admission.media_type, None, upload_id))

    def _check_upload_blob(self, admission: UploadAdmission, blob: StoredBlob) -> None:
        if blob.security_domain != self.actor.security_domain:
            raise DomainError("not_found", 404)
        if blob.sha256 != admission.expected_sha256 or blob.size_bytes != admission.size_bytes:
            raise DomainError("upload_hash_or_size_mismatch", 409)

    def _commit_upload(self, admission: UploadAdmission, blob: StoredBlob, register: Callable[[], dict]) -> dict:
        row = self.db.get(m.MaterialUpload, admission.upload_id)
        if row.file_id is not None:
            return self.get_file(row.file_id)
        receipt = register()
        file_id = UUID(receipt["file_id"])
        row.file_id, row.file_receipt = file_id, receipt
        self.db.add(m.MaterialVersion(owner_id=self.actor.owner_id, file_id=file_id, security_domain=self.actor.security_domain,
                                      source_sha256=blob.sha256, media_type=admission.media_type))
        ingestion = m.MaterialIngestion(id=uuid4(), owner_id=self.actor.owner_id, file_id=file_id, state="queued", revision=1)
        self.db.add(ingestion)
        self.db.flush()
        self.db.add(m.MaterialEvent(ingestion_id=ingestion.id, seq=1, owner_id=self.actor.owner_id, state="queued", occurred_at=datetime.now(UTC)))
        return checked("File", receipt, response=True)

    def put_prepared_upload(self, upload_id: UUID, prepared: PreparedBlobReceipt,
                            storage: BusinessStorage, *, operation_id: UUID | None = None) -> dict:
        """Current ACL and prepared-pointer recheck; no object IO in this UOW."""
        self.catalog.lock_owner()
        admission = self.upload(upload_id)
        blob = StoredBlob(prepared.identity.security_domain, prepared.identity.sha256, prepared.identity.size_bytes)
        self._check_upload_blob(admission, blob)
        operation_id = operation_id or material_blob_operation("upload", upload_id)
        return self._commit_upload(admission, blob, lambda: self.catalog.register_prepared_file(
            storage, prepared, blob, admission.media_type, None, upload_id, operation_id=operation_id))

    @staticmethod
    def text_bytes(body: dict) -> tuple[bytes, str]:
        checked("TextMaterialInput", body)
        if not body["title"].strip() or any(ord(c) < 32 for c in body["title"]):
            raise DomainError("invalid_material_title", 400)
        data = body["text"].encode("utf-8")
        media = "text/markdown" if body["format"] == "md" else "text/plain"
        try:
            prepare_text(data, media, load_limits(ROOT / "contracts/v2/policy.json"))
        except MaterialError as exc:
            raise DomainError(exc.code.lower(), 413 if "EXCEEDED" in exc.code else 400) from exc
        return data, media

    def text_replay(self, body: dict, *, key: UUID) -> dict | None:
        """Reject changed keys and project a committed resource before any IO."""
        previous = self.db.get(core.Operation, (self.actor.owner_id, key))
        if previous is None:
            return None
        if previous.payload_sha256 != digest({"operation": "ingest_text", **body}) or previous.result is None:
            raise DomainError("idempotency_conflict", 409)
        return self.get_file(UUID(previous.result["file_id"]))

    def ingest_prepared_text(self, body: dict, *, key: UUID, prepared: PreparedBlobReceipt,
                             storage: BusinessStorage) -> dict:
        """The public text operation shares its original ticket/File effect."""
        data, media = self.text_bytes(body)
        expected = StoredBlob(self.actor.security_domain, hashlib.sha256(data).hexdigest(), len(data))
        if (prepared.identity.security_domain, prepared.identity.sha256, prepared.identity.size_bytes) != (
                expected.security_domain, expected.sha256, expected.size_bytes):
            raise DomainError("material_prepared_text_identity_mismatch", 409)

        def effect() -> dict:
            ticket = self.create_upload({"file_name": "pasted-material.md" if media == "text/markdown" else "pasted-material.txt",
                "media_type": media, "size_bytes": len(data), "expected_sha256": expected.sha256, "project_id": None},
                key=uuid5(NAMESPACE_URL, str(key) + ":text-upload"))
            upload_id = UUID(ticket["upload_id"])
            self.db.get(m.MaterialUpload, upload_id).display_title = body["title"]
            return self.put_prepared_upload(upload_id, prepared, storage, operation_id=material_blob_operation("text", key))

        result = self.catalog.operation(key, {"operation": "ingest_text", **body}, effect)
        return self.get_file(UUID(result["file_id"]))

    def ingest_text(self, body: dict, *, key: UUID) -> dict:
        if self.store is None:
            raise DomainError("material_local_storage_not_configured", 503)
        data, media = self.text_bytes(body)

        def effect() -> dict:
            blob = self.store.put(self.actor.security_domain, io.BytesIO(data), len(data))
            ticket = self.create_upload({"file_name": "pasted-material.md" if media == "text/markdown" else "pasted-material.txt", "media_type": media, "size_bytes": len(data),
                                         "expected_sha256": blob.sha256, "project_id": None}, key=uuid5(NAMESPACE_URL, str(key) + ":text-upload"))
            self.db.get(m.MaterialUpload, UUID(ticket["upload_id"])).display_title = body["title"]
            return self.put_upload_content(UUID(ticket["upload_id"]), blob)

        result = self.catalog.operation(key, {"operation": "ingest_text", **body}, effect)
        self.catalog.file(UUID(result["file_id"]))
        return result

    def _ingestion(self, file_id: UUID) -> m.MaterialIngestion:
        self.catalog.file(file_id)
        row = self.db.scalar(select(m.MaterialIngestion).where(m.MaterialIngestion.owner_id == self.actor.owner_id, m.MaterialIngestion.file_id == file_id))
        if row is None:
            raise DomainError("ingestion_not_found", 404)
        return row

    def _binding(self, ingestion: m.MaterialIngestion) -> m.MaterialBindingCheck | None:
        if ingestion.binding_check_id is None:
            return None
        return self.db.scalar(select(m.MaterialBindingCheck).where(m.MaterialBindingCheck.id == ingestion.binding_check_id,
                                                                  m.MaterialBindingCheck.owner_id == self.actor.owner_id,
                                                                  m.MaterialBindingCheck.file_id == ingestion.file_id))

    def get_file(self, file_id: UUID) -> dict:
        file = self.catalog.file(file_id)
        payload = base_file(self.catalog, file)
        ingestion = self.db.scalar(select(m.MaterialIngestion).where(m.MaterialIngestion.owner_id == self.actor.owner_id, m.MaterialIngestion.file_id == file_id))
        if ingestion and ingestion.representation_id:
            rep = self.db.get(m.ParsedRepresentation, ingestion.representation_id)
            if rep and rep.owner_id == self.actor.owner_id:
                if rep.source_sha256 != file.sha256 or rep.security_domain != file.security_domain:
                    raise DomainError("representation_source_mismatch", 409)
                payload["coverage"] = rep.payload["coverage"]
                payload["page_count"] = rep.page_count
                # S04 exposes readable/searchable sources; the Agent endpoint is
                # not implemented until S09. Do not turn source readiness into
                # an advertised working QA capability.
                payload["can_qa"] = False
        if file.status in {"ready", "partial"}:
            payload["can_download"] = True
            payload["can_preview"] = file.media_type in {"application/pdf", "text/plain", "text/markdown"}
        return checked("File", payload, response=True)

    def get_representation(self, file_id: UUID) -> dict:
        file = self.catalog.file(file_id)
        ingestion = self._ingestion(file_id)
        if ingestion.representation_id is None:
            raise DomainError("representation_not_ready", 409)
        row = self.db.get(m.ParsedRepresentation, ingestion.representation_id)
        if row is None or row.owner_id != self.actor.owner_id or row.file_id != file_id or row.source_sha256 != file.sha256:
            raise DomainError("not_found", 404)
        return checked("ParsedDocument", row.payload, response=True)

    def get_ingestion(self, file_id: UUID) -> dict:
        row = self._ingestion(file_id)
        file = self.catalog.file(file_id)
        grade = self._binding(row)
        material_ready = row.representation_id is not None and self.db.scalar(select(m.MaterialFragment.id).where(
            m.MaterialFragment.owner_id == self.actor.owner_id, m.MaterialFragment.file_id == file_id,
            m.MaterialFragment.representation_id == row.representation_id).limit(1)) is not None
        formal_ready = bool(material_ready and grade and grade.identity_status == "hash_verified" and grade.version_id == file.version_id and self.db.scalar(select(m.MaterialEvidence.id)
                            .join(m.MaterialChunk, m.MaterialChunk.id == m.MaterialEvidence.chunk_id)
                            .join(m.MaterialFragment, m.MaterialFragment.id == m.MaterialChunk.fragment_id)
                            .where(m.MaterialEvidence.owner_id == self.actor.owner_id, m.MaterialFragment.file_id == file_id,
                                   m.MaterialChunk.version_id == grade.version_id).limit(1)) is not None)
        events = self.db.scalars(select(m.MaterialEvent).where(m.MaterialEvent.owner_id == self.actor.owner_id, m.MaterialEvent.ingestion_id == row.id).order_by(m.MaterialEvent.seq)).all()
        return {"ingestion_id": str(row.id), "file_id": str(file_id), "state": row.state, "revision": row.revision, "error_code": row.error_code,
                "representation_id": row.representation_id, "lease_active": row.lease_id is not None and row.lease_expires_at is not None and row.lease_expires_at > datetime.now(UTC),
                "binding_source": grade.binding_source if grade else "none", "identity_status": grade.identity_status if grade else "unverified",
                "material_source_ready": bool(material_ready), "paper_source_ready": formal_ready, "formal_evidence_ready": formal_ready,
                "execution_outcome": row.execution_outcome, "active_elapsed_seconds": row.active_elapsed_seconds,
                "execution_budget_seconds": row.execution_budget_seconds, "deadline_at": row.deadline_at.isoformat() if row.deadline_at else None,
                "execution_success": row.execution_outcome == "within_deadline" and row.state == "ready"
                                     and row.active_elapsed_seconds is not None and row.execution_budget_seconds is not None
                                     and 0 <= row.active_elapsed_seconds <= row.execution_budget_seconds,
                "events": [{"seq": event.seq, "state": event.state, "at": event.occurred_at.isoformat(), "error_code": event.error_code} for event in events]}

    def get_fragments(self, file_id: UUID, after: int = 0, limit: int = 50, query: str | None = None) -> dict:
        if type(after) is not int or type(limit) is not int or after < 0 or not 1 <= limit <= 50:
            raise DomainError("invalid_page", 400)
        file = self.catalog.file(file_id)
        representation = self.get_representation(file_id)
        statement = select(m.MaterialFragment).where(m.MaterialFragment.owner_id == self.actor.owner_id, m.MaterialFragment.file_id == file_id,
                                                     m.MaterialFragment.representation_id == representation["representation_id"], m.MaterialFragment.ordinal > after)
        if query is not None:
            if not query.strip() or len(query) > 2000 or "\x00" in query:
                raise DomainError("invalid_query", 400)
            literal = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            statement = statement.where(or_(m.MaterialFragment.search_document.op("@@")(func.plainto_tsquery("simple", query)),
                                            m.MaterialFragment.index_text.ilike("%" + literal + "%", escape="\\")))
        rows = self.db.scalars(statement.order_by(m.MaterialFragment.ordinal).limit(limit + 1)).all()
        return {"schema_version": "cq.material.fragments/1", "file_id": str(file_id), "representation_id": representation["representation_id"],
                "items": [{**r.payload, "work_version_id": str(file.version_id) if file.version_id else None} for r in rows[:limit]],
                "next_after": rows[limit - 1].ordinal if len(rows) > limit else None, "coverage": representation["coverage"], "missing_ranges": representation["missing_ranges"]}

    def read_content(self, file_id: UUID) -> tuple[StoredBlob, str, str]:
        file = self.catalog.file(file_id)
        if not self.get_file(file_id)["can_download"]:
            raise DomainError("file_not_available", 409)
        blob = self.db.get(core.Blob, (file.security_domain, file.sha256))
        upload = self.db.scalar(select(m.MaterialUpload).where(m.MaterialUpload.owner_id == self.actor.owner_id, m.MaterialUpload.file_id == file_id))
        return StoredBlob(file.security_domain, file.sha256, blob.size_bytes), file.media_type, upload.file_name if upload else "document"

    def get_artifacts(self, file_id: UUID) -> dict:
        representation = self.get_representation(file_id)
        rows = self.db.scalars(select(m.ArtifactAsset).where(m.ArtifactAsset.owner_id == self.actor.owner_id,
                               m.ArtifactAsset.representation_id == representation["representation_id"]).order_by(m.ArtifactAsset.role)).all()
        return {"schema_version": "cq.material.artifacts/1", "file_id": str(file_id), "representation_id": representation["representation_id"],
                "items": [{"role": row.role, "sha256": row.sha256, "size_bytes": row.size_bytes, "availability": "registered",
                           "provenance": row.provenance} for row in rows]}

    def read_artifact(self, file_id: UUID, role: str) -> tuple[StoredBlob, str, str]:
        if len(role) > 60 or not role.replace("_", "").isalnum():
            raise DomainError("invalid_artifact_role", 400)
        representation = self.get_representation(file_id)
        row = self.db.get(m.ArtifactAsset, (self.actor.owner_id, representation["representation_id"], role))
        if row is None or row.security_domain != self.actor.security_domain:
            raise DomainError("not_found", 404)
        return StoredBlob(row.security_domain, row.sha256, row.size_bytes), "application/json", role + ".json"

    def _advance(self, row: m.MaterialIngestion, file: core.File, target: str, error_code: str | None = None) -> None:
        transitions = json.loads((ROOT / "contracts/v2/policy.json").read_text(encoding="utf-8"))["upload_transitions"]
        if target == row.state:
            return
        if target not in transitions[row.state]:
            raise DomainError("ingestion_transition_conflict", 409)
        row.state, row.error_code, row.revision = target, error_code, row.revision + 1
        file.status, file.revision = target, file.revision + 1
        if target in {"ready", "partial", "failed", "cancelled"}:
            row.lease_id = row.lease_expires_at = None
        if target in {"failed", "cancelled"}:
            row.execution_outcome = target
        self.db.add(m.MaterialEvent(ingestion_id=row.id, seq=row.revision, owner_id=self.actor.owner_id,
                                   state=target, occurred_at=datetime.now(UTC), error_code=error_code))
        self.db.flush()
        # Revocation/deletion remains authoritative even if a native worker was
        # already running. Never resurrect an inaccessible File in the outbox.
        snapshot = base_file(self.catalog, file) if target == "cancelled" else self.get_file(file.id)
        self.catalog.outbox(file.id, file.revision, "revoke" if file.deleted_at else "upsert", snapshot, entity_type="File")

    def cancel(self, file_id: UUID) -> dict:
        self.catalog.lock_owner()
        file = self.catalog.file(file_id)
        row = self._ingestion(file_id)
        if row.state == "cancelled":
            return self.get_file(file_id)
        self._advance(row, file, "cancelled", "user_cancelled")
        return self.get_file(file_id)

    def bind_version(self, file_id: UUID, version_id: UUID) -> dict:
        """A user association remains unverified; no formal paper evidence."""
        self.catalog.lock_owner()
        file, version = self.catalog.file(file_id), self.catalog.version(version_id)
        row = self._ingestion(file_id)
        if file.version_id is not None and file.version_id != version_id:
            raise DomainError("version_binding_conflict", 409)
        previous = self._binding(row)
        if previous is not None:
            return self.get_file(file_id)
        check = m.MaterialBindingCheck(id=uuid4(), owner_id=self.actor.owner_id, file_id=file_id, version_id=version_id,
                                       binding_source="user_explicit", identity_status="unverified", source_record_id=None,
                                       proof={"schema_version": "cq.material.binding/1", "file_sha256": file.sha256,
                                              "version_source_sha256": version.source_sha256, "machine_identity_checks": []})
        self.db.add(check)
        self.db.flush()
        file.version_id, file.revision, row.binding_check_id = version_id, file.revision + 1, check.id
        self.catalog.outbox(file.id, file.revision, "upsert", self.get_file(file_id), entity_type="File")
        return self.get_file(file_id)

    def bind_verified_source(self, file_id: UUID, version_id: UUID) -> dict:
        """Internal publisher route only; the HTTP user binding never calls it.

        The verified source must contain these exact file bytes and already be
        a trusted public full-text publication, not merely metadata about a PDF.
        """
        self.catalog.lock_owner()
        self.catalog.require_publisher()
        file, version = self.catalog.file(file_id), self.catalog.version(version_id)
        row = self._ingestion(file_id)
        if file.version_id is not None and file.version_id != version_id:
            raise DomainError("version_binding_conflict", 409)
        source = self.db.scalar(select(core.SourceRecord).where(
            core.SourceRecord.work_id == version.work_id, core.SourceRecord.security_domain == "public",
            core.SourceRecord.raw_sha256 == file.sha256,
            core.SourceRecord.raw_sha256 == version.source_sha256, core.SourceRecord.rights_status == "verified",
            core.SourceRecord.source.in_(["trusted_pdf_download", "licensed_corpus_file"])))
        if (source is None or version.security_domain != "public" or version.kind not in {"published", "accepted", "preprint"}
                or not source.source_url or not source.source_url.startswith("https://")):
            raise DomainError("source_identity_proof_unavailable", 409)
        raw_blob = self.db.get(core.Blob, (source.raw_security_domain, source.raw_sha256))
        if raw_blob is None:
            raise DomainError("source_identity_proof_unavailable", 409)
        self.store.verify(StoredBlob(source.raw_security_domain, source.raw_sha256, raw_blob.size_bytes))
        original_blob = self.db.get(core.Blob, (file.security_domain, file.sha256))
        self.store.verify(StoredBlob(file.security_domain, file.sha256, original_blob.size_bytes))
        previous = self._binding(row)
        if previous is not None and previous.identity_status == "hash_verified":
            return self.get_file(file_id)
        check = m.MaterialBindingCheck(id=uuid4(), owner_id=self.actor.owner_id, file_id=file_id, version_id=version_id,
                                       binding_source="trusted_source", identity_status="hash_verified", source_record_id=source.id,
                                       proof={"schema_version": "cq.material.binding/1", "file_sha256": file.sha256, "source_sha256": source.raw_sha256,
                                              "version_source_sha256": version.source_sha256, "source_record_id": str(source.id),
                                              "machine_identity_checks": ["trusted_public_source", "source_bytes_reverified", "exact_file_source_version_sha256"]})
        self.db.add(check)
        self.db.flush()
        file.version_id, row.binding_check_id, file.revision = version_id, check.id, file.revision + 1
        self._publish_formal(file, row)
        self.catalog.outbox(file.id, file.revision, "upsert", self.get_file(file_id), entity_type="File")
        return self.get_file(file_id)

    def _publish_formal(self, file: core.File, ingestion: m.MaterialIngestion) -> None:
        grade = self._binding(ingestion)
        if grade is None or grade.identity_status != "hash_verified" or grade.version_id != file.version_id or not ingestion.representation_id:
            return
        self.catalog.version(file.version_id)
        fragments = self.db.scalars(select(m.MaterialFragment).where(m.MaterialFragment.owner_id == self.actor.owner_id,
                                      m.MaterialFragment.file_id == file.id, m.MaterialFragment.representation_id == ingestion.representation_id)).all()
        for fragment in fragments:
            data = fragment.payload
            chunk_base = {"representation_id": ingestion.representation_id, "version_id": str(file.version_id), "security_domain": self.actor.security_domain,
                          "source_text": fragment.source_text, "source_text_sha256": hashlib.sha256(fragment.source_text.encode()).hexdigest(),
                          "index_text": fragment.index_text, "granularity": data.get("granularity", "paragraph"), "locator": data["locator"],
                          "normalization_map_sha256": None, "chunker_version": data["chunker_version"], "tokenizer_version": data["tokenizer_version"], "token_count": fragment.token_count}
            chunk_id = digest({"owner_id": str(self.actor.owner_id), "file_id": str(file.id), "fragment_id": fragment.id, "chunk": chunk_base})
            if self.db.get(m.MaterialChunk, chunk_id) is None:
                self.db.add(m.MaterialChunk(id=chunk_id, owner_id=self.actor.owner_id, fragment_id=fragment.id, version_id=file.version_id,
                                           payload=checked("Chunk", {"chunk_id": chunk_id, **chunk_base}, response=True)))
                self.db.flush()
                evidence_base = {"chunk_id": chunk_id, "version_id": str(file.version_id), "quote": fragment.source_text,
                                 "quote_sha256": chunk_base["source_text_sha256"], "locator": data["locator"], "locator_version": "cq.material.source-view/1",
                                 "granularity": "fulltext", "relation": "insufficient", "conditions": ["Source identity/locator verified; no claim relation assessed"],
                                 "judgment_source": "rule", "check_revision": 1}
                evidence_id = digest({"owner_id": str(self.actor.owner_id), **evidence_base})
                self.db.add(m.MaterialEvidence(id=evidence_id, owner_id=self.actor.owner_id, chunk_id=chunk_id,
                                              payload=checked("Evidence", {"evidence_id": evidence_id, **evidence_base}, response=True)))
        self.db.flush()
