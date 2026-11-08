"""Transactional product operations. Caller owns one Session.begin boundary.

Every private lookup includes server-derived owner; Work metadata has its own ACL.
No LLM, Milvus, implicit commit, checkpoint maintenance or capability fallback.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4, uuid5

from sqlalchemy import delete, exists, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import models as m
from .domain import (
    Actor,
    CollectionCreate,
    CollectionPatch,
    DomainError,
    LibraryPatch,
    LibrarySave,
    NoteCreate,
    NotePatch,
    SourceInput,
    WorkInput,
)
from .storage import CAS, StoredBlob

if TYPE_CHECKING:
    from citequest.business_storage_v3.contracts import PreparedBlobReceipt
    from citequest.business_storage_v3.service import BusinessStorage


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def normalize_identifier(namespace: str, raw: str) -> str:
    value = raw.strip()
    if namespace == "doi":
        value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I).lower()
        if not re.fullmatch(r"10\.\d{4,9}/\S+", value):
            raise DomainError("invalid_doi", 400)
    if not value or any(c.isspace() for c in value) or len(value) > 512:
        raise DomainError("invalid_identifier", 400)
    return value


class Catalog:
    def __init__(self, db: Session, actor: Actor) -> None:
        self.db, self.actor = db, actor

    def work(self, work_id: UUID) -> m.Work:
        access = exists(
            select(m.WorkAccess.work_id).where(
                m.WorkAccess.work_id == m.Work.id,
                m.WorkAccess.owner_id == self.actor.owner_id,
                m.WorkAccess.revoked.is_(False),
            )
        )
        work = self.db.scalar(select(m.Work).where(m.Work.id == work_id, or_(m.Work.public.is_(True), access)))
        if work is None:
            raise DomainError("not_found", 404)
        return work

    def project(self, project_id: UUID) -> m.Project:
        project = self.db.scalar(
            select(m.Project).where(m.Project.id == project_id, m.Project.owner_id == self.actor.owner_id)
        )
        if project is None:
            raise DomainError("not_found", 404)
        return project

    def require_publisher(self) -> None:
        owner = self.db.get(m.Owner, self.actor.owner_id)
        if not self.actor.publisher or owner is None or not owner.can_publish:
            raise DomainError("publisher_required", 403)

    def version(self, version_id: UUID) -> m.WorkVersion:
        version = self.db.scalar(
            select(m.WorkVersion).where(
                m.WorkVersion.id == version_id,
                m.WorkVersion.security_domain.in_([self.actor.security_domain, "public"]),
            )
        )
        if version is None:
            raise DomainError("not_found", 404)
        self.work(version.work_id)
        return version

    def record_source(
        self, work_id: UUID, source: SourceInput, raw_bytes: bytes, store: CAS, *, public: bool = False
    ) -> dict:
        self.lock_owner()
        self.work(work_id)
        domain = "public" if public else self.actor.security_domain
        if public:
            self.require_publisher()
            if not self.work(work_id).public or source.rights_status != "verified" or source.license is None:
                raise DomainError("public_source_requires_verified_rights", 422)
        if hashlib.sha256(raw_bytes).hexdigest() != source.raw_sha256:
            raise DomainError("source_hash_mismatch", 422)
        blob = store.put(self.actor.security_domain, io.BytesIO(raw_bytes), max_bytes=50 * 1024 * 1024)
        self.lock_owner()
        identity = self.db.get(m.SourceIdentity, (domain, source.source, source.native_id))
        if identity is None:
            self.db.add(
                m.SourceIdentity(
                    security_domain=domain, source=source.source, native_id=source.native_id, work_id=work_id
                )
            )
            self.db.flush()
        elif identity.work_id != work_id:
            raise DomainError("source_identity_conflict")
        prior = self.db.scalar(
            select(m.SourceRecord).where(
                m.SourceRecord.security_domain == domain,
                m.SourceRecord.source == source.source,
                m.SourceRecord.native_id == source.native_id,
                m.SourceRecord.source_revision == source.source_revision,
                m.SourceRecord.raw_sha256 == source.raw_sha256,
            )
        )
        if prior is not None:
            if prior.work_id != work_id:
                raise DomainError("source_identity_conflict")
            from .presentation import source_record

            return source_record(prior)
        if self.db.get(m.Blob, (blob.security_domain, blob.sha256)) is None:
            self.db.add(m.Blob(security_domain=blob.security_domain, sha256=blob.sha256, size_bytes=blob.size_bytes))
            self.db.flush()
        obj = m.SourceRecord(
            id=uuid4(),
            work_id=work_id,
            security_domain=domain,
            raw_security_domain=blob.security_domain,
            **source.model_dump(),
            payload={"raw_blob_sha256": blob.sha256},
        )
        self.db.add(obj)
        self.db.flush()
        from .presentation import source_record

        return source_record(obj)

    def record_version(
        self, work_id: UUID, source_sha256: str, kind: str, version_label: str | None, *, public: bool = False
    ) -> dict:
        self.lock_owner()
        self.work(work_id)
        if version_label is not None and (
            not isinstance(version_label, str) or not version_label.strip() or len(version_label) > 100
        ):
            raise DomainError("invalid_version_label", 400)
        domain = "public" if public else self.actor.security_domain
        if public:
            self.require_publisher()
        if kind not in {"preprint", "accepted", "published", "source_unknown"}:
            raise DomainError("invalid_version_kind", 400)
        source = self.db.scalar(
            select(m.SourceRecord).where(
                m.SourceRecord.work_id == work_id,
                m.SourceRecord.raw_sha256 == source_sha256,
                m.SourceRecord.security_domain.in_([domain, "public"]),
            )
        )
        if source is None:
            raise DomainError("source_not_recorded", 422)
        self.lock_owner()
        version_id = uuid5(work_id, digest([domain, source_sha256, kind, version_label]))
        obj = self.db.get(m.WorkVersion, version_id)
        if obj is None:
            obj = m.WorkVersion(
                id=version_id,
                work_id=work_id,
                security_domain=domain,
                source_sha256=source_sha256,
                kind=kind,
                version_label=version_label,
            )
            self.db.add(obj)
            self.db.flush()
        from .presentation import version

        return version(obj)

    def item(self, item_id: UUID, *, deleted: bool = False) -> m.LibraryItem:
        query = select(m.LibraryItem).where(m.LibraryItem.id == item_id, m.LibraryItem.owner_id == self.actor.owner_id)
        if not deleted:
            query = query.where(m.LibraryItem.deleted_at.is_(None))
        item = self.db.scalar(query)
        if item is None:
            raise DomainError("not_found", 404)
        self.work(item.work_id)  # Current revocation overrides saved associations.
        return item

    def collection(self, collection_id: UUID) -> m.Collection:
        obj = self.db.scalar(
            select(m.Collection).where(
                m.Collection.id == collection_id,
                m.Collection.owner_id == self.actor.owner_id,
                m.Collection.deleted_at.is_(None),
            )
        )
        if obj is None:
            raise DomainError("not_found", 404)
        return obj

    def lock_owner(self) -> None:
        if self.db.scalar(select(m.Owner.id).where(m.Owner.id == self.actor.owner_id).with_for_update()) is None:
            raise DomainError("not_found", 404)

    def operation(self, key: UUID, payload: dict, effect: Callable[[], dict]) -> dict:
        """Unique claim waits for competing commit; savepoint protects outer UOW.

        PostgreSQL READ COMMITTED sees the committed winner on the next SELECT.
        Exceptions from the effect propagate; the HTTP transaction rolls back all.
        """
        self.lock_owner()  # One lock order for claims and all related mutations.
        payload_hash = digest(payload)

        def replay(previous: m.Operation | None) -> dict:
            if previous is None or previous.payload_sha256 != payload_hash or previous.result is None:
                raise DomainError("idempotency_conflict")
            result = previous.result
            operation = payload["operation"]
            if operation == "create_work":
                self.work(UUID(result["work_id"]))
            elif operation == "save":
                self.item(UUID(result["library_item_id"]), deleted=True)
            elif operation == "create_collection":
                self.collection(UUID(result["collection_id"]))
            elif operation == "create_note":
                self.note(UUID(result["note_id"]))
            elif operation == "register_file":
                self.file(UUID(result["file_id"]))
            return result

        previous = self.db.get(m.Operation, (self.actor.owner_id, key))
        if previous is not None:
            return replay(previous)
        claim_inserted = False
        try:
            with self.db.begin_nested():
                claim = m.Operation(owner_id=self.actor.owner_id, key=key, payload_sha256=payload_hash, result=None)
                self.db.add(claim)
                self.db.flush()
                claim_inserted = True
                result = effect()
                claim.result = result
                self.db.flush()
            return result
        except IntegrityError:
            if claim_inserted:
                raise  # An effect constraint failure is not a replay receipt.
            return replay(self.db.get(m.Operation, (self.actor.owner_id, key)))

    def outbox(
        self,
        entity: UUID,
        revision: int,
        operation: str,
        payload: dict,
        *,
        entity_type: str = "Work",
        security_domain: str | None = None,
    ) -> None:
        domain = security_domain or self.actor.security_domain
        envelope = {
            "schema_version": "cq.outbox.payload/1",
            "entity_type": entity_type,
            "entity_schema": "contracts/v2/domain.schema.json#/$defs/" + entity_type,
            "entity_id": str(entity),
            "entity_revision": revision,
            "security_domain": domain,
            "operation": operation,
            "data": payload,
        }
        self.db.add(
            m.Outbox(
                idempotency_key=f"{entity_type}:{entity}:{revision}:{operation}",
                entity_id=str(entity),
                entity_revision=revision,
                operation=operation,
                payload_sha256=digest(envelope),
                payload=envelope,
                schema_version="cq.outbox.payload/1",
                entity_type=entity_type,
                security_domain=domain,
            )
        )

    def create_work(self, request: WorkInput, *, public: bool = False) -> dict:
        if public:
            self.require_publisher()
        domain = "public" if public else self.actor.security_domain

        def effect() -> dict:
            self.lock_owner()
            identifiers = [(i.namespace, i.raw, normalize_identifier(i.namespace, i.raw)) for i in request.identifiers]
            if len({(n, v) for n, _r, v in identifiers}) != len(identifiers):
                raise DomainError("duplicate_identifier", 400)
            public_by_key = {
                (namespace, value): i.work_id
                for namespace, _raw, value in identifiers
                if (i := self.db.get(m.Identifier, ("public", namespace, value))) is not None
            }
            local_by_key = {
                (namespace, value): i.work_id
                for namespace, _raw, value in identifiers
                if (i := self.db.get(m.Identifier, (domain, namespace, value))) is not None
            }
            known_public, known_local = set(public_by_key.values()), set(local_by_key.values())
            same_identifier_private_copies = {local_by_key[key] for key in public_by_key if key in local_by_key}
            if known_public and known_local - known_public - same_identifier_private_copies:
                raise DomainError("identifier_conflict")
            chosen = known_public or known_local
            if len(chosen) > 1:
                raise DomainError("identifier_conflict")
            if chosen:
                known = self.work(next(iter(chosen)))
                from .presentation import work as present_work

                return present_work(self, known, public_only=public)
            work = m.Work(
                id=uuid4(),
                title=request.title,
                title_normalized=unicodedata.normalize("NFKC", request.title).casefold(),
                authors=request.authors,
                year=request.year,
                public=public,
                security_domain=domain,
            )
            self.db.add(work)
            self.db.flush()
            self.db.add(m.WorkAccess(owner_id=self.actor.owner_id, work_id=work.id))
            for namespace, raw, value in identifiers:
                self.db.add(
                    m.Identifier(
                        security_domain=domain,
                        namespace=namespace,
                        normalized=value,
                        raw=raw,
                        work_id=work.id,
                        normalizer_version="identifier-v1",
                    )
                )
            from .presentation import work as present_work

            self.db.flush()
            snapshot = present_work(self, work, public_only=public)
            self.outbox(work.id, 1, "upsert", snapshot, entity_type="Work", security_domain=work.security_domain)
            return snapshot

        return self.operation(
            request.client_operation_id,
            {"operation": "create_work", "public": public, **request.model_dump(mode="json")},
            effect,
        )

    def create_collection(self, request: CollectionCreate) -> dict:
        def effect() -> dict:
            self.lock_owner()
            if request.parent_id:
                self.collection(request.parent_id)
            obj = m.Collection(id=uuid4(), owner_id=self.actor.owner_id, name=request.name, parent_id=request.parent_id)
            self.db.add(obj)
            self.db.flush()
            from .presentation import collection

            return collection(obj)

        return self.operation(
            request.client_operation_id, {"operation": "create_collection", **request.model_dump(mode="json")}, effect
        )

    def patch_collection(self, collection_id: UUID, request: CollectionPatch) -> None:
        self.lock_owner()  # Serialize same-owner tree changes; reject write-skew cycles.
        self.collection(collection_id)
        visited = {collection_id}
        parent = request.parent_id
        while parent is not None:
            if parent in visited:
                raise DomainError("collection_cycle", 400)
            visited.add(parent)
            parent = self.collection(parent).parent_id
        changed = self.db.execute(
            update(m.Collection)
            .where(
                m.Collection.id == collection_id,
                m.Collection.owner_id == self.actor.owner_id,
                m.Collection.revision == request.expected_revision,
            )
            .values(name=request.name, parent_id=request.parent_id, revision=m.Collection.revision + 1)
        )
        if changed.rowcount != 1:
            raise DomainError("revision_conflict")

    def save(self, request: LibrarySave) -> dict:
        self.lock_owner()
        self.work(request.work_id)
        if request.collection_id:
            self.collection(request.collection_id)

        def effect() -> dict:
            self.lock_owner()  # Different operation keys still converge on one saved item.
            item = self.db.scalar(
                select(m.LibraryItem).where(
                    m.LibraryItem.owner_id == self.actor.owner_id, m.LibraryItem.work_id == request.work_id
                )
            )
            if item is None:
                item = m.LibraryItem(id=uuid4(), owner_id=self.actor.owner_id, work_id=request.work_id)
                self.db.add(item)
                self.db.flush()
            elif item.deleted_at is not None:
                raise DomainError("item_deleted_use_restore")
            if (
                request.collection_id
                and self.db.get(m.ItemCollection, (self.actor.owner_id, item.id, request.collection_id)) is None
            ):
                self.db.add(
                    m.ItemCollection(owner_id=self.actor.owner_id, item_id=item.id, collection_id=request.collection_id)
                )
            from .presentation import item as present_item

            self.db.flush()
            return present_item(self, item)

        return self.operation(
            request.client_operation_id, {"operation": "save", **request.model_dump(mode="json")}, effect
        )

    def patch_item(self, item_id: UUID, request: LibraryPatch) -> None:
        self.lock_owner()
        self.item(item_id)
        if len(set(request.tags)) != len(request.tags) or len(set(request.collection_ids)) != len(
            request.collection_ids
        ):
            raise DomainError("duplicate_association", 400)
        for collection in request.collection_ids:
            self.collection(collection)
        changed = self.db.execute(
            update(m.LibraryItem)
            .where(
                m.LibraryItem.id == item_id,
                m.LibraryItem.owner_id == self.actor.owner_id,
                m.LibraryItem.revision == request.expected_revision,
                m.LibraryItem.deleted_at.is_(None),
            )
            .values(revision=m.LibraryItem.revision + 1)
        )
        if changed.rowcount != 1:
            raise DomainError("revision_conflict")
        self.db.execute(delete(m.ItemTag).where(m.ItemTag.item_id == item_id))
        self.db.execute(
            delete(m.ItemCollection).where(
                m.ItemCollection.item_id == item_id, m.ItemCollection.owner_id == self.actor.owner_id
            )
        )
        self.db.add_all([m.ItemTag(item_id=item_id, tag=tag) for tag in request.tags])
        self.db.add_all(
            [
                m.ItemCollection(owner_id=self.actor.owner_id, item_id=item_id, collection_id=c)
                for c in request.collection_ids
            ]
        )

    def delete_item(self, item_id: UUID) -> None:
        self.lock_owner()
        item = self.item(item_id, deleted=True)
        if item.deleted_at is None:
            item.deleted_at, item.revision = datetime.now(UTC), item.revision + 1

    def restore_item(self, item_id: UUID) -> None:
        self.lock_owner()
        item = self.item(item_id, deleted=True)
        if item.deleted_at is not None:
            item.deleted_at, item.revision = None, item.revision + 1

    def attach(self, item_id: UUID, file_id: UUID) -> None:
        self.lock_owner()
        item = self.item(item_id)
        file = self.db.scalar(
            select(m.File).where(
                m.File.id == file_id,
                m.File.owner_id == self.actor.owner_id,
                m.File.security_domain == self.actor.security_domain,
                m.File.deleted_at.is_(None),
            )
        )
        if file is None:
            raise DomainError("not_found", 404)
        version = self.version(file.version_id) if file.version_id else None
        if version is None or version.work_id != item.work_id:
            raise DomainError("attachment_work_mismatch", 422)
        if self.db.get(m.ItemFile, (self.actor.owner_id, item_id, file_id)) is None:
            self.db.add(m.ItemFile(owner_id=self.actor.owner_id, item_id=item_id, file_id=file_id))

    def detach(self, item_id: UUID, file_id: UUID) -> None:
        self.lock_owner()
        self.item(item_id)
        self.db.execute(delete(m.ItemFile).where(m.ItemFile.item_id == item_id, m.ItemFile.file_id == file_id))

    def create_note(self, request: NoteCreate) -> dict:
        self.lock_owner()
        self.item(request.library_item_id)
        if request.expected_revision is not None:
            raise DomainError("create_note_requires_null_revision_use_patch", 400)
        if request.evidence_ids:
            raise DomainError("evidence_links_not_available_until_S04", 422)

        def effect() -> dict:
            note = m.Note(
                id=uuid4(), owner_id=self.actor.owner_id, library_item_id=request.library_item_id, body=request.body
            )
            self.db.add(note)
            self.db.flush()
            from .presentation import note as present_note

            return present_note(note)

        return self.operation(
            request.client_operation_id, {"operation": "create_note", **request.model_dump(mode="json")}, effect
        )

    def note(self, note_id: UUID) -> m.Note:
        note = self.db.scalar(
            select(m.Note).where(
                m.Note.id == note_id, m.Note.owner_id == self.actor.owner_id, m.Note.deleted_at.is_(None)
            )
        )
        if note is None:
            raise DomainError("not_found", 404)
        self.item(note.library_item_id)
        return note

    def patch_note(self, note_id: UUID, request: NotePatch) -> None:
        self.lock_owner()
        self.note(note_id)
        changed = self.db.execute(
            update(m.Note)
            .where(
                m.Note.id == note_id,
                m.Note.owner_id == self.actor.owner_id,
                m.Note.revision == request.expected_revision,
            )
            .values(body=request.body, revision=m.Note.revision + 1)
        )
        if changed.rowcount != 1:
            raise DomainError("revision_conflict")

    def delete_note(self, note_id: UUID) -> None:
        self.lock_owner()
        note = self.note(note_id)
        note.deleted_at, note.revision = datetime.now(UTC), note.revision + 1

    def delete_collection(self, collection_id: UUID) -> None:
        self.lock_owner()
        obj = self.collection(collection_id)
        if self.db.scalar(
            select(m.Collection.id)
            .where(
                m.Collection.owner_id == self.actor.owner_id,
                m.Collection.parent_id == collection_id,
                m.Collection.deleted_at.is_(None),
            )
            .limit(1)
        ):
            raise DomainError("collection_has_children")
        self.db.execute(
            delete(m.ItemCollection).where(
                m.ItemCollection.owner_id == self.actor.owner_id, m.ItemCollection.collection_id == collection_id
            )
        )
        obj.deleted_at, obj.revision = datetime.now(UTC), obj.revision + 1

    def file(self, file_id: UUID) -> m.File:
        obj = self.db.scalar(
            select(m.File).where(
                m.File.id == file_id,
                m.File.owner_id == self.actor.owner_id,
                m.File.security_domain == self.actor.security_domain,
                m.File.deleted_at.is_(None),
            )
        )
        if obj is None:
            raise DomainError("not_found", 404)
        if obj.version_id:
            self.version(obj.version_id)
        return obj

    def register_file(self, store: CAS, blob: StoredBlob, media_type: str, version_id: UUID | None, key: UUID) -> dict:
        """Register verified bytes, still queued/unparsed; S04 validates content."""
        self._check_file_blob(blob, media_type)
        store.verify(blob)
        return self._register_verified_file(blob, media_type, version_id, key)

    def register_prepared_file(
        self, storage: BusinessStorage, prepared: PreparedBlobReceipt, blob: StoredBlob,
        media_type: str, version_id: UUID | None, key: UUID, *, operation_id: UUID,
    ) -> dict:
        """SQL-only registration of a current durable, server-issued receipt.

        This shares the existing File/outbox/idempotency effect. Neither this
        internal method nor the receipt is an HTTP capability or an ACL bypass.
        """
        from citequest.platform_v3.blob_store import BlobIdentity

        self._check_file_blob(blob, media_type)
        storage.validate_prepared(self.db, self.actor, prepared, operation_id=operation_id,
            expected_identity=BlobIdentity(blob.security_domain, blob.sha256, blob.size_bytes))
        return self._register_verified_file(blob, media_type, version_id, key)

    def _check_file_blob(self, blob: StoredBlob, media_type: str) -> None:
        self.lock_owner()
        if blob.security_domain != self.actor.security_domain:
            raise DomainError("not_found", 404)
        if media_type not in {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "text/plain",
            "text/markdown",
        }:
            raise DomainError("unsupported_media", 415)

    def _register_verified_file(self, blob: StoredBlob, media_type: str, version_id: UUID | None, key: UUID) -> dict:
        """Shared effect; callers must establish byte verification first."""
        if version_id:
            self.version(version_id)

        def effect() -> dict:
            self.lock_owner()
            known = self.db.get(m.Blob, (blob.security_domain, blob.sha256))
            if known is None:
                self.db.add(
                    m.Blob(security_domain=blob.security_domain, sha256=blob.sha256, size_bytes=blob.size_bytes)
                )
                self.db.flush()
            elif known.size_bytes != blob.size_bytes:
                raise DomainError("blob_integrity_conflict")
            file = m.File(
                id=uuid4(),
                owner_id=self.actor.owner_id,
                version_id=version_id,
                security_domain=blob.security_domain,
                sha256=blob.sha256,
                media_type=media_type,
            )
            self.db.add(file)
            self.db.flush()
            from .presentation import file as present_file

            snapshot = present_file(self, file)
            self.outbox(file.id, 1, "upsert", snapshot, entity_type="File")
            return snapshot

        return self.operation(
            key,
            {
                "operation": "register_file",
                "sha256": blob.sha256,
                "size_bytes": blob.size_bytes,
                "media_type": media_type,
                "version_id": str(version_id) if version_id else None,
            },
            effect,
        )

    def delete_file(self, file_id: UUID) -> None:
        self.lock_owner()
        obj = self.file(file_id)
        obj.deleted_at, obj.revision = datetime.now(UTC), obj.revision + 1
        self.db.execute(delete(m.ItemFile).where(m.ItemFile.file_id == file_id))
        from .presentation import file as present_file

        self.outbox(file_id, obj.revision, "revoke", present_file(self, obj), entity_type="File")

    def append_history(self, kind: str, target_id: str, snapshot_id: UUID | None = None) -> UUID:
        if (
            kind not in {"search", "view", "download_requested", "download_completed", "export"}
            or not target_id
            or len(target_id) > 100
        ):
            raise DomainError("invalid_history", 400)
        obj = m.History(
            id=uuid4(), owner_id=self.actor.owner_id, kind=kind, target_id=target_id, snapshot_id=snapshot_id
        )
        self.db.add(obj)
        return obj.id

    def delete_history(self, history_id: UUID) -> None:
        self.lock_owner()
        row = self.db.scalar(
            select(m.History).where(m.History.id == history_id, m.History.owner_id == self.actor.owner_id)
        )
        if row is None:
            raise DomainError("not_found", 404)
        self.db.delete(row)
