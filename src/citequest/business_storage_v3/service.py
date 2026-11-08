"""Durable object admission around the existing verified BlobStore.

Storage IO never occurs inside this service's SQL transactions. A logical blob
or prepared receipt is NOT end-user authorization; callers must recheck current
File/Work/source permissions in their own short transaction before publishing.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.orm import Session, sessionmaker

from citequest.platform_v3.blob_config import BlobStoreConfig, create_blob_store
from citequest.platform_v3.blob_store import BlobIdentity, BlobReference
from citequest.v2.domain import Actor, DomainError
from citequest.v2.storage import StoredBlob

from .contracts import (
    OperationDeadline,
    PreparedBlobReceipt,
    StorageBinding,
    authorize_domain,
    digest,
    parse_reference,
    reference_value,
)
from .io_process import SupervisedBlobIO


class BusinessStorage:
    def __init__(self, factory: sessionmaker[Session], config: BlobStoreConfig,
                 *, io_executor: SupervisedBlobIO | None = None) -> None:
        self.config = BlobStoreConfig.model_validate(config.model_dump(mode="python"), strict=True)
        self.binding = StorageBinding.configured(self.config)
        # B1/B2 prove PG + official SDK/emulator behavior. A supervised cloud IO
        # executor and product authorization orchestration are separate gates.
        if config.mode == "cloud":
            raise DomainError("business_cloud_io_supervision_not_integrated", 503)
        self.factory = factory
        self.io_executor = io_executor
        if io_executor is not None and (io_executor.binding != self.binding
                or io_executor.profile.maximum_object_bytes != self.config.maximum_object_bytes):
            raise DomainError("business_blob_io_executor_binding_mismatch", 503)
        # A supervised path must not construct an SDK client/perform ADC in the
        # parent. The child imports/constructs it only after durable sent + GO.
        self.store = None if io_executor is not None else create_blob_store(self.config)

    def _lock(self, db: Session, identity: BlobIdentity) -> None:
        value = digest({"domain": identity.security_domain, "sha256": identity.sha256, "store_id": self.binding.store_id})
        key = int.from_bytes(bytes.fromhex(value)[:8], "big", signed=True)
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})

    def _identity_params(self, identity: BlobIdentity) -> dict:
        return {"domain": identity.security_domain, "sha": identity.sha256, "size": identity.size_bytes,
                "store": self.binding.store_id, "binding": self.binding.sha256}

    def _registered(self, db: Session, *, allow_register: bool = False) -> None:
        params = {"store": self.binding.store_id, "binding": self.binding.sha256,
                  "backend": self.binding.backend, "bucket": self.binding.bucket}
        if allow_register:
            db.execute(text("""INSERT INTO cq_app.business_blob_stores(store_id,binding_sha256,backend,bucket)
                VALUES(:store,:binding,:backend,:bucket) ON CONFLICT(store_id) DO NOTHING"""), params)
        row = db.execute(text("SELECT * FROM cq_app.business_blob_stores WHERE store_id=:store"), params).mappings().one_or_none()
        if row is None:
            raise DomainError("business_blob_store_not_registered", 503)
        if (row["binding_sha256"] != self.binding.sha256 or row["backend"] != self.binding.backend
                or row["bucket"] != self.binding.bucket):
            raise DomainError("business_blob_store_binding_conflict", 409)

    def _location(self, db: Session, identity: BlobIdentity) -> BlobReference | None:
        self._registered(db)
        row = db.execute(text("""SELECT l.*, b.size_bytes AS logical_size FROM cq_app.business_blob_locations l
            JOIN cq_app.blobs b USING(security_domain,sha256)
            WHERE l.security_domain=:domain AND l.sha256=:sha AND l.store_id=:store"""),
            self._identity_params(identity)).mappings().one_or_none()
        if row is None:
            return None
        reference = parse_reference(row["reference"])
        self.binding.validate(reference, identity)
        if (row["binding_sha256"] != self.binding.sha256 or row["size_bytes"] != identity.size_bytes
                or row["logical_size"] != identity.size_bytes or row["reference_sha256"] != digest(reference_value(reference))):
            raise DomainError("business_blob_location_integrity_conflict", 409)
        return reference

    def resolve_location(self, db: Session, identity: BlobIdentity) -> BlobReference:
        """SQL-only; call only after current business authorization, not from raw HTTP identity."""
        reference = self._location(db, identity)
        if reference is None:
            raise DomainError("business_blob_location_missing", 503)
        return reference

    @staticmethod
    def _event(db: Session, row: dict, event: str, payload: dict | None = None) -> None:
        db.execute(text("""INSERT INTO cq_app.business_blob_admission_events(admission_id,seq,event_type,payload)
            VALUES(:id,:revision,:event,CAST(:payload AS jsonb))"""),
            {"id": row["id"], "revision": row["revision"], "event": event,
             "payload": json.dumps(payload or {}, separators=(",", ":"))})

    def _receipt(self, db: Session, row: dict) -> PreparedBlobReceipt:
        identity = BlobIdentity(row["security_domain"], row["sha256"], row["size_bytes"])
        reference = self.resolve_location(db, identity)
        reference_sha = digest(reference_value(reference))
        if (row["state"] != "verified" or row["binding_sha256"] != self.binding.sha256
                or row["location_reference_sha256"] != reference_sha):
            raise DomainError("business_blob_admission_not_verified", 409)
        return PreparedBlobReceipt(row["id"], row["owner_id"], row["operation_id"], identity,
                                   self.binding.sha256, reference_sha, row["deadline_at"])

    def _reserve(self, actor: Actor, operation_id: UUID, identity: BlobIdentity, deadline: OperationDeadline) -> dict:
        payload = digest({"identity": asdict(identity), "binding": self.binding.sha256})
        with self.factory.begin() as db:
            self._registered(db, allow_register=True)
            operation_key = int.from_bytes(bytes.fromhex(digest({"owner": str(actor.owner_id), "operation": str(operation_id)}))[:8],
                                           "big", signed=True)
            db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": operation_key})
            self._lock(db, identity)
            existing = db.execute(text("""SELECT * FROM cq_app.business_blob_admissions
                WHERE owner_id=:owner AND operation_id=:op FOR UPDATE"""),
                {"owner": actor.owner_id, "op": operation_id}).mappings().one_or_none()
            if existing is not None:
                if existing["payload_sha256"] != payload:
                    raise DomainError("business_blob_operation_payload_conflict", 409)
                return dict(existing)
            # Only intent proves that no GO/data IO was authorized. Closing an
            # expired intent permits a new explicit operation, never a replay
            # of sent/unknown or a reset of the original deadline/history.
            expired = db.execute(text("""UPDATE cq_app.business_blob_admissions
                SET state='failed',revision=revision+1,completed_at=clock_timestamp(),
                    error_code='deadline_expired_before_any_sent'
                WHERE security_domain=:domain AND sha256=:sha AND store_id=:store
                AND state='intent' AND deadline_at<=clock_timestamp() RETURNING *"""),
                self._identity_params(identity)).mappings().all()
            for closed in expired:
                self._event(db, closed, "expired_intent_without_physical_dispatch")
            inflight = db.execute(text("""SELECT 1 FROM cq_app.business_blob_admissions
                WHERE security_domain=:domain AND sha256=:sha AND store_id=:store
                AND state IN ('intent','sent','unknown') LIMIT 1"""), self._identity_params(identity)).first()
            if inflight:
                raise DomainError("business_blob_identity_inflight_or_unknown", 409)
            row = db.execute(text("""INSERT INTO cq_app.business_blob_admissions
                (id,owner_id,operation_id,payload_sha256,security_domain,sha256,size_bytes,store_id,binding_sha256,state,deadline_at)
                VALUES(:id,:owner,:op,:payload,:domain,:sha,:size,:store,:binding,'intent',
                    clock_timestamp()+make_interval(secs=>CAST(:remaining AS double precision))) RETURNING *"""),
                {**self._identity_params(identity), "id": uuid4(), "owner": actor.owner_id,
                 "op": operation_id, "payload": payload,
                 "remaining": deadline.remaining()}).mappings().one()
            self._event(db, row, "intent_created")
            return dict(row)

    def _dispatch(self, admission_id: UUID) -> None:
        with self.factory.begin() as db:
            row = db.execute(text("""UPDATE cq_app.business_blob_admissions SET state='sent',revision=revision+1,
                dispatch_started_at=clock_timestamp() WHERE id=:id AND state='intent' AND deadline_at>clock_timestamp() RETURNING *"""),
                {"id": admission_id}).mappings().one_or_none()
            if row is None:
                expired = db.execute(text("""UPDATE cq_app.business_blob_admissions SET state='failed',revision=revision+1,
                    completed_at=clock_timestamp(),error_code='deadline_before_physical_dispatch'
                    WHERE id=:id AND state='intent' AND deadline_at<=clock_timestamp() RETURNING *"""),
                    {"id": admission_id}).mappings().one_or_none()
                if expired is not None:
                    self._event(db, expired, "expired_without_physical_dispatch")
            else:
                self._event(db, row, "physical_io_may_start")
        if row is None:
            raise DomainError("business_blob_dispatch_not_claimed", 409)

    def _unknown(self, admission_id: UUID, code: str) -> None:
        with self.factory.begin() as db:
            row = db.execute(text("""UPDATE cq_app.business_blob_admissions SET state='unknown',revision=revision+1,
                error_code=:code WHERE id=:id AND state IN ('sent','unknown') RETURNING *"""),
                {"id": admission_id, "code": code}).mappings().one_or_none()
            if row is not None:
                self._event(db, row, "unknown_retained", {"code": code})

    def _verified(self, admission_id: UUID, reference: BlobReference) -> PreparedBlobReceipt:
        identity = reference.identity
        self.binding.validate(reference, identity)
        value = reference_value(reference)
        reference_sha = digest(value)
        with self.factory.begin() as db:
            self._lock(db, identity)
            row = db.execute(text("SELECT * FROM cq_app.business_blob_admissions WHERE id=:id FOR UPDATE"),
                             {"id": admission_id}).mappings().one()
            if row["state"] == "verified":
                return self._receipt(db, row)
            if (row["state"] not in {"sent", "unknown"} or row["binding_sha256"] != self.binding.sha256
                    or (row["security_domain"], row["sha256"], row["size_bytes"])
                    != (identity.security_domain, identity.sha256, identity.size_bytes)):
                raise DomainError("business_blob_verified_identity_conflict", 409)
            db.execute(text("""INSERT INTO cq_app.blobs(security_domain,sha256,size_bytes)
                VALUES(:domain,:sha,:size) ON CONFLICT(security_domain,sha256) DO NOTHING"""),
                self._identity_params(identity))
            logical_size = db.execute(text("SELECT size_bytes FROM cq_app.blobs WHERE security_domain=:domain AND sha256=:sha"),
                                      self._identity_params(identity)).scalar_one()
            if logical_size != identity.size_bytes:
                raise DomainError("business_blob_logical_size_conflict", 409)
            previous = self._location(db, identity)
            if previous is not None and previous != reference:
                raise DomainError("business_blob_immutable_location_conflict", 409)
            if previous is None:
                db.execute(text("""INSERT INTO cq_app.business_blob_locations
                    (security_domain,sha256,store_id,size_bytes,binding_sha256,reference_sha256,reference)
                    VALUES(:domain,:sha,:store,:size,:binding,:reference_sha,CAST(:reference AS jsonb))"""),
                    {**self._identity_params(identity), "reference_sha": reference_sha, "reference": json.dumps(value)})
            updated = db.execute(text("""UPDATE cq_app.business_blob_admissions SET state='verified',revision=revision+1,
                location_reference_sha256=:ref,error_code=NULL,completed_at=clock_timestamp() WHERE id=:id RETURNING *"""),
                {"id": admission_id, "ref": reference_sha}).mappings().one()
            self._event(db, updated, "verified_location_recorded", {"reference_sha256": reference_sha})
            return self._receipt(db, updated)

    @contextmanager
    def _admitted(self, identity: BlobIdentity, source: BinaryIO, deadline: OperationDeadline) -> Iterator[BinaryIO]:
        if identity.size_bytes > self.config.maximum_object_bytes:
            raise DomainError("business_blob_size_exceeds_profile", 413)
        with tempfile.SpooledTemporaryFile(max_size=2 * 1024**2, mode="w+b", dir=self.config.spool_dir) as spool:
            sha, size = hashlib.sha256(), 0
            while True:
                deadline.remaining()
                part = source.read(65536)
                if not part:
                    break
                size += len(part)
                if size > identity.size_bytes:
                    raise DomainError("business_blob_upload_size_mismatch", 422)
                sha.update(part)
                spool.write(part)
            if size != identity.size_bytes or sha.hexdigest() != identity.sha256:
                raise DomainError("business_blob_upload_identity_mismatch", 422)
            spool.seek(0)
            yield spool

    @contextmanager
    def _preparation_transport(self, identity: BlobIdentity, source: BinaryIO, deadline: OperationDeadline
                              ) -> Iterator[tuple[Callable, Callable]]:
        if self.io_executor is None:
            with self._admitted(identity, source, deadline) as admitted:
                def start(_existing: BlobReference | None) -> None:
                    deadline.remaining()

                def finish(existing: BlobReference | None) -> BlobReference:
                    if existing is not None:
                        with self.store.open_verified(existing):
                            pass
                        return existing
                    return self.store.put(identity, admitted)

                yield start, finish
        else:
            with self.io_executor.operation(deadline, identity=identity, source=source) as operation, operation.physical():
                def start(existing: BlobReference | None) -> None:
                    if existing is not None:
                        operation.start("read", reference=existing)
                    else:
                        operation.start("put", identity=identity)

                def finish(existing: BlobReference | None) -> BlobReference:
                    value = operation.go()
                    reference = parse_reference(value["reference"])
                    if existing is not None:
                        with operation.verified_stream(existing):
                            pass
                    return reference

                yield start, finish

    def prepare_upload(self, actor: Actor, operation_id: UUID, identity: BlobIdentity, source: BinaryIO,
                       *, deadline: OperationDeadline) -> PreparedBlobReceipt:
        authorize_domain(actor, identity)
        if not isinstance(operation_id, UUID):
            raise DomainError("business_blob_operation_id_invalid", 422)
        with self._preparation_transport(identity, source, deadline) as (start, finish):
            row = self._reserve(actor, operation_id, identity, deadline)
            if row["state"] == "verified":
                with self.factory.begin() as db:
                    receipt = self._receipt(db, row)
                    self.validate_prepared(db, actor, receipt, operation_id=operation_id, expected_identity=identity)
                    return receipt
            if row["state"] != "intent":
                raise DomainError("business_blob_reconciliation_required", 409)
            deadline.remaining()
            with self.factory.begin() as db:
                existing = self._location(db, identity)
            start(existing)  # Supervised child READY is local-only, no client/ADC/socket.
            self._dispatch(row["id"])  # Commit must complete before physical IO.
            try:
                deadline.remaining()
                reference = finish(existing)  # GO only after durable sent.
                deadline.remaining()
                receipt = self._verified(row["id"], reference)
                deadline.remaining()
                with self.factory.begin() as db:
                    self.validate_prepared(db, actor, receipt, operation_id=operation_id, expected_identity=identity)
                deadline.remaining()
                return receipt
            except Exception:
                # If SQL is unavailable, sent remains a durable unknown barrier.
                # Never replace this original exception with a fabricated success.
                with suppress(Exception):
                    self._unknown(row["id"], "physical_io_or_verification_unresolved")
                raise

    def reconcile(self, actor: Actor, admission_id: UUID, *, deadline: OperationDeadline) -> PreparedBlobReceipt | None:
        deadline.remaining()
        with self.factory.begin() as db:
            row = db.execute(text("""SELECT * FROM cq_app.business_blob_admissions
                WHERE id=:id AND owner_id=:owner"""), {"id": admission_id, "owner": actor.owner_id}).mappings().one_or_none()
            if row is None:
                raise DomainError("not_found", 404)
            identity = BlobIdentity(row["security_domain"], row["sha256"], row["size_bytes"])
            authorize_domain(actor, identity)
            if row["binding_sha256"] != self.binding.sha256:
                raise DomainError("business_blob_admission_binding_mismatch", 409)
            if row["state"] == "verified":
                return self._receipt(db, row)
            if row["state"] not in {"sent", "unknown"}:
                raise DomainError("business_blob_admission_not_reconcilable", 409)
        try:
            if self.io_executor is not None:
                with self.io_executor.operation(deadline) as operation, operation.physical():
                    value = operation.run("reconcile", identity=identity)
                    reference = parse_reference(value["reference"]) if value["reference"] is not None else None
            elif self.binding.backend == "gcs":
                reference = self.store.reconcile(identity)
            else:
                reference = BlobReference(identity, "filesystem", self.binding.store_id, identity.key)
                try:
                    with self.store.open_verified(reference):
                        pass
                except FileNotFoundError:
                    reference = None
            deadline.remaining()
            if reference is None:
                self._unknown(admission_id, "reconcile_not_observed_late_completion_possible")
                return None
            return self._verified(admission_id, reference)
        except Exception:
            with suppress(Exception):
                self._unknown(admission_id, "reconcile_unresolved")
            raise

    def validate_prepared(self, db: Session, actor: Actor, receipt: PreparedBlobReceipt,
                          *, operation_id: UUID, expected_identity: BlobIdentity) -> StoredBlob:
        """SQL-only, current verified pointer; business permissions remain the caller's job."""
        authorize_domain(actor, expected_identity)
        if receipt.owner_id != actor.owner_id or receipt.operation_id != operation_id or receipt.identity != expected_identity:
            raise DomainError("business_blob_prepared_receipt_mismatch", 409)
        row = db.execute(text("""SELECT * FROM cq_app.business_blob_admissions
            WHERE id=:id AND owner_id=:owner AND operation_id=:op AND deadline_at>clock_timestamp()"""),
            {"id": receipt.admission_id, "owner": actor.owner_id, "op": operation_id}).mappings().one_or_none()
        if row is None or self._receipt(db, row) != receipt:
            raise DomainError("business_blob_prepared_receipt_mismatch", 409)
        return StoredBlob(expected_identity.security_domain, expected_identity.sha256, expected_identity.size_bytes)

    @contextmanager
    def open_verified(self, reference: BlobReference, *, deadline: OperationDeadline) -> Iterator[BinaryIO]:
        self.binding.validate(reference, reference.identity)
        deadline.remaining()
        manager = (self.io_executor.open_verified(reference, deadline=deadline) if self.io_executor is not None
                   else self.store.open_verified(reference))
        with manager as stream:
            deadline.remaining()
            yield stream
            deadline.remaining()

    @contextmanager
    def materialized(self, reference: BlobReference, *, deadline: OperationDeadline) -> Iterator[Path]:
        """Single-operation local spool, not a persistent CAS path or public response field."""
        with tempfile.TemporaryDirectory(prefix="cq-blob-", dir=self.config.spool_dir) as temporary:
            directory = Path(temporary)
            directory.chmod(0o700)
            path = directory / "source.bin"
            with self.open_verified(reference, deadline=deadline) as verified, path.open("xb") as output:
                while True:
                    deadline.remaining()
                    part = verified.read(65536)
                    if not part:
                        break
                    output.write(part)
            path.chmod(0o400)
            try:
                yield path
            finally:
                path.chmod(0o600)
