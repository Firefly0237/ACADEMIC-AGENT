"""One ingestion's storage adapter, with current SQL guards around external IO.

Paths are ephemeral execution handles. No private object identity or receipt is
an end-user capability, and this class never publishes a parser result itself.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO
from uuid import UUID, uuid5

from sqlalchemy.orm import Session

from citequest.business_storage_v3.contracts import OperationDeadline, PreparedBlobReceipt
from citequest.business_storage_v3.service import BusinessStorage
from citequest.materials_v3.broker_protocol import rooted
from citequest.platform_v3.blob_store import BlobIdentity, BlobReference
from citequest.v2.domain import Actor, DomainError
from citequest.v2.storage import StoredBlob


class MaterialExecutionStorage:
    """Trusted parent only; never passed to an untrusted parser process."""

    def __init__(self, storage: BusinessStorage, actor: Actor, execution_id: UUID,
                 deadline: OperationDeadline, authorize: Callable[[Session], object]) -> None:
        self.storage, self.actor, self.execution_id = storage, actor, execution_id
        self.deadline, self.authorize = deadline, authorize
        self.prepared: dict[str, PreparedBlobReceipt] = {}
        self.observed: dict[str, BlobReference] = {}

    def _identity(self, blob: StoredBlob) -> BlobIdentity:
        if blob.security_domain != self.actor.security_domain:
            raise DomainError("not_found", 404)
        identity = BlobIdentity(blob.security_domain, blob.sha256, blob.size_bytes)
        if identity.size_bytes > self.storage.config.maximum_object_bytes:
            raise DomainError("material_object_exceeds_storage_profile", 413)
        return identity

    def _capture(self, blob: StoredBlob) -> BlobReference:
        self.deadline.remaining()
        identity = self._identity(blob)
        with self.storage.factory.begin() as db:
            self.authorize(db)
            return self.storage.resolve_location(db, identity)

    def _recheck(self, blob: StoredBlob, reference: BlobReference) -> None:
        self.deadline.remaining()
        if self._capture(blob) != reference:
            raise DomainError("material_storage_location_changed", 409)
        self.observed[blob.sha256] = reference

    @contextmanager
    def opened(self, blob: StoredBlob) -> Iterator[BinaryIO]:
        reference = self._capture(blob)
        with self.storage.open_verified(reference, deadline=self.deadline) as stream:
            self._recheck(blob, reference)
            yield stream
            self._recheck(blob, reference)

    def verify(self, blob: StoredBlob) -> None:
        # open_verified already checks all bytes and yields the same verified FD.
        with self.opened(blob):
            pass

    def read(self, blob: StoredBlob) -> bytes:
        with self.opened(blob) as stream:
            raw = stream.read(blob.size_bytes + 1)
            if len(raw) != blob.size_bytes:
                raise DomainError("material_storage_original_size_changed", 409)
            return raw

    @contextmanager
    def materialized(self, blob: StoredBlob) -> Iterator[Path]:
        reference = self._capture(blob)
        # Keep the actual supervised local quota for the entire parser handoff,
        # rather than copying to an uncounted permanent CAS-like path.
        materialize = (self.storage.io_executor.materialized if self.storage.io_executor is not None
                       else self.storage.materialized)
        with materialize(reference, deadline=self.deadline) as path:
            self._recheck(blob, reference)
            yield path
            self._recheck(blob, reference)

    @contextmanager
    def handoff(self, blob: StoredBlob, *, root: Path, ingestion_id: UUID) -> Iterator[Path]:
        """Owned same-lease transport to a no-secret/no-network parser mount.

        This is an ephemeral copy, not an alternate CAS or a persistent task
        locator. Keep the verified IO lease/quota until the parser has stopped.
        """
        if not isinstance(ingestion_id, UUID):
            raise ValueError("typed_ingestion_id_required")
        root = root.resolve(strict=True)
        relative = f"data/runtime-s04-linux/artifacts/{ingestion_id}/{self.execution_id}/input"
        directory = rooted(root, relative)
        directory.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory.mkdir(mode=0o700, exist_ok=False)
        source = directory / "source.bin"
        try:
            with self.opened(blob) as stream:
                sha, size = hashlib.sha256(), 0
                with source.open("xb") as output:
                    while True:
                        self.deadline.remaining()
                        part = stream.read(65536)
                        if not part:
                            break
                        size += len(part)
                        if size > blob.size_bytes:
                            raise DomainError("material_handoff_source_size_changed", 409)
                        sha.update(part)
                        output.write(part)
                    output.flush()
                    os.fsync(output.fileno())
                if size != blob.size_bytes or sha.hexdigest() != blob.sha256:
                    raise DomainError("material_handoff_source_hash_changed", 409)
                source.chmod(0o400)
                yield source
        finally:
            if source.exists():
                source.chmod(0o600)
                source.unlink()
            directory.rmdir()

    def put(self, domain: str, source: BinaryIO, max_bytes: int) -> StoredBlob:
        self.deadline.remaining()
        if domain != self.actor.security_domain:
            raise DomainError("not_found", 404)
        if type(max_bytes) is not int or not 1 <= max_bytes <= self.storage.config.maximum_object_bytes:
            raise DomainError("invalid_material_object_limit", 413)
        with self.storage.factory.begin() as db:
            self.authorize(db)
        with tempfile.SpooledTemporaryFile(max_size=1024**2, mode="w+b", dir=self.storage.config.spool_dir) as spool:
            sha, size = hashlib.sha256(), 0
            while True:
                self.deadline.remaining()
                part = source.read(65536)
                if not part:
                    break
                size += len(part)
                if size > max_bytes:
                    raise DomainError("material_asset_object_too_large", 413)
                sha.update(part)
                spool.write(part)
            identity = BlobIdentity(domain, sha.hexdigest(), size)
            operation_id = uuid5(self.execution_id, "material-asset:" + identity.sha256)
            spool.seek(0)
            prepared = self.storage.prepare_upload(self.actor, operation_id, identity, spool, deadline=self.deadline)
        with self.storage.factory.begin() as db:
            self.authorize(db)
            blob = self.storage.validate_prepared(db, self.actor, prepared,
                operation_id=operation_id, expected_identity=identity)
        self.prepared[blob.sha256] = prepared
        return blob

    def validate_in(self, db: Session, blob: StoredBlob) -> None:
        """Final publication transaction: SQL only, no verify/network callback."""
        self.deadline.remaining()
        identity = self._identity(blob)
        self.authorize(db)
        prepared = self.prepared.get(blob.sha256)
        if prepared is not None:
            self.storage.validate_prepared(db, self.actor, prepared,
                operation_id=uuid5(self.execution_id, "material-asset:" + identity.sha256), expected_identity=identity)
        else:
            reference = self.observed.get(blob.sha256)
            if reference is None or self.storage.resolve_location(db, identity) != reference:
                raise DomainError("material_source_not_verified_in_this_execution", 409)
