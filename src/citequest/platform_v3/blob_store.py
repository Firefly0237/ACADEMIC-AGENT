"""Verified, immutable blob IO through local CAS or the official GCS SDK.

This module does not authorize an end user. Services must check current SQL
owner/project permissions before resolving a reference. No delete, public URL,
filesystem path or credentials are exposed by this storage interface.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.metadata
import io
import re
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, Literal, Protocol
from uuid import UUID

from citequest.v2.storage import CAS, StoredBlob

if TYPE_CHECKING:
    from google.cloud.storage import Blob, Client


class BlobStoreError(ValueError):
    pass


class BlobWriteUncertain(BlobStoreError):
    """The call may have stored bytes; reconcile the deterministic key first."""


@dataclass(frozen=True)
class BlobIdentity:
    security_domain: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.security_domain, str):
            raise BlobStoreError("invalid_security_domain")
        if self.security_domain != "public":
            if not self.security_domain.startswith("owner:"):
                raise BlobStoreError("invalid_security_domain")
            try:
                owner = UUID(self.security_domain[6:])
            except ValueError as exc:
                raise BlobStoreError("invalid_security_domain") from exc
            if str(owner) != self.security_domain[6:]:
                raise BlobStoreError("noncanonical_security_domain")
        if not isinstance(self.sha256, str) or re.fullmatch(r"[a-f0-9]{64}", self.sha256) is None:
            raise BlobStoreError("invalid_blob_sha256")
        if type(self.size_bytes) is not int or not 0 < self.size_bytes < 2**63:
            raise BlobStoreError("invalid_blob_size")

    @property
    def domain_sha256(self) -> str:
        return hashlib.sha256(self.security_domain.encode()).hexdigest()

    @property
    def key(self) -> str:
        return f"blobs/{self.domain_sha256}/{self.sha256}"


@dataclass(frozen=True)
class BlobReference:
    identity: BlobIdentity
    backend: Literal["filesystem", "gcs"]
    store_id: str
    object_key: str
    bucket: str | None = None
    generation: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.store_id, str) or not self.store_id or len(self.store_id) > 100 or self.object_key != self.identity.key:
            raise BlobStoreError("invalid_blob_locator")
        if self.backend == "gcs":
            if not self.bucket or type(self.generation) is not int or self.generation < 1:
                raise BlobStoreError("immutable_gcs_generation_required")
        elif self.backend != "filesystem" or self.bucket is not None or self.generation is not None:
            raise BlobStoreError("invalid_blob_backend")


class BlobStore(Protocol):
    def put(self, identity: BlobIdentity, stream: BinaryIO) -> BlobReference: ...
    def open_verified(self, reference: BlobReference) -> AbstractContextManager[BinaryIO]: ...


class LimitedSink:
    """Bound downloaded bytes before they can grow an unbounded spool."""
    def __init__(self, stream: BinaryIO, limit: int) -> None:
        self.stream, self.limit = stream, limit

    def write(self, data: bytes) -> int:
        if self.stream.tell() + len(data) > self.limit:
            raise BlobStoreError("blob_download_exceeds_declared_size")
        return self.stream.write(data)

    def tell(self) -> int:
        return self.stream.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        result = self.stream.seek(offset, whence)
        if not 0 <= result <= self.limit:
            raise BlobStoreError("blob_download_seek_outside_limit")
        return result

    def flush(self) -> None:
        self.stream.flush()


class _VerifiedStore:
    def __init__(self, *, store_id: str, spool_dir: Path, maximum_object_bytes: int = 64 * 1024**2) -> None:
        if not isinstance(store_id, str) or not store_id or len(store_id) > 100 or type(maximum_object_bytes) is not int or maximum_object_bytes < 1:
            raise BlobStoreError("invalid_blob_store_configuration")
        self.store_id, self.spool_dir, self.maximum = store_id, spool_dir.resolve(), maximum_object_bytes
        self.spool_dir.mkdir(parents=True, exist_ok=True)
        if spool_dir.is_symlink():
            raise BlobStoreError("blob_spool_symlink_rejected")

    def _limit(self, identity: BlobIdentity) -> None:
        if identity.size_bytes > self.maximum:
            raise BlobStoreError("blob_exceeds_store_profile")

    def _reference(self, reference: BlobReference, backend: str) -> None:
        self._limit(reference.identity)
        if reference.store_id != self.store_id or reference.backend != backend:
            raise BlobStoreError("blob_store_reference_mismatch")

    @contextmanager
    def _spool(self, identity: BlobIdentity, stream: BinaryIO | None = None) -> Iterator[BinaryIO]:
        self._limit(identity)
        with tempfile.SpooledTemporaryFile(max_size=2 * 1024**2, mode="w+b", dir=self.spool_dir) as spool:
            if stream is not None:
                sink = LimitedSink(spool, identity.size_bytes)
                while part := stream.read(65536):
                    sink.write(part)
                self._verify_spool(identity, spool)
            yield spool

    @staticmethod
    def _verify_spool(identity: BlobIdentity, spool: BinaryIO) -> None:
        spool.seek(0)
        value, size = hashlib.sha256(), 0
        while block := spool.read(65536):
            value.update(block)
            size += len(block)
            if size > identity.size_bytes:
                raise BlobStoreError("blob_size_mismatch")
        if size != identity.size_bytes or value.hexdigest() != identity.sha256:
            raise BlobStoreError("blob_identity_mismatch")
        spool.seek(0)


class LocalBlobStore(_VerifiedStore):
    """Adapter over the existing no-overwrite CAS, not another file format."""
    def __init__(self, root: Path, *, store_id: str, spool_dir: Path, maximum_object_bytes: int = 64 * 1024**2) -> None:
        super().__init__(store_id=store_id, spool_dir=spool_dir, maximum_object_bytes=maximum_object_bytes)
        self.cas = CAS(root)

    def put(self, identity: BlobIdentity, stream: BinaryIO) -> BlobReference:
        with self._spool(identity, stream) as verified:
            result = self.cas.put(identity.security_domain, verified, identity.size_bytes)
        if result != StoredBlob(identity.security_domain, identity.sha256, identity.size_bytes):
            raise BlobStoreError("local_CAS_identity_mismatch")
        return BlobReference(identity, "filesystem", self.store_id, identity.key)

    @contextmanager
    def open_verified(self, reference: BlobReference) -> Iterator[BinaryIO]:
        self._reference(reference, "filesystem")
        identity = reference.identity
        # Verify and deliver the same private spool. A separate verify(path)
        # followed by reopening an attacker-replaceable path is insufficient.
        with self.cas.path(identity.security_domain, identity.sha256).open("rb") as source, self._spool(identity, source) as verified:
            yield verified


class GCSBlobStore(_VerifiedStore):
    """SDK client supplied by a separately validated runtime factory.

    SDK retries are explicitly disabled here. A write exception remains
    uncertain; callers may reconcile the deterministic key without re-upload.
    IAM/real cloud behavior must be verified on GCP separately from emulation.
    """
    def __init__(self, client: Client, bucket: str, *, store_id: str, spool_dir: Path, maximum_object_bytes: int = 64 * 1024**2) -> None:
        # Upload integrity behavior is sensitive to the SDK's resumable path.
        # A dependency upgrade requires review, not an implicit new behavior.
        if importlib.metadata.version("google-cloud-storage") != "3.17.1":
            raise BlobStoreError("reviewed_gcs_sdk_3_17_1_required")
        super().__init__(store_id=store_id, spool_dir=spool_dir, maximum_object_bytes=maximum_object_bytes)
        if re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket) is None:
            raise BlobStoreError("explicit_simple_gcs_bucket_required")
        self.client, self.bucket_name = client, bucket
        self.bucket = client.bucket(bucket)

    @staticmethod
    def _crc32c(stream: BinaryIO) -> str:
        import google_crc32c

        checksum = google_crc32c.Checksum()
        stream.seek(0)
        while part := stream.read(65536):
            checksum.update(part)
        stream.seek(0)
        return base64.b64encode(checksum.digest()).decode("ascii")

    def _describe(self, identity: BlobIdentity, blob: Blob, *, expected_crc32c: str | None = None) -> BlobReference:
        try:
            valid_crc = isinstance(blob.crc32c, str) and len(base64.b64decode(blob.crc32c, validate=True)) == 4
        except ValueError:
            valid_crc = False
        if (
            blob.size != identity.size_bytes
            or (blob.metadata or {}).get("cq_sha256") != identity.sha256
            or (blob.metadata or {}).get("cq_domain_sha256") != identity.domain_sha256
            or type(blob.generation) is not int or blob.generation < 1
            or not valid_crc or (expected_crc32c is not None and blob.crc32c != expected_crc32c)
        ):
            raise BlobStoreError("gcs_blob_metadata_identity_mismatch")
        return BlobReference(identity, "gcs", self.store_id, identity.key, self.bucket_name, blob.generation)

    def put(self, identity: BlobIdentity, stream: BinaryIO) -> BlobReference:
        from google.api_core.exceptions import PreconditionFailed

        with self._spool(identity, stream) as verified:
            crc32c = self._crc32c(verified)
            blob = self.bucket.blob(identity.key, chunk_size=8 * 1024**2)
            blob.metadata = {"cq_sha256": identity.sha256, "cq_domain_sha256": identity.domain_sha256}
            try:
                try:
                    # SDK 3.17.1 may issue an unconditional DELETE after a
                    # resumable client-checksum mismatch. Avoid that path:
                    # provide the complete CRC32C for GCS server validation,
                    # then independently verify the fixed generation below.
                    # Multipart retains the SDK's client checksum as well.
                    blob.upload_from_file(verified, size=identity.size_bytes, rewind=True,
                        content_type="application/octet-stream", if_generation_match=0,
                        checksum="crc32c" if identity.size_bytes <= 8 * 1024**2 else None,
                        crc32c_checksum_value=crc32c, timeout=(5, 30), retry=None)
                except PreconditionFailed:
                    existing = self.reconcile(identity)
                    if existing is None:
                        raise BlobWriteUncertain("conditional_create_conflict_not_yet_reconciled") from None
                    return existing
                reference = self._describe(identity, blob, expected_crc32c=crc32c)
                with self.open_verified(reference):
                    pass
                return reference
            except Exception as exc:
                # Includes metadata/readback failures after a successful write.
                # No verified reference is returned and no delete is attempted.
                raise BlobWriteUncertain("gcs_write_result_uncertain") from exc

    def reconcile(self, identity: BlobIdentity) -> BlobReference | None:
        self._limit(identity)
        blob = self.bucket.get_blob(identity.key, timeout=(5, 30), retry=None)
        if blob is None:
            # Merely not visible at this observation. Late completion of a
            # previous unknown request is not ruled out by this return value.
            return None
        reference = self._describe(identity, blob)
        with self.open_verified(reference):
            pass
        return reference

    @contextmanager
    def open_verified(self, reference: BlobReference) -> Iterator[BinaryIO]:
        self._reference(reference, "gcs")
        if reference.bucket != self.bucket_name:
            raise BlobStoreError("gcs_bucket_reference_mismatch")
        blob = self.bucket.blob(reference.object_key, generation=reference.generation)
        blob.reload(if_generation_match=reference.generation, timeout=(5, 30), retry=None)
        if self._describe(reference.identity, blob) != reference:
            raise BlobStoreError("gcs_read_generation_mismatch")
        with self._spool(reference.identity) as spool:
            blob.download_to_file(LimitedSink(spool, reference.identity.size_bytes),
                raw_download=True, if_generation_match=reference.generation,
                checksum="crc32c", timeout=(5, 30), retry=None)
            self._verify_spool(reference.identity, spool)
            if self._crc32c(spool) != blob.crc32c:
                raise BlobStoreError("gcs_download_crc32c_mismatch")
            yield spool


def identity_for_bytes(domain: str, data: bytes) -> BlobIdentity:
    """Trusted caller convenience; input/resource admission remains upstream."""
    return BlobIdentity(domain, hashlib.sha256(data).hexdigest(), len(data))


def read_bytes(store: BlobStore, reference: BlobReference) -> bytes:
    with store.open_verified(reference) as stream:
        return stream.read()


def put_bytes(store: BlobStore, domain: str, data: bytes) -> BlobReference:
    return store.put(identity_for_bytes(domain, data), io.BytesIO(data))
