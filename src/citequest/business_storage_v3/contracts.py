"""Trusted storage identities and internal receipts, never HTTP capabilities."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from uuid import UUID

from citequest.platform_v3.blob_config import BlobStoreConfig
from citequest.platform_v3.blob_store import BlobIdentity, BlobReference
from citequest.v2.domain import Actor, DomainError


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class StorageBinding:
    store_id: str
    sha256: str
    backend: str
    bucket: str | None

    @classmethod
    def configured(cls, config: BlobStoreConfig) -> StorageBinding:
        config = BlobStoreConfig.model_validate(config.model_dump(mode="python"), strict=True)
        value = {"version": "citequest-business-blob-binding/1", "backend": config.backend,
                 "store_id": config.store_id, "layout": "domain-sha256/content-sha256/1",
                 "project": config.gcp_project, "bucket": config.gcs_bucket,
                 "emulator_endpoint": config.emulator_endpoint,
                 "filesystem_root": str(config.filesystem_root.resolve()) if config.filesystem_root else None}
        return cls(config.store_id, digest(value), "filesystem" if config.backend == "filesystem" else "gcs",
                   config.gcs_bucket)

    def validate(self, reference: BlobReference, identity: BlobIdentity) -> None:
        if (reference.identity != identity or reference.store_id != self.store_id
                or reference.backend != self.backend or reference.bucket != self.bucket):
            raise DomainError("business_blob_reference_binding_mismatch", 409)


@dataclass(frozen=True)
class PreparedBlobReceipt:
    admission_id: UUID
    owner_id: UUID
    operation_id: UUID
    identity: BlobIdentity
    binding_sha256: str
    reference_sha256: str
    deadline_at: datetime


@dataclass(frozen=True)
class OperationDeadline:
    """Publication gate, not by itself proof that an SDK socket was terminated."""
    expires_at: float
    clock: Callable[[], float] = time.monotonic

    def remaining(self) -> float:
        value = self.expires_at - self.clock()
        if not math.isfinite(value) or value <= 0:
            raise DomainError("business_blob_deadline_exceeded", 408)
        return value


def authorize_domain(actor: Actor, identity: BlobIdentity) -> None:
    if identity.security_domain != actor.security_domain and not (identity.security_domain == "public" and actor.publisher):
        raise DomainError("not_found", 404)


def reference_value(reference: BlobReference) -> dict:
    return asdict(reference)


def parse_reference(value: dict) -> BlobReference:
    try:
        if not isinstance(value, dict) or set(value) != {"identity", "backend", "store_id", "object_key", "bucket", "generation"}:
            raise ValueError("invalid_reference_shape")
        identity = BlobIdentity(**value["identity"])
        return BlobReference(identity, **{key: item for key, item in value.items() if key != "identity"})
    except (ValueError, TypeError, KeyError) as exc:
        raise DomainError("business_blob_reference_shape_invalid", 409) from exc

