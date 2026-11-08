"""Host-only immutable retrieval bindings; never model or HTTP capabilities.

Persistence and Task ownership are the caller's responsibility. A valid shape
does not authorize a read: SearchService rechecks source/ACL, proof and runtime.
"""
from __future__ import annotations

import hashlib
from typing import Literal, Protocol
from uuid import UUID

from pydantic import Field

from .schemas import SHA, CatalogName, Strict


class CatalogSnapshot(Strict):
    version: Literal["task-catalog-snapshot/1"] = "task-catalog-snapshot/1"
    catalog: CatalogName
    generation_id: UUID
    epoch: int = Field(ge=1)
    collection_name: str = Field(min_length=1, max_length=128)
    source_snapshot_id: UUID
    manifest_sha256: SHA
    source_seal_sha256: SHA
    server_proof_sha256: SHA
    index_profile_sha256: SHA
    tokenizer_profile_sha256: SHA
    policy_sha256: SHA
    evaluation_sha256: SHA
    native_runtime_sha256: SHA


class NativeRuntime(Protocol):
    uri: str
    database: str
    tei_info_url: str
    function_endpoint: str
    backend_profile_sha256: str | None
    tls_ca: bytes | None


def native_runtime_identity(native: NativeRuntime) -> dict:
    """Bind endpoint/DB/TLS/model routing, without credentials or SDK clients."""
    return {
        "uri": native.uri,
        "database": native.database,
        "tei_info_url": native.tei_info_url,
        "function_endpoint": native.function_endpoint,
        "backend_profile_sha256": native.backend_profile_sha256,
        "tls_ca_sha256": hashlib.sha256(native.tls_ca).hexdigest() if native.tls_ca else None,
    }
