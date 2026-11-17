"""Explicit CF-01 runtime profile; no cloud activation or implicit fallback."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.engine import make_url


class PoolBudget(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    pool_size: Annotated[int, Field(ge=1, le=32)] = 2
    max_overflow: Annotated[int, Field(ge=0, le=16)] = 0
    pool_timeout: Annotated[float, Field(gt=0, le=60, allow_inf_nan=False)] = 3.0
    connect_timeout: Annotated[int, Field(ge=1, le=30)] = 3
    max_api_instances: Annotated[int, Field(ge=1, le=32)] = 2
    worker_connections: Annotated[int, Field(ge=0, le=128)] = 0
    migrator_connections: Annotated[int, Field(ge=3, le=16)] = 3
    operator_connections: Annotated[int, Field(ge=1, le=16)] = 2
    database_connection_budget: Annotated[int, Field(ge=8, le=256)] = 24

    @property
    def planned_connections(self) -> int:
        return self.max_api_instances * (self.pool_size + self.max_overflow) + self.worker_connections + self.migrator_connections + self.operator_connections

    @model_validator(mode="after")
    def bounded(self) -> PoolBudget:
        if not math.isfinite(self.pool_timeout) or self.planned_connections > self.database_connection_budget:
            raise ValueError("runtime_database_connection_budget_exceeded")
        return self


class StorageDeclaration(BaseModel):
    """Configuration declaration only; domain services are not BlobStore-wired."""
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    backend: Literal["filesystem", "gcs"] = "filesystem"
    store_id: Annotated[str, Field(pattern=r"^[a-z0-9_-]{1,80}$")] = "cf01-shared-local-cas"
    shared_root: Path | None = None
    bucket: str | None = None

    @model_validator(mode="after")
    def shape(self) -> StorageDeclaration:
        if self.backend == "filesystem" and (self.shared_root is None or self.bucket is not None):
            raise ValueError("explicit_local_storage_declaration_required")
        if self.backend == "gcs" and (self.shared_root is not None or not self.bucket):
            raise ValueError("explicit_gcs_storage_declaration_required")
        return self


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    mode: Literal["development", "test", "cloud"]
    component: Literal["domain_api"] = "domain_api"
    instance_id: Annotated[str, Field(pattern=r"^[a-z0-9_-]{1,40}$")]
    port: Annotated[int, Field(ge=1024, le=65535)] = 8080
    origin: Literal["http://127.0.0.1:18000", "http://localhost:18000"] = "http://127.0.0.1:18000"
    auth_mode: Literal["trusted_local_sessions", "oidc"] = "trusted_local_sessions"
    database_dsn_file: Path
    cursor_key_file: Path
    contract_path: Path
    temporary_directory: Path
    storage: StorageDeclaration
    pool: PoolBudget = Field(default_factory=PoolBudget)
    expected_schema_revision: Literal["s03_0005"] = "s03_0005"
    file_content_io: bool = False
    search: bool = False
    agent: bool = False
    ingestion: bool = False
    milvus_endpoint: str | None = None
    tei_endpoint: str | None = None
    index_profile: str | None = None

    @model_validator(mode="after")
    def implemented_boundary(self) -> RuntimeConfig:
        if self.mode == "cloud":
            raise ValueError("cloud_domain_oidc_https_and_gcs_business_bridge_not_integrated")
        if self.auth_mode != "trusted_local_sessions":
            raise ValueError("oidc_entry_not_integrated")
        if self.storage.backend != "filesystem" or any((self.file_content_io, self.search, self.agent, self.ingestion)):
            raise ValueError("requested_capability_not_integrated_in_cf01")
        if any((self.milvus_endpoint, self.tei_endpoint, self.index_profile)):
            raise ValueError("disabled_search_must_not_have_active_dependency_settings")
        return self

    def effective(self) -> dict:
        """Only references/configuration, never DSN, credential or key contents."""
        value = self.model_dump(mode="json")
        value["database_dsn_file"] = "<file-reference>"
        value["cursor_key_file"] = "<file-reference>"
        value["planned_database_connections"] = self.pool.planned_connections
        value["capabilities"] = {"domain_crud": "enabled_local", "blob_business": "not_integrated", "search": "disabled", "agent": "disabled", "parser": "disabled", "oidc": "not_integrated"}
        return value

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.effective(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_runtime(path: Path, *, port: str | None = None) -> RuntimeConfig:
    config = RuntimeConfig.model_validate_json(path.read_text(encoding="utf-8"), strict=True)
    if port is not None:
        if not port.isascii() or not port.isdecimal():
            raise ValueError("PORT_must_be_a_decimal_integer")
        config = RuntimeConfig.model_validate({**config.model_dump(mode="python"), "port": int(port)}, strict=True)
    return config


def owned_cf01_dsn(value: str, role: Literal["app", "migrator", "admin"]) -> str:
    """This local package must never connect to the existing product databases."""
    parsed = make_url(value.strip())
    if (parsed.drivername != "postgresql+psycopg" or parsed.database != "citequest_cf01"
        or parsed.username != "cq_cf01_" + role or not parsed.password or parsed.query
        or (parsed.host, parsed.port) != ("postgres", 5432)):
        raise ValueError("explicit_owned_cf01_database_required")
    return value.strip()
