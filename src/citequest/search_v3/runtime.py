"""Trusted deployment configuration. Secrets are file/env references, not HTTP input."""
from __future__ import annotations
import json
import os
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, model_validator
from citequest.materials_v2.chunking import SourceChunker
from citequest.v2.db import engine, sessions
from citequest.v2.storage_config import app_storage
from .native import SearchNative
from .service import SearchService
from .schemas import NativePDFAssetProfile


class SearchRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: str = Field(pattern=r"^citequest-search-runtime/1$")
    workspace: str
    dsn_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{2,100}$")
    dsn_file: str | None = None
    cursor_key_file: str
    milvus_uri: str
    milvus_database: str = Field(default="default", pattern=r"^[a-z][a-z0-9_]{0,63}$")
    tei_info_url: str
    backend_profile_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    function_endpoint: str = "http://tei-strict:8080"
    milvus_token_env: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9_]{2,100}$")
    tls_ca_file: str | None = None
    source_storage_mode: str = Field(pattern=r"^application_local_cas$")
    policy_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    evaluation_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    native_pdf_asset_profile_file: str | None = None

    @model_validator(mode="after")
    def one_dsn(self):
        if (self.dsn_env is None) == (self.dsn_file is None):
            raise ValueError("exactly one trusted DSN secret reference required")
        return self


def service_from_config(config_path: Path) -> SearchService:
    # Only trusted application code chooses this path. No dynamic imports or
    # model-controlled endpoint/DSN and no alternate store on source failure.
    config_path = Path(config_path).resolve(strict=True)
    config = SearchRuntimeConfig.model_validate_json(config_path.read_bytes())
    def configured_path(value):
        path = Path(value)
        return (path if path.is_absolute() else config_path.parent/path).resolve(strict=True)
    workspace = configured_path(config.workspace)
    key = configured_path(config.cursor_key_file).read_bytes()
    chunker = SourceChunker(workspace)
    chunker.verify_assets()
    dsn = os.environ[config.dsn_env] if config.dsn_env else configured_path(config.dsn_file).read_text(encoding="utf-8").strip()
    db = engine(dsn)
    native = SearchNative(uri=config.milvus_uri, tei_info_url=config.tei_info_url,
        database=config.milvus_database,
        backend_profile_sha256=config.backend_profile_sha256, function_endpoint=config.function_endpoint,
        token=os.environ[config.milvus_token_env] if config.milvus_token_env else "",
        tls_ca=configured_path(config.tls_ca_file).read_bytes() if config.tls_ca_file else None)
    return SearchService(sessions(db), native, app_storage(workspace).cas(),
        policy_sha256=config.policy_sha256, evaluation_sha256=config.evaluation_sha256,
        cursor_key=key, query_token_counter=chunker.count,
        query_tokenizer_profile_sha256=chunker.profile_sha256,
        native_pdf_asset_profile=NativePDFAssetProfile.model_validate_json(configured_path(config.native_pdf_asset_profile_file).read_bytes())
            if config.native_pdf_asset_profile_file else None)
