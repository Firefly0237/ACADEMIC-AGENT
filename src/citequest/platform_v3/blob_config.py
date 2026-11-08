"""Explicit storage runtime selection; emulation cannot activate in cloud mode."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, model_validator

from .blob_store import BlobStore, GCSBlobStore, LocalBlobStore


class BlobStoreConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    mode: Literal["development", "test", "cloud"]
    backend: Literal["filesystem", "gcs", "gcs_emulator"]
    store_id: str
    spool_dir: Path
    maximum_object_bytes: int = 64 * 1024**2
    filesystem_root: Path | None = None
    gcp_project: str | None = None
    gcs_bucket: str | None = None
    emulator_endpoint: str | None = None

    @model_validator(mode="after")
    def explicit_backend(self) -> BlobStoreConfig:
        if not self.store_id or len(self.store_id) > 100 or self.maximum_object_bytes < 1:
            raise ValueError("invalid_blob_store_profile")
        if self.mode == "cloud" and self.backend != "gcs":
            raise ValueError("cloud_profile_requires_real_gcs_backend")
        if self.backend == "filesystem":
            if self.filesystem_root is None or any((self.gcp_project, self.gcs_bucket, self.emulator_endpoint)):
                raise ValueError("filesystem_profile_must_not_contain_gcs_settings")
        elif self.filesystem_root is not None or not self.gcp_project or not self.gcs_bucket:
            raise ValueError("explicit_gcs_project_and_bucket_required_without_filesystem_fallback")
        if self.backend == "gcs_emulator":
            if self.mode == "cloud" or not self.emulator_endpoint:
                raise ValueError("emulator_is_development_only")
            url = urlsplit(self.emulator_endpoint)
            if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "gcs-emulator"}
                or url.username or url.password or url.path not in {"", "/"} or url.query or url.fragment or not url.port):
                raise ValueError("explicit_local_emulator_endpoint_required")
        elif self.emulator_endpoint is not None:
            raise ValueError("emulator_endpoint_forbidden_for_real_backend")
        return self


def create_blob_store(config: BlobStoreConfig) -> BlobStore:
    """Never create buckets or discover cloud credentials for local emulation."""
    # Pydantic model_copy(update=...) intentionally skips validation. Validate
    # the final configuration at this IO boundary as well as at initial load.
    config = BlobStoreConfig.model_validate(config.model_dump(mode="python"), strict=True)
    common = {"store_id": config.store_id, "spool_dir": config.spool_dir, "maximum_object_bytes": config.maximum_object_bytes}
    if config.backend == "filesystem":
        return LocalBlobStore(config.filesystem_root, **common)
    from google.cloud import storage

    if config.backend == "gcs_emulator":
        from google.auth.credentials import AnonymousCredentials

        client = storage.Client(project=config.gcp_project, credentials=AnonymousCredentials(),
                                client_options={"api_endpoint": config.emulator_endpoint})
    else:
        if any(os.environ.get(name) for name in ("STORAGE_EMULATOR_HOST", "FIREBASE_STORAGE_EMULATOR_HOST", "API_ENDPOINT_OVERRIDE")):
            raise ValueError("ambient_emulator_environment_forbidden_for_real_gcs")
        # Application Default Credentials: runtime identity on GCP, never an
        # embedded service-account key or an anonymous fallback.
        client = storage.Client(project=config.gcp_project,
                                client_options={"api_endpoint": "https://storage.googleapis.com"})
    expected_endpoint = config.emulator_endpoint if config.backend == "gcs_emulator" else "https://storage.googleapis.com"
    if client.api_endpoint.rstrip("/") != expected_endpoint.rstrip("/"):
        client.close()
        raise ValueError("effective_gcs_endpoint_differs_from_explicit_profile")
    return GCSBlobStore(client, config.gcs_bucket, **common)
