"""One product CAS root; temporary unit-only CAS remains a separate low-level utility."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .storage import CAS, StoredBlob


class ApplicationStorage(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    workspace: Path

    @property
    def blob_root(self) -> Path:
        base = self.workspace.resolve(strict=True)
        target = base
        for part in ("data", "runtime-v2-app", "blobs"):
            target = target / part
            if target.is_symlink():
                raise ValueError("application_storage_symlink_root_rejected")
        return target

    def cas(self) -> CAS:
        return CAS(self.blob_root)

    def assert_store(self, store: CAS) -> None:
        if store.root != self.blob_root.resolve():
            raise ValueError("application_storage_root_mismatch")

    def readonly_path(self, blob: StoredBlob) -> Path:
        """Use the same CAS layout without creating domain directories during inventory."""
        if not blob.security_domain or not re.fullmatch(r"[a-f0-9]{64}", blob.sha256):
            raise ValueError("invalid_blob_identity")
        parent = self.blob_root / hashlib.sha256(blob.security_domain.encode()).hexdigest()
        target = parent / blob.sha256
        if parent.is_symlink() or target.is_symlink():
            raise ValueError("application_storage_symlink_blob_rejected")
        return target


def app_storage(workspace: Path | None = None) -> ApplicationStorage:
    return ApplicationStorage(workspace=workspace or Path(__file__).resolve().parents[3])
