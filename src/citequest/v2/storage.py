"""Domain-scoped CAS. Filesystem publish precedes the database reference.

An interrupted transaction may leave an orphan: inventory reports it; no implicit
GC/delete or claim of atomicity across disk and PostgreSQL is provided.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


@dataclass(frozen=True)
class StoredBlob:
    security_domain: str
    sha256: str
    size_bytes: int


class CAS:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, domain: str, sha256: str) -> Path:
        if not domain or not re.fullmatch(r"[a-f0-9]{64}", sha256):
            raise ValueError("invalid_blob_identity")
        parent = self.root / hashlib.sha256(domain.encode()).hexdigest()
        if parent.is_symlink():
            raise ValueError("symlink_domain_rejected")
        parent.mkdir(exist_ok=True)
        path = parent / sha256
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("blob_path_escape")
        return path

    def put(self, domain: str, stream: BinaryIO, max_bytes: int) -> StoredBlob:
        if max_bytes <= 0:
            raise ValueError("invalid_size_limit")
        sha, size = hashlib.sha256(), 0
        fd, temporary = tempfile.mkstemp(prefix=".staging-", dir=self.root)
        temp = Path(temporary)
        try:
            with os.fdopen(fd, "wb") as output:
                while part := stream.read(65536):
                    size += len(part)
                    if size > max_bytes:
                        raise ValueError("file_too_large")
                    sha.update(part)
                    output.write(part)
                output.flush()
                os.fsync(output.fileno())
            if not size:
                raise ValueError("empty_file")
            blob = StoredBlob(domain, sha.hexdigest(), size)
            target = self.path(domain, blob.sha256)
            try:
                os.link(temp, target)  # Atomic no-overwrite publish on one volume.
            except FileExistsError:
                self.verify(blob)
            return blob
        finally:
            temp.unlink(missing_ok=True)

    def verify(self, blob: StoredBlob) -> None:
        path = self.path(blob.security_domain, blob.sha256)
        sha, size = hashlib.sha256(), 0
        with path.open("rb") as source:
            while part := source.read(65536):
                sha.update(part)
                size += len(part)
        if size != blob.size_bytes or sha.hexdigest() != blob.sha256:
            raise ValueError("stored_blob_corrupt")

    def read(self, blob: StoredBlob) -> bytes:
        data = self.path(blob.security_domain, blob.sha256).read_bytes()
        if len(data) != blob.size_bytes or hashlib.sha256(data).hexdigest() != blob.sha256:
            raise ValueError("stored_blob_corrupt")
        return data
