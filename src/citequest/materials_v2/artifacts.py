"""Immutable, owner-scoped parser assets; CAS publish precedes SQL references."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from sqlalchemy.orm import Session

from citequest.v2 import models as core
from citequest.v2.domain import DomainError
from citequest.v2.storage import CAS, StoredBlob

from . import models as m
from .provenance import native_pdf_view
from .text import MaterialError

ROLES = {"docling-document.json": "native_document", "source-representation.json": "source_representation", "source-pages.json": "source_pages",
         "preflight-pages.json": "preflight_pages", "source-paragraphs.json": "source_paragraphs", "normalized-text.json": "normalized_text",
         "docling-options.json": "parser_options", "unmapped-blocks.json": "unmapped_blocks", "receipt.json": "parser_resource_receipt"}


@dataclass(frozen=True)
class Asset:
    role: str
    blob: StoredBlob
    provenance: dict


def capture_assets(output: Path, parsed: dict, original: StoredBlob, result: StoredBlob, store: CAS, *,
                   mode: str, check_remaining: Callable[[], object] | None = None) -> list[Asset]:
    check = check_remaining or (lambda: None)
    check()
    parse = (output / "parse").resolve(strict=True)
    if (parsed.get("source_sha256") != original.sha256 or result.security_domain != original.security_domain):
        raise MaterialError("ASSET_SOURCE_BINDING_MISMATCH")
    if json.loads(store.read(result)) != parsed:
        raise MaterialError("ASSET_PARSER_RESULT_MISMATCH")
    assets = [Asset("parser_result", result, {"schema_version": "cq.material.asset/1", "acquisition_mode": mode, "source_sha256": original.sha256})]
    source_view = parse / "source-representation.json"
    view, view_sha = None, None
    if source_view.exists():
        if source_view.is_symlink() or source_view.stat().st_size > 64 * 1024 * 1024:
            raise MaterialError("ASSET_SOURCE_VIEW_SIZE_OR_PATH_INVALID")
        view_raw = source_view.read_bytes()
        view, view_sha = json.loads(view_raw), hashlib.sha256(view_raw).hexdigest()
        if (view.get("file_sha256") != original.sha256 or view.get("representation_id") != parsed.get("representation_id")
                or view.get("blocks") != parsed.get("blocks") or view.get("missing_ranges") != parsed.get("missing_ranges")):
            raise MaterialError("ASSET_NATIVE_SOURCE_VIEW_MISMATCH")
    total = result.size_bytes
    for filename, role in ROLES.items():
        path = parse / filename
        if not path.is_file():
            continue
        check()
        if path.is_symlink() or not path.resolve().is_relative_to(parse):
            raise MaterialError("ASSET_PATH_ESCAPE")
        total += path.stat().st_size
        if total > 128 * 1024 * 1024:
            raise MaterialError("ASSET_CAPTURE_BYTES_EXCEEDED")
        with path.open("rb") as stream:
            blob = store.put(original.security_domain, stream, 64 * 1024 * 1024)
        assets.append(Asset(role, blob, {"schema_version": "cq.material.asset/1", "acquisition_mode": mode, "source_sha256": original.sha256,
                                        "source_filename": filename, "source_view_id": parsed.get("representation_id")}))
    chunk_receipt = output / "chunk/receipt.json"
    if chunk_receipt.is_file():
        check()
        if chunk_receipt.is_symlink() or not chunk_receipt.resolve().is_relative_to(output.resolve()):
            raise MaterialError("ASSET_PATH_ESCAPE")
        total += chunk_receipt.stat().st_size
        if total > 128 * 1024 * 1024:
            raise MaterialError("ASSET_CAPTURE_BYTES_EXCEEDED")
        with chunk_receipt.open("rb") as stream:
            blob = store.put(original.security_domain, stream, 64 * 1024 * 1024)
        assets.append(Asset("chunk_resource_receipt", blob, {"schema_version": "cq.material.asset/1", "acquisition_mode": mode, "source_sha256": original.sha256}))
    roles = {asset.role for asset in assets}
    if parsed.get("blocks") and parsed["blocks"][0]["locator"]["kind"] == "pdf" and not {"native_document", "source_representation", "source_pages"} <= roles:
        raise MaterialError("ASSET_REQUIRED_NATIVE_PDF_PROOF_MISSING")
    if parsed.get("blocks") and parsed["blocks"][0]["locator"]["kind"] == "docx" and not {"native_document", "source_paragraphs"} <= roles:
        raise MaterialError("ASSET_REQUIRED_DOCX_PROOF_MISSING")
    if view is not None:
        captured = {asset.role: asset.blob for asset in assets}
        if (not {"native_document", "source_representation", "source_pages"} <= captured.keys()
                or captured["native_document"].sha256 != view["native_document_sha256"]
                or captured["source_representation"].sha256 != view_sha
                or json.loads(store.read(captured["source_pages"])) != view["pages"]):
            raise MaterialError("ASSET_CAPTURED_NATIVE_PROOF_CHANGED")
        # Comparing hashes supplied together is insufficient: a changed native
        # text can be accompanied by its new hash while retaining old pages,
        # blocks, offsets and representation_id. Rebuild from the exact native
        # bytes now in CAS, using the original source identity and view config.
        # Worker profile/page metadata, when present, must agree as well.
        config_sha = view.get("config_sha256")
        page_count = len(view.get("pages", []))
        if (type(config_sha) is not str or len(config_sha) != 64 or any(c not in "0123456789abcdef" for c in config_sha)
                or not 1 <= page_count <= 200 or parsed.get("profile_sha256", config_sha) != config_sha
                or type(parsed.get("pages", page_count)) is not int or parsed.get("pages", page_count) != page_count):
            raise MaterialError("ASSET_NATIVE_VIEW_CONFIGURATION_MISMATCH")
        check()
        try:
            rebuilt = native_pdf_view(json.loads(store.read(captured["native_document"])), source_sha256=original.sha256,
                                      native_sha256=captured["native_document"].sha256, config_sha256=config_sha, page_count=page_count,
                                      structure_version=view.get("native_structure_version"))
        except (ValueError, KeyError, TypeError) as exc:
            raise MaterialError("ASSET_NATIVE_VIEW_RECONSTRUCTION_FAILED") from exc
        if rebuilt != view:
            raise MaterialError("ASSET_NATIVE_VIEW_RECONSTRUCTION_MISMATCH")
    check()
    return assets


def register_assets(db: Session, store: CAS | None, *, owner_id: UUID, representation_id: str, assets: list[Asset],
                    verify_in: Callable[[Session, StoredBlob], None] | None = None) -> None:
    representation = db.get(m.ParsedRepresentation, representation_id)
    domain = "owner:" + str(owner_id)
    if representation is None or representation.owner_id != owner_id or representation.security_domain != domain:
        raise DomainError("not_found", 404)
    for asset in assets:
        if asset.blob.security_domain != domain or asset.provenance.get("source_sha256") != representation.source_sha256:
            raise DomainError("artifact_asset_source_mismatch", 409)
        if verify_in is not None:
            verify_in(db, asset.blob)  # SQL-only prepared/observed pointer guard.
        elif store is not None:
            store.verify(asset.blob)
        else:
            raise DomainError("material_asset_storage_not_configured", 503)
        previous = db.get(m.ArtifactAsset, (owner_id, representation_id, asset.role))
        if previous is not None:
            if previous.sha256 != asset.blob.sha256 or previous.size_bytes != asset.blob.size_bytes:
                raise DomainError("artifact_asset_immutable_conflict", 409)
            continue
        blob = db.get(core.Blob, (domain, asset.blob.sha256))
        if blob is None:
            db.add(core.Blob(security_domain=domain, sha256=asset.blob.sha256, size_bytes=asset.blob.size_bytes))
            db.flush()
        elif blob.size_bytes != asset.blob.size_bytes:
            raise DomainError("artifact_asset_blob_size_conflict", 409)
        db.add(m.ArtifactAsset(owner_id=owner_id, representation_id=representation_id, role=asset.role, security_domain=domain,
                               sha256=asset.blob.sha256, size_bytes=asset.blob.size_bytes, provenance=asset.provenance))
    db.flush()
