"""PostgreSQL FTS restricted to selected files BEFORE ranking and limiting.

This branch is intentionally not a CanonicalWork or a published-paper index.
Partial parser coverage remains partial. Cloud model consent is checked at every
read, including restored Task results; a former consent is never an ACL cache.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from citequest.materials_v2.models import MaterialFragment, ParsedRepresentation
from citequest.materials_v2.services import MaterialService
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor, DomainError
from citequest.v2.models import Blob, File
from citequest.v2.storage import CAS, StoredBlob

if TYPE_CHECKING:
    from .native import MaterialNativeIndex


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MaterialEvidenceService:
    def __init__(
        self,
        factory: sessionmaker,
        store: CAS,
        *,
        native_index: MaterialNativeIndex | None = None,
        retrieval_mode: Literal["native_hybrid", "fts_diagnostic"] = "native_hybrid",
    ) -> None:
        if retrieval_mode not in {"native_hybrid", "fts_diagnostic"}:
            raise ValueError("explicit_material_retrieval_mode_required")
        self.factory, self.store, self.native_index = factory, store, native_index
        self.retrieval_mode = retrieval_mode

    def validate_selection(self, actor: Actor, file_ids: list[UUID]) -> None:
        if not 1 <= len(file_ids) <= 20 or len(set(file_ids)) != len(file_ids):
            raise DomainError("invalid_material_evidence_scope", 400)
        if self.retrieval_mode != "native_hybrid" or self.native_index is None:
            raise DomainError("material_semantic_index_not_configured", 503)
        with self.factory.begin() as db:
            for file_id in file_ids:
                self._source(db, actor, file_id)
        self.native_index.preflight(actor, file_ids)

    def _source(self, db: Session, actor: Actor, file_id: UUID) -> tuple[File, ParsedRepresentation, dict]:
        # Repeated authorization in one SQLAlchemy Session must not reuse stale
        # identity-map policy/revision attributes after another transaction revokes consent.
        db.expire_all()
        material = MaterialService(db, actor, self.store)
        file = material.catalog.file(file_id)  # owner, domain, deletion and bound Work ACL
        if file.outbound_policy != "selected_cloud_processing_allowed" or file.rights_status == "restricted":
            raise DomainError("material_cloud_processing_not_permitted", 403)
        if file.status not in {"ready", "partial"}:
            raise DomainError("material_not_readable", 409)
        parsed = material.get_representation(file_id)
        rep = db.get(ParsedRepresentation, parsed["representation_id"])
        if rep.material_version_id is None or rep.source_sha256 != file.sha256:
            raise DomainError("material_source_version_changed", 409)
        blob = db.get(Blob, (file.security_domain, file.sha256))
        if blob is None:
            raise DomainError("material_blob_missing", 503)
        self.store.verify(StoredBlob(file.security_domain, file.sha256, blob.size_bytes))
        return file, rep, parsed

    def _dto(self, file: File, rep: ParsedRepresentation, parsed: dict, row: MaterialFragment) -> dict:
        if (
            row.payload.get("source_text_sha256") != sha(row.source_text)
            or row.payload.get("source_text") != row.source_text
        ):
            raise DomainError("material_fragment_integrity_failure", 503)
        if len(row.source_text.encode("utf-8")) > 20000:
            # No hidden truncation of a formula or a source boundary.
            raise DomainError("material_fragment_exceeds_context_window", 413)
        return {
            "ref_id": "material:" + row.id,
            "text": row.source_text,
            "source_ref": {"kind": "material_fragment", "id": row.id},
            "file_id": str(file.id),
            "material_version_id": str(rep.material_version_id),
            "representation_id": rep.id,
            "source_file_sha256": file.sha256,
            "source_sha256": sha(row.source_text),
            "unit_text_sha256": sha(row.source_text),
            "span_start": 0,
            "span_end": len(row.source_text),
            "coordinate_system": "unicode_codepoints_in_material_fragment",
            "locator": row.payload["locator"],
            "coverage": parsed["coverage"],
            "missing_ranges": parsed["missing_ranges"],
            "file_revision": file.revision,
            "rights_status": file.rights_status,
            "outbound_policy": file.outbound_policy,
            "source_url": f"/api/v2/files/{file.id}/content",
            "fixture": False,
        }

    def search_selected(self, actor: Actor, file_ids: list[UUID], query: str, limit: int = 2) -> list[dict]:
        if (
            not 1 <= len(file_ids) <= 20
            or len(set(file_ids)) != len(file_ids)
            or not query.strip()
            or len(query) > 2000
            or "\x00" in query
            or type(limit) is not int
            or not 1 <= limit <= 3
        ):
            raise DomainError("invalid_material_evidence_scope", 400)
        with self.factory.begin() as db:
            sources = {file_id: self._source(db, actor, file_id) for file_id in file_ids}
            if self.retrieval_mode == "native_hybrid":
                if self.native_index is None:
                    raise DomainError("material_semantic_index_not_configured", 503)
                hits = self.native_index.search(actor, file_ids, query, limit)
                output = []
                for hit in hits:
                    # A second current read closes the retrieval-to-return ACL/version gap.
                    source = self._source(db, actor, hit.file_id)
                    row = db.get(MaterialFragment, hit.fragment_id)
                    if (
                        row is None
                        or row.owner_id != actor.owner_id
                        or row.file_id != hit.file_id
                        or row.representation_id != hit.representation_id
                        or source[1].id != hit.representation_id
                        or sha(row.source_text) != hit.source_text_sha256
                    ):
                        raise DomainError("material_retrieval_source_changed", 409)
                    output.append(self._dto(*source, row))
                return output
            ids = [value[1].id for value in sources.values()]
            tsquery = func.plainto_tsquery("simple", query)
            rank = func.ts_rank_cd(MaterialFragment.search_document, tsquery)
            statement = (
                select(MaterialFragment)
                .where(
                    MaterialFragment.owner_id == actor.owner_id,
                    MaterialFragment.file_id.in_(file_ids),
                    MaterialFragment.representation_id.in_(ids),
                    MaterialFragment.search_document.op("@@")(tsquery),
                )
                .order_by(rank.desc(), MaterialFragment.id)
                .limit(limit)
            )
            rows = db.scalars(statement).all()
            return [self._dto(*sources[row.file_id], row) for row in rows]

    def read_pinned(self, actor: Actor, pinned: dict) -> dict:
        ref = pinned.get("source_ref", {})
        if ref.get("kind") != "material_fragment" or pinned.get("ref_id") != "material:" + str(ref.get("id", "")):
            raise DomainError("invalid_material_evidence_reference", 400)
        with self.factory.begin() as db:
            file_id = UUID(pinned["file_id"])
            source = self._source(db, actor, file_id)
            row = db.scalar(
                select(MaterialFragment).where(
                    MaterialFragment.id == ref["id"],
                    MaterialFragment.owner_id == actor.owner_id,
                    MaterialFragment.file_id == file_id,
                    MaterialFragment.representation_id == source[1].id,
                )
            )
            if row is None:
                raise DomainError("material_evidence_not_found", 404)
            value = self._dto(*source, row)
            if value != pinned:
                raise DomainError("material_evidence_pin_changed", 409)
            return value


class Config(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: str
    dsn_file: str
    cas_root: str
    retrieval_mode: Literal["native_hybrid", "fts_diagnostic"] = "native_hybrid"
    native_config_file: str | None = None


def service_from_config(path: Path) -> MaterialEvidenceService:
    path = path.resolve(strict=True)
    config = Config.model_validate_json(path.read_bytes())
    if config.version != "citequest-material-evidence/1":
        raise ValueError("material_evidence_config_version")
    native = None

    def configured(value: str) -> Path:
        candidate = Path(value)
        return (candidate if candidate.is_absolute() else path.parent / candidate).resolve(strict=True)

    if config.native_config_file:
        from .native import index_from_config

        native = index_from_config(configured(config.native_config_file))
    dsn = configured(config.dsn_file).read_text(encoding="utf-8").strip()
    if dsn.startswith("postgresql://"):
        dsn = dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    return MaterialEvidenceService(
        sessions(engine(dsn)),
        CAS(configured(config.cas_root)),
        native_index=native,
        retrieval_mode=config.retrieval_mode,
    )
