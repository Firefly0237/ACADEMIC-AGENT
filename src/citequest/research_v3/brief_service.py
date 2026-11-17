"""Transactional project intake, reusing the product owner lock and operation ledger."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from sqlalchemy import select, tuple_

from citequest.agents_v3.contracts import EvidenceSelection, canonical, digest
from citequest.v2.domain import Actor, DomainError
from citequest.v2.pagination import Pager
from citequest.v2.services import Catalog

from .brief import (
    BriefCreate,
    BriefDocument,
    BriefResolution,
    BriefTurn,
    apply_turn,
    create_document,
    planning_payload,
    readiness,
    resolve,
)
from .brief_models import ResearchBriefRevision, ResearchBriefRow


def material_bibliography_pin(catalog: Catalog, file_id: UUID) -> dict:
    """Read the current owned parser identity without changing its payload or rights."""
    from citequest.materials_v2.models import MaterialIngestion, MaterialUpload, ParsedRepresentation

    catalog.db.expire_all()
    file = catalog.file(file_id)
    if file.status not in {"ready", "partial"}:
        raise DomainError("material_not_readable", 409)
    ingestion = catalog.db.scalar(select(MaterialIngestion).where(
        MaterialIngestion.file_id == file_id, MaterialIngestion.owner_id == catalog.actor.owner_id))
    representation = catalog.db.get(ParsedRepresentation, ingestion.representation_id) if ingestion and ingestion.representation_id else None
    if representation is None or representation.owner_id != catalog.actor.owner_id or representation.file_id != file_id or representation.source_sha256 != file.sha256:
        raise DomainError("material_bibliography_source_changed", 409)
    upload = catalog.db.scalar(select(MaterialUpload).where(
        MaterialUpload.file_id == file_id, MaterialUpload.owner_id == catalog.actor.owner_id))
    return {"file_id": str(file_id), "file_sha256": file.sha256, "representation_id": representation.id,
        "material_name": upload.display_title if upload else "研究材料"}


def read_confirmed_material_bibliography(catalog: Catalog, brief_id: UUID, file_id: UUID,
                                        *, expected_brief_sha256: str) -> dict:
    row = catalog.db.scalar(select(ResearchBriefRow).where(ResearchBriefRow.id == brief_id,
        ResearchBriefRow.owner_id == catalog.actor.owner_id, ResearchBriefRow.deleted_at.is_(None))
        .execution_options(populate_existing=True))
    if row is None:
        raise DomainError("research_brief_not_found", 404)
    catalog.project(row.project_id)
    document = BriefService.document(row)
    if row.document_sha256 != expected_brief_sha256:
        raise DomainError("research_brief_revision_conflict", 409)
    identifier = str(file_id)
    if identifier in {str(value) for value in document.draft_source_ids}:
        raise DomainError("research_draft_is_not_bibliographic_evidence", 422)
    record = (document.material_bibliography or {}).get(identifier)
    if record is None:
        raise DomainError("research_material_bibliography_required", 422)
    pin = material_bibliography_pin(catalog, file_id)
    if record.file_sha256 != pin["file_sha256"] or record.representation_id != pin["representation_id"]:
        raise DomainError("material_bibliography_source_changed", 409)
    metadata = record.model_dump(mode="json")
    return {"bibliography": metadata, "bibliography_sha256": digest(metadata), "provenance": "user_confirmed", "source": pin}


def read_historical_confirmed_material_bibliography(
    catalog: Catalog, brief_id: UUID, brief_revision: int, file_id: UUID, *, expected_brief_sha256: str,
) -> dict:
    """A fixed intake version keeps current file rights and confirmation guards."""
    head = catalog.db.scalar(select(ResearchBriefRow).where(ResearchBriefRow.id == brief_id,
        ResearchBriefRow.owner_id == catalog.actor.owner_id, ResearchBriefRow.deleted_at.is_(None))
        .execution_options(populate_existing=True))
    if head is None:
        raise DomainError("research_brief_not_found", 404)
    catalog.project(head.project_id)
    revision = catalog.db.scalar(select(ResearchBriefRevision).where(
        ResearchBriefRevision.brief_id == brief_id, ResearchBriefRevision.owner_id == catalog.actor.owner_id,
        ResearchBriefRevision.revision == brief_revision).execution_options(populate_existing=True))
    if revision is None:
        raise DomainError("research_brief_revision_unavailable", 404)
    document = BriefDocument.model_validate_json(canonical(revision.document), strict=True)
    if (document.id != brief_id or document.project_id != head.project_id or document.revision != brief_revision
            or digest(revision.document) != revision.document_sha256
            or revision.document_sha256 != expected_brief_sha256):
        raise DomainError("research_brief_revision_integrity_failure", 409)
    current = BriefService.document(head)
    identifier = str(file_id)
    old_record = (document.material_bibliography or {}).get(identifier)
    current_record = (current.material_bibliography or {}).get(identifier)
    if (old_record is None or old_record != current_record
            or identifier in {str(value) for value in document.draft_source_ids}
            or identifier in {str(value) for value in current.draft_source_ids}):
        raise DomainError("research_source_bibliography_changed", 409)
    pin = material_bibliography_pin(catalog, file_id)
    if old_record.file_sha256 != pin["file_sha256"] or old_record.representation_id != pin["representation_id"]:
        raise DomainError("material_bibliography_source_changed", 409)
    metadata = old_record.model_dump(mode="json")
    return {"bibliography": metadata, "bibliography_sha256": digest(metadata), "provenance": "user_confirmed", "source": pin}


class BriefService:
    """Caller owns Session.begin; source admission is mandatory at planning handoff."""

    def __init__(self, catalog: Catalog, authorize_sources: Callable[[Actor, list[EvidenceSelection]], None]) -> None:
        self.catalog, self.db, self.actor = catalog, catalog.db, catalog.actor
        self.authorize_sources = authorize_sources

    def row(self, brief_id: UUID, *, include_deleted: bool = False) -> ResearchBriefRow:
        query = select(ResearchBriefRow).where(
            ResearchBriefRow.id == brief_id, ResearchBriefRow.owner_id == self.actor.owner_id
        )
        if not include_deleted:
            query = query.where(ResearchBriefRow.deleted_at.is_(None))
        row = self.db.scalar(query.execution_options(populate_existing=True))
        if row is None:
            raise DomainError("not_found", 404)
        self.catalog.project(row.project_id)
        return row

    @staticmethod
    def document(row: ResearchBriefRow) -> BriefDocument:
        value = BriefDocument.model_validate(row.document)
        if (
            value.id != row.id
            or value.project_id != row.project_id
            or value.revision != row.revision
            or digest(row.document) != row.document_sha256
        ):
            raise DomainError("research_brief_integrity_failure", 500)
        return value

    def present(self, row: ResearchBriefRow) -> dict:
        document = self.document(row)
        return {
            "brief": document.model_dump(mode="json"),
            "brief_sha256": row.document_sha256,
            "deleted": row.deleted_at is not None,
            "readiness": readiness(document),
        }

    def get(self, brief_id: UUID) -> dict:
        return self.present(self.row(brief_id))

    def bibliography_source(self, brief_id: UUID, file_id: UUID) -> dict:
        document = self.document(self.row(brief_id))
        identifier = str(file_id)
        references = {source.id for source in document.selected_sources if source.kind == "material_file"}
        if identifier not in references or identifier in {str(value) for value in document.draft_source_ids}:
            raise DomainError("material_not_in_research_reference_scope", 403)
        pin = material_bibliography_pin(self.catalog, file_id)
        record = (document.material_bibliography or {}).get(identifier)
        return {"source": pin, "bibliography": record.model_dump(mode="json") if record else None,
            "source_matches_confirmation": bool(record and record.file_sha256 == pin["file_sha256"]
                and record.representation_id == pin["representation_id"])}

    def list(self, project_id: UUID, pager: Pager, *, limit: int = 20, cursor: str | None = None) -> dict:
        self.catalog.project(project_id)
        if not 1 <= limit <= 50:
            raise DomainError("invalid_page", 400)
        identity = {
            "owner": str(self.actor.owner_id),
            "project_id": str(project_id),
            "kind": "research_briefs",
            "limit": limit,
        }
        query = select(ResearchBriefRow).where(
            ResearchBriefRow.owner_id == self.actor.owner_id,
            ResearchBriefRow.project_id == project_id,
            ResearchBriefRow.deleted_at.is_(None),
        )
        if cursor:
            last = pager.decode(cursor, identity)
            query = query.where(
                tuple_(ResearchBriefRow.created_at, ResearchBriefRow.id)
                < tuple_(datetime.fromisoformat(last["created"]), UUID(last["id"]))
            )
        rows = list(
            self.db.scalars(
                query.order_by(ResearchBriefRow.created_at.desc(), ResearchBriefRow.id.desc()).limit(limit + 1)
            )
        )
        next_cursor = None
        if len(rows) > limit:
            last = rows[limit - 1]
            next_cursor = pager.encode(
                {
                    "identity": identity,
                    "created": last.created_at.isoformat(),
                    "id": str(last.id),
                    "expires": datetime.now(UTC).timestamp() + 900,
                }
            )
        return {
            "items": [
                {
                    "brief_id": str(row.id),
                    "project_id": str(row.project_id),
                    "revision": row.revision,
                    "created_at": row.created_at.isoformat(),
                    "topic": self.document(row).facts.get("topic").value
                    if "topic" in self.document(row).facts
                    else None,
                    "readiness": readiness(self.document(row)),
                }
                for row in rows[:limit]
            ],
            "next_cursor": next_cursor,
        }

    def _store(self, row: ResearchBriefRow, document: BriefDocument, operation_id: UUID, kind: str) -> dict:
        row.revision, row.document = document.revision, document.model_dump(mode="json")
        row.document_sha256 = digest(row.document)
        self.db.add(
            ResearchBriefRevision(
                brief_id=row.id,
                revision=row.revision,
                owner_id=self.actor.owner_id,
                operation_id=operation_id,
                document=row.document,
                document_sha256=row.document_sha256,
                change_kind=kind,
            )
        )
        self.db.flush()
        return self.present(row)

    def create(self, request: BriefCreate) -> dict:
        self.catalog.project(request.project_id)
        self.authorize_sources(self.actor, request.selected_sources)

        def effect() -> dict:
            self.authorize_sources(self.actor, request.selected_sources)
            value = create_document(request)
            row = ResearchBriefRow(
                id=value.id,
                owner_id=self.actor.owner_id,
                project_id=value.project_id,
                revision=1,
                document=value.model_dump(mode="json"),
                document_sha256=digest(value.model_dump(mode="json")),
            )
            self.db.add(row)
            self.db.flush()
            return self._store(row, value, request.operation_id, "created")

        result = self.catalog.operation(
            request.operation_id, {"operation": "create_research_brief", **request.model_dump(mode="json")}, effect
        )
        # The operation ledger is immutable execution history, not current read
        # authority. Replays must not move the UI back to an older revision.
        return self.present(self.row(UUID(result["brief"]["id"])))

    def turn(self, brief_id: UUID, request: BriefTurn) -> dict:
        self.row(brief_id)
        if request.selected_sources is not None:
            self.authorize_sources(self.actor, request.selected_sources)

        def effect() -> dict:
            row = self.row(brief_id)
            if request.selected_sources is not None:
                self.authorize_sources(self.actor, request.selected_sources)
            document = apply_turn(self.document(row), request)
            for identifier, record in (request.material_bibliography or {}).items():
                if record is None:
                    continue
                pin = material_bibliography_pin(self.catalog, UUID(identifier))
                if record.file_sha256 != pin["file_sha256"] or record.representation_id != pin["representation_id"]:
                    raise DomainError("material_bibliography_source_changed", 409)
            return self._store(row, document, request.operation_id, "turn")

        self.catalog.operation(
            request.operation_id,
            {"operation": "research_brief_turn", "brief_id": str(brief_id), **request.model_dump(mode="json")},
            effect,
        )
        return self.present(self.row(brief_id))

    def resolution(self, brief_id: UUID, request: BriefResolution) -> dict:
        self.row(brief_id)

        def effect() -> dict:
            row = self.row(brief_id)
            return self._store(row, resolve(self.document(row), request), request.operation_id, "resolved")

        self.catalog.operation(
            request.operation_id,
            {"operation": "resolve_research_brief", "brief_id": str(brief_id), **request.model_dump(mode="json")},
            effect,
        )
        return self.present(self.row(brief_id))

    def planning(self, brief_id: UUID, expected_revision: int, action: Literal["plan", "experiment"] = "plan") -> dict:
        document = self.document(self.row(brief_id))
        if expected_revision != document.revision:
            raise DomainError("brief_revision_conflict")
        self.authorize_sources(self.actor, document.selected_sources)
        payload = planning_payload(document, action=action)
        for identifier in ((document.material_bibliography or {}) if action == "plan" else ()):
            read_confirmed_material_bibliography(self.catalog, brief_id, UUID(identifier),
                expected_brief_sha256=digest(document.model_dump(mode="json")))
        return payload

    def set_deleted(self, brief_id: UUID, operation_id: UUID, expected_revision: int, *, deleted: bool) -> dict:
        self.row(brief_id, include_deleted=True)

        def effect() -> dict:
            row = self.row(brief_id, include_deleted=True)
            document = self.document(row)
            if expected_revision != document.revision:
                raise DomainError("brief_revision_conflict")
            if deleted == (row.deleted_at is not None):
                raise DomainError("research_brief_state_conflict")
            row.deleted_at = datetime.now(UTC) if deleted else None
            value = document.model_copy(update={"revision": document.revision + 1})
            return self._store(row, value, operation_id, "deleted" if deleted else "restored")

        self.catalog.operation(
            operation_id,
            {
                "operation": "delete_or_restore_research_brief",
                "brief_id": str(brief_id),
                "expected_revision": expected_revision,
                "deleted": deleted,
            },
            effect,
        )
        return self.present(self.row(brief_id, include_deleted=True))
