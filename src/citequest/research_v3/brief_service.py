"""Transactional project intake, reusing the product owner lock and operation ledger."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID

from sqlalchemy import select, tuple_

from citequest.agents_v3.contracts import EvidenceSelection, digest
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
            return self._store(row, apply_turn(self.document(row), request), request.operation_id, "turn")

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

    def planning(self, brief_id: UUID, expected_revision: int) -> dict:
        document = self.document(self.row(brief_id))
        if expected_revision != document.revision:
            raise DomainError("brief_revision_conflict")
        self.authorize_sources(self.actor, document.selected_sources)
        return planning_payload(document)

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
