"""Owner-scoped typed bookmarks; saved metadata is never an ACL fallback.

Source title queries use bounded *current* authorized projections, not a search
oracle over withdrawn snapshots. A partial scan has an explicit continuation.
The caller owns Session.begin and supplies the current SearchService.detail.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import delete, exists, func, select, text, tuple_
from sqlalchemy.exc import DBAPIError

from citequest.agents_v3.contracts import digest
from citequest.search_v3.batch_headers import CurrentPaperHeader, locator_key
from citequest.search_v3.schemas import PaperDetail, PaperLocator
from citequest.v2 import models as core
from citequest.v2.domain import Actor, DomainError
from citequest.v2.pagination import Pager
from citequest.v2.services import Catalog

from .saved_paper_contracts import (
    SavedPaperCreate,
    SavedPaperDTO,
    SavedPaperLifecycle,
    SavedPaperNoteCreate,
    SavedPaperNoteDelete,
    SavedPaperNotePatch,
    SavedPaperPatch,
)
from .saved_paper_models import SavedPaperCollection, SavedPaperNote, SavedPaperReference, SavedPaperTag


class SavedPaperService:
    MAX_QUERY_SCAN = 200

    def __init__(
        self,
        catalog: Catalog,
        pager: Pager,
        read_paper: Callable[[Actor, PaperLocator], PaperDetail],
        *,
        read_many: Callable[[Actor, list[PaperLocator]], dict[str, CurrentPaperHeader]] | None = None,
    ) -> None:
        self.catalog, self.pager, self.read_paper = catalog, pager, read_paper
        self.db, self.actor = catalog.db, catalog.actor
        self.read_many = read_many

    def current_many(self, papers: list[PaperLocator]) -> tuple[dict[str, tuple[PaperDetail | None, str]], list[str]]:
        if not papers:
            return {}, []
        if self.read_many is None:
            # Explicit compatibility profile, never a fallback after a failed
            # registered batch reader. Product composition selects its reader.
            return {locator_key(paper): self.current(paper) for paper in papers}, ["scalar_registered"]
        try:
            result = self.read_many(self.actor, papers)
            if set(result) != {locator_key(paper) for paper in papers} or any(
                not isinstance(result[locator_key(paper)], CurrentPaperHeader)
                or result[locator_key(paper)].paper != paper
                for paper in papers
            ):
                raise DomainError("current_header_batch_contract_invalid", 503)
            return {key: (header.detail, header.source_availability) for key, header in result.items()}, sorted(
                {header.read_mode for header in result.values()}
            )
        except DomainError as error:
            state = "unavailable" if error.status in {401, 403, 404, 410, 451} else "temporarily_unavailable"
        except (DBAPIError, OSError, TimeoutError):
            state = "temporarily_unavailable"
        return {locator_key(paper): (None, state) for paper in papers}, ["batch_failed_no_scalar_fallback"]

    def row(self, identifier: UUID) -> SavedPaperReference:
        row = self.db.scalar(
            select(SavedPaperReference)
            .where(SavedPaperReference.id == identifier, SavedPaperReference.owner_id == self.actor.owner_id)
            .execution_options(populate_existing=True)
        )
        if row is None:
            raise DomainError("not_found", 404)
        return row

    def current(self, paper: PaperLocator) -> tuple[PaperDetail | None, str]:
        try:
            value = self.read_paper(self.actor, paper)
            if not isinstance(value, PaperDetail) or value.paper != paper:
                raise DomainError("current_paper_detail_contract_invalid", 503)
            return value, "available"
        except DomainError as error:
            return None, "unavailable" if error.status in {401, 403, 404, 410, 451} else "temporarily_unavailable"
        except (DBAPIError, OSError, TimeoutError):
            return None, "temporarily_unavailable"

    def decorations(self, identifiers: list[UUID]) -> dict[UUID, dict]:
        result = {identifier: {"collection_ids": [], "tags": [], "note_count": 0} for identifier in identifiers}
        if not identifiers:
            return result
        for identifier, collection in self.db.execute(
            select(SavedPaperCollection.saved_paper_id, SavedPaperCollection.collection_id)
            .join(
                core.Collection,
                (core.Collection.id == SavedPaperCollection.collection_id)
                & (core.Collection.owner_id == self.actor.owner_id),
            )
            .where(
                SavedPaperCollection.owner_id == self.actor.owner_id,
                SavedPaperCollection.saved_paper_id.in_(identifiers),
                core.Collection.deleted_at.is_(None),
            )
            .order_by(SavedPaperCollection.collection_id)
        ):
            result[identifier]["collection_ids"].append(collection)
        for identifier, tag in self.db.execute(
            select(SavedPaperTag.saved_paper_id, SavedPaperTag.tag)
            .where(SavedPaperTag.owner_id == self.actor.owner_id, SavedPaperTag.saved_paper_id.in_(identifiers))
            .order_by(SavedPaperTag.tag)
        ):
            result[identifier]["tags"].append(tag)
        for identifier, count in self.db.execute(
            select(SavedPaperNote.saved_paper_id, func.count())
            .where(
                SavedPaperNote.owner_id == self.actor.owner_id,
                SavedPaperNote.saved_paper_id.in_(identifiers),
                SavedPaperNote.deleted_at.is_(None),
            )
            .group_by(SavedPaperNote.saved_paper_id)
        ):
            result[identifier]["note_count"] = count
        return result

    def present(
        self,
        row: SavedPaperReference,
        decoration: dict | None = None,
        projection: tuple[PaperDetail | None, str] | None = None,
    ) -> dict:
        locator = PaperLocator(kind=row.paper_kind, id=row.paper_id)
        detail, availability = projection if projection is not None else self.current(locator)
        decoration = decoration or self.decorations([row.id])[row.id]
        native = detail is not None and detail.coverage == "native_literal" and detail.literal_text_available
        dto = SavedPaperDTO(
            saved_paper_id=row.id,
            paper=locator,
            revision=row.revision,
            deleted=row.deleted_at is not None,
            source_availability=availability,
            metadata=detail,
            **decoration,
            qa_scope="unavailable" if detail is None else "native_text" if native else "metadata_only",
            fulltext_summary_available=native,
        )
        return dto.model_dump(mode="json")

    def get(self, identifier: UUID) -> dict:
        return self.present(self.row(identifier))

    def collections(self, identifiers: list[UUID]) -> None:
        for identifier in identifiers:
            self.catalog.collection(identifier)

    def replace_organization(
        self, row: SavedPaperReference, collections: list[UUID], tags: list[str], *, validate: bool = True
    ) -> None:
        if validate:
            self.collections(collections)
        for model in (SavedPaperCollection, SavedPaperTag):
            self.db.execute(delete(model).where(model.owner_id == self.actor.owner_id, model.saved_paper_id == row.id))
        self.db.add_all(
            SavedPaperCollection(owner_id=self.actor.owner_id, saved_paper_id=row.id, collection_id=value)
            for value in collections
        )
        self.db.add_all(SavedPaperTag(owner_id=self.actor.owner_id, saved_paper_id=row.id, tag=value) for value in tags)

    def create(self, request: SavedPaperCreate) -> dict:
        def effect() -> dict:
            row = self.db.scalar(
                select(SavedPaperReference)
                .where(
                    SavedPaperReference.owner_id == self.actor.owner_id,
                    SavedPaperReference.paper_kind == request.paper.kind,
                    SavedPaperReference.paper_id == request.paper.id,
                )
                .execution_options(populate_existing=True)
            )
            if row is None:
                self.collections(request.collection_ids)
                detail = self.read_paper(self.actor, request.paper)
                if not isinstance(detail, PaperDetail) or detail.paper != request.paper:
                    raise DomainError("current_paper_detail_contract_invalid", 503)
                captured = detail.model_dump(mode="json")
                size = self.db.execute(
                    text("SELECT octet_length(CAST(CAST(:payload AS jsonb) AS text))"),
                    {"payload": json.dumps(captured, ensure_ascii=False)},
                ).scalar_one()
                if size > 131072:
                    raise DomainError("saved_paper_metadata_byte_limit", 413)
                work_id = detail.work_id
                if work_id is not None:
                    try:
                        self.catalog.work(work_id)
                    except DomainError:
                        work_id = None  # optional association only; never create a Work
                row = SavedPaperReference(
                    id=uuid4(),
                    owner_id=self.actor.owner_id,
                    paper_kind=request.paper.kind,
                    paper_id=request.paper.id,
                    work_id=work_id,
                    source_snapshot_id=detail.source_snapshot_id,
                    captured_metadata=captured,
                    captured_metadata_sha256=digest(captured),
                    source_projection_sha256=detail.source.projection_sha256,
                    source_pointer_sha256=detail.source.source_pointer_sha256,
                    revision=1,
                )
                self.db.add(row)
                self.db.flush()
                self.replace_organization(row, request.collection_ids, request.tags)
            else:
                # Repeated save adds requested organization; it never erases prior notes/tags.
                before = self.decorations([row.id])[row.id]
                self.collections(request.collection_ids)
                prior_collections = list(
                    self.db.scalars(
                        select(SavedPaperCollection.collection_id).where(
                            SavedPaperCollection.owner_id == self.actor.owner_id,
                            SavedPaperCollection.saved_paper_id == row.id,
                        )
                    )
                )
                collections = sorted(set(prior_collections) | set(request.collection_ids), key=str)
                tags = sorted(set(before["tags"]) | set(request.tags))
                # The total must remain editable by the same 30-item command
                # contract. Per-request limits alone do not bound repeat-save.
                if len(collections) > 30 or len(tags) > 30:
                    raise DomainError("saved_paper_organization_limit", 422)
                if set(collections) != set(prior_collections) or tags != before["tags"] or row.deleted_at is not None:
                    self.replace_organization(row, collections, tags, validate=False)
                    row.deleted_at, row.revision = None, row.revision + 1
            self.db.flush()
            return {"saved_paper_id": str(row.id)}

        pointer = self.catalog.operation(
            request.operation_id, {"operation": "saved_paper_create", **request.model_dump(mode="json")}, effect
        )
        return self.get(UUID(pointer["saved_paper_id"]))

    def patch(self, identifier: UUID, request: SavedPaperPatch) -> dict:
        self.row(identifier)

        def effect() -> dict:
            row = self.row(identifier)
            if row.revision != request.expected_revision:
                raise DomainError("revision_conflict")
            self.replace_organization(row, request.collection_ids, request.tags)
            row.revision += 1
            self.db.flush()
            return {"saved_paper_id": str(row.id)}

        pointer = self.catalog.operation(
            request.operation_id,
            {"operation": "saved_paper_patch", "id": str(identifier), **request.model_dump(mode="json")},
            effect,
        )
        return self.get(UUID(pointer["saved_paper_id"]))

    def lifecycle(self, identifier: UUID, request: SavedPaperLifecycle) -> dict:
        self.row(identifier)

        def effect() -> dict:
            row = self.row(identifier)
            if row.revision != request.expected_revision:
                raise DomainError("revision_conflict")
            if request.deleted == (row.deleted_at is not None):
                raise DomainError("lifecycle_conflict")
            row.deleted_at = datetime.now(UTC) if request.deleted else None
            row.revision += 1
            self.db.flush()
            return {"saved_paper_id": str(row.id)}

        pointer = self.catalog.operation(
            request.operation_id,
            {"operation": "saved_paper_lifecycle", "id": str(identifier), **request.model_dump(mode="json")},
            effect,
        )
        return self.get(UUID(pointer["saved_paper_id"]))

    def cursor(self, row: SavedPaperReference | SavedPaperNote | core.Note, identity: dict) -> str:
        return self.pager.encode(
            {
                "identity": identity,
                "created": row.created_at.isoformat(),
                "id": str(row.id),
                "expires": datetime.now(UTC).timestamp() + 900,
            }
        )

    def page(
        self,
        *,
        limit: int = 20,
        cursor: str | None = None,
        query: str = "",
        collection_id: UUID | None = None,
        tag: str | None = None,
    ) -> dict:
        if not 1 <= limit <= 50 or len(query) > 200 or (tag is not None and len(tag) > 100):
            raise DomainError("invalid_page", 400)
        if self.db.execute(text("SHOW transaction_isolation")).scalar_one() != "read committed":
            raise DomainError("library_requires_read_committed_projection", 503)
        if collection_id:
            self.catalog.collection(collection_id)
        identity = {
            "owner": str(self.actor.owner_id),
            "kind": "typed_saved_papers",
            "limit": limit,
            "query": query,
            "collection_id": str(collection_id) if collection_id else None,
            "tag": tag,
        }
        statement = select(SavedPaperReference).where(
            SavedPaperReference.owner_id == self.actor.owner_id, SavedPaperReference.deleted_at.is_(None)
        )
        if collection_id:
            statement = statement.where(
                exists(
                    select(SavedPaperCollection.saved_paper_id).where(
                        SavedPaperCollection.owner_id == self.actor.owner_id,
                        SavedPaperCollection.saved_paper_id == SavedPaperReference.id,
                        SavedPaperCollection.collection_id == collection_id,
                    )
                )
            )
        if tag is not None:
            statement = statement.where(
                exists(
                    select(SavedPaperTag.saved_paper_id).where(
                        SavedPaperTag.owner_id == self.actor.owner_id,
                        SavedPaperTag.saved_paper_id == SavedPaperReference.id,
                        SavedPaperTag.tag == tag,
                    )
                )
            )
        if cursor:
            last = self.pager.decode(cursor, identity)
            statement = statement.where(
                tuple_(SavedPaperReference.created_at, SavedPaperReference.id)
                < tuple_(datetime.fromisoformat(last["created"]), UUID(last["id"]))
            )
        normalized = query.strip().casefold()
        maximum = self.MAX_QUERY_SCAN if normalized else limit + 1
        rows = list(
            self.db.scalars(
                statement.order_by(SavedPaperReference.created_at.desc(), SavedPaperReference.id.desc()).limit(
                    maximum + 1
                )
            )
        )
        candidates = rows[:maximum]
        initial_revisions = {row.id: row.revision for row in candidates}
        projections, read_modes = self.current_many(
            [PaperLocator(kind=row.paper_kind, id=row.paper_id) for row in candidates]
        )
        # Source reads use independent short transactions. Repeat the original
        # owner/lifecycle/organization predicate afterwards so a concurrently
        # removed bookmark or changed collection/tag cannot leak an old item.
        current_rows = (
            {
                row.id: row
                for row in self.db.scalars(
                    statement.where(SavedPaperReference.id.in_(initial_revisions)).execution_options(
                        populate_existing=True
                    )
                )
            }
            if candidates
            else {}
        )
        library_changed = len(current_rows) != len(candidates) or any(
            row.revision != initial_revisions[identifier] for identifier, row in current_rows.items()
        )
        decorations = self.decorations(list(current_rows))
        source_status_counts = {"available": 0, "unavailable": 0, "temporarily_unavailable": 0}
        matches, scanned = [], 0
        for row in candidates:
            scanned += 1
            if row.id not in current_rows:
                continue
            row = current_rows[row.id]
            detail = projections[locator_key(PaperLocator(kind=row.paper_kind, id=row.paper_id))]
            dto = self.present(row, decorations[row.id], detail)
            source_status_counts[dto["source_availability"]] += 1
            # Only currently authorized metadata participates in title/abstract matching.
            text_value = (
                " ".join(str(dto["metadata"].get(key) or "") for key in ("title", "abstract"))
                if dto["metadata"]
                else ""
            )
            if (
                not normalized
                or normalized in text_value.casefold()
                or any(normalized in value.casefold() for value in dto["tags"])
            ):
                matches.append((row, dto))
                if len(matches) > limit:
                    break
        partial = bool(normalized and len(matches) <= limit and len(rows) > maximum)
        next_cursor = (
            self.cursor(matches[limit - 1][0], identity)
            if len(matches) > limit
            else self.cursor(candidates[-1], identity)
            if len(rows) > maximum
            else None
        )
        return {
            "items": [dto for _, dto in matches[:limit]],
            "next_cursor": next_cursor,
            "partial_scan": partial,
            "scanned": scanned,
            "scan_limit": maximum,
            "query_scope": "current_authorized_title_abstract_and_personal_tags",
            "source_read_modes": read_modes,
            "library_changed_during_read": library_changed,
            "source_status_counts": source_status_counts,
            "source_check_incomplete": bool(
                normalized and (source_status_counts["unavailable"] or source_status_counts["temporarily_unavailable"])
            ),
        }

    def note_row(self, identifier: UUID, *, deleted: bool = False) -> SavedPaperNote:
        statement = select(SavedPaperNote).where(
            SavedPaperNote.id == identifier, SavedPaperNote.owner_id == self.actor.owner_id
        )
        if not deleted:
            statement = statement.where(SavedPaperNote.deleted_at.is_(None))
        row = self.db.scalar(statement.execution_options(populate_existing=True))
        if row is None:
            raise DomainError("not_found", 404)
        self.row(row.saved_paper_id)
        return row

    def note_dto(self, row: SavedPaperNote) -> dict:
        return {
            "note_id": str(row.id),
            "saved_paper_id": str(row.saved_paper_id),
            "body": row.body,
            "revision": row.revision,
            "deleted": row.deleted_at is not None,
        }

    def notes(self, identifier: UUID, *, limit: int = 20, cursor: str | None = None) -> dict:
        saved = self.row(identifier)
        if not 1 <= limit <= 50:
            raise DomainError("invalid_page", 400)
        identity = {
            "owner": str(self.actor.owner_id),
            "kind": "typed_saved_paper_notes",
            "saved_paper_id": str(identifier),
            "limit": limit,
        }
        statement = select(SavedPaperNote).where(
            SavedPaperNote.owner_id == self.actor.owner_id,
            SavedPaperNote.saved_paper_id == identifier,
            SavedPaperNote.deleted_at.is_(None),
        )
        if cursor:
            last = self.pager.decode(cursor, identity)
            statement = statement.where(
                tuple_(SavedPaperNote.created_at, SavedPaperNote.id)
                < tuple_(datetime.fromisoformat(last["created"]), UUID(last["id"]))
            )
        rows = list(
            self.db.scalars(
                statement.order_by(SavedPaperNote.created_at.desc(), SavedPaperNote.id.desc()).limit(limit + 1)
            )
        )
        _, available = self.current(PaperLocator(kind=saved.paper_kind, id=saved.paper_id))
        return {
            "items": [self.note_dto(row) for row in rows[:limit]],
            "next_cursor": self.cursor(rows[limit - 1], identity) if len(rows) > limit else None,
            "source_availability": available,
        }

    def create_note(self, identifier: UUID, request: SavedPaperNoteCreate) -> dict:
        self.row(identifier)

        def effect() -> dict:
            saved = self.row(identifier)
            if saved.deleted_at:
                raise DomainError("saved_paper_deleted")
            row = SavedPaperNote(
                id=uuid4(), owner_id=self.actor.owner_id, saved_paper_id=identifier, body=request.body, revision=1
            )
            self.db.add(row)
            self.db.flush()
            return {"note_id": str(row.id)}

        pointer = self.catalog.operation(
            request.operation_id,
            {
                "operation": "saved_paper_note_create",
                "saved_paper_id": str(identifier),
                **request.model_dump(mode="json"),
            },
            effect,
        )
        row = self.note_row(UUID(pointer["note_id"]), deleted=True)
        _, availability = self.current(
            PaperLocator(kind=self.row(identifier).paper_kind, id=self.row(identifier).paper_id)
        )
        return {**self.note_dto(row), "source_availability": availability}

    def patch_note(self, identifier: UUID, request: SavedPaperNotePatch) -> dict:
        self.note_row(identifier, deleted=True)

        def effect() -> dict:
            row = self.note_row(identifier)
            if self.row(row.saved_paper_id).deleted_at:
                raise DomainError("saved_paper_deleted")
            if row.revision != request.expected_revision:
                raise DomainError("revision_conflict")
            row.body, row.revision = request.body, row.revision + 1
            self.db.flush()
            return {"note_id": str(row.id)}

        pointer = self.catalog.operation(
            request.operation_id,
            {"operation": "saved_paper_note_patch", "note_id": str(identifier), **request.model_dump(mode="json")},
            effect,
        )
        row = self.note_row(UUID(pointer["note_id"]), deleted=True)
        saved = self.row(row.saved_paper_id)
        _, availability = self.current(PaperLocator(kind=saved.paper_kind, id=saved.paper_id))
        return {**self.note_dto(row), "source_availability": availability}

    def delete_note(self, identifier: UUID, request: SavedPaperNoteDelete) -> dict:
        self.note_row(identifier, deleted=True)

        def effect() -> dict:
            row = self.note_row(identifier, deleted=True)
            if row.revision != request.expected_revision:
                raise DomainError("revision_conflict")
            if row.deleted_at:
                raise DomainError("lifecycle_conflict")
            row.deleted_at, row.revision = datetime.now(UTC), row.revision + 1
            self.db.flush()
            return {"note_id": str(row.id)}

        pointer = self.catalog.operation(
            request.operation_id,
            {"operation": "saved_paper_note_delete", "note_id": str(identifier), **request.model_dump(mode="json")},
            effect,
        )
        row = self.note_row(UUID(pointer["note_id"]), deleted=True)
        saved = self.row(row.saved_paper_id)
        _, availability = self.current(PaperLocator(kind=saved.paper_kind, id=saved.paper_id))
        return {
            "note_id": str(row.id),
            "revision": row.revision,
            "deleted": row.deleted_at is not None,
            "source_availability": availability,
        }

    def legacy_notes(self, identifier: UUID, *, limit: int = 20, cursor: str | None = None) -> dict:
        self.catalog.item(identifier)
        if not 1 <= limit <= 50:
            raise DomainError("invalid_page", 400)
        identity = {
            "owner": str(self.actor.owner_id),
            "kind": "legacy_item_notes",
            "library_item_id": str(identifier),
            "limit": limit,
        }
        statement = select(core.Note).where(
            core.Note.owner_id == self.actor.owner_id,
            core.Note.library_item_id == identifier,
            core.Note.deleted_at.is_(None),
        )
        if cursor:
            last = self.pager.decode(cursor, identity)
            statement = statement.where(
                tuple_(core.Note.created_at, core.Note.id)
                < tuple_(datetime.fromisoformat(last["created"]), UUID(last["id"]))
            )
        rows = list(
            self.db.scalars(statement.order_by(core.Note.created_at.desc(), core.Note.id.desc()).limit(limit + 1))
        )
        return {
            "items": [
                {
                    "note_id": str(row.id),
                    "library_item_id": str(row.library_item_id),
                    "body": row.body,
                    "revision": row.revision,
                }
                for row in rows[:limit]
            ],
            "next_cursor": self.cursor(rows[limit - 1], identity) if len(rows) > limit else None,
        }

    def record_view(self, operation_id: UUID, paper: PaperLocator) -> dict:
        # A view is not a save or completed download; replays still verify current source.
        detail = self.read_paper(self.actor, paper)
        if not isinstance(detail, PaperDetail) or detail.paper != paper:
            raise DomainError("current_paper_detail_contract_invalid", 503)

        def effect() -> dict:
            identifier = self.catalog.append_history("view", "paper:" + paper.kind + ":" + paper.id)
            self.db.flush()
            return {"history_id": str(identifier), "kind": "view", "recorded": "view_only"}

        return self.catalog.operation(
            operation_id, {"operation": "typed_paper_view", "paper": paper.model_dump(mode="json")}, effect
        )
