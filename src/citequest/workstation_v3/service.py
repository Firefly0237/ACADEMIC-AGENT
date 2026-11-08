"""Compose existing authorized product services without a second library store.

No method commits, calls a model, runs a task or returns unguarded task output.
The HTTP adapter owns the authenticated actor, CSRF and Session.begin boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid5

from sqlalchemy import delete, exists, func, or_, select, text, tuple_

from citequest.v2 import models as m
from citequest.v2.domain import DomainError, LibrarySave
from citequest.v2.pagination import Pager
from citequest.v2.services import Catalog

from .contracts import Activity, BatchSave, ClearHistory


class WorkstationService:
    def __init__(self, catalog: Catalog, pager: Pager) -> None:
        self.catalog, self.pager = catalog, pager
        self.db, self.actor = catalog.db, catalog.actor

    def save_batch(self, request: BatchSave) -> dict:
        # Recheck current access even when replaying an already completed operation.
        for work_id in request.work_ids:
            self.catalog.work(work_id)
        if request.collection_id:
            self.catalog.collection(request.collection_id)

        def effect() -> dict:
            items = [
                self.catalog.save(
                    LibrarySave(
                        work_id=work_id,
                        collection_id=request.collection_id,
                        client_operation_id=uuid5(request.operation_id, str(work_id)),
                    )
                )
                for work_id in request.work_ids
            ]
            return {"items": items, "count": len(items), "atomic": True}

        value = self.catalog.operation(
            request.operation_id, {"operation": "workstation_batch_save", **request.model_dump(mode="json")}, effect
        )
        for item in value["items"]:
            self.catalog.item(UUID(item["library_item_id"]))
        return value

    def library(
        self,
        *,
        limit: int = 20,
        cursor: str | None = None,
        query: str = "",
        collection_id: UUID | None = None,
        tag: str | None = None,
    ) -> dict:
        """Bounded owner search with batched decorations, avoiding per-item SQL."""
        if not 1 <= limit <= 50 or len(query) > 200 or (tag is not None and len(tag) > 500):
            raise DomainError("invalid_page", 400)
        if collection_id:
            self.catalog.collection(collection_id)
        identity = {
            "owner": str(self.actor.owner_id),
            "kind": "workstation_library",
            "limit": limit,
            "query": query,
            "collection_id": str(collection_id) if collection_id else None,
            "tag": tag,
        }
        allowed = exists(
            select(m.WorkAccess.work_id).where(
                m.WorkAccess.work_id == m.Work.id,
                m.WorkAccess.owner_id == self.actor.owner_id,
                m.WorkAccess.revoked.is_(False),
            )
        )
        statement = (
            select(m.LibraryItem, m.Work)
            .join(m.Work, m.Work.id == m.LibraryItem.work_id)
            .where(
                m.LibraryItem.owner_id == self.actor.owner_id,
                m.LibraryItem.deleted_at.is_(None),
                or_(m.Work.public.is_(True), allowed),
            )
        )
        if query.strip():
            statement = statement.where(func.lower(m.Work.title).contains(query.strip().lower(), autoescape=True))
        if collection_id:
            statement = statement.where(
                exists(
                    select(m.ItemCollection.item_id).where(
                        m.ItemCollection.owner_id == self.actor.owner_id,
                        m.ItemCollection.item_id == m.LibraryItem.id,
                        m.ItemCollection.collection_id == collection_id,
                    )
                )
            )
        if tag is not None:
            statement = statement.where(
                exists(select(m.ItemTag.item_id).where(m.ItemTag.item_id == m.LibraryItem.id, m.ItemTag.tag == tag))
            )
        if cursor:
            last = self.pager.decode(cursor, identity)
            statement = statement.where(
                tuple_(m.LibraryItem.created_at, m.LibraryItem.id)
                < tuple_(datetime.fromisoformat(last["created"]), UUID(last["id"]))
            )
        rows = list(
            self.db.execute(
                statement.order_by(m.LibraryItem.created_at.desc(), m.LibraryItem.id.desc()).limit(limit + 1)
            )
        )
        next_cursor = None
        if len(rows) > limit:
            end = rows[limit - 1][0]
            next_cursor = self.pager.encode(
                {
                    "identity": identity,
                    "created": end.created_at.isoformat(),
                    "id": str(end.id),
                    "expires": datetime.now(UTC).timestamp() + 900,
                }
            )
        items = {
            row[0].id: {
                "library_item_id": str(row[0].id),
                "work_id": str(row[1].id),
                "title": row[1].title,
                "authors": row[1].authors,
                "year": row[1].year,
                "revision": row[0].revision,
                "file_ids": [],
                "collection_ids": [],
                "tags": [],
            }
            for row in rows[:limit]
        }
        if items:
            ids = list(items)
            for item_id, value in self.db.execute(
                select(m.ItemTag.item_id, m.ItemTag.tag).where(m.ItemTag.item_id.in_(ids)).order_by(m.ItemTag.tag)
            ):
                items[item_id]["tags"].append(value)
            for item_id, value in self.db.execute(
                select(m.ItemCollection.item_id, m.ItemCollection.collection_id)
                .join(m.Collection, m.Collection.id == m.ItemCollection.collection_id)
                .where(
                    m.ItemCollection.owner_id == self.actor.owner_id,
                    m.ItemCollection.item_id.in_(ids),
                    m.Collection.deleted_at.is_(None),
                )
                .order_by(m.ItemCollection.collection_id)
            ):
                items[item_id]["collection_ids"].append(str(value))
            for item_id, value in self.db.execute(
                select(m.ItemFile.item_id, m.ItemFile.file_id)
                .join(m.File, m.File.id == m.ItemFile.file_id)
                .where(
                    m.ItemFile.owner_id == self.actor.owner_id,
                    m.ItemFile.item_id.in_(ids),
                    m.File.owner_id == self.actor.owner_id,
                    m.File.deleted_at.is_(None),
                )
                .order_by(m.ItemFile.file_id)
            ):
                items[item_id]["file_ids"].append(str(value))
        return {"items": list(items.values()), "next_cursor": next_cursor}

    def _task_metadata(self, task_id: UUID) -> dict:
        row = (
            self.db.execute(
                text("""SELECT t.id,t.project_id,t.kind,t.state,t.revision,t.created_at
            FROM cq_app.agent_tasks t JOIN cq_app.projects p
              ON (p.id,p.owner_id)=(t.project_id,t.owner_id)
            WHERE t.id=:task_id AND t.owner_id=:owner"""),
                {"task_id": task_id, "owner": self.actor.owner_id},
            )
            .mappings()
            .first()
        )
        if row is None:
            raise DomainError("not_found", 404)
        return dict(row)

    def record_activity(self, request: Activity) -> dict:
        def authorize() -> None:
            if request.target_kind == "work":
                self.catalog.work(request.target_id)
            elif request.target_kind == "file":
                self.catalog.file(request.target_id)
            else:
                self._task_metadata(request.target_id)

        authorize()

        def effect() -> dict:
            authorize()
            history_id = self.catalog.append_history(request.kind, f"{request.target_kind}:{request.target_id}")
            self.db.flush()
            return {"history_id": str(history_id), "kind": request.kind, "recorded": "request_or_view_only"}

        return self.catalog.operation(
            request.operation_id, {"operation": "workstation_activity", **request.model_dump(mode="json")}, effect
        )

    def clear_history(self, request: ClearHistory) -> dict:
        if request.through > datetime.now(UTC):
            raise DomainError("history_cutoff_in_future", 400)

        def effect() -> dict:
            result = self.db.execute(
                delete(m.History).where(
                    m.History.owner_id == self.actor.owner_id,
                    m.History.created_at <= request.through,
                    m.History.kind.in_(request.kinds),
                )
            )
            return {"deleted": result.rowcount, "through": request.through.isoformat(), "kinds": request.kinds}

        return self.catalog.operation(
            request.operation_id, {"operation": "clear_workstation_history", **request.model_dump(mode="json")}, effect
        )

    def tasks(self, *, project_id: UUID | None = None, limit: int = 20, cursor: str | None = None) -> dict:
        if not 1 <= limit <= 50:
            raise DomainError("invalid_page", 400)
        if project_id is not None:
            self.catalog.project(project_id)
        identity = {
            "owner": str(self.actor.owner_id),
            "kind": "workstation_tasks",
            "project_id": str(project_id) if project_id else None,
            "limit": limit,
        }
        last = self.pager.decode(cursor, identity) if cursor else None
        conditions = ["t.owner_id=:owner"]
        values = {"owner": self.actor.owner_id, "limit": limit + 1}
        if project_id:
            conditions.append("t.project_id=:project")
            values["project"] = project_id
        if last:
            conditions.append("(t.created_at,t.id)<(:created,:id)")
            values.update(created=datetime.fromisoformat(last["created"]), id=UUID(last["id"]))
        # Result, source text and memory-derived content require the existing Agent
        # result-read guards, so they are deliberately absent from this listing.
        rows = list(
            self.db.execute(
                text(
                    """SELECT t.id,t.project_id,t.kind,t.state,t.revision,t.created_at
            FROM cq_app.agent_tasks t JOIN cq_app.projects p
              ON (p.id,p.owner_id)=(t.project_id,t.owner_id) WHERE """
                    + " AND ".join(conditions)
                    + " ORDER BY t.created_at DESC,t.id DESC LIMIT :limit"
                ),
                values,
            ).mappings()
        )
        next_cursor = None
        if len(rows) > limit:
            end = rows[limit - 1]
            next_cursor = self.pager.encode(
                {
                    "identity": identity,
                    "created": end["created_at"].isoformat(),
                    "id": str(end["id"]),
                    "expires": datetime.now(UTC).timestamp() + 900,
                }
            )
        return {
            "items": [
                {
                    "task_id": str(row["id"]),
                    "project_id": str(row["project_id"]),
                    "kind": row["kind"],
                    "state": row["state"],
                    "revision": row["revision"],
                    "created_at": row["created_at"].isoformat(),
                    "detail_path": f"/api/agent/tasks/{row['id']}",
                    "artifact_content": "requires_current_result_authorization",
                }
                for row in rows[:limit]
            ],
            "next_cursor": next_cursor,
        }
