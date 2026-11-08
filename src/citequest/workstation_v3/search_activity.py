"""Owner search history outlives result cursors without pinning old indexes."""
from __future__ import annotations

from uuid import UUID

from pydantic import Field
from sqlalchemy import ForeignKey, String, UniqueConstraint, Uuid, func, select
from sqlalchemy.orm import Mapped, mapped_column

from citequest.search_v3.models import SearchSnapshot
from citequest.search_v3.schemas import SearchRequest, Strict
from citequest.v2.domain import DomainError
from citequest.v2.models import SCHEMA, Base, History, J
from citequest.v2.services import Catalog, digest


class SearchActivityInput(Strict):
    operation_id: UUID = Field(strict=False)
    snapshot_id: UUID = Field(strict=False)


class SearchActivity(Base):
    __tablename__ = "search_activities"
    __table_args__ = (UniqueConstraint("owner_id", "search_snapshot_id"),)
    history_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.history.id", ondelete="CASCADE"), primary_key=True)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"))
    # Deliberately no FK to the ephemeral search snapshot or index generation.
    # Clearing expired results must not erase user history or pin a retired index.
    search_snapshot_id: Mapped[UUID] = mapped_column(Uuid)
    generation_id: Mapped[UUID] = mapped_column(Uuid)
    epoch: Mapped[int]
    request: Mapped[dict] = mapped_column(J)
    request_sha256: Mapped[str] = mapped_column(String(64))


def record_search(catalog: Catalog, value: SearchActivityInput) -> dict:
    catalog.lock_owner()

    def effect() -> dict:
        snapshot = catalog.db.scalar(select(SearchSnapshot).where(
            SearchSnapshot.id == value.snapshot_id, SearchSnapshot.owner_id == catalog.actor.owner_id,
            SearchSnapshot.expires_at > func.clock_timestamp()))
        if snapshot is None:
            raise DomainError("owned_live_search_snapshot_required", 404)
        if digest(snapshot.request) != snapshot.query_sha256:
            raise DomainError("search_history_snapshot_hash_mismatch", 409)
        request = SearchRequest.model_validate(snapshot.request, strict=True)
        old = catalog.db.scalar(select(SearchActivity).where(
            SearchActivity.owner_id == catalog.actor.owner_id, SearchActivity.search_snapshot_id == snapshot.id))
        if old is not None:
            return {"history_id": str(old.history_id), "kind": "search"}
        identifier = catalog.append_history("search", "search:" + str(snapshot.id), snapshot_id=snapshot.id)
        catalog.db.flush()
        body = request.model_dump(mode="json")
        catalog.db.add(SearchActivity(history_id=identifier, owner_id=catalog.actor.owner_id,
            search_snapshot_id=snapshot.id, generation_id=snapshot.generation_id, epoch=snapshot.epoch,
            request=body, request_sha256=digest(body)))
        catalog.db.flush()
        return {"history_id": str(identifier), "kind": "search"}

    receipt = catalog.operation(value.operation_id, {"operation": "record_search_history", **value.model_dump(mode="json")}, effect)
    # An old network replay after the user cleared history must not recreate it
    # or return an earlier query from an immutable operation receipt.
    row = catalog.db.scalar(select(History).where(History.id == UUID(receipt["history_id"]),
                                                  History.owner_id == catalog.actor.owner_id))
    return {**receipt, "recorded": row is not None, "availability": "available" if row is not None else "history_cleared"}


def project_searches(catalog: Catalog, rows: list[dict]) -> list[dict]:
    ids = [UUID(row["history_id"]) for row in rows if row["kind"] == "search"]
    if not ids:
        return rows
    records = {str(row.history_id): row for row in catalog.db.scalars(select(SearchActivity).where(
        SearchActivity.owner_id == catalog.actor.owner_id, SearchActivity.history_id.in_(ids)))}
    result = []
    for value in rows:
        record = records.get(value["history_id"])
        if record is None:
            result.append(value)
            continue
        if digest(record.request) != record.request_sha256:
            raise DomainError("search_history_record_hash_mismatch", 409)
        request = SearchRequest.model_validate(record.request, strict=True)
        result.append({**value, "search": {
            "query": request.q, "catalog": request.catalog, "mode": request.mode, "field": request.field,
            "sort": request.sort, "filters": request.filters.model_dump(),
            "original_generation_id": str(record.generation_id), "original_epoch": record.epoch,
            "requires_source_reselection": bool(request.selected_papers),
            "repeat_semantics": "new_search_on_current_catalog_not_historical_result_replay"}})
    return result
