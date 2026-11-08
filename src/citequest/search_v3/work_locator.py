"""Resolve library Work IDs to admitted source locators, without inventing IDs.

The default contract requires actual native text. Metadata is selectable only
when the caller explicitly permits that coverage. This is a SQL/source check;
it neither searches a vector index nor grants a subsequent Agent permission.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import exists, or_, select, text

from citequest.index_v2 import models as index
from citequest.v2 import models as core
from citequest.v2.domain import Actor, DomainError

from .schemas import PaperLocator
from .visibility import visible_unit

if TYPE_CHECKING:
    from .service import SearchService


class WorkSelectionResolution(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    work_id: UUID
    status: Literal["resolved", "unavailable"]
    reason: str
    paper: PaperLocator | None = None
    coverage: Literal["native_literal", "metadata"] | None = None
    catalog: Literal["application_papers", "standard_metadata"] | None = None
    generation_id: UUID | None = None
    epoch: int | None = None
    source_snapshot_id: UUID | None = None
    work_version_id: UUID | None = None


def resolve_work_selections(
    search: SearchService,
    actor: Actor,
    work_ids: list[UUID],
    *,
    require_full_text: bool = True,
) -> list[WorkSelectionResolution]:
    """Preserve request order; unavailable items remain visible as explicit gaps."""
    if (
        not isinstance(work_ids, list)
        or not 1 <= len(work_ids) <= 10
        or any(type(value) is not UUID for value in work_ids)
    ):
        raise DomainError("bounded_typed_work_selection_required", 400)
    if len(set(work_ids)) != len(work_ids) or type(require_full_text) is not bool:
        raise DomainError("duplicate_work_or_invalid_coverage_policy", 400)
    results: list[WorkSelectionResolution] = []
    with search.factory.begin() as db:
        access = exists(
            select(core.WorkAccess.work_id).where(
                core.WorkAccess.work_id == core.Work.id,
                core.WorkAccess.owner_id == actor.owner_id,
                core.WorkAccess.revoked.is_(False),
            )
        )
        visible = set(
            db.scalars(
                select(core.Work.id).where(
                    core.Work.id.in_(work_ids),
                    core.Work.canonical_redirect_work_id.is_(None),
                    or_(core.Work.public.is_(True), access),
                )
            )
        )
        catalogs = ("application_papers",) if require_full_text else ("application_papers", "standard_metadata")
        active = db.execute(
            select(index.IndexGeneration, index.IndexCatalog.epoch)
            .join(
                index.IndexCatalog,
                index.IndexCatalog.active_generation_id == index.IndexGeneration.id,
            )
            .where(
                index.IndexCatalog.name.in_(catalogs),
                index.IndexCatalog.security_domain == "public",
                index.IndexGeneration.state == "active",
                index.IndexGeneration.security_domain == "public",
                index.IndexGeneration.build_scope == "full",
            )
        ).all()
        ready = {}
        repo = search._repo(db, actor)
        for generation, epoch in active:
            # Use the same immutable source/readback gate plus the DB's backend-adoption guard.
            try:
                repo._read_ready(generation)
            except DomainError:
                continue
            adopted = db.execute(
                text("SELECT cq_app.index_backend_publication_allowed(:generation)"), {"generation": generation.id}
            ).scalar_one()
            if adopted is True:
                ready[generation.catalog_name] = generation, epoch
        candidates: dict[UUID, dict[str, list[index.IndexUnit]]] = {value: {} for value in visible}
        for catalog, (generation, _epoch) in ready.items():
            rows = db.scalars(
                select(index.IndexUnit).where(
                    index.IndexUnit.generation_id == generation.id,
                    index.IndexUnit.work_id.in_(visible),
                    index.IndexUnit.searchable.is_(True),
                    index.IndexUnit.chunk_count > 0,
                    visible_unit(actor, generation.security_domain),
                )
            ).all()
            for unit in rows:
                candidates[unit.work_id].setdefault(catalog, []).append(unit)
        for work_id in work_ids:

            def unavailable(reason: str, identifier: UUID = work_id) -> WorkSelectionResolution:
                return WorkSelectionResolution(work_id=identifier, status="unavailable", reason=reason)

            if work_id not in visible:
                results.append(unavailable("not_found_or_not_visible"))
                continue
            selected = next((catalog for catalog in catalogs if candidates[work_id].get(catalog)), None)
            if selected is None:
                reason = (
                    "fulltext_catalog_unavailable"
                    if require_full_text and "application_papers" not in ready
                    else (
                        "no_published_catalog"
                        if not ready
                        else "fulltext_unavailable"
                        if require_full_text
                        else "no_admitted_source_mapping"
                    )
                )
                results.append(unavailable(reason))
                continue
            units = candidates[work_id][selected]
            if len(units) != 1:
                results.append(unavailable("ambiguous_source_mapping_select_explicit_version"))
                continue
            unit = units[0]
            generation, epoch = ready[selected]
            if unit.source_reference_kind == "native_representation" and unit.native_representation_id:
                locator = PaperLocator(kind="native_representation", id=unit.native_representation_id)
                coverage = "native_literal"
            elif (
                not require_full_text
                and unit.source_reference_kind == "metadata_projection"
                and unit.metadata_projection_id
            ):
                locator = PaperLocator(kind="metadata_projection", id=str(unit.metadata_projection_id))
                coverage = "metadata"
            else:
                results.append(unavailable("source_coverage_not_allowed"))
                continue
            try:
                current, snapshot = search._current_source(db, actor, locator)
            except DomainError as exc:
                results.append(unavailable("source_not_found_or_denied" if exc.status in {403, 404} else exc.code))
                continue
            if (
                snapshot.id != generation.snapshot_id
                or current.work_id != work_id
                or current.work_version_id != unit.work_version_id
                or current.projection_sha256 != unit.projection_sha256
                or current.source_pointer_sha256 != unit.source_pointer_sha256
                or not current.literal_text
            ):
                results.append(unavailable("current_source_version_or_content_mismatch"))
                continue
            # A changed catalog or ACL cannot silently retarget this response to another source.
            still_active = db.execute(
                select(index.IndexCatalog.active_generation_id, index.IndexCatalog.epoch).where(
                    index.IndexCatalog.name == selected,
                    index.IndexCatalog.security_domain == "public",
                )
            ).one_or_none()
            still_visible = db.scalar(
                select(index.IndexUnit.projection_id).where(
                    index.IndexUnit.generation_id == generation.id,
                    index.IndexUnit.projection_id == unit.projection_id,
                    visible_unit(actor, generation.security_domain),
                )
            )
            if still_active is None or tuple(still_active) != (generation.id, epoch) or still_visible is None:
                results.append(unavailable("source_or_catalog_changed_retry_resolution"))
                continue
            results.append(
                WorkSelectionResolution(
                    work_id=work_id,
                    status="resolved",
                    paper=locator,
                    coverage=coverage,
                    catalog=selected,
                    generation_id=generation.id,
                    epoch=epoch,
                    source_snapshot_id=snapshot.id,
                    work_version_id=unit.work_version_id,
                    reason="native_fulltext_resolved"
                    if coverage == "native_literal"
                    else "metadata_only_explicitly_allowed",
                )
            )
    return results
