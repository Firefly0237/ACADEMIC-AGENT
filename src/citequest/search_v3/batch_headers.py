"""Bounded current metadata headers with request-local, two-snapshot guards.

Only ordinary metadata projections are batched. Native representations retain
the existing scalar reader and are explicitly labelled. No saved/captured text,
index results, cross-request authorization cache or alternate source is used.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from types import SimpleNamespace
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator
from sqlalchemy import exists, or_, select, text
from sqlalchemy.exc import DBAPIError

from citequest.data_v2 import models as data
from citequest.data_v2.schemas import IndexProjection
from citequest.index_v2.source_variants import metadata_unit
from citequest.v2 import models as core
from citequest.v2.domain import Actor, DomainError
from citequest.v2.services import digest

from .schemas import PaperDetail, PaperLocator
from .service import SearchService


class CurrentPaperHeader(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    paper: PaperLocator
    detail: PaperDetail | None
    source_availability: Literal["available", "unavailable", "temporarily_unavailable"]
    read_mode: Literal["metadata_batch", "native_scalar"]
    code: str | None = None

    @model_validator(mode="after")
    def current_only(self) -> CurrentPaperHeader:
        if self.source_availability == "available":
            if self.detail is None or self.detail.paper != self.paper or self.code is not None:
                raise ValueError("available header must contain matching current detail")
        elif self.detail is not None or not self.code:
            raise ValueError("unavailable headers withhold all metadata and name the failure")
        return self


def locator_key(paper: PaperLocator) -> str:
    return paper.kind + ":" + paper.id


class BatchPaperHeaders:
    """Read-only Python service API; caller still owns user identity and limits.

    Each metadata pass is a new PostgreSQL REPEATABLE READ READ ONLY snapshot.
    The second pass revalidates profile/seal, projection hashes, latest resolution,
    current Work ACL/metadata and current denials. Changed candidates are withheld
    rather than silently substituted. The final snapshot defines observed current
    authorization; no network response can promise atomicity with later writers.
    """

    MAX_PAPERS = 200

    def __init__(self, service: SearchService, *, between_snapshots: Callable[[], None] | None = None) -> None:
        self.service = service
        # Server-owned fault-test seam; never an HTTP/model-controlled callback.
        self.between_snapshots = between_snapshots

    @staticmethod
    def failed(paper: PaperLocator, error: DomainError, *, mode: str = "metadata_batch") -> CurrentPaperHeader:
        return CurrentPaperHeader(
            paper=paper,
            detail=None,
            source_availability="unavailable"
            if error.status in {401, 403, 404, 410, 451}
            else "temporarily_unavailable",
            code=error.code,
            read_mode=mode,
        )

    def metadata_pass(
        self, actor: Actor, papers: list[PaperLocator]
    ) -> tuple[dict[str, CurrentPaperHeader], dict[str, str]]:
        headers: dict[str, CurrentPaperHeader] = {}
        fingerprints: dict[str, str] = {}
        if not papers:
            return headers, fingerprints
        with self.service.factory.begin() as db:
            # All data/ACL within a pass sees one coherent committed snapshot;
            # this is request-local and never reused for a later request.
            db.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY"))
            rows = db.execute(
                select(data.SourceProjection, data.SourceRow, data.SourceSnapshot)
                .join(data.SourceRow, data.SourceRow.id == data.SourceProjection.source_row_id)
                .join(data.SourceSnapshot, data.SourceSnapshot.id == data.SourceRow.snapshot_id)
                .where(data.SourceProjection.id.in_([UUID(paper.id) for paper in papers]))
            ).all()
            by_id = {projection.id: (projection, row, snapshot) for projection, row, snapshot in rows}
            resolutions = {
                resolution.source_row_id: resolution
                for resolution in db.scalars(
                    select(data.SourceResolution)
                    .where(data.SourceResolution.source_row_id.in_({row.id for _, row, _ in rows}))
                    .distinct(data.SourceResolution.source_row_id)
                    .order_by(data.SourceResolution.source_row_id, data.SourceResolution.revision.desc())
                )
            }
            work_ids = {resolution.work_id for resolution in resolutions.values() if resolution.work_id is not None}
            # Same authoritative ACL predicate as Catalog.work; a source identity
            # or a saved bookmark is never permission to read a private Work.
            access = exists(
                select(core.WorkAccess.work_id).where(
                    core.WorkAccess.work_id == core.Work.id,
                    core.WorkAccess.owner_id == actor.owner_id,
                    core.WorkAccess.revoked.is_(False),
                )
            )
            allowed_works = {
                work.id: work
                for work in db.scalars(
                    select(core.Work).where(core.Work.id.in_(work_ids), or_(core.Work.public.is_(True), access))
                )
            }
            readers = self.service._repo(db, actor).readers
            roots, root_errors = {}, {}
            groups: dict[tuple[str, str], list[tuple[str, str | None]]] = defaultdict(list)
            candidates = []
            for paper in papers:
                key = locator_key(paper)
                try:
                    if UUID(paper.id) not in by_id:
                        raise DomainError("not_found", 404)
                    projection, row, snapshot = by_id[UUID(paper.id)]
                    if snapshot.id not in roots and snapshot.id not in root_errors:
                        try:
                            if snapshot.status != "staged" or not snapshot.profile_adopted or not snapshot.seal_sha256:
                                raise DomainError("source_not_formally_adopted", 409)
                            roots[snapshot.id] = readers.root(
                                SimpleNamespace(
                                    snapshot_id=snapshot.id,
                                    source_root_kind="metadata_snapshot",
                                    catalog_name="standard_metadata",
                                    security_domain=snapshot.security_domain,
                                    source_namespace=snapshot.source_namespace,
                                    representation_profile=snapshot.representation_profile,
                                    source_manifest_sha256=snapshot.source_manifest_sha256,
                                )
                            )
                        except DomainError as error:
                            root_errors[snapshot.id] = error
                    if snapshot.id in root_errors:
                        raise root_errors[snapshot.id]
                    root = roots[snapshot.id]
                    if root.state != "staged" or not root.profile_adopted or not root.source_seal_sha256:
                        raise DomainError("source_not_formally_adopted", 409)
                    resolution = resolutions.get(row.id)
                    work_id = resolution.work_id if resolution is not None else None
                    if work_id is not None and work_id not in allowed_works:
                        raise DomainError("not_found", 404)
                    # Reuse the full projection hash/text presence validator and
                    # canonical metadata builder, rather than captured headers.
                    source = IndexProjection(
                        projection_id=projection.id,
                        source_row_id=row.id,
                        snapshot_id=snapshot.id,
                        native_id=row.native_id,
                        source_namespace=snapshot.source_namespace,
                        source_revision=snapshot.source_revision,
                        security_domain=snapshot.security_domain,
                        representation_profile=snapshot.representation_profile,
                        source_pointer_sha256=row.source_pointer_sha256,
                        projection_sha256=projection.projection_sha256,
                        title=projection.title,
                        abstract=projection.abstract,
                        field_provenance=projection.field_provenance,
                        year=None,
                        work_id=work_id,
                        searchable_text_present=projection.searchable_text_present,
                        profile_adopted=snapshot.profile_adopted,
                        snapshot_state=snapshot.status,
                    )
                    unit = metadata_unit(source, root)
                    candidates.append((paper, unit, snapshot, resolution))
                    groups[(root.security_domain, root.source_namespace)].append(
                        (row.native_id, snapshot.source_revision)
                    )
                except DomainError as error:
                    headers[key] = self.failed(paper, error)
                except ValueError:
                    headers[key] = self.failed(paper, DomainError("current_metadata_projection_contract_invalid", 503))
            denied = set()
            for (domain, namespace), items in groups.items():
                for receipt in readers.updates.current_denials(namespace, items, domain):
                    denied.add((domain, namespace, receipt["native_id"], receipt["source_revision"]))
            safe = []
            for paper, unit, snapshot, resolution in candidates:
                lineage = (unit.root.security_domain, unit.root.source_namespace, unit.native_id)
                if (*lineage, "*") in denied or (*lineage, unit.source_revision) in denied:
                    headers[locator_key(paper)] = self.failed(paper, DomainError("source_currently_denied", 403))
                else:
                    safe.append((paper, unit, snapshot, resolution))
            summaries = (
                self.service._summaries(db, [self.service._summary_unit(unit, paper) for paper, unit, _, _ in safe])
                if safe
                else {}
            )
            for paper, unit, snapshot, resolution in safe:
                summary = summaries[unit.source_unit_id]
                detail = PaperDetail(
                    **summary.model_dump(),
                    source_snapshot_id=snapshot.id,
                    literal_text_available=unit.searchable_text_present,
                    coordinate_system=unit.coordinate_system,
                    links_status="source_provided" if summary.source_url else "not_in_source",
                    citation_status="complete"
                    if summary.title and summary.authors and summary.year
                    else "incomplete_metadata",
                )
                key = locator_key(paper)
                headers[key] = CurrentPaperHeader(
                    paper=paper, detail=detail, source_availability="available", read_mode="metadata_batch"
                )
                fingerprints[key] = digest(
                    {
                        "detail": detail.model_dump(mode="json"),
                        "root": unit.root.model_dump(mode="json"),
                        "source_row_id": str(unit.original_source_dto["source_row_id"]),
                        "resolution": None
                        if resolution is None
                        else [str(resolution.id), resolution.revision, resolution.proof_sha256],
                    }
                )
        return headers, fingerprints

    def read_many(self, actor: Actor, papers: list[PaperLocator]) -> dict[str, CurrentPaperHeader]:
        if not 1 <= len(papers) <= self.MAX_PAPERS or len({locator_key(paper) for paper in papers}) != len(papers):
            raise DomainError("bounded_unique_header_batch_required", 422)
        result: dict[str, CurrentPaperHeader] = {}
        metadata = [paper for paper in papers if paper.kind == "metadata_projection"]
        for paper in papers:
            if paper.kind != "native_representation":
                continue
            try:
                result[locator_key(paper)] = CurrentPaperHeader(
                    paper=paper,
                    detail=self.service.detail(actor, paper),
                    source_availability="available",
                    read_mode="native_scalar",
                )
            except DomainError as error:
                result[locator_key(paper)] = self.failed(paper, error, mode="native_scalar")
            except (DBAPIError, OSError, TimeoutError):
                result[locator_key(paper)] = self.failed(
                    paper, DomainError("current_header_dependency_unavailable", 503), mode="native_scalar"
                )
        if metadata:
            try:
                observed, before = self.metadata_pass(actor, metadata)
                if self.between_snapshots is not None:
                    self.between_snapshots()
                current, after = self.metadata_pass(actor, metadata)
                for paper in metadata:
                    key = locator_key(paper)
                    if current[key].detail is not None and (
                        observed[key].detail is None or before.get(key) != after.get(key)
                    ):
                        result[key] = self.failed(paper, DomainError("current_header_binding_changed", 409))
                    else:
                        result[key] = current[key]
            except (DBAPIError, OSError, TimeoutError):
                result.update(
                    {
                        locator_key(paper): self.failed(
                            paper, DomainError("current_header_dependency_unavailable", 503)
                        )
                        for paper in metadata
                    }
                )
        return result
