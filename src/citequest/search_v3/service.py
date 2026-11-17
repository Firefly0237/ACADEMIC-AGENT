"""SQL-authoritative, current-ACL search and literal evidence read service."""
from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import UUID, uuid4

from sqlalchemy import Text, cast, func, or_, select, text
from sqlalchemy.orm import Session
from citequest.data_v2 import models as data
from citequest.data_v2.application_members import ApplicationMember
from citequest.data_v2.application_freeze import source_record_sha
from citequest.index_v2 import models as index
from citequest.index_v2.repository_v2 import VariantIndexRepository
from citequest.index_v2.rows_v2 import canonical_sha
from citequest.index_v2.source_readers import SourceReaders
from citequest.index_v2.source_variants import CanonicalSourceUnit, DeniedSourceUnit, MetadataReference, NativeReference, UpdateReference
from citequest.v2 import models as core
from citequest.v2.domain import Actor, DomainError
from citequest.v2.services import Catalog, normalize_identifier
from citequest.v2.storage import StoredBlob
from . import models as m
from .cursor import CursorSigner
from .catalog_snapshot import CatalogSnapshot, native_runtime_identity
from .native import NativeHit
from .schemas import EvidenceDTO, NativePDFAssetProfile, PaperDetail, PaperLocator, PaperSummary, SearchHit, SearchPage, SearchRequest, SimilarRequest, SourceCitation
from .visibility import visible_unit
from .query_projection import AUDIT_KEY, MaterialQueryProjection, snapshot_search_request
from .citation import project_csl


@dataclass(frozen=True)
class PaperPDFAsset:
    path: Path
    filename: str
    sha256: str
    size_bytes: int
    source_url: str | None
    license: str


class SearchService:
    def __init__(self, factory, native, store, *, policy_sha256: str, evaluation_sha256: str,
                 cursor_key: bytes, query_token_counter=None, query_tokenizer_profile_sha256: str | None = None,
                 native_pdf_asset_profile: NativePDFAssetProfile | None = None):
        self.factory, self.native, self.store = factory, native, store
        self.policy_sha256, self.evaluation_sha256 = policy_sha256, evaluation_sha256
        self.signer, self.query_token_counter = CursorSigner(cursor_key), query_token_counter
        self.query_tokenizer_profile_sha256 = query_tokenizer_profile_sha256
        self.native_pdf_asset_profile = native_pdf_asset_profile

    def _repo(self, db, actor):
        return VariantIndexRepository(Catalog(db, actor), store=self.store,
                                      policy_sha256=self.policy_sha256, evaluation_sha256=self.evaluation_sha256)

    @staticmethod
    def _paper_condition(paper):
        return (index.IndexUnit.native_representation_id == paper.id if paper.kind == "native_representation" else
            or_(index.IndexUnit.metadata_projection_id == UUID(paper.id), index.IndexUnit.update_observation_id == UUID(paper.id)))

    @staticmethod
    def _row_keys(db, generation, identifiers):
        if generation.source_root_kind != "publication_snapshot":
            return [canonical_sha({"snapshot_id": str(generation.snapshot_id), "projection_id": str(key)}) for key in identifiers]
        rows = db.execute(select(index.IndexUnit.projection_id, index.IndexUnit.compilation["source_root_id"].as_string())
            .where(index.IndexUnit.generation_id == generation.id, index.IndexUnit.projection_id.in_(identifiers))).all()
        if len(rows) != len(identifiers) or any(root is None for _, root in rows):
            raise DomainError("published_metadata_origin_row_key_missing", 409)
        return [canonical_sha({"snapshot_id": root, "projection_id": str(key)}) for key, root in rows]

    def _validate_query(self, request):
        if self.query_token_counter is None:
            raise DomainError("pinned_query_tokenizer_unavailable", 503)
        if not 1 <= self.query_token_counter(request.q) <= 2048:
            raise DomainError("query_exceeds_pinned_embedding_window", 422)

    def _runtime_binding(self, generation):
        proof = generation.server_proof or {}
        if proof.get("backend_profile_sha256") != getattr(self.native, "backend_profile_sha256", None):
            raise DomainError("native_runtime_backend_binding_mismatch", 503)
        endpoints = [value.get("params", {}).get("endpoint") for value in proof.get("schema_signature", {}).get("functions", [])
                     if value.get("name") == "text_dense"]
        if endpoints != [getattr(self.native, "function_endpoint", None)]:
            raise DomainError("native_runtime_function_endpoint_binding_mismatch", 503)

    def preflight(self, actor: Actor, request_scope_catalog: str, selected_papers: list[PaperLocator]) -> dict:
        """SQL-only admission before Agent model use; no native/embedding call."""
        if request_scope_catalog not in {"application_papers", "standard_metadata"} or type(selected_papers) is not list or len(selected_papers) > 20:
            raise DomainError("invalid_search_preflight_scope", 422)
        if len({(paper.kind, paper.id) for paper in selected_papers}) != len(selected_papers):
            raise DomainError("selected_papers_must_be_unique", 422)
        with self.factory.begin() as db:
            repo = self._repo(db, actor)
            try:
                captured = repo.capture_reader(catalog_name=request_scope_catalog)
            except DomainError as error:
                if error.code in {"search_index_not_published", "not_found", "adopted_sealed_source_and_complete_native_index_required"}:
                    raise DomainError("search_catalog_requires_published_full_generation", 503) from None
                raise
            generation = db.get(index.IndexGeneration, UUID(captured["generation_id"]))
            self._runtime_binding(generation)
            if generation.tokenizer_profile_sha256 != self.query_tokenizer_profile_sha256:
                raise DomainError("query_and_generation_tokenizer_profile_mismatch", 503)
            if selected_papers:
                conditions = [self._paper_condition(paper) for paper in selected_papers]
                count = db.scalar(select(func.count()).select_from(index.IndexUnit).where(
                    index.IndexUnit.generation_id == generation.id, or_(*conditions), visible_unit(actor, generation.security_domain)))
                if count != len(selected_papers):
                    raise DomainError("selected_paper_outside_current_catalog_or_access", 404)
            repo.close_reader(UUID(captured["lease_id"]))
            return {"catalog": request_scope_catalog, "generation_id": captured["generation_id"], "epoch": captured["epoch"],
                    "selected_papers": len(selected_papers), "embedding_calls": 0, "model_calls": 0}

    def search(self, actor: Actor, request: SearchRequest) -> SearchPage:
        return self._search(actor, request)

    def capture_catalog_snapshot(self, actor: Actor, catalog: str, selected: list[PaperLocator]) -> CatalogSnapshot:
        """SQL-only capture for durable server-owned Task admission, no RPC.

        The caller must persist the returned binding before any model/tool
        dispatch, under its Task lock and a generation FK. Never accept it from
        user/model arguments. A capture alone is neither a durable pin nor ACL.
        """
        return self.capture_catalog_snapshots(actor, {catalog: selected})[catalog]

    def capture_catalog_snapshots(self, actor: Actor, scopes: dict[str, list[PaperLocator]]) -> dict[str, CatalogSnapshot]:
        """Capture all initially required catalogs in one SQL statement snapshot.

        Later introduction of another catalog is an explicitly later admission;
        it cannot turn an existing binding into a new generation.
        """
        if type(scopes) is not dict or not scopes or not set(scopes) <= {"application_papers", "standard_metadata"}:
            raise DomainError("invalid_task_catalog_scope", 422)
        for selected in scopes.values():
            if type(selected) is not list or len(selected) > 20 or any(not isinstance(paper, PaperLocator) for paper in selected):
                raise DomainError("invalid_task_catalog_scope", 422)
            if len({(paper.kind, paper.id) for paper in selected}) != len(selected):
                raise DomainError("selected_papers_must_be_unique", 422)
        with self.factory.begin() as db:
            repo = self._repo(db, actor)
            pointers = db.execute(select(index.IndexGeneration, index.IndexCatalog.epoch).join(index.IndexCatalog,
                (index.IndexCatalog.active_generation_id == index.IndexGeneration.id) &
                (index.IndexCatalog.security_domain == index.IndexGeneration.security_domain) &
                (index.IndexCatalog.name == index.IndexGeneration.catalog_name)).where(
                    index.IndexCatalog.name.in_(sorted(scopes)), index.IndexCatalog.security_domain == "public")
                .execution_options(populate_existing=True, cq_atomic_capture=True)).all()
            if len(pointers) != len(scopes) or any(generation.state != "active" for generation, _ in pointers):
                raise DomainError("search_catalog_requires_published_full_generation", 503)
            snapshots = {}
            for generation, epoch in pointers:
                repo._validate_generation_value(generation)
                repo._read_ready(generation)
                self._runtime_binding(generation)
                if generation.tokenizer_profile_sha256 != self.query_tokenizer_profile_sha256:
                    raise DomainError("query_and_generation_tokenizer_profile_mismatch", 503)
                self._check_snapshot_selection(db, actor, generation, scopes[generation.catalog_name])
                self._check_published_epoch(db, generation, epoch)
                snapshots[generation.catalog_name] = self._snapshot_binding(generation, epoch)
            return snapshots

    def _snapshot_binding(self, generation: index.IndexGeneration, epoch: int) -> CatalogSnapshot:
        return CatalogSnapshot(catalog=generation.catalog_name, generation_id=generation.id, epoch=epoch,
            collection_name=generation.collection_name, source_snapshot_id=generation.snapshot_id,
            manifest_sha256=generation.expected_manifest_sha256, source_seal_sha256=generation.source_seal_sha256,
            server_proof_sha256=canonical_sha(generation.server_proof), index_profile_sha256=generation.index_profile_sha256,
            tokenizer_profile_sha256=generation.tokenizer_profile_sha256, policy_sha256=generation.policy_sha256,
            evaluation_sha256=generation.evaluation_sha256, native_runtime_sha256=canonical_sha(native_runtime_identity(self.native)))

    @staticmethod
    def _check_snapshot_selection(db: Session, actor: Actor, generation: index.IndexGeneration, selected: list[PaperLocator]) -> None:
        if not selected:
            return
        expected_kind = "native_representation" if generation.catalog_name == "application_papers" else "metadata_projection"
        if any(paper.kind != expected_kind for paper in selected):
            raise DomainError("selected_paper_outside_pinned_catalog", 404)
        conditions = [SearchService._paper_condition(paper) for paper in selected]
        count = db.scalar(select(func.count()).select_from(index.IndexUnit).where(
            index.IndexUnit.generation_id == generation.id, or_(*conditions), visible_unit(actor, generation.security_domain)))
        if count != len(selected):
            raise DomainError("selected_paper_outside_pinned_catalog_or_access", 404)

    @staticmethod
    def _check_published_epoch(db: Session, generation: index.IndexGeneration, epoch: int) -> None:
        committed = db.scalar(select(index.IndexPublishIntent.id).where(
            index.IndexPublishIntent.target_generation_id == generation.id,
            index.IndexPublishIntent.catalog_name == generation.catalog_name,
            index.IndexPublishIntent.security_domain == generation.security_domain,
            index.IndexPublishIntent.state == "committed", index.IndexPublishIntent.expected_epoch + 1 == epoch).limit(1))
        if committed is None:
            raise DomainError("task_catalog_epoch_not_published", 409)

    def _check_catalog_snapshot(self, db: Session, actor: Actor, snapshot: CatalogSnapshot,
                                selected: list[PaperLocator]) -> tuple[VariantIndexRepository, index.IndexGeneration]:
        if not isinstance(snapshot, CatalogSnapshot):
            raise DomainError("server_owned_catalog_snapshot_required", 422)
        repo = self._repo(db, actor)
        generation = repo.generation(snapshot.generation_id)
        repo._read_ready(generation)
        self._check_published_epoch(db, generation, snapshot.epoch)
        self._runtime_binding(generation)
        if self._snapshot_binding(generation, snapshot.epoch) != snapshot:
            raise DomainError("task_catalog_snapshot_binding_changed", 409)
        if generation.tokenizer_profile_sha256 != self.query_tokenizer_profile_sha256:
            raise DomainError("query_and_generation_tokenizer_profile_mismatch", 503)
        self._check_snapshot_selection(db, actor, generation, selected)
        return repo, generation

    def recheck_catalog_snapshot(self, actor: Actor, snapshot: CatalogSnapshot, selected: list[PaperLocator]) -> None:
        """SQL-only current source/runtime/selection barrier, never recaptures."""
        with self.factory.begin() as db:
            self._check_catalog_snapshot(db, actor, snapshot, selected)

    def search_pinned(self, actor: Actor, request: SearchRequest, snapshot: CatalogSnapshot) -> SearchPage:
        """Internal trusted Task path. Runtime mismatch fails, no active fallback."""
        if not isinstance(snapshot, CatalogSnapshot) or snapshot.catalog != request.catalog:
            raise DomainError("task_search_catalog_binding_required", 409)
        return self._search(actor, request, catalog_snapshot=snapshot)

    def _search(self, actor: Actor, request: SearchRequest, *, catalog_snapshot: CatalogSnapshot | None = None) -> SearchPage:
        if request.field == "text" and re.match(r"(?i)^(?:https?://(?:dx\.)?doi\.org/|doi:\s*|10\.\d{4,9}/)", request.q.strip()):
            normalize_identifier("doi", request.q)
            request = request.model_copy(update={"field": "doi"})
        self._validate_query(request)
        lease_id = None
        try:
            with self.factory.begin() as db:
                # PostgreSQL17 JIT compilation dominates this bounded OLTP
                # visibility plan; keep every current ACL/denial predicate.
                db.execute(text("SET LOCAL jit = off"))
                repo = self._repo(db, actor)
                try:
                    if catalog_snapshot is None:
                        pointer = repo.capture_reader(catalog_name=request.catalog)
                    else:
                        repo, pinned_generation = self._check_catalog_snapshot(db, actor, catalog_snapshot, request.selected_papers)
                        lease = index.IndexReaderLease(generation_id=pinned_generation.id, owner_id=actor.owner_id,
                            epoch=catalog_snapshot.epoch, expires_at=datetime.now(UTC) + timedelta(seconds=120))
                        db.add(lease)
                        db.flush()
                        pointer = {"lease_id": str(lease.id), "generation_id": str(pinned_generation.id), "epoch": catalog_snapshot.epoch}
                except DomainError as error:
                    if error.code in {"search_index_not_published", "index_unavailable", "not_found", "adopted_sealed_source_and_complete_native_index_required", "full_standard_64183_required", "formal_application_500_to_2000_required"}:
                        raise DomainError("search_catalog_requires_published_full_generation", 503) from None
                    raise
                lease_id = UUID(pointer["lease_id"])
                generation = db.get(index.IndexGeneration, UUID(pointer["generation_id"]))
                self._runtime_binding(generation)
                if generation.tokenizer_profile_sha256 != self.query_tokenizer_profile_sha256:
                    raise DomainError("query_and_generation_tokenizer_profile_mismatch", 503)
                denied = list(db.scalars(select(index.IndexUnit.projection_id).where(
                    index.IndexUnit.generation_id == generation.id, ~visible_unit(actor, generation.security_domain)).limit(5001)))
                if len(denied) > 5000:
                    raise DomainError("visibility_prefilter_exceeds_bounded_native_plan", 503)
                excluded = self._row_keys(db, generation, denied)
                included = None
                if request.selected_papers:
                    identities = {(paper.kind, paper.id) for paper in request.selected_papers}
                    if len(identities) != len(request.selected_papers):
                        raise DomainError("selected_papers_must_be_unique", 422)
                    conditions = [self._paper_condition(paper) for paper in request.selected_papers]
                    selected = list(db.scalars(select(index.IndexUnit.projection_id).where(
                        index.IndexUnit.generation_id == generation.id, or_(*conditions), visible_unit(actor, generation.security_domain))))
                    if len(selected) != len(identities):
                        raise DomainError("selected_paper_outside_current_catalog_or_access", 404)
                    included = self._row_keys(db, generation, selected)
                native_args = dict(collection=generation.collection_name, request=request,
                                   security_domain=generation.security_domain, excluded_row_keys=excluded,
                                   expected_proof=generation.server_proof, included_row_keys=included)
                structured = self._structured_hits(db, actor, generation, request, included) if request.field != "text" else None
                generation_id, epoch = generation.id, pointer["epoch"]
            hits = self.native.search(**native_args) if structured is None else structured
            with self.factory.begin() as db:
                repo = self._repo(db, actor)
                if catalog_snapshot is not None:
                    # Revalidate the same identity after native I/O; publishing
                    # a new active pointer cannot change this binding.
                    self._check_catalog_snapshot(db, actor, catalog_snapshot, request.selected_papers)
                allowed = set(repo.recheck_hits(lease_id, [hit.chunk_id for hit in hits])["allowed_chunk_ids"])
                rows = db.execute(select(index.IndexChunk, index.IndexUnit).join(index.IndexUnit,
                    (index.IndexUnit.generation_id == index.IndexChunk.generation_id) &
                    (index.IndexUnit.projection_id == index.IndexChunk.projection_id)).where(
                    index.IndexChunk.generation_id == generation_id,
                    index.IndexChunk.chunk_id.in_(allowed), visible_unit(actor, native_args["security_domain"]))).all()
                by_chunk = {chunk.chunk_id: (chunk, unit) for chunk, unit in rows}
                ranked, units_seen = [], set()
                for hit in hits:
                    if hit.chunk_id not in by_chunk:
                        continue
                    chunk, unit = by_chunk[hit.chunk_id]
                    if hit.row_key != chunk.payload["row_key"]:
                        raise DomainError("native_hit_SQL_identity_mismatch", 503)
                    if unit.projection_id in units_seen:
                        raise DomainError("native_paper_grouping_failed", 503)
                    units_seen.add(unit.projection_id)
                    ranked.append((hit, chunk, unit))
                summaries = self._summaries(db, [unit for _, _, unit in ranked])
                if request.sort == "year_desc":
                    ranked.sort(key=lambda row: (-(summaries[row[2].projection_id].year or 0), row[2].projection_id.hex))
                elif request.sort == "title":
                    ranked.sort(key=lambda row: ((summaries[row[2].projection_id].title or "\U0010ffff").casefold(), row[2].projection_id.hex))
                snapshot = m.SearchSnapshot(id=uuid4(), owner_id=actor.owner_id, generation_id=generation_id,
                    epoch=epoch, request=request.model_dump(mode="json"), query_sha256=canonical_sha(request.model_dump(mode="json")),
                    page_size=request.page_size, candidate_limit=request.candidate_limit, candidate_count=len(ranked),
                    expires_at=datetime.now(UTC) + timedelta(seconds=request.ttl_seconds))
                db.add(snapshot)
                db.flush()
                db.add_all([m.SearchSnapshotHit(snapshot_id=snapshot.id, ordinal=rank, generation_id=generation_id,
                    projection_id=unit.projection_id, chunk_id=chunk.chunk_id, score=hit.score)
                    for rank, (hit, chunk, unit) in enumerate(ranked, 1)])
                repo.close_reader(lease_id)
                snapshot_id = snapshot.id
            lease_id = None
            return self.page(actor, snapshot_id)
        finally:
            if lease_id is not None:
                with self.factory.begin() as db:
                    repo = self._repo(db, actor)
                    # Close only our lease even if it expired; never retarget.
                    lease = db.get(index.IndexReaderLease, lease_id)
                    if lease is not None and lease.owner_id == actor.owner_id:
                        db.delete(lease)

    def _structured_hits(self, db, actor, generation, request, included):
        # Known canonical bibliographic fields only. Most clean benchmark rows
        # have no resolved Work, hence no invented author/DOI identity.
        query = select(index.IndexUnit).join(core.Work, core.Work.id == index.IndexUnit.work_id).where(
            index.IndexUnit.generation_id == generation.id, visible_unit(actor, generation.security_domain))
        if request.field == "doi":
            doi = normalize_identifier("doi", request.q)
            query = query.join(core.Identifier, core.Identifier.work_id == core.Work.id).where(
                core.Identifier.namespace == "doi", core.Identifier.normalized == doi,
                core.Identifier.security_domain.in_(["public", actor.security_domain]))
        else:
            pattern = "%" + request.q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
            query = query.where(cast(core.Work.authors, Text).ilike(pattern, escape="\\"))
        if request.filters.year_from is not None:
            query = query.where(index.IndexUnit.projection["year"].as_integer() >= request.filters.year_from)
        if request.filters.year_to is not None:
            query = query.where(index.IndexUnit.projection["year"].as_integer() <= request.filters.year_to)
        if request.selected_papers:
            query = query.where(or_(*[self._paper_condition(paper) for paper in request.selected_papers]))
        units = list(db.scalars(query.distinct().order_by(index.IndexUnit.projection_id).limit(request.candidate_limit)))
        # Pick the real first literal chunk for each selected paper. The field
        # match is metadata, so no fabricated full-text match score is assigned.
        first = db.execute(select(index.IndexChunk.projection_id, func.min(index.IndexChunk.chunk_id)).where(
            index.IndexChunk.generation_id == generation.id, index.IndexChunk.projection_id.in_([unit.projection_id for unit in units]))
            .group_by(index.IndexChunk.projection_id)).all()
        identifiers = dict(first)
        rows = {row.chunk_id: row for row in db.scalars(select(index.IndexChunk).where(index.IndexChunk.generation_id == generation.id,
            index.IndexChunk.chunk_id.in_(identifiers.values())))}
        return [NativeHit(rows[identifiers[unit.projection_id]].chunk_id, rows[identifiers[unit.projection_id]].payload["row_key"], 1.0)
                for unit in units if unit.projection_id in identifiers]

    def _snapshot(self, db, actor, snapshot_id):
        snapshot = db.get(m.SearchSnapshot, snapshot_id)
        if snapshot is None or snapshot.owner_id != actor.owner_id:
            raise DomainError("not_found", 404)
        if snapshot.expires_at <= datetime.now(UTC):
            raise DomainError("search_snapshot_expired", 410)
        repo = self._repo(db, actor)
        generation = repo.generation(snapshot.generation_id)
        repo._read_ready(generation)
        # Durable snapshot, not a stale 120-second lease, pins this exact old
        # generation. Publication may advance; membership/order cannot retarget.
        return snapshot, generation, repo

    def _visible_rows(self, db, actor, snapshot, generation):
        return db.execute(select(m.SearchSnapshotHit, index.IndexChunk, index.IndexUnit).join(index.IndexChunk,
            (index.IndexChunk.generation_id == m.SearchSnapshotHit.generation_id) &
            (index.IndexChunk.chunk_id == m.SearchSnapshotHit.chunk_id)).join(index.IndexUnit,
            (index.IndexUnit.generation_id == m.SearchSnapshotHit.generation_id) &
            (index.IndexUnit.projection_id == m.SearchSnapshotHit.projection_id)).where(
            m.SearchSnapshotHit.snapshot_id == snapshot.id, visible_unit(actor, generation.security_domain))
            .order_by(m.SearchSnapshotHit.ordinal)).all()

    def page(self, actor: Actor, snapshot_id: UUID, cursor: str | None = None) -> SearchPage:
        with self.factory.begin() as db:
            snapshot, generation, _ = self._snapshot(db, actor, snapshot_id)
            after = self.signer.decode(cursor, snapshot=snapshot, owner_id=actor.owner_id) if cursor else 0
            rows = self._visible_rows(db, actor, snapshot, generation)
            summaries = self._summaries(db, [unit for _, _, unit in rows])
            chosen = [(hit, chunk, unit) for hit, chunk, unit in rows if hit.ordinal > after][:snapshot.page_size]
            more = bool(chosen and any(hit.ordinal > chosen[-1][0].ordinal for hit, _, _ in rows))
            request = snapshot_search_request(snapshot.request)
            facets = {"year": dict(Counter(str(item.year) if item.year is not None else "unknown" for item in summaries.values())),
                      "coverage": dict(Counter(item.coverage for item in summaries.values()))}
            return SearchPage(snapshot_id=snapshot.id, generation_id=generation.id, epoch=snapshot.epoch,
                query=request.q, catalog=request.catalog, mode=request.mode, sort=request.sort,
                retrieval_method=("structured_" + request.field if request.field != "text" else {"hybrid": "native_rrf", "dense": "native_dense", "bm25": "native_bm25"}[request.mode]),
                candidate_limit=snapshot.candidate_limit, candidate_count=snapshot.candidate_count,
                facets=facets, masked_since_search=snapshot.candidate_count-len(rows),
                results=[SearchHit(rank=hit.ordinal, score=float(hit.score), paper=summaries[unit.projection_id],
                    snippet=chunk.payload["text"], evidence_refs=[f"{snapshot.id}:{chunk.chunk_id}"])
                    for hit, chunk, unit in chosen],
                next_cursor=self.signer.encode(snapshot=snapshot, after=chosen[-1][0].ordinal) if more else None,
                expires_at=snapshot.expires_at, generation_profile_sha256=generation.index_profile_sha256)

    def _summaries(self, db, units):
        works = {work.id: work for work in db.scalars(select(core.Work).where(core.Work.id.in_({unit.work_id for unit in units if unit.work_id})))}
        members = {member.representation_id: source for member, source in db.execute(select(ApplicationMember, core.SourceRecord)
            .join(core.SourceRecord, core.SourceRecord.id == ApplicationMember.source_record_id)
            .where(ApplicationMember.representation_id.in_({unit.native_representation_id for unit in units if unit.native_representation_id}))).all()}
        result = {}
        for unit in units:
            dto = unit.projection.get("original_source_dto", {})
            work = works.get(unit.work_id)
            native = unit.source_reference_kind == "native_representation"
            observed = unit.source_reference_kind == "update_observation"
            reference_id = unit.native_representation_id if native else str(unit.update_observation_id if observed else unit.metadata_projection_id)
            source = members.get(reference_id)
            result[unit.projection_id] = PaperSummary(paper=PaperLocator(kind="native_representation" if native else "metadata_projection", id=reference_id),
                title=work.title if native and work is not None else dto.get("title"), abstract=dto.get("abstract"),
                authors=work.authors if work is not None else None, year=unit.projection.get("year"),
                work_id=unit.work_id, work_version_id=unit.work_version_id,
                coverage="native_literal" if native else "metadata",
                source=SourceCitation(source_namespace=unit.source_namespace, native_id=unit.native_id,
                    source_revision=unit.source_revision, projection_sha256=unit.projection_sha256,
                    source_pointer_sha256=unit.source_pointer_sha256),
                source_url=source.source_url if source is not None else (dto.get("locator") or {}).get("source_url") if observed else None,
                metadata_revision=work.metadata_revision if work is not None else None)
        return result

    def read_evidence(self, actor: Actor, ref_id: str) -> EvidenceDTO:
        try:
            snapshot_text, chunk_id = ref_id.split(":")
            snapshot_id = UUID(snapshot_text)
            if re.fullmatch(r"[a-f0-9]{64}", chunk_id) is None:
                raise ValueError()
        except (ValueError, AttributeError):
            raise DomainError("invalid_search_evidence_reference", 400) from None
        with self.factory.begin() as db:
            snapshot, generation, repo = self._snapshot(db, actor, snapshot_id)
            row = next((row for row in self._visible_rows(db, actor, snapshot, generation) if row[0].chunk_id == chunk_id), None)
            if row is None:
                raise DomainError("not_found", 404)
            hit, chunk, unit = row
            return self._literal_evidence(db, actor, generation, chunk, unit,
                snapshot_id=snapshot.id, epoch=snapshot.epoch, ref_id=ref_id)

    def _literal_evidence(self, db: Session, actor: Actor, generation: index.IndexGeneration,
                          chunk: index.IndexChunk, unit: index.IndexUnit, *, snapshot_id: UUID,
                          epoch: int, ref_id: str) -> EvidenceDTO:
            """One shared original-source barrier for paper and chunk retrieval."""
            repo = self._repo(db, actor)
            reference = (NativeReference(native_representation_id=unit.native_representation_id)
                         if unit.source_reference_kind == "native_representation" else UpdateReference(update_observation_id=unit.update_observation_id)
                         if unit.source_reference_kind == "update_observation" else MetadataReference(metadata_projection_id=unit.metadata_projection_id))
            current = repo.readers.read(repo.readers.root(generation), reference)
            if isinstance(current, DeniedSourceUnit):
                raise DomainError("source_currently_denied", 403)
            payload = chunk.payload
            start, end = payload["span_start"], payload["span_end"]
            if (current.unit_text_sha256 != payload["unit_text_sha256"] or not 0 <= start < end <= len(current.literal_text)
                    or current.literal_text[start:end] != payload["text"]
                    or hashlib.sha256(payload["text"].encode()).hexdigest() != payload["source_text_sha256"]):
                raise DomainError("evidence_source_hash_or_span_mismatch", 409)
            # Citation identity is already pinned in this IndexUnit. Hydrating
            # paper display metadata here adds two unrelated queries per chunk.
            native = unit.source_reference_kind == "native_representation"
            source_ref = PaperLocator(kind="native_representation" if native else "metadata_projection",
                id=unit.native_representation_id if native else str(unit.update_observation_id if unit.source_reference_kind == "update_observation" else unit.metadata_projection_id))
            source = SourceCitation(source_namespace=unit.source_namespace, native_id=unit.native_id,
                source_revision=unit.source_revision, projection_sha256=unit.projection_sha256,
                source_pointer_sha256=unit.source_pointer_sha256)
            return EvidenceDTO(ref_id=ref_id, snapshot_id=snapshot_id, generation_id=generation.id, epoch=epoch,
                source_ref=source_ref, source=source, source_sha256=payload["source_text_sha256"],
                unit_text_sha256=current.unit_text_sha256, text=payload["text"], span_start=start, span_end=end,
                coordinate_system=current.coordinate_system, work_id=current.work_id, work_version_id=current.work_version_id,
                coverage="native_literal" if native else "metadata")

    def _current_source(self, db, actor, locator):
            if locator.kind == "metadata_projection":
                from .metadata_origin import published_observation
                observed = published_observation(db, self._repo(db, actor), UUID(locator.id))
                if observed is not None:
                    if isinstance(observed[0], DeniedSourceUnit):
                        raise DomainError("source_currently_denied", 403)
                    return observed
                projection = db.get(data.SourceProjection, UUID(locator.id))
                source_row = db.get(data.SourceRow, projection.source_row_id) if projection is not None else None
                reference = MetadataReference(metadata_projection_id=UUID(locator.id))
            else:
                native = db.get(data.NativeRepresentation, locator.id)
                source_row = db.get(data.SourceRow, native.source_row_id) if native is not None else None
                reference = NativeReference(native_representation_id=locator.id)
            if source_row is None:
                raise DomainError("not_found", 404)
            snapshot = db.get(data.SourceSnapshot, source_row.snapshot_id)
            if snapshot.status != "staged" or not snapshot.profile_adopted or not snapshot.seal_sha256:
                raise DomainError("source_not_formally_adopted", 409)
            source_generation = SimpleNamespace(snapshot_id=snapshot.id,
                source_root_kind="native_snapshot" if locator.kind == "native_representation" else "metadata_snapshot",
                catalog_name="application_papers" if locator.kind == "native_representation" else "standard_metadata",
                security_domain=snapshot.security_domain, source_namespace=snapshot.source_namespace,
                representation_profile=snapshot.representation_profile, source_manifest_sha256=snapshot.source_manifest_sha256,
                policy_sha256=snapshot.policy_sha256, evaluation_sha256=snapshot.evaluation_sha256)
            repo = self._repo(db, actor)
            current = repo.readers.read(repo.readers.root(source_generation), reference)
            if isinstance(current, DeniedSourceUnit):
                raise DomainError("source_currently_denied", 403)
            return current, snapshot

    @staticmethod
    def _summary_unit(current, locator):
            # A transient hydration DTO, not a fabricated SQL IndexUnit.
            observed = isinstance(current.source_reference, UpdateReference)
            return SimpleNamespace(projection_id=current.source_unit_id, projection=current.model_dump(mode="json"),
                source_reference_kind="update_observation" if observed else locator.kind,
                update_observation_id=current.source_unit_id if observed else None,
                metadata_projection_id=current.source_unit_id if locator.kind == "metadata_projection" and not observed else None,
                native_representation_id=locator.id if locator.kind == "native_representation" else None,
                work_id=current.work_id, work_version_id=current.work_version_id,
                source_namespace=current.root.source_namespace, native_id=current.native_id,
                source_revision=current.source_revision, projection_sha256=current.projection_sha256,
                source_pointer_sha256=current.source_pointer_sha256)
    def detail(self, actor: Actor, locator: PaperLocator) -> PaperDetail:
        # Details need a real adopted source, but do not require a search index.
        with self.factory.begin() as db:
            current, snapshot = self._current_source(db, actor, locator)
            summary = self._summaries(db, [self._summary_unit(current, locator)])[current.source_unit_id]
            pdf_license, pdf_is_unmodified = None, None
            citation_metadata, verified_paper = {}, False
            if isinstance(current.source_reference, UpdateReference):
                citation_metadata = dict(current.original_source_dto.get("metadata_payload") or {})
            if locator.kind == "native_representation":
                member, source, blob = self._pdf_source(db, actor, current, snapshot, locator, require_asset_access=False)
                resolution = db.get(data.SourceResolution, member.resolution_id, populate_existing=True)
                if resolution is not None:
                    # _current_source revalidates the sealed resolution against
                    # its original publisher chain; _pdf_source pins this member.
                    citation_metadata = dict(resolution.proof.get("attribution") or {})
                    verified_paper = (resolution.state == "paper_identity_verified"
                        and resolution.representation_id == locator.id
                        and (current.original_source_dto.get("source_reference") or {}).get("source_kind") == "paper")
                attribution_url = (resolution.proof.get("attribution") or {}).get("source_url") if resolution is not None else None
                if self._official_acl_url(attribution_url):
                    summary.source_url = attribution_url
                elif self._official_acl_url(source.source_url):
                    summary.source_url = source.source_url
                if blob is not None and self._can_read_pdf(db, actor, source):
                    summary.pdf_url = f"/api/v3/papers/native_representation/{locator.id}/pdf"
                    pdf_license, pdf_is_unmodified = source.license, True
                elif self._official_acl_url(source.source_url) and urlsplit(source.source_url).path.lower().endswith(".pdf"):
                    # A verified original-site link does not reclassify the
                    # importer-owned raw PDF as a public product blob grant.
                    summary.pdf_url = source.source_url
            csl, citation_missing = project_csl(identifier=locator.kind + ":" + locator.id,
                title=summary.title, authors=summary.authors, year=summary.year, url=summary.source_url,
                metadata=citation_metadata, verified_paper=verified_paper)
            return PaperDetail(**summary.model_dump(), source_snapshot_id=snapshot.id,
                pdf_license=pdf_license, pdf_is_unmodified=pdf_is_unmodified,
                literal_text_available=current.searchable_text_present, coordinate_system=current.coordinate_system,
                links_status="source_provided" if summary.source_url else "not_in_source",
                citation_status="complete" if csl is not None and not citation_missing else "incomplete_metadata",
                csl=csl, citation_missing_fields=citation_missing or None,
                metadata_observation=getattr(snapshot, "metadata_observation", None))

    def _can_read_pdf(self, db, actor, source):
        profile = self.native_pdf_asset_profile
        if source.raw_security_domain == actor.security_domain:
            return True
        if profile is None or profile.state != "adopted" or source.license != profile.exact_source_license:
            return False
        # The separately adopted source-PDF profile covers public Works only.
        # Private Work grants never expand into raw-blob mirroring permission.
        return bool(db.scalar(select(core.Work.public).where(core.Work.id == source.work_id)))

    @staticmethod
    def _official_acl_url(value):
        if type(value) is not str or "\r" in value or "\n" in value:
            return False
        try:
            url = urlsplit(value)
            return url.scheme == "https" and url.hostname == "aclanthology.org" and not url.username and not url.password
        except ValueError:
            return False

    def _pdf_source(self, db, actor, current, snapshot, locator, *, require_asset_access):
        member = db.scalar(select(ApplicationMember).where(ApplicationMember.snapshot_id == snapshot.id,
            ApplicationMember.representation_id == locator.id))
        source = db.get(core.SourceRecord, member.source_record_id, populate_existing=True) if member is not None else None
        if (member is None or source is None or member.work_id != current.work_id or member.work_version_id != current.work_version_id
                or source.work_id != current.work_id or source_record_sha(source) != member.source_record_sha256
                or source.raw_sha256 != member.raw_PDF_sha256 or source.rights_status != "verified" or source.license != "CC-BY-4.0"):
            raise DomainError("paper_original_PDF_source_or_rights_conflict", 409)
        if require_asset_access and not self._can_read_pdf(db, actor, source):
            raise DomainError("paper_pdf_asset_profile_not_adopted", 403)
        blob = db.get(core.Blob, (source.raw_security_domain, source.raw_sha256), populate_existing=True)
        if require_asset_access and (blob is None or not 0 < blob.size_bytes <= 50*1024**2):
            raise DomainError("paper_original_PDF_blob_missing_or_oversize", 503)
        return member, source, blob

    def read_pdf(self, actor: Actor, locator: PaperLocator) -> PaperPDFAsset:
        if locator.kind != "native_representation":
            raise DomainError("paper_source_has_no_verified_original_PDF", 404)
        with self.factory.begin() as db:
            # Repeats formal member/resolution/WorkVersion/current Work ACL and
            # monotone source/DOI/ACL-Anthology denial checks for every request.
            current, snapshot = self._current_source(db, actor, locator)
            _, source, blob = self._pdf_source(db, actor, current, snapshot, locator, require_asset_access=True)
            stored = StoredBlob(source.raw_security_domain, source.raw_sha256, blob.size_bytes)
            self.store.verify(stored)
            path = self.store.path(stored.security_domain, stored.sha256)
            with path.open("rb") as stream:
                if stream.read(5) != b"%PDF-":
                    raise DomainError("paper_original_PDF_signature_invalid", 503)
            return PaperPDFAsset(path, "paper-"+locator.id[:16]+".pdf", stored.sha256, stored.size_bytes,
                                 source.source_url, source.license)

    def read_pinned_evidence(self, actor: Actor, pinned: EvidenceDTO) -> EvidenceDTO:
        """Internal task-issued pin revalidation, independent of search cursor TTL.

        The caller MUST load this DTO from its owner/task-authorized issuance
        table, never accept an arbitrary model-supplied DTO as a capability. Text
        is re-read from the adopted source under current ACL/denials and compared
        to every immutable locator/hash. No cached text becomes permission.
        """
        with self.factory.begin() as db:
            current, _ = self._current_source(db, actor, pinned.source_ref)
            source = SourceCitation(source_namespace=current.root.source_namespace, native_id=current.native_id,
                source_revision=current.source_revision, projection_sha256=current.projection_sha256,
                source_pointer_sha256=current.source_pointer_sha256)
            start, end = pinned.span_start, pinned.span_end
            if (source != pinned.source or current.unit_text_sha256 != pinned.unit_text_sha256
                    or current.work_id != pinned.work_id or current.work_version_id != pinned.work_version_id
                    or current.coordinate_system != pinned.coordinate_system
                    or not 0 <= start < end <= len(current.literal_text)):
                raise DomainError("pinned_evidence_source_binding_changed", 409)
            text = current.literal_text[start:end]
            if hashlib.sha256(text.encode()).hexdigest() != pinned.source_sha256 or text != pinned.text:
                raise DomainError("pinned_evidence_literal_mismatch", 409)
            return pinned.model_copy(update={"text": text})

    def search_selected_evidence(self, actor: Actor, selected: list[PaperLocator], query: str, limit: int = 5,
                                 *, maximum: int = 5) -> list[EvidenceDTO]:
        return self._search_selected_evidence(actor, selected, query, limit, maximum=maximum)

    def search_selected_pinned_evidence(self, actor: Actor, selected: list[PaperLocator], query: str, limit: int,
                                       snapshots: dict[str, CatalogSnapshot], *, maximum: int = 5) -> list[EvidenceDTO]:
        """One server-owned binding per actual catalog in the explicit selection."""
        required = {"application_papers" if paper.kind == "native_representation" else "standard_metadata" for paper in selected}
        if set(snapshots) != required:
            raise DomainError("task_selected_catalog_bindings_required", 409)
        return self._search_selected_evidence(actor, selected, query, limit, snapshots=snapshots, maximum=maximum)

    def _search_selected_evidence(self, actor: Actor, selected: list[PaperLocator], query: str, limit: int,
                                  *, snapshots: dict[str, CatalogSnapshot] | None = None,
                                  maximum: int = 5) -> list[EvidenceDTO]:
        """Native hybrid retrieval scoped BEFORE both branches, never post-filtered global top-k.

        A mixed metadata/native selection is searched in two explicitly scoped
        catalogs and merged by rank round-robin; incompatible raw scores are not
        compared. Every returned excerpt then crosses the current-source barrier.
        """
        if type(maximum) is not int or maximum not in {5, 8} or type(limit) is not int or not 1 <= limit <= maximum or not 1 <= len(selected) <= 20:
            raise DomainError("bounded_selected_evidence_request_required", 422)
        groups = [[paper for paper in selected if paper.kind == kind]
                  for kind in ("native_representation", "metadata_projection")]
        pages = []
        for group in groups:
            if group:
                catalog = "application_papers" if group[0].kind == "native_representation" else "standard_metadata"
                request = SearchRequest(q=query, selected_papers=group, catalog=catalog,
                    page_size=limit, candidate_limit=max(limit, len(group)), ttl_seconds=1800)
                pages.append(self.search(actor, request) if snapshots is None else self.search_pinned(actor, request, snapshots[catalog]))
        references = [page.results[index].evidence_refs[0] for index in range(limit) for page in pages if index < len(page.results)][:limit]
        return [self.read_evidence(actor, ref_id) for ref_id in references]

    def similar(self, actor: Actor, locator: PaperLocator, request: SimilarRequest) -> SearchPage:
        detail = self.detail(actor, locator)
        if not detail.title or len(detail.title) > 2000:
            raise DomainError("source_has_no_bounded_title_for_similarity_query", 422)
        # The request explicitly searches the actual title. It does not silently
        # truncate a 200k-codepoint native body into a different query.
        result = self.search(actor, SearchRequest(q=detail.title, mode=request.mode,
            catalog="application_papers" if locator.kind == "native_representation" else "standard_metadata",
            page_size=min(request.page_size, request.candidate_limit), candidate_limit=request.candidate_limit))
        return self.exclude_similarity_source(actor, result.snapshot_id, locator=locator, work_id=detail.work_id)

    def exclude_similarity_source(self, actor: Actor, snapshot_id: UUID, *, locator: PaperLocator | None = None,
                                  work_id: UUID | None = None, projection: MaterialQueryProjection | None = None) -> SearchPage:
        """Remove the source from all persisted pages and bind private query provenance."""
        with self.factory.begin() as db:
            snapshot, _, _ = self._snapshot(db, actor, snapshot_id)
            if projection is not None:
                if projection.query != snapshot.request["q"] or projection.source.work_id != (str(work_id) if work_id else None):
                    raise DomainError("material_similarity_snapshot_query_changed", 409)
                snapshot.request = {**snapshot.request, AUDIT_KEY: projection.model_dump(mode="json")}
                snapshot.query_sha256 = canonical_sha(snapshot.request)
            rows = db.execute(select(m.SearchSnapshotHit, index.IndexUnit).join(index.IndexUnit,
                (index.IndexUnit.generation_id == m.SearchSnapshotHit.generation_id) &
                (index.IndexUnit.projection_id == m.SearchSnapshotHit.projection_id)).where(
                m.SearchSnapshotHit.snapshot_id == snapshot_id)).all()
            removed = 0
            for hit, unit in rows:
                same = locator is not None and (unit.projection_id == (UUID(locator.id) if locator.kind == "metadata_projection" else None)
                    or unit.native_representation_id == locator.id)
                if same or (work_id is not None and unit.work_id == work_id):
                    db.delete(hit)
                    removed += 1
            snapshot.candidate_count -= removed
        return self.page(actor, snapshot_id)
