"""Scoped RAG chunk retrieval with independent immutable query receipts.

The caller loads a durable server-owned CatalogSnapshot; no user/model can
select a collection/endpoint. This service does not create or retarget Task
pins. Paper search and its unique-paper snapshots keep their original path.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4

from pydantic import Field, model_validator
from sqlalchemy import or_, select, text

from citequest.index_v2 import models as index
from citequest.index_v2.rows_v2 import canonical_sha
from citequest.v2.domain import Actor, DomainError

from . import evidence_models as m
from .catalog_snapshot import CatalogSnapshot
from .schemas import EvidenceDTO, PaperLocator, SearchRequest, Strict
from .service import SearchService
from .visibility import visible_unit


class ChunkEvidencePage(Strict):
    retrieval_policy: Literal["source-chunk-hybrid/1"] = "source-chunk-hybrid/1"
    retrieval_id: UUID
    generation_id: UUID
    epoch: int = Field(ge=1)
    evidence: list[EvidenceDTO] = Field(max_length=20)
    candidates: int = Field(ge=0, le=20)
    masked_before_delivery: int = Field(ge=0, le=20)

    @model_validator(mode="after")
    def bound_delivery(self) -> ChunkEvidencePage:
        if len(self.evidence) + self.masked_before_delivery != self.candidates:
            raise ValueError("chunk candidate/delivery/masked accounting mismatch")
        if len({item.ref_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("chunk evidence references must be unique")
        if any(item.generation_id != self.generation_id or item.epoch != self.epoch
               or item.snapshot_id != self.retrieval_id for item in self.evidence):
            raise ValueError("chunk evidence must belong to the issued retrieval snapshot")
        return self


class ChunkEvidenceSearch:
    def __init__(self, search: SearchService) -> None:
        self.search = search

    def retrieve_selected(self, actor: Actor, selected: list[PaperLocator], query: str,
                          pins: dict[str, CatalogSnapshot], *, limit: int = 8) -> list[ChunkEvidencePage]:
        """Bound each native query to one selected document before both branches.

        Allocate at least one candidate slot to each document, then distribute
        the remaining slots in the user's stable selection order. This is a
        retrieval allocation, never a full-document coverage or support claim.
        No raw scores from different queries/catalogs are compared.
        """
        if (not isinstance(selected, list) or any(not isinstance(item, PaperLocator) for item in selected)
                or type(limit) is not int or not 1 <= limit <= 20 or not 1 <= len(selected) <= limit
                or len({(item.kind, item.id) for item in selected}) != len(selected)):
            raise DomainError("selected_chunk_retrieval_requires_capacity_for_every_document", 422)
        catalogs = {"application_papers" if item.kind == "native_representation" else "standard_metadata"
                    for item in selected}
        if set(pins) != catalogs or any(not isinstance(pin, CatalogSnapshot) or pin.catalog != catalog
                                        for catalog, pin in pins.items()):
            raise DomainError("task_selected_catalog_bindings_required", 409)
        quotient, remainder = divmod(limit, len(selected))
        groups = {catalog: [item for item in selected if ("application_papers" if item.kind == "native_representation"
                  else "standard_metadata") == catalog] for catalog in catalogs}
        for catalog in sorted(catalogs):
            self.search.recheck_catalog_snapshot(actor, pins[catalog], groups[catalog])
        pages = []
        for ordinal, paper in enumerate(selected):
            catalog = "application_papers" if paper.kind == "native_representation" else "standard_metadata"
            depth = quotient + int(ordinal < remainder)
            request = SearchRequest(q=query, catalog=catalog, selected_papers=[paper],
                                    candidate_limit=depth, page_size=depth)
            pages.append(self.retrieve(actor, request, pins[catalog], limit=depth))
        # An earlier document may be revoked while a later one is read. Recheck
        # all selected scopes at the end; independent receipts remain audit
        # records, never stale permission to return an earlier excerpt.
        for catalog in sorted(catalogs):
            self.search.recheck_catalog_snapshot(actor, pins[catalog], groups[catalog])
        return pages

    def retrieve(self, actor: Actor, request: SearchRequest, pin: CatalogSnapshot, *, limit: int = 8) -> ChunkEvidencePage:
        if (type(limit) is not int or not 1 <= limit <= 20 or request.field != "text" or request.sort != "relevance"
                or not isinstance(pin, CatalogSnapshot) or request.catalog != pin.catalog):
            raise DomainError("bounded_server_owned_chunk_retrieval_required", 422)
        request = request.model_copy(update={"page_size": min(limit, request.page_size), "candidate_limit": limit})
        self.search._validate_query(request)
        lease_id = None
        try:
            with self.search.factory.begin() as db:
                db.execute(text("SET LOCAL jit = off"))
                if not db.execute(text("SELECT to_regclass('cq_app.evidence_retrievals') IS NOT NULL AND "
                                       "to_regclass('cq_app.evidence_retrieval_hits') IS NOT NULL")).scalar_one():
                    raise DomainError("chunk_evidence_receipt_schema_unavailable", 503)
                repo, generation = self.search._check_catalog_snapshot(db, actor, pin, request.selected_papers)
                denied = list(db.scalars(select(index.IndexUnit.projection_id).where(
                    index.IndexUnit.generation_id == generation.id, ~visible_unit(actor, generation.security_domain)).limit(5001)))
                if len(denied) > 5000:
                    raise DomainError("visibility_prefilter_exceeds_bounded_native_plan", 503)
                excluded = self.search._row_keys(db, generation, denied)
                included = None
                if request.selected_papers:
                    conditions = [self.search._paper_condition(paper) for paper in request.selected_papers]
                    selected = list(db.scalars(select(index.IndexUnit.projection_id).where(
                        index.IndexUnit.generation_id == generation.id, or_(*conditions), visible_unit(actor, generation.security_domain))))
                    included = self.search._row_keys(db, generation, selected)
                lease = index.IndexReaderLease(generation_id=generation.id, owner_id=actor.owner_id, epoch=pin.epoch,
                    expires_at=datetime.now(UTC) + timedelta(seconds=120))
                db.add(lease)
                db.flush()
                lease_id = lease.id
                native_args = {"collection": generation.collection_name, "request": request, "security_domain": generation.security_domain,
                    "excluded_row_keys": excluded, "included_row_keys": included, "expected_proof": generation.server_proof}
            hits = self.search.native.search_chunks(**native_args)
            with self.search.factory.begin() as db:
                db.execute(text("SET LOCAL jit = off"))
                repo, generation = self.search._check_catalog_snapshot(db, actor, pin, request.selected_papers)
                allowed = set(repo.recheck_hits(lease_id, [hit.chunk_id for hit in hits])["allowed_chunk_ids"])
                rows = db.execute(select(index.IndexChunk, index.IndexUnit).join(index.IndexUnit,
                    (index.IndexUnit.generation_id == index.IndexChunk.generation_id) &
                    (index.IndexUnit.projection_id == index.IndexChunk.projection_id)).where(
                        index.IndexChunk.generation_id == generation.id, index.IndexChunk.chunk_id.in_(allowed),
                        visible_unit(actor, generation.security_domain))).all()
                by_chunk = {chunk.chunk_id: (chunk, unit) for chunk, unit in rows}
                retrieval_id, evidence, ranked = uuid4(), {}, []
                for hit in hits:
                    if hit.chunk_id not in by_chunk:
                        continue
                    chunk, unit = by_chunk[hit.chunk_id]
                    if chunk.payload["row_key"] != hit.row_key or hit.chunk_id in evidence:
                        raise DomainError("native_chunk_SQL_identity_mismatch", 503)
                    evidence[hit.chunk_id] = self.search._literal_evidence(db, actor, generation, chunk, unit,
                        snapshot_id=retrieval_id, epoch=pin.epoch, ref_id=f"rag:{retrieval_id}:{hit.chunk_id}")
                    ranked.append((hit, chunk, unit))
                # Recheck after source materialization as well as after native
                # IO. An immutable generation is never a cached access grant.
                final = set(repo.recheck_hits(lease_id, list(evidence))["allowed_chunk_ids"])
                # recheck_hits covers its lease/root/direct-denial contract.
                # Repeat the broader visibility predicate in a new statement:
                # DOI/Anthology aliases and native rights may change while
                # literal sources are being materialized.
                final &= set(db.scalars(select(index.IndexChunk.chunk_id).join(index.IndexUnit,
                    (index.IndexUnit.generation_id == index.IndexChunk.generation_id) &
                    (index.IndexUnit.projection_id == index.IndexChunk.projection_id)).where(
                        index.IndexChunk.generation_id == generation.id, index.IndexChunk.chunk_id.in_(final),
                        visible_unit(actor, generation.security_domain))))
                ranked = [item for item in ranked if item[0].chunk_id in final]
                payload = {"retrieval_policy": "source-chunk-hybrid/1", "search": request.model_dump(mode="json")}
                db.add(m.EvidenceRetrieval(id=retrieval_id, owner_id=actor.owner_id, generation_id=generation.id,
                    epoch=pin.epoch, request=payload, request_sha256=canonical_sha(payload),
                    catalog_binding_sha256=canonical_sha(pin.model_dump(mode="json")), candidate_count=len(hits),
                    delivered_count=len(ranked), masked_count=len(hits) - len(ranked),
                    expires_at=datetime.now(UTC) + timedelta(seconds=request.ttl_seconds)))
                db.flush()
                db.add_all([m.EvidenceRetrievalHit(retrieval_id=retrieval_id, ordinal=ordinal, generation_id=generation.id,
                    projection_id=unit.projection_id, chunk_id=chunk.chunk_id, score=hit.score)
                    for ordinal, (hit, chunk, unit) in enumerate(ranked, 1)])
                repo.close_reader(lease_id)
                result = ChunkEvidencePage(retrieval_id=retrieval_id, generation_id=generation.id, epoch=pin.epoch,
                    evidence=[evidence[hit.chunk_id] for hit, _, _ in ranked], candidates=len(hits),
                    masked_before_delivery=len(hits) - len(ranked))
            lease_id = None
            return result
        finally:
            if lease_id is not None:
                with self.search.factory.begin() as db:
                    lease = db.get(index.IndexReaderLease, lease_id)
                    if lease is not None and lease.owner_id == actor.owner_id:
                        db.delete(lease)
