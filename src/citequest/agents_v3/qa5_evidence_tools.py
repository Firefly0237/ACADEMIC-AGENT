"""QA5's explicit chunk policy over existing current-authority evidence services."""
from __future__ import annotations

from collections import Counter
import time
from uuid import UUID

from citequest.research_v3.evidence import EvidenceAccess, reference_selections
from citequest.research_v3.qa_v5_contracts import result_limits
from citequest.search_v3.chunk_evidence import ChunkEvidenceSearch
from citequest.search_v3.schemas import PaperLocator, SearchRequest
from citequest.v2.domain import Actor

from .catalog_scope import POLICY, task_policy
from .contracts import AgentError, canonical, digest
from .product_profiles import retrieval_rerank_protocol
from .chunk_tool_policy import uses_typed_chunk_tools


class QA5EvidenceTools:
    def __init__(self, database, search, material, context, registered, register, catalog_pins, *, reranker=None):
        self.database, self.search, self.material = database, search, material
        self.context, self.registered, self.register, self.catalog_pins = context, registered, register, catalog_pins
        self.access = EvidenceAccess(database, search, material)
        self.reranker = reranker

    def task(self) -> dict:
        task = self.context()
        if not uses_typed_chunk_tools(task) or task_policy(task) != POLICY:
            raise AgentError("qa5_chunk_profile_and_task_catalog_pins_required", 503)
        return task

    def verify(self, ref_ids: list[str]) -> dict:
        if (not isinstance(ref_ids, list) or not 1 <= len(ref_ids) <= 64 or len(set(ref_ids)) != len(ref_ids)
                or any(not isinstance(ref, str) or not 1 <= len(ref) <= 300 for ref in ref_ids)
                or len(canonical({"ref_ids": ref_ids})) > 20000):
            raise AgentError("invalid_evidence_verification_batch")
        task = self.task()
        actor = Actor(task["owner_id"])
        reads, results = Counter(), []
        for ref_id in ref_ids:
            # EvidenceAccess loads this Task's issued DTO and uses current pin,
            # selection, literal-source and material ACL/version authorities.
            # No client/model DTO is accepted as a read capability.
            value = self.access.read(actor, task["id"], ref_id)
            kind = "source_window" if value.get("evidence_type") == "source_window" else value["source_ref"]["kind"]
            reads[kind] += 1
            results.append({"ref_id": ref_id, "source_sha256": value["source_sha256"],
                            "unit_text_sha256": value["unit_text_sha256"]})
        self.task()  # An epoch/cancellation change during batch IO is terminal.
        return {"status":"ok", "verified":len(results), "evidence":results,
            "actual_reference_read_calls":sum(reads.values()), "source_read_calls_by_kind":dict(reads),
            "count_basis":"actual EvidenceAccess.read invocations, not disk IO or SQL operation counts",
            "authority":"current_Task_issued_refs_source_ACL_literal_and_catalog_pins"}

    def search_chunks(self, query: str, limit: int, *, remaining_ms: int | None = None) -> dict:
        started = time.monotonic()
        if not isinstance(query, str) or not 1 <= len(query) <= 2000 or type(limit) is not int or not 1 <= limit <= 20:
            raise AgentError("qa5_chunk_search_arguments_invalid")
        task = self.task()
        rerank = retrieval_rerank_protocol(task) is not None
        if rerank and (self.reranker is None
                or self.reranker.configuration_sha256 != task["profile"]["rerank_config_sha256"]
                or type(remaining_ms) is not int or not 1 <= remaining_ms <= 30000):
            raise AgentError("rerank_server_configuration_or_deadline_required", 503)
        actor = Actor(task["owner_id"])
        selected = reference_selections(task["brief"])
        if selected and len(selected) > limit:
            raise AgentError("qa5_result_capacity_must_include_every_selected_document")
        if limit not in result_limits(selected):
            raise AgentError("qa5_query_candidate_allocation_unsupported")
        pins = self.catalog_pins.ensure(actor, task["id"])
        evidence, documents, receipts = [], [], []
        if selected:
            quotient, remainder = divmod(limit, len(selected))
            allocations = [(item, quotient + int(index < remainder)) for index, item in enumerate(selected)]
            papers = [PaperLocator.model_validate(item, strict=True) for item, _ in allocations if item["kind"] != "material_file"]
            paper_limit = sum(size for item, size in allocations if item["kind"] != "material_file")
            by_document = {}
            if papers:
                if self.search is None:
                    raise AgentError("paper_evidence_bridge_unavailable", 503)
                catalogs = {"application_papers" if paper.kind == "native_representation" else "standard_metadata" for paper in papers}
                pages = ChunkEvidenceSearch(self.search).retrieve_selected(actor, papers, query,
                    {name: pins[name] for name in catalogs}, limit=paper_limit)
                for paper, page in zip(papers, pages, strict=True):
                    values = [value.model_dump(mode="json") for value in page.evidence]
                    by_document[(paper.kind,paper.id)] = values
                    receipts.append({"retrieval_id":str(page.retrieval_id),"generation_id":str(page.generation_id),
                        "epoch":page.epoch,"candidates":page.candidates,"masked_before_delivery":page.masked_before_delivery})
            for item, size in allocations:
                key = (item["kind"],item["id"])
                if item["kind"] == "material_file":
                    if self.material is None:
                        raise AgentError("material_evidence_bridge_unavailable", 503)
                    by_document[key] = self.material.search_selected(actor, [UUID(item["id"])], query, size)
                documents.append({"selection":item,"candidate_slots":size,"delivered":len(by_document[key]),
                                  "full_document_coverage_claimed":False})
            # Stable per-document round robin, never incomparable score fusion
            # or filtering a global top-k after selecting the papers.
            for rank in range(max(size for _, size in allocations)):
                for item, _ in allocations:
                    values = by_document[(item["kind"],item["id"])]
                    if rank < len(values):
                        evidence.append(values[rank])
        else:
            if self.search is None:
                raise AgentError("paper_evidence_bridge_unavailable", 503)
            catalog = task["brief"]["catalog"]
            if catalog not in pins:
                raise AgentError("task_catalog_search_not_in_active_role", 403)
            page = ChunkEvidenceSearch(self.search).retrieve(actor,
                SearchRequest(q=query,catalog=catalog,page_size=limit,candidate_limit=limit),pins[catalog],limit=limit)
            evidence = [value.model_dump(mode="json") for value in page.evidence]
            receipts.append({"retrieval_id":str(page.retrieval_id),"generation_id":str(page.generation_id),
                "epoch":page.epoch,"candidates":page.candidates,"masked_before_delivery":page.masked_before_delivery})
        if len(evidence) > limit or len({value["ref_id"] for value in evidence}) != len(evidence):
            raise AgentError("qa5_chunk_delivery_identity_or_capacity_invalid", 503)
        self.register(evidence)
        verification = self.verify([value["ref_id"] for value in evidence]) if evidence else None
        ranking = {}
        if rerank:
            def scope_hash(value):
                return digest({**{key: str(value[key]) for key in ("id", "owner_id", "project_id")},
                    **{key: value[key] for key in ("epoch", "revision", "brief", "profile")}})

            initial_scope = scope_hash(task)

            def check_scope():
                if scope_hash(self.task()) != initial_scope:
                    raise AgentError("rerank_Task_scope_changed", 409)

            left_ms = remaining_ms - int((time.monotonic() - started) * 1000)
            if left_ms <= 0:
                raise AgentError("rerank_active_budget_exhausted", 503)
            evidence, receipt = self.reranker.rerank(query, evidence, remaining_ms=left_ms, check_scope=check_scope)
            verification = self.verify([value["ref_id"] for value in evidence]) if evidence else None
            ranking = {"rerank_receipt": receipt}
        self.task()
        return {"status":"ok", "empty":not evidence,"evidence":evidence,"fixture":False,
            "retrieval_policy":"source-chunk-hybrid/1", "scope":"per_selected_document" if selected else "bounded_pinned_catalog",
            "document_coverage":documents,"retrieval_receipts":receipts,"fresh_verification":verification,
            "full_document_coverage_claimed":False, **ranking}
