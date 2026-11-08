"""Opt-in navigation over this Task's real, already issued source evidence.

This concrete bridge repeats EvidenceAccess.read's current ACL/catalog/source
barriers around inference. An issued frontier obtained from hybrid search is a
hybrid engineering mode, never retriever-free full-corpus paper reproduction.
It is not exposed as a caller-selected HTTP mode or activated by default.
"""
from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import UUID

from citequest.agents_v3.contracts import AgentError, digest

from .decisionrag import BinaryJudge, DecisionRequest, Document, NavigationLimits, Passage, navigate

if TYPE_CHECKING:
    from citequest.v2.domain import Actor

    from .evidence import EvidenceAccess


def _source(item: dict) -> tuple[str, str]:
    source = item.get("source_ref", {})
    if item.get("evidence_type") == "source_window":
        selection = item.get("selection", {})
        key = str(selection.get("kind")) + ":" + str(selection.get("id"))
        version = item.get("source_version_sha256")
    elif source.get("kind") == "material_fragment":
        if any(not item.get(key) for key in ("file_id", "representation_id", "material_version_id", "source_file_sha256", "file_revision")):
            raise AgentError("decisionrag_material_source_binding_missing")
        key = "material_file:" + item["file_id"]
        version = digest({key:item[key] for key in ("representation_id","material_version_id","source_file_sha256","file_revision")})
    elif source.get("kind") in {"native_representation", "metadata_projection"}:
        if any(not item.get(key) for key in ("unit_text_sha256", "generation_id", "epoch", "source")) or not source.get("id"):
            raise AgentError("decisionrag_paper_source_binding_missing")
        key = source["kind"] + ":" + source["id"]
        version = digest({key:item[key] for key in ("unit_text_sha256","generation_id","epoch","source")})
    else:
        raise AgentError("decisionrag_product_source_required")
    if (key.startswith("None:") or key.endswith(":None") or not isinstance(version,str)
            or len(version) != 64):
        raise AgentError("decisionrag_source_version_required")
    return key, version


def navigate_issued_evidence(query: str, ref_ids: list[str], judge: BinaryJudge, *,
                             access: EvidenceAccess, actor: Actor, task_id: UUID,
                             current_task_scope: Callable[[], str],
                             before_decision: Callable[[DecisionRequest], None],
                             limits: NavigationLimits) -> dict:
    """Server-only bridge; `current_task_scope` must call the live Task guard.

    Pass the enclosing Execution's current ownership/cancellation/scope guard,
    not a cached hash supplied by the model. `before_decision` is the explicit
    local-compute admission hook; a paid provider needs its wallet settlement
    adapter before it can be used here. No new Task or budget window is created.
    Full source_window refs preserve their original source block/locator and
    paper/material chunks preserve their original Unicode spans in the issued
    registry. Navigation returns their exact original refs for QuoteRegistry.
    """
    started = time.monotonic()
    from .evidence import EvidenceAccess
    limits.validate()
    if (not isinstance(access,EvidenceAccess) or type(ref_ids) is not list
            or not 1 <= len(ref_ids) <= limits.max_passages
            or any(type(ref) is not str or not 1 <= len(ref) <= 300 for ref in ref_ids)
            or len(set(ref_ids)) != len(ref_ids)):
        raise AgentError("decisionrag_actual_Task_evidence_access_required")
    # EvidenceAccess loads the issued DTO from PostgreSQL itself; a caller/model
    # cannot replace a ref's source text, hierarchy, ACL or generation payload.
    def in_window() -> None:
        if time.monotonic()-started >= limits.active_seconds:
            raise AgentError("decisionrag_active_budget_exhausted")

    def read_current() -> list[dict]:
        values = []
        for ref in ref_ids:
            in_window()
            values.append(access.read(actor,task_id,ref))
            in_window()
        return values

    baseline = read_current()
    baseline_hash = digest(baseline)
    in_window()
    task_binding = current_task_scope()
    in_window()
    if not isinstance(task_binding,str) or not task_binding:
        raise AgentError("decisionrag_current_Task_scope_required")
    groups = OrderedDict()
    for ref_id,item in zip(ref_ids,baseline,strict=True):
        text, start, end = item.get("text"), item.get("span_start"), item.get("span_end")
        if (item.get("ref_id") != ref_id or type(text) is not str or not text.strip()
                or type(start) is not int or type(end) is not int or not 0 <= start < end
                or end-start != len(text) or hashlib.sha256(text.encode()).hexdigest() != item.get("source_sha256")):
            raise AgentError("decisionrag_literal_issued_evidence_required")
        key,version = _source(item)
        entry = groups.setdefault(key,{"version":version,"passages":[]})
        if entry["version"] != version:
            raise AgentError("decisionrag_document_has_mixed_source_versions")
        entry["passages"].append(Passage(ref_id,text,key,version,ref_id))
    # Each document description is ALL of its supplied literal child passages,
    # never an LLM summary or invented taxonomy. The frontier is explicitly
    # bounded; its completeness with respect to the entire document is unknown.
    documents = tuple(Document(key,"\n\n".join(p.text for p in group["passages"]),group["version"],
                               tuple(group["passages"])) for key,group in groups.items())
    binding = digest({"task_scope":task_binding,"issued_evidence":baseline_hash})

    def check_scope() -> str:
        in_window()
        if current_task_scope() != task_binding:
            raise AgentError("decisionrag_Task_scope_changed")
        in_window()
        current = read_current()
        if digest(current) != baseline_hash:
            raise AgentError("decisionrag_issued_source_or_version_changed")
        return binding

    result = navigate(query,documents,judge,scope_binding=binding,check_scope=check_scope,
                      before_decision=before_decision,limits=limits,started_at=started)
    return {**result,"integration_mode":"issued_frontier_hybrid_engineering",
            "full_document_or_corpus_coverage_claimed":False,
            "current_authority":"EvidenceAccess.read_and_live_Task_guard_before_after_inference",
            "input_frontier_ref_ids":list(ref_ids)}
