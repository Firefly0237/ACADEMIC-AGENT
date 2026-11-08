"""Bounded document-to-passage DecisionRAG mechanism, independent of generation.

Implements the two-level instance of arXiv:2610.35868 Algorithm 1: judge ALL
documents together, prune to top-k, expand actual children, then judge those
passages together. No embedding retriever, synthetic clustering, generated
scores or fallback is hidden here. This is an opt-in mechanism component;
caller-owned ACL/version/Task accounting and real-model evaluation are required.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from citequest.agents_v3.contracts import AgentError, canonical, digest, validate_user_text

VERSION = "citequest-decisionrag-navigation/1"


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    document_id: str
    source_version: str
    evidence_ref_id: str


@dataclass(frozen=True)
class Document:
    id: str
    description: str
    source_version: str
    passages: tuple[Passage, ...]


@dataclass(frozen=True)
class DecisionRequest:
    query: str
    stage: str
    candidates: tuple[tuple[str, str], ...]
    scope_binding: str

    def wire_data(self) -> tuple[dict, dict, dict[str, str]]:
        labels = {"N" + str(index):identifier for index,(identifier,_) in enumerate(self.candidates, 1)}
        state = {"query":self.query, "candidates":[{"id":label, "text":text}
            for label,(_,text) in zip(labels, self.candidates, strict=True)]}
        questions = {label:{"type":"noul", "instructions":f"Does candidate {label} contain information needed to answer the query?",
            "criteria":{"true":"Contains information that addresses the query.",
                        "false":"Does not contain information that addresses the query."}} for label in labels}
        return state, questions, labels

    def fingerprint(self) -> str:
        return digest({"protocol":VERSION, "stage":self.stage, "query":self.query,
            "candidates":self.candidates, "scope_binding":self.scope_binding})


@dataclass(frozen=True)
class DecisionReceipt:
    request_sha256: str
    probabilities: dict[str, float]
    model_binding: dict
    usage: dict


class BinaryJudge(Protocol):
    def preflight(self, request: DecisionRequest) -> None: ...
    def score(self, request: DecisionRequest) -> DecisionReceipt: ...


@dataclass(frozen=True)
class NavigationLimits:
    document_keep: int = 2
    passage_keep: int = 5
    max_documents: int = 8
    max_passages: int = 32
    max_shared_bytes: int = 32768
    active_seconds: float = 30

    def validate(self) -> None:
        values = (self.document_keep, self.passage_keep, self.max_documents, self.max_passages, self.max_shared_bytes)
        if (any(type(value) is not int or value < 1 for value in values)
                or self.document_keep > self.max_documents or self.passage_keep > self.max_passages
                or not isinstance(self.active_seconds, (int, float)) or isinstance(self.active_seconds, bool)
                or not math.isfinite(self.active_seconds) or self.active_seconds <= 0):
            raise AgentError("decisionrag_invalid_resource_profile")


def validate_receipt(request: DecisionRequest, receipt: DecisionReceipt) -> None:
    expected = {identifier for identifier,_ in request.candidates}
    if (not isinstance(receipt, DecisionReceipt) or receipt.request_sha256 != request.fingerprint()
            or not isinstance(receipt.probabilities, dict) or set(receipt.probabilities) != expected):
        raise AgentError("decisionrag_response_identity_mismatch")
    if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
           for value in receipt.probabilities.values()):
        raise AgentError("decisionrag_invalid_probability")
    if (not isinstance(receipt.model_binding, dict) or not receipt.model_binding
            or not isinstance(receipt.usage, dict) or not receipt.usage):
        raise AgentError("decisionrag_model_or_usage_binding_missing")


def navigate(query: str, documents: tuple[Document, ...], judge: BinaryJudge, *,
             scope_binding: str, check_scope: Callable[[], str],
             before_decision: Callable[[DecisionRequest], None],
             limits: NavigationLimits | None = None, started_at: float | None = None) -> dict:
    """Return only original server-issued leaf refs after current scope checks.

    `check_scope` must repeat actual current ACL/source/generation checks and
    return the server-owned binding; it is not supplied by a model or HTTP user.
    `before_decision` admits/logs each real local/remote decision in the enclosing
    Task harness. Remote paid judges also require their provider wallet adapter.
    Wall checks reject late results but do not kill a stuck native forward: the
    caller's process supervisor must enforce the physical execution deadline.
    """
    now = time.monotonic()
    if (started_at is not None and (type(started_at) not in (int,float)
            or not math.isfinite(started_at) or started_at > now)):
        raise AgentError("decisionrag_invalid_parent_execution_window")
    started = now if started_at is None else started_at
    limits = limits if limits is not None else NavigationLimits()
    limits.validate()
    if (type(query) is not str or not query.strip() or len(query) > 8000
            or type(documents) is not tuple or not 1 <= len(documents) <= limits.max_documents
            or type(scope_binding) is not str or not scope_binding
            or not callable(check_scope) or not callable(before_decision)):
        raise AgentError("decisionrag_bounded_server_scope_required")
    validate_user_text(query)
    if (any(not isinstance(document, Document) for document in documents)
            or any(type(document.id) is not str or not document.id
                   or type(document.description) is not str or not document.description.strip()
                   or type(document.source_version) is not str or not document.source_version
                   or type(document.passages) is not tuple or not document.passages for document in documents)
            or len({document.id for document in documents}) != len(documents)):
        raise AgentError("decisionrag_document_identity_invalid")
    passages = tuple(passage for document in documents for passage in document.passages)
    if (not passages or len(passages) > limits.max_passages
            or any(not isinstance(passage, Passage)
                   or any(type(value) is not str or not value for value in
                          (passage.id, passage.text, passage.document_id, passage.source_version, passage.evidence_ref_id))
                   or not passage.text.strip() for passage in passages)
            or len({passage.id for passage in passages}) != len(passages)
            or len({passage.evidence_ref_id for passage in passages}) != len(passages)):
        raise AgentError("decisionrag_passage_identity_or_capacity_invalid")
    for document in documents:
        if (not document.id or not document.description.strip() or not document.source_version
                or not document.passages or any(not isinstance(passage, Passage)
                    or not passage.id or not passage.text.strip() or not passage.evidence_ref_id
                    or passage.document_id != document.id or passage.source_version != document.source_version
                    for passage in document.passages)):
            raise AgentError("decisionrag_actual_source_hierarchy_required")
    def scope() -> None:
        if time.monotonic() - started >= limits.active_seconds:
            raise AgentError("decisionrag_active_budget_exhausted")
        if check_scope() != scope_binding:
            raise AgentError("decisionrag_scope_or_version_changed")
        # Actual authorization IO is part of this execution window too. A slow
        # final read cannot turn an expired navigation into a successful result.
        if time.monotonic() - started >= limits.active_seconds:
            raise AgentError("decisionrag_active_budget_exhausted")

    def request(stage: str, values: tuple[tuple[str,str], ...]) -> DecisionRequest:
        value = DecisionRequest(query, stage, values, scope_binding)
        state, questions, _ = value.wire_data()
        if len(canonical({"state":state, "questions":questions})) > limits.max_shared_bytes:
            raise AgentError("decisionrag_shared_context_byte_capacity")
        return value

    document_request = request("documents", tuple((document.id,document.description) for document in documents))
    # Admit the complete unpruned source frontier before doing any model work.
    # This bounded component deliberately refuses oversize frontiers; it never
    # splits them into incomparable requests or silently clips source content.
    full_passage_request = request("passages", tuple((passage.id,passage.text) for passage in passages))
    scope()
    judge.preflight(document_request)
    judge.preflight(full_passage_request)
    trace = []
    model_binding = None

    def score(value: DecisionRequest) -> dict[str,float]:
        nonlocal model_binding
        scope()
        judge.preflight(value)
        scope()
        before_decision(value)
        scope()
        receipt = judge.score(value)
        validate_receipt(value, receipt)
        scope()
        if model_binding is not None and receipt.model_binding != model_binding:
            raise AgentError("decisionrag_judge_changed_between_levels")
        model_binding = deepcopy(receipt.model_binding)
        trace.append({"stage":value.stage, "request_sha256":value.fingerprint(),
            "candidate_ids":[identifier for identifier,_ in value.candidates],
            "probabilities":deepcopy(receipt.probabilities), "model_binding":deepcopy(receipt.model_binding),
            "usage":deepcopy(receipt.usage)})
        return dict(receipt.probabilities)

    document_scores = score(document_request)
    ordinal = {document.id:index for index,document in enumerate(documents)}
    kept = sorted(documents, key=lambda document:(-document_scores[document.id],ordinal[document.id]))[:limits.document_keep]
    selected = tuple(passage for document in kept for passage in document.passages)
    passage_request = request("passages", tuple((passage.id,passage.text) for passage in selected))
    passage_scores = score(passage_request)
    ordinal = {passage.id:index for index,passage in enumerate(selected)}
    ranked = sorted(selected, key=lambda passage:(-passage_scores[passage.id],ordinal[passage.id]))[:limits.passage_keep]
    scope()
    return {"retrieval_policy":VERSION, "hierarchy":"actual_documents_to_actual_passages",
        "paper_benchmark_reproduction":False, "pruning_reversible":False,
        "document_candidates":len(documents), "document_ids_kept":[document.id for document in kept],
        "passage_candidates":len(selected), "pruned_document_ids":[document.id for document in documents if document not in kept],
        "evidence_ref_ids":[passage.evidence_ref_id for passage in ranked],
        "selected_passage_ids":[passage.id for passage in ranked], "decision_calls":len(trace),
        "elapsed_seconds":time.monotonic()-started, "trace":trace,
        "coverage":"bounded_navigation_not_full_document_summary_or_entailment"}
