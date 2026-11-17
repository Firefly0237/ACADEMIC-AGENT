"""Whole-candidate screening followed by a single shared final comparison.

The split-level/rescore pattern follows DecisionRAG section 3.4. Different
screening batches never contribute directly comparable final scores.
"""
from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass

from citequest.agents_v3.contracts import AgentError, canonical, validate_user_text

from .decisionrag import DecisionRequest, Document, NavigationLimits, validate_receipt

VERSION = "citequest-decisionrag-navigation/2"
CAPACITY_ERRORS = frozenset({"laya_complete_context_exceeds_token_budget", "laya_shared_question_capacity",
    "decisionrag_shared_context_byte_capacity", "decisionrag_remote_request_byte_limit"})


@dataclass(frozen=True)
class BatchedNavigationLimits(NavigationLimits):
    shortlist_count: int = 8
    max_batches_per_level: int = 4
    max_forward_calls: int = 10

    def validate(self) -> None:
        super().validate()
        if (not 1 <= self.max_documents <= 20 or not 1 <= self.max_passages <= 20
                or type(self.shortlist_count) is not int or not 1 <= self.shortlist_count <= 8
                or type(self.max_batches_per_level) is not int or not 1 <= self.max_batches_per_level <= 4
                or self.max_batches_per_level > self.shortlist_count
                or max(self.document_keep, self.passage_keep) > self.shortlist_count
                or type(self.max_forward_calls) is not int or not 2 <= self.max_forward_calls <= 10):
            raise AgentError("decisionrag_invalid_batched_resource_profile")


def shortlist_quotas(sizes: list[int], maximum: int) -> list[int]:
    """Every batch participates; allocate fixed slots without pooling scores."""
    if not sizes or any(type(size) is not int or size < 1 for size in sizes) or not len(sizes) <= maximum:
        raise AgentError("decisionrag_shared_shortlist_capacity")
    quotas = [1] * len(sizes)
    remaining = min(maximum, sum(sizes)) - len(sizes)
    while remaining:
        for index, size in enumerate(sizes):
            if quotas[index] < size:
                quotas[index] += 1
                remaining -= 1
                if not remaining:
                    break
    return quotas


def navigate_batched(query, documents, judge, *, scope_binding, check_scope, before_decision,
                     limits: BatchedNavigationLimits, started_at=None) -> dict:
    started = time.monotonic() if started_at is None else started_at
    trace, selections = [], []
    pending_score = False
    model_binding = None
    result = {"retrieval_policy": VERSION, "hierarchy": "actual_documents_to_actual_passages",
        "paper_benchmark_reproduction": False, "pruning_reversible": False,
        "score_comparison": "within_screening_batch_then_shared_final_request",
        "coverage": "bounded_navigation_not_full_document_summary_or_entailment",
        "document_candidates": len(documents) if isinstance(documents, tuple) else 0,
        "document_ids_kept": [], "pruned_document_ids": [], "passage_candidates": 0,
        "evidence_ref_ids": [], "selected_passage_ids": []}

    def scope():
        if time.monotonic() - started >= limits.active_seconds:
            raise AgentError("decisionrag_active_budget_exhausted")
        if check_scope() != scope_binding:
            raise AgentError("decisionrag_scope_or_version_changed")
        if time.monotonic() - started >= limits.active_seconds:
            raise AgentError("decisionrag_active_budget_exhausted")

    def request(stage, values):
        return DecisionRequest(query, stage, tuple(values), scope_binding)

    def preflight(value):
        state, questions, _ = value.wire_data()
        if len(value.candidates) > 8:
            raise AgentError("laya_shared_question_capacity")
        if len(canonical({"state": state, "questions": questions})) > limits.max_shared_bytes:
            raise AgentError("decisionrag_shared_context_byte_capacity")
        judge.preflight(value)

    def batches(stage, values):
        packed, current = [], []
        for candidate in values:
            scope()
            proposed = [*current, candidate]
            try:
                preflight(request(stage, proposed))
            except AgentError as error:
                if error.code not in CAPACITY_ERRORS:
                    raise
                if not current:
                    raise AgentError("decisionrag_single_candidate_capacity_exceeded") from None
                packed.append(tuple(current))
                current = [candidate]
                try:
                    preflight(request(stage, current))
                except AgentError as single_error:
                    if single_error.code not in CAPACITY_ERRORS:
                        raise
                    raise AgentError("decisionrag_single_candidate_capacity_exceeded") from None
            else:
                current = proposed
        if current:
            packed.append(tuple(current))
        if len(packed) > limits.max_batches_per_level:
            raise AgentError("decisionrag_batch_count_capacity_exceeded")
        return packed

    def score(value, phase, index):
        nonlocal pending_score, model_binding
        scope()
        preflight(value)
        if len(trace) >= limits.max_forward_calls:
            raise AgentError("decisionrag_forward_budget_exhausted")
        before_decision(value)
        scope()
        pending_score = True
        receipt = judge.score(value)
        validate_receipt(value, receipt)
        usage = receipt.usage
        if (usage.get("observed_forward_calls") != 1 or usage.get("state_tokens_dropped") != 0
                or usage.get("truncated") is not False or type(usage.get("input_tokens")) is not int
                or usage["input_tokens"] < 1 or usage.get("external_API_cash_ncny") != 0):
            raise AgentError("decisionrag_batched_usage_invalid")
        if model_binding is not None and receipt.model_binding != model_binding:
            raise AgentError("decisionrag_judge_changed_between_levels")
        model_binding = deepcopy(receipt.model_binding)
        trace.append({"stage": value.stage, "phase": phase, "batch_index": index,
            "request_sha256": value.fingerprint(), "candidate_ids": [row[0] for row in value.candidates],
            "probabilities": deepcopy(receipt.probabilities), "model_binding": deepcopy(model_binding),
            "usage": deepcopy(usage)})
        pending_score = False
        scope()
        return receipt.probabilities

    def rank(values, probabilities, keep):
        order = {value[0]: index for index, value in enumerate(values)}
        return sorted(values, key=lambda value: (-probabilities[value[0]], order[value[0]]))[:keep]

    def level(stage, values, keep, packed):
        if len(packed) == 1:
            final_values = packed[0]
        else:
            quotas = shortlist_quotas([len(group) for group in packed], limits.shortlist_count)
            chosen = set()
            for index, (group, quota) in enumerate(zip(packed, quotas, strict=True)):
                probabilities = score(request(stage, group), "screen", index)
                chosen.update(row[0] for row in rank(group, probabilities, quota))
            # Source order is retained; scores from different batches are not sorted.
            final_values = tuple(value for value in values if value[0] in chosen)
            try:
                preflight(request(stage, final_values))
            except AgentError as error:
                if error.code not in CAPACITY_ERRORS:
                    raise
                raise AgentError("decisionrag_shared_final_capacity_exceeded") from None
        probabilities = score(request(stage, final_values), "final", 0)
        kept = rank(final_values, probabilities, keep)
        selections.append({"stage": stage, "screening_batches": [[row[0] for row in group] for group in packed],
            "shared_final_candidate_ids": [row[0] for row in final_values], "kept_ids": [row[0] for row in kept]})
        return [row[0] for row in kept]

    try:
        limits.validate()
        if (type(started) not in (int, float) or not math.isfinite(started) or started > time.monotonic()
                or type(query) is not str or not query.strip() or len(query) > 8000
                or type(documents) is not tuple or not 1 <= len(documents) <= limits.max_documents
                or not isinstance(scope_binding, str) or not scope_binding):
            raise AgentError("decisionrag_bounded_server_scope_required")
        validate_user_text(query)
        passages = tuple(passage for document in documents for passage in document.passages)
        if (any(not isinstance(document, Document) or not document.id or not document.description.strip()
                or not document.source_version or not document.passages for document in documents)
                or len({document.id for document in documents}) != len(documents)
                or not 1 <= len(passages) <= limits.max_passages
                or len({passage.id for passage in passages}) != len(passages)
                or len({passage.evidence_ref_id for passage in passages}) != len(passages)
                or any(passage.document_id != document.id or passage.source_version != document.source_version
                    or not passage.id or not passage.text.strip() or not passage.evidence_ref_id
                    for document in documents for passage in document.passages)):
            raise AgentError("decisionrag_actual_source_hierarchy_required")
        document_values = tuple((document.id, document.description) for document in documents)
        passage_values = tuple((passage.id, passage.text) for passage in passages)
        document_batches = batches("documents", document_values)
        # Reject oversized single leaves before doing any inference, including
        # leaves whose document might later be pruned.
        full_passage_batches = batches("passages", passage_values)
        forecast = sum(len(group) + int(len(group) > 1) for group in (document_batches, full_passage_batches))
        if forecast > limits.max_forward_calls:
            raise AgentError("decisionrag_forward_budget_exhausted")
        document_ids = level("documents", document_values, limits.document_keep, document_batches)
        result.update(document_ids_kept=document_ids,
            pruned_document_ids=[document.id for document in documents if document.id not in document_ids])
        selected = tuple(passage for document in documents if document.id in document_ids for passage in document.passages)
        values = tuple((passage.id, passage.text) for passage in selected)
        selected_ids = level("passages", values, limits.passage_keep, batches("passages", values))
        by_id = {passage.id: passage for passage in selected}
        scope()
        result.update(status="ok", passage_candidates=len(selected), selected_passage_ids=selected_ids,
            evidence_ref_ids=[by_id[identifier].evidence_ref_id for identifier in selected_ids])
    except Exception as error:
        result.update(status="error", error_code=error.code if isinstance(error, AgentError)
            else "decisionrag_navigation_internal_error", evidence_ref_ids=[], selected_passage_ids=[])
    return {**result, "decision_calls": None if pending_score else len(trace), "trace": trace,
        "selection_steps": selections, "unknown_outcome": pending_score,
        "elapsed_seconds": time.monotonic() - started}
