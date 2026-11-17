"""Prepared Workset5 source-preserving batching, raw packs and resource math.

No execution, model client, retrieval fallback or source store exists here.
Original FullSourceWindows responses must be authorized/read by the caller;
grouping calls its actual full-payload encoder before accepting a group.
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass

from citequest.agents_v3.contracts import AgentError, digest

from .quote_spans import QuoteRegistry, QuoteScope

VERSION = "citequest-workset-planning/5"
REQUEST_BYTES = 131072


@dataclass(frozen=True)
class WindowGroup:
    group_id: str
    window_ids: tuple[str, ...]
    wire_bytes: int


def group_windows(window_ids: list[str], read_windows: dict[str, dict], encoded_bytes: Callable[[list[dict]], int],
        *, maximum: int = REQUEST_BYTES) -> list[WindowGroup]:
    """Greedy deterministic groups of up to four complete original windows.

    The callback must build the same full request used by the eventual caller,
    including schema, current requirements, Skills/Memory and source identities.
    The maximum is not interpreted as a token count. Oversized single windows
    fail explicitly; no text or source part is removed to make them fit.
    """
    if (not window_ids or len(set(window_ids)) != len(window_ids) or set(read_windows) != set(window_ids)
            or type(maximum) is not int or maximum < 1):
        raise AgentError("workset_group_window_manifest_mismatch")
    for identifier in window_ids:
        if not isinstance(read_windows[identifier], dict) or not read_windows[identifier].get("evidence"):
            raise AgentError("workset_group_original_window_unread")
    groups, cursor = [], 0
    while cursor < len(window_ids):
        selected, size = None, None
        for count in range(min(4, len(window_ids) - cursor), 0, -1):
            candidate = window_ids[cursor:cursor + count]
            size = encoded_bytes([copy.deepcopy(read_windows[key]) for key in candidate])
            if type(size) is not int or size < 1:
                raise AgentError("workset_group_encoder_result_invalid")
            if size <= maximum:
                selected = candidate
                break
        if selected is None:
            raise AgentError("workset_group_single_window_exceeds_request_capacity", 413)
        groups.append(WindowGroup(digest({"version": VERSION, "windows": selected,
            "response_hashes": [digest(read_windows[key]) for key in selected]}), tuple(selected), size))
        cursor += len(selected)
    if [key for group in groups for key in group.window_ids] != window_ids:
        raise AgentError("workset_group_original_window_coverage_changed")
    return groups


def raw_quote_pack(registry: QuoteRegistry, edges: list[dict], *, margin: int = 160,
        minimum_documents: int = 2) -> tuple[QuoteScope, list[str]]:
    """Use canonical exact locators to extract bounded original context.

    Source objects/ref IDs/hashes stay original. context_span_* selects a view
    into the already read source buffer, which QuoteScope independently checks.
    The returned parent refs must still be passed through the caller's existing
    fresh verify_evidence_batch and provider scope guard. Summary prose is never
    accepted as source text. Duplicate wording uses its exact locator, not find().
    """
    if not 1 <= len(edges) <= 5 or type(margin) is not int or not 0 <= margin <= 160:
        raise AgentError("workset_atomic_evidence_pack_bounds")
    full_scope = registry.scope(list(registry.sources.values()))
    contexts, parents, documents, identities = [], set(), set(), set()
    for edge in edges:
        if not isinstance(edge.get("quote_locator"), dict):
            raise AgentError("workset_atomic_exact_quote_locator_required")
        ref, start, end = full_scope.locate(edge)
        identity = (ref, start, end)
        if identity in identities:
            raise AgentError("workset_atomic_duplicate_evidence_locator")
        identities.add(identity)
        source = registry.sources[ref]
        document = source.get("document_id")
        if not isinstance(document, str) or not document:
            raise AgentError("workset_atomic_document_identity_missing")
        documents.add(document)
        parents.add(ref)
        left, right = max(source["span_start"], start - margin), min(source["span_end"], end + margin)
        contexts.append({**copy.deepcopy(source), "context_span_start": left, "context_span_end": right,
            "text": source["text"][left - source["span_start"]:right - source["span_start"]],
            "review_scope": "exact_cited_span_and_bounded_original_context"})
    if len(documents) < minimum_documents:
        raise AgentError("workset_atomic_comparison_requires_distinct_documents")
    # Supplying candidates binds every selected full span even when splitter
    # boundaries divide it. QuoteScope verifies all locator/version fields.
    scope = registry.scope(contexts, candidates=[{"claims": [{"id": "C1", "edges": copy.deepcopy(edges)}]}])
    return scope, sorted(parents)


def resource_plan(*, documents: int, windows: int, groups: int, comparison_units: int,
        source_parts: int, revision_credits: int = 2, kind: str = "compare",
        review_output_cap: int | None = None) -> dict:
    if (kind not in {"summary", "compare"} or any(type(x) is not int for x in
            (documents, windows, groups, comparison_units, source_parts, revision_credits))
            or not 1 <= documents <= 10 or not documents <= groups <= windows
            or not groups <= windows <= 4 * groups or not source_parts >= windows
            or not 0 <= comparison_units <= 12 or not 0 <= revision_credits <= 2
            or kind == "summary" and comparison_units != 0 or kind == "compare" and not 1 <= comparison_units <= 12
            or review_output_cap is not None and (type(review_output_cap) is not int or not 2048 <= review_output_cap <= 4096)):
        raise AgentError("workset_v5_resource_dimensions_invalid")
    document_review = 1536 if review_output_cap is None else review_output_cap
    atomic_review = 2048 if review_output_cap is None else review_output_cap
    revision_review = 2048 if review_output_cap is None else review_output_cap
    doc_output = (2 * groups - documents) * 3072 + documents * document_review
    if kind == "compare":
        models = 2 * groups + 2 * comparison_units + 3 + 2 * revision_credits
        output = doc_output + 3072 + 2048 + 3072 + comparison_units * (3072 + atomic_review) + revision_credits * (4096 + revision_review)
    else:
        models = 2 * groups + 1 + 2 * revision_credits
        output = doc_output + 3072 + revision_credits * (4096 + revision_review)
    tools = 1 + windows + comparison_units + revision_credits + (source_parts + 63) // 64
    return {"schema_version": VERSION, "kind": kind, "dimensions": {"documents": documents, "windows": windows,
        "groups": groups, "comparison_units": comparison_units, "source_parts": source_parts, "unused_revision_credits": revision_credits},
        "required_models": models, "required_tools": tools, "reserved_output_tokens": output,
        "candidate_model_limit": 64, "candidate_tool_limit": 64, "candidate_output_limit": 160000,
        "capacity_admitted": models <= 64 and tools <= 64 and output <= 160000,
        "remaining_checks": ["actual_input_usage_accounting", "complete_encoder_per_actual_request", "same_task_active_600_seconds",
            "known_usage_and_shared_campaign_wallet", "current_source_memory_skill_snapshot_guards", "independent_content_review"],
        "latency_or_semantic_success_guaranteed": False}
