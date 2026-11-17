"""Bounded probability routing over complete selected-source structure indexes.

Compact descriptions guide routing. Only freshly authorized literal leaves
become evidence; existing retrieval is retained as an independent access path.
"""
from __future__ import annotations

import math
import hashlib
import time

from citequest.agents_v3.contracts import AgentError, canonical, digest
from citequest.v2.domain import DomainError

from .decisionrag import DecisionRequest, validate_receipt
from .decisionrag_batched import shortlist_quotas
from .source_navigation_sources import scope_evidence


class SourceNavigator:
    def __init__(self, sources, config):
        self.sources, self.config = sources, config

    def run(self, query, actor, task_id, selections, initial, judge, register, check_task, *, started=None):
        started = time.monotonic() if started is None else started
        config = self.config
        protocol = getattr(config, "navigation_protocol", "structure1")
        trace, steps, admissions = [], [], []
        issued, visited, routed, pending = [], set(), set(), False
        manifest, loaded = self.sources.capture(actor, task_id, selections, check_task)
        scope_pin = scope_evidence(actor, task_id, manifest)
        register([scope_pin])
        binding = digest(manifest)
        records = {row["document_id"]: row for row in manifest["sources"]}
        all_nodes, node_document = {}, {}
        for document_id, (_, _, structure) in loaded.items():
            for node in structure.nodes:
                all_nodes[node["id"]] = node
                node_document[node["id"]] = document_id
        roots = [loaded[row["document_id"]][2].root_id for row in manifest["sources"]]
        visited.update(roots)
        base = initial[:config.hybrid_keep]

        def guard():
            check_task()
            if time.monotonic()-started >= config.active_seconds:
                raise AgentError("structure_navigation_active_budget_exhausted")
            self.sources.verify(actor, task_id, manifest, selections)
            check_task()

        def score(level, values, phase, batch_index):
            nonlocal pending
            guard()
            if len(trace) >= config.maximum_forward_calls:
                raise AgentError("structure_navigation_forward_budget_exhausted")
            request = DecisionRequest(query, "documents" if level == "documents" else "passages", tuple(values), binding)
            if len(canonical(request.wire_data()[:2])) > config.max_shared_bytes:
                raise AgentError("structure_navigation_request_capacity")
            checks = judge.preflight(request)
            tokens = (sum(row["sequence_tokens"] for row in checks.values()) if isinstance(checks, dict)
                else getattr(judge, "preflight_tokens", {}).get(request.fingerprint()))
            if type(tokens) is not int or tokens < 1:
                raise AgentError("structure_navigation_exact_token_receipt_required")
            if sum(row["complete_input_tokens"] for row in admissions)+tokens > config.local_input_token_limit:
                raise AgentError("structure_navigation_input_budget_exhausted")
            guard()
            admission = {"request_sha256": request.fingerprint(), "complete_input_tokens": tokens,
                "candidate_count": len(values), "level": level, "execution_status": "pending"}
            admissions.append(admission)
            pending = True
            try:
                receipt = judge.score(request)
            except AgentError as error:
                if getattr(error, "forward_not_started", False):
                    pending = False
                    admission["execution_status"] = "not_executed"
                raise
            validate_receipt(request, receipt)
            usage = receipt.usage
            if (usage.get("observed_forward_calls") != 1 or usage.get("truncated") is not False
                    or usage.get("state_tokens_dropped") != 0 or usage.get("input_tokens") != tokens):
                raise AgentError("structure_navigation_usage_receipt_invalid")
            trace.append({"stage": request.stage, "navigation_level": level, "phase": phase,
                "batch_index": batch_index, "request_sha256": request.fingerprint(),
                "candidate_ids": [row[0] for row in values], "probabilities": receipt.probabilities,
                "model_binding": receipt.model_binding, "usage": usage})
            pending = False
            admission["execution_status"] = "completed"
            guard()
            return receipt.probabilities

        def rank(level, identifiers, keep):
            values = [(identifier, all_nodes[identifier]["description"]) for identifier in identifiers
                if all_nodes[identifier]["description"].strip()]
            if not values:
                return []
            if len(values) > config.max_batches_per_level*8:
                raise AgentError("structure_navigation_level_capacity")
            visited.update(identifier for identifier, _ in values)
            batches = [values[index:index+8] for index in range(0, len(values), 8)]
            order = {identifier: index for index, (identifier, _) in enumerate(values)}
            shortlist = values
            if len(batches) > 1:
                quotas = shortlist_quotas([len(group) for group in batches], config.shortlist_count)
                chosen = set()
                for index, (group, quota) in enumerate(zip(batches, quotas, strict=True)):
                    probabilities = score(level, group, "screen", index)
                    chosen.update(identifier for identifier, _ in sorted(group,
                        key=lambda row: (-probabilities[row[0]], order[row[0]]))[:quota])
                shortlist = [row for row in values if row[0] in chosen]
            probabilities = score(level, shortlist, "final", 0)
            result = [identifier for identifier, _ in sorted(shortlist,
                key=lambda row: (-probabilities[row[0]], order[row[0]]))[:keep]
                if protocol == "structure2" or probabilities[identifier] >= config.minimum_route_probability]
            routed.update(result)
            steps.append({"level": level, "screening_batches": [[row[0] for row in group] for group in batches],
                "shared_final_candidate_ids": [row[0] for row in shortlist], "kept_ids": result})
            return result

        status, error_code = "ok", None
        try:
            documents = rank("documents", roots, config.document_keep)
            sections = [child for identifier in documents for child in all_nodes[identifier]["child_ids"]]
            selected_sections = rank("sections", sections, config.section_keep)
            leaves = [child for identifier in selected_sections for child in all_nodes[identifier]["child_ids"]]
            seeds = rank("leaves", leaves, config.passage_keep)
            expand = list(seeds)
            # Adjacent ranges preserve local conditions; never jump to an unselected source.
            for identifier in seeds:
                siblings = all_nodes[all_nodes[identifier]["parent_id"]]["child_ids"]
                index = siblings.index(identifier)
                for direction in (-1, 1):
                    position = index+direction
                    while 0 <= position < len(siblings) and not all_nodes[siblings[position]]["description"].strip():
                        position += direction
                    if 0 <= position < len(siblings) and siblings[position] not in expand:
                        expand.append(siblings[position])
            for identifier in expand[:config.maximum_leaf_reads]:
                guard()
                record = records[node_document[identifier]]
                evidence = self.sources.expand(actor, record, identifier)
                register([evidence])
                issued.append(evidence)
                visited.add(identifier)
                guard()
        except Exception as error:
            error_code = error.code if isinstance(error, (AgentError, DomainError)) else "structure_navigation_internal_failure"
            safe_capacity = error_code in {
                "structure_navigation_active_budget_exhausted", "structure_navigation_forward_budget_exhausted",
                "structure_navigation_request_capacity", "structure_navigation_level_capacity",
                "structure_navigation_input_budget_exhausted", "laya_complete_context_exceeds_token_budget",
                "navigation_question_rows_capacity", "navigation_sequence_length_capacity", "navigation_padded_shape_capacity",
                "navigation_attention_shape_capacity", "navigation_estimated_peak_memory_capacity",
                "navigation_remaining_deadline_capacity", "navigation_calibrated_latency_exceeds_deadline",
                "navigation_latency_calibration_required", "navigation_concurrency_capacity", "navigation_deadline_context_required",
            }
            status = "partial" if not pending and safe_capacity else "error"
        guard_after_error = status == "partial" and not pending
        if guard_after_error:
            # Capacity failure may retain retrieval; authority failure may not.
            check_task()
            self.sources.verify(actor, task_id, manifest, selections)
        if status == "ok":
            guard()
        result_evidence = []
        for item in [*base, *issued]:
            if item["ref_id"] not in {row["ref_id"] for row in result_evidence}:
                result_evidence.append(item)
        if status == "error":
            result_evidence = []

        def leaf_refs(identifier):
            node = all_nodes[identifier]
            span = node.get("source_span")
            if not span:
                return []
            document_id = node_document[identifier]
            source = loaded[document_id][1]
            unit = source.units[span["unit_index"]]
            return [row["ref_id"] for row in result_evidence if row.get("source_ref") == unit.reference
                and row.get("span_start", -1) < span["span_end"] and row.get("span_end", -1) > span["span_start"]]

        def fully_returned(identifier, refs):
            span = all_nodes[identifier].get("source_span")
            if not span:
                return False
            cursor = span["span_start"]
            intervals = sorted((row["span_start"], row["span_end"]) for row in result_evidence if row["ref_id"] in refs)
            for start, end in intervals:
                if start > cursor:
                    return False
                cursor = max(cursor, end)
                if cursor >= span["span_end"]:
                    return True
            return False

        refs_by_node = {identifier: leaf_refs(identifier) for identifier, node in all_nodes.items() if node["kind"] == "leaf"}
        def node_refs(identifier):
            if identifier in refs_by_node:
                return refs_by_node[identifier]
            values = [ref for child in all_nodes[identifier]["child_ids"] for ref in node_refs(child)]
            refs_by_node[identifier] = list(dict.fromkeys(values))
            return refs_by_node[identifier]
        display = set(roots) | visited
        for identifier in list(display):
            parent = all_nodes[identifier]["parent_id"]
            while parent:
                display.add(parent)
                parent = all_nodes[parent]["parent_id"]
        display_nodes = []
        for identifier, node in all_nodes.items():
            if identifier not in display:
                continue
            document_id = node_document[identifier]
            refs = node_refs(identifier)
            kind = "passage" if node["kind"] == "leaf" else node["kind"]
            reason = "capacity" if error_code else "not_selected" if identifier not in routed else "not_requested"
            display_nodes.append({"id": identifier, "parent_id": node["parent_id"], "kind": kind,
                "label": (records[document_id]["title"] if kind == "document" else node.get("title"))
                    or ("原文位置" if kind == "passage" else "未命名章节"),
                "read_state": "source_text_returned" if kind == "passage" and fully_returned(identifier, refs) else "partially_read" if refs else "not_read",
                "evidence_ref_ids": refs, "why_not_expanded": None if refs else reason,
                "metadata_unknown": node.get("metadata_unknown", [])})
        known_tokens = sum(row["usage"]["input_tokens"] for row in trace)
        supplemental = [row["ref_id"] for row in issued if row["ref_id"] not in {item["ref_id"] for item in initial}]
        local_usage = {"completed_forward_calls": len(trace), "known_input_tokens": known_tokens,
            "observed_forward_calls": None if pending else len(trace), "input_tokens": None if pending else known_tokens,
            "source_tokens_dropped": None if pending else 0, "external_API_cash_ncny": 0,
            "physical_compute_cost_not_measured": True, "elapsed_seconds": time.monotonic()-started}
        return {"protocol": protocol, "navigation_protocol": protocol, "status": status,
            **({"routing_policy": "rank_then_read/1", "absolute_probability_gate": False,
                "raw_text_requires_independent_review": True} if protocol == "structure2" else {}),
            "error_code": error_code, "unknown_outcome": pending,
            "input_frontier_ref_ids": [row["ref_id"] for row in initial],
            "evidence_ref_ids": [row["ref_id"] for row in result_evidence], "evidence": result_evidence,
            "decision_calls": None if pending else len(trace), "trace": trace, "selection_steps": steps,
            "admissions": admissions, "local_usage": local_usage, "elapsed_seconds": time.monotonic()-started,
            "resource_admission": {"status": "rejected" if status == "partial" else "admitted" if trace else "not_recorded",
                "reason_codes": [error_code] if status == "partial" else []},
            "structure": {"version": "citequest-source-navigation/1", "scope_ref_id": scope_pin["ref_id"],
                "scope": {"documents": len(roots), "sections": sum(row["kind"] == "section" for row in all_nodes.values()),
                    "passages": sum(row["kind"] == "leaf" for row in all_nodes.values()), "coverage": "bounded_current_sources"},
                "nodes": display_nodes[:200], "nodes_total": len(all_nodes), "nodes_truncated": len(display_nodes) > 200,
                "supplemental_evidence_ref_ids": list(dict.fromkeys(supplemental)),
                "limitations": ["范围仅限本次明确选择资料的可用原文；不代表完整文件或整个资料库。",
                    "目录与原文摘录用于路由；已返回原文不表示结论已核验。",
                    "未展开内容不能视为不相关；结构层级缺失时按原文顺序提示分组。",
                    ("相关概率仅用于本轮预算内的候选排序，不设绝对门槛；原文仍需独立核验。"
                     if protocol == "structure2" else "相关概率和路由阈值尚未在各学科完成校准。") ]}}


def validate_structure_receipt(config, body, initial_refs):
    protocol = getattr(config, "navigation_protocol", "structure1")
    if (not isinstance(body, dict) or body.get("protocol") != protocol or body.get("navigation_protocol") != protocol
            or body.get("status") not in {"ok", "partial", "error"}
            or body.get("input_frontier_ref_ids") != initial_refs or type(body.get("unknown_outcome")) is not bool):
        raise AgentError("structure_navigation_receipt_invalid")
    if protocol == "structure2" and (body.get("routing_policy") != "rank_then_read/1"
            or body.get("absolute_probability_gate") is not False or body.get("raw_text_requires_independent_review") is not True):
        raise AgentError("structure_navigation_routing_contract_invalid")
    refs, evidence = body.get("evidence_ref_ids"), body.get("evidence")
    if (not isinstance(refs, list) or len(refs) > config.hybrid_keep+config.maximum_leaf_reads
            or len(set(refs)) != len(refs) or not isinstance(evidence, list)
            or [row.get("ref_id") for row in evidence] != refs):
        raise AgentError("structure_navigation_evidence_receipt_invalid")
    for item in evidence:
        if (not isinstance(item.get("text"), str)
                or hashlib.sha256(item["text"].encode()).hexdigest() != item.get("source_sha256")
                or type(item.get("span_start")) is not int or type(item.get("span_end")) is not int
                or item["span_end"]-item["span_start"] != len(item["text"])):
            raise AgentError("structure_navigation_literal_receipt_invalid")
    trace, usage = body.get("trace"), body.get("local_usage")
    if not isinstance(trace, list) or len(trace) > config.maximum_forward_calls or not isinstance(usage, dict):
        raise AgentError("structure_navigation_usage_receipt_invalid")
    if usage.get("completed_forward_calls") != len(trace):
        raise AgentError("structure_navigation_usage_receipt_invalid")
    for row in trace:
        probabilities = row.get("probabilities", {})
        if (set(probabilities) != set(row["candidate_ids"]) or not 1 <= len(probabilities) <= 8
                or any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities.values())
                or row["usage"].get("observed_forward_calls") != 1 or row["usage"].get("state_tokens_dropped") != 0
                or row["usage"].get("truncated") is not False
                or type(row["usage"].get("input_tokens")) is not int or row["usage"]["input_tokens"] < 1):
            raise AgentError("structure_navigation_trace_invalid")
    known = sum(row["usage"]["input_tokens"] for row in trace)
    if usage.get("known_input_tokens") != known or known > config.local_input_token_limit or usage.get("external_API_cash_ncny") != 0:
        raise AgentError("structure_navigation_usage_receipt_invalid")
    if body["unknown_outcome"]:
        if (body["status"] != "error" or refs or body.get("decision_calls") is not None or usage.get("input_tokens") is not None
                or usage.get("observed_forward_calls") is not None or usage.get("source_tokens_dropped") is not None):
            raise AgentError("structure_navigation_unknown_receipt_invalid")
    elif (body.get("decision_calls") != len(trace) or usage.get("input_tokens") != known
            or usage.get("observed_forward_calls") != len(trace) or usage.get("source_tokens_dropped") != 0):
        raise AgentError("structure_navigation_usage_receipt_invalid")
    if body["status"] == "error" and (refs or evidence):
        raise AgentError("structure_navigation_error_contains_evidence")
    structure = body.get("structure", {})
    if (structure.get("version") != "citequest-source-navigation/1"
            or not isinstance(structure.get("scope_ref_id"), str) or not structure["scope_ref_id"].startswith("structure-scope:")
            or not isinstance(structure.get("nodes"), list) or len(structure["nodes"]) > 200):
        raise AgentError("structure_navigation_scope_receipt_invalid")
