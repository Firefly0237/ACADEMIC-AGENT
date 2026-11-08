"""Official graph/selector controls with explicit model and ledger fixtures."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.research_literature import ResearchLiteratureFlow
from citequest.agents_v3.research_workflow import ResearchWorkflow
from citequest.research_v3.brief import REQUIRED
from citequest.research_v3.literature_grounding import LiteratureGrounding


def source() -> dict:
    text = "EXPLICIT SOURCE FIXTURE: this hypothesis is unverified. Formula glyph \uf8ff remains."
    return {"ref_id": "TASK-ISSUED:fragment-1", "source_ref": {"kind": "native_representation", "id": "a" * 64},
        "text": text, "span_start": 11, "span_end": 11 + len(text),
        "coordinate_system": "native_literal_unicode_codepoints", "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "unit_text_sha256": "b" * 64}


class LedgerFixture:
    def __init__(self) -> None:
        self.task_id = uuid4()
        self.command = {"id": uuid4()}
        self.database = None
        facts = {name: "Explicit synthetic constraint: " + name for name in REQUIRED}
        self.intake = {"brief_id": str(uuid4()), "brief_revision": 1, "brief_sha256": digest(facts),
            "user_confirmed": facts, "explicit_unknowns": {}, "selected_sources": []}
        self.value = {"profile": {"profile_id": "research-v3", "model_calls": 64, "tool_calls": 64,
            "output_tokens": 96000, "input_tokens": 512000, "active_seconds": 1800, "cash_ncny": 80000000000,
            "per_call_output": 2048, "request_bytes": 131072}, "brief": {"question": "明确合成研究问题"},
            "graph_version": "citequest-agent-graph/17-server-Q-literature-and-shared-revision",
            "profile_hash": digest("EXPLICIT-FIXTURE"), "owner_id": uuid4(), "project_id": uuid4(),
            "model_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
        self.known: set[str] = set()
        self.credits: dict[str, dict] = {}
        self.unknown = False
        self.requests: dict[str, dict] = {}

    def snapshot(self) -> dict:
        return copy.deepcopy(self.value)

    def progress(self, *_args: str) -> None:
        pass

    def content_revision_count(self) -> int:
        return len(self.credits)

    def settled_model_request(self, key: str) -> dict | None:
        if key not in self.requests:
            return None
        payload = self.requests[key]["payload"]
        return {"request": copy.deepcopy(payload), "request_hash": digest(payload)}

    def reserve_content_revision(self, key: str, *, phase: str, error_codes: list[str], source_logical_key: str) -> dict:
        if key in self.credits:
            return self.credits[key]
        if self.unknown:
            raise AgentError("attempt_outcome_unresolved")
        if source_logical_key not in self.known:
            raise AgentError("content_revision_requires_known_receipt")
        if len(self.credits) >= 2:
            raise AgentError("content_revision_limit")
        value = {"logical_key": key, "ordinal": len(self.credits) + 1, "limit": 2,
            "source": source_logical_key, "phase": phase, "error_codes": error_codes}
        self.credits[key] = value
        return value


class ProviderFixture:
    def __init__(self, ledger: LedgerFixture, *, failure: str = "", repeat: bool = False) -> None:
        self.ledger, self.failure, self.repeat = ledger, failure, repeat
        self.requests: dict[str, dict] = {}
        self.ledger.requests = self.requests
        self.calls: list[str] = []
        self.author_calls = self.review_calls = 0

    async def complete(self, payload: dict, key: str) -> dict:
        if key in self.requests:
            assert self.requests[key]["payload"] == payload
            return self.requests[key]["response"]
        if self.failure == "unknown":
            self.ledger.unknown = True
            raise AgentError("provider_usage_missing")
        prompt = json.loads(payload["input"][0]["content"])
        reviewer = prompt["role"] == "reviewer"
        self.calls.append(key)
        if not reviewer:
            self.author_calls += 1
            quote = sorted(LiteratureGrounding([source()]).scope.allowed)[0]
            value = {"kind": "qa", "requested_direction": "neutral", "claims": [{"id": "C1", "statement": "该合成假设尚未验证。",
                "status": "supported", "edges": [{"quote_id": quote, "relation": "supports", "rationale": "明确合成意见。", "conditions": []}],
                "limitations": ["明确合成来源，不代表真实研究结论。"]}, {"id": "C2", "statement": "来源包含保留的公式字符。", "status": "supported",
                "edges": [{"quote_id": quote, "relation": "supports", "rationale": "明确合成意见。", "conditions": []}], "limitations": []}],
                "gaps": [], "retrieval_scope": "仅明确合成片段。", "evidence_status": "supported"}
            if self.failure == "invalid_status" and (self.author_calls == 1 or self.repeat):
                value["claims"][0]["status"] = "support"
            if self.failure == "change_unaffected" and self.author_calls > 1:
                value["claims"][1]["statement"] = "擅自改变未拒绝的第二个命题。"
        else:
            self.review_calls += 1
            value = {"claims": {identifier: {"verdict": "entailed", "conditions_preserved": True,
                "explanation": "明确合成意见。", "quotes": {quote: {"relation": "matches_declared_relations", "conditions_preserved": True,
                    "explanation": "明确合成意见。"} for quote in quotes}} for identifier, quotes in prompt["review_grid"]["claim_quote_ids"].items()},
                "source_instruction_injection_detected": False, "requested_stance_bias_detected": False,
                "request_coverage": "complete", "uncovered_requests": []}
            if self.failure == "missing_quote" and self.review_calls == 1:
                value["claims"]["C1"]["quotes"] = {}
            if self.failure in {"semantic", "change_unaffected"} and (self.review_calls == 1 or self.repeat):
                value["claims"]["C1"]["verdict"] = "invalid"
            if self.failure == "injection":
                value["source_instruction_injection_detected"] = True
        response = {"output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": json.dumps(value)}]}]}
        self.requests[key] = {"payload": payload, "response": response}
        self.ledger.known.add(key)
        self.ledger.value["model_calls"] += 1
        return response


def fixture(*, failure: str = "", repeat: bool = False) -> tuple[ResearchWorkflow, dict, ProviderFixture]:
    ledger = LedgerFixture()
    host = SimpleNamespace(execution=ledger, memory=SimpleNamespace(current=lambda *_args: []), wait=asyncio.run)
    provider = host.provider = ProviderFixture(ledger, failure=failure, repeat=repeat)
    boundary = ResearchWorkflow(host, lambda: copy.deepcopy(ledger.intake), writing_reserve_model_calls=0)
    intake = boundary.current_input().model_dump(mode="json")
    anchor = {"intake": intake, "hard_memory_refs": [], "current_evidence": [source()], "evidence": [source()], "literature": [], "handoffs": [],
        "rq_index": 0, "outline": {"questions": [{"id": "RQ1", "question": "该假设是否已验证，来源中的字符是什么？"}]}}
    return boundary, anchor, provider


def run(boundary: ResearchWorkflow, anchor: dict) -> dict:
    flow = ResearchLiteratureFlow(boundary, checkpointer=InMemorySaver())
    return flow.workflow.invoke({"anchor": anchor}, {"configurable": {"thread_id": str(uuid4())}}, durability="sync")["outcome"]


def test_server_quote_and_independent_grid_are_on_the_actual_child_graph() -> None:
    boundary, anchor, provider = fixture()
    result = run(boundary, anchor)
    assert result["accepted"] and len(provider.calls) == 2
    edge = result["candidate"]["claims"][0]["edges"][0]
    assert edge["ref_id"] == source()["ref_id"] and "\uf8ff" in edge["quote"]
    assert edge["quote_locator"]["span_start"] == 11
    review_prompt = json.loads(list(provider.requests.values())[1]["payload"]["input"][0]["content"])
    assert review_prompt["selected_user_skills"] == []
    assert set(review_prompt["review_grid"]["claim_quote_ids"]) == {"C1", "C2"}


@pytest.mark.parametrize("failure", ["invalid_status", "missing_quote", "semantic"])
def test_known_content_failure_uses_one_shared_receipt_bound_credit(failure: str) -> None:
    boundary, anchor, provider = fixture(failure=failure)
    result = run(boundary, anchor)
    assert result["accepted"] and len(boundary.host.execution.credits) == 1
    credit = next(iter(boundary.host.execution.credits.values()))
    assert credit["source"] in provider.requests and credit["limit"] == 2
    assert result["candidate"]["claims"][1]["statement"] == "来源包含保留的公式字符。"


def test_unaffected_claim_cannot_be_rewritten_during_directed_repair() -> None:
    boundary, anchor, _provider = fixture(failure="change_unaffected")
    result = run(boundary, anchor)
    assert not result["accepted"] and result["candidate"] is None
    assert result["audit"][-1]["error_code"] == "typed_feedback_unaffected_claim_changed"


def test_two_credits_are_task_wide_and_exhaustion_never_publishes_candidate() -> None:
    boundary, anchor, provider = fixture(failure="invalid_status", repeat=True)
    result = run(boundary, anchor)
    assert not result["accepted"] and result["candidate"] is None
    assert len(boundary.host.execution.credits) == 2 and provider.author_calls == 3
    assert result["audit"][-1]["error_code"] == "research_shared_content_revision_limit"


def test_transport_unknown_never_enters_content_revision() -> None:
    boundary, anchor, _provider = fixture(failure="unknown")
    with pytest.raises(AgentError, match="provider_usage_missing"):
        run(boundary, anchor)
    assert not boundary.host.execution.credits


def test_review_source_injection_rejects_without_trying_to_rewrite_it_away() -> None:
    boundary, anchor, provider = fixture(failure="injection")
    result = run(boundary, anchor)
    assert not result["accepted"] and not boundary.host.execution.credits and len(provider.calls) == 2


def test_checkpoint_resume_after_credit_commit_uses_same_credit_and_model_identity() -> None:
    boundary, anchor, provider = fixture(failure="invalid_status")
    saver = InMemorySaver()
    flow = ResearchLiteratureFlow(boundary, checkpointer=saver)
    original = flow.author
    # The compiled node captures this method via observe; construct after patch.
    crash = {"pending": True}
    class CrashFlow(ResearchLiteratureFlow):
        def author(self, state: dict) -> dict:
            if state["revision"] == 1 and crash["pending"]:
                crash["pending"] = False
                raise RuntimeError("EXPLICIT interruption after durable revision")
            return original(state)
    flow = CrashFlow(boundary, checkpointer=saver)
    config = {"configurable": {"thread_id": str(uuid4())}}
    with pytest.raises(RuntimeError, match="EXPLICIT interruption"):
        flow.workflow.invoke({"anchor": anchor}, config, durability="sync")
    restored = ResearchLiteratureFlow(boundary, checkpointer=saver)
    result = restored.workflow.invoke(None, config, durability="sync")["outcome"]
    assert result["accepted"] and len(boundary.host.execution.credits) == 1
    assert provider.author_calls == 2 and provider.review_calls == 1


def test_settled_author_replay_reconstructs_original_cap_after_output_budget_exhausted() -> None:
    boundary, anchor, provider = fixture()
    flow = ResearchLiteratureFlow(boundary, checkpointer=InMemorySaver())
    initial = flow.initialize({"anchor": anchor})
    first = flow.author(initial)
    boundary.host.execution.value["output_tokens"] = boundary.host.execution.value["profile"]["output_tokens"]
    replay = flow.author(initial)
    assert replay == first and len(provider.calls) == 1
    changed = copy.deepcopy(initial)
    changed["anchor"]["outline"]["questions"][0]["question"] = "未经新命令更换相同逻辑键的问题。"
    with pytest.raises(AgentError, match="research_settled_request_changed"):
        flow.author(changed)
    assert len(provider.calls) == 1


def test_child_propagates_changed_intake_before_another_dispatch_or_revision() -> None:
    from citequest.agents_v3.research_workflow import IntakeChanged
    boundary, anchor, provider = fixture()
    flow = ResearchLiteratureFlow(boundary, checkpointer=InMemorySaver())
    state = flow.initialize({"anchor": anchor})
    state.update(flow.author(state))
    boundary.host.execution.intake["brief_revision"] += 1
    boundary.host.execution.intake["brief_sha256"] = "c" * 64
    with pytest.raises(IntakeChanged):
        flow.observe("validate_author", flow.validate_author)(state)
    assert len(provider.calls) == 1 and not boundary.host.execution.credits


def test_parent_node_uses_inherited_official_child_saver_and_retains_quote_grid() -> None:
    from langgraph.graph import END, START, StateGraph

    from citequest.agents_v3.research_workflow import ResearchState
    boundary, anchor, provider = fixture()
    builder = StateGraph(ResearchState)
    builder.add_node("prepare", boundary.observe("literature_v3_prepare", boundary.literature_v3_prepare))
    builder.add_node("literature_v3", boundary.literature_flow.workflow)
    builder.add_node("accept", boundary.observe("literature_v3_accept", boundary.literature_v3_accept))
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "literature_v3")
    builder.add_edge("literature_v3", "accept")
    builder.add_edge("accept", END)
    graph = builder.compile(checkpointer=InMemorySaver())
    assert "literature_v3" in dict(graph.get_subgraphs())
    state = graph.invoke(anchor, {"configurable": {"thread_id": str(uuid4())}}, durability="sync")
    artifact = state["literature"][0]
    assert artifact["review_grid"]["opinions"]["claims"]["C1"]["quotes"]
    assert artifact["source_protocol"]["binding_sha256"] and len(provider.calls) == 2
    projected = boundary.project_result(state, {"status": "partial"})
    assert projected["context_manifest"]["literature_protocol_bindings"][0]["review_grid_binding_sha256"] == artifact["review_grid"]["binding"]["binding_sha256"]


def test_literature_and_plan_spend_shared_credits_then_school_repair_is_blocked() -> None:
    from citequest.agents_v3.research_workflow import TargetRepair
    from citequest.agents_v3.school_document import SchoolDocumentWorkflow
    from citequest.research_v3.research_plan import ResearchPlan
    boundary, anchor, provider = fixture(failure="invalid_status")
    outcome = run(boundary, anchor)
    artifact = boundary.accept_literature(anchor, outcome["candidate"], outcome["review"], audit=outcome["audit"],
        source_protocol=outcome["source_protocol"], review_grid=outcome["review_grid"])
    question = {"id": "RQ1", "question": "明确合成拟议问题", "motivation_claim_ids": ["RQ1:C1"],
        "experiment_ids": ["E1"], "expected_outputs": ["拟议对照，不是已完成实验"], "known_gaps": []}
    experiment = {"id": "E1", "research_question_id": "RQ1", "evidence_ref_ids": [source()["ref_id"]],
        **dict.fromkeys(("hypothesis", "data_and_access", "method_and_independent_variables", "baselines_and_controls",
            "metrics_and_units", "repetitions_and_uncertainty", "resources_and_upper_bound", "risks_and_stopping_rules"), "明确合成拟议约束")}
    plan = ResearchPlan(brief_sha256=anchor["intake"]["brief_sha256"], questions=[question], experiments=[experiment],
        innovation_hypotheses=[], literature_gaps=[], feasibility="proposed_feasible", feasibility_reasons=["明确合成资源意见"])
    state = {**anchor, **artifact, "plan": plan.model_dump(), "repair_targets": ["RQ1"], "repair_index": 0,
        "revision_round": 1, "unit_reviews": [], "global_review": {"conflicts": []}, "review_index": 0}
    review_key = boundary.model_key(state, "global-review:0")
    boundary.host.execution.known.add(review_key)  # Explicit known-receipt ledger fixture.
    repair = TargetRepair(brief_sha256=plan.brief_sha256, target_id="RQ1", replacement=plan.questions[0],
        resolved_conflict_ids=[], evidence_ref_ids=[source()["ref_id"]], basis="current_evidence_and_user_constraints", public_reason="明确合成修订")
    with patch.object(boundary, "complete", return_value=repair) as model:
        revised = boundary.revise_target(state)
    assert revised["next"] == "review_unit" and model.call_count == 1
    assert [item["phase"] for item in boundary.host.execution.credits.values()] == ["research_literature_content_revision", "research_plan_content_revision"]
    school = SchoolDocumentWorkflow(boundary)
    school_state = {"index": 0, "specs": [{}] * 14, "repairs": 0, "writing_start_model_calls": boundary.task()["model_calls"], "sections": {}}
    stopped = school.schedule_repair(school_state, "review_scope", ["explicit_fixture_failure"], "advance")
    assert stopped["result"]["school_writing"]["reason"] == "research_shared_content_revision_limit:review_scope"
    assert len(provider.calls) == 3 and len(boundary.host.execution.credits) == 2
