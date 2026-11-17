"""Pure capacity and complete-delivery contracts; no inference or database."""
import copy

import pytest

from citequest.agents_v3.contracts import AgentError
from citequest.agents_v3.research_capacity import RESEARCH_OUTPUT_CAP, SCHOOL_NORMAL_CALLS, research_v5_capacity, reviewed_outline_capacity
from citequest.research_v3.experiment_delivery import EXPERIMENT_FIELDS, experiment_result
from citequest.research_v3.research_plan import Experiment, ResearchPlan


def task(profile_id="research-v6", action="experiment"):
    return {"kind": "research", "profile": {"profile_id": profile_id, "model_calls": 1000,
        "tool_calls": 1000, "output_tokens": 2000000, "per_call_output": 2048},
        "model_calls": 0, "tool_calls": 0, "output_tokens": 0,
        "brief": {"kind": "research", "research_action": action, "selected_papers": []}}


def reviewed():
    experiment = Experiment(id="E1", research_question_id="RQ1", evidence_ref_ids=["synthetic:source:1"],
        **{field: f"{field}: α = β + 1；拟议设计。" for field, _ in EXPERIMENT_FIELDS})
    plan = ResearchPlan(brief_sha256="a" * 64, questions=[{"id": "RQ1", "question": "合成研究问题？",
        "motivation_claim_ids": ["RQ1:C1"], "experiment_ids": ["E1"],
        "expected_outputs": ["拟交付可复现实验记录。"], "known_gaps": ["数据访问待确认。"]}],
        experiments=[experiment], innovation_hypotheses=["待证实的合成假设。"],
        literature_gaps=["合成文献缺口。"], feasibility="proposed_feasible", feasibility_reasons=["仅为拟议方案。"])
    identity = {"brief_id": "synthetic-brief", "brief_revision": 1, "brief_sha256": "a" * 64}
    return {"status": "structured_plan_ready", "publication_status": "validated_ai_review",
        "performed_experiments": False, "plan": plan.model_dump(mode="json"),
        "research_intake": {**identity, "explicit_unknowns": {"data_sources": "用户尚不确定数据来源。"}},
        "context_manifest": identity, "citations": [{"ref_id": "synthetic:source:1", "quote": "原始合成引文。"}],
        "literature_ledger": [{"research_question_id": "RQ1", "answer": {"claims": [{"id": "C1",
            "statement": "受限定的合成文献声明。", "status": "supported", "limitations": ["适用范围有限。"],
            "edges": [{"ref_id": "synthetic:source:1", "relation": "supports", "quote": "原始合成引文。",
                "rationale": "合成理由。", "conditions": ["指定条件下。"]}]}], "gaps": ["仍有证据缺口。"]}}]}


def test_experiment_removes_only_school_costs_and_old_plan_capacity_is_unchanged():
    old = research_v5_capacity(task("research-v5", "plan"), 2, 3, outline_calls=2)
    assert research_v5_capacity(task(action="plan"), 2, 3, outline_calls=2) == old
    experimental = research_v5_capacity(task(), 2, 3, outline_calls=2)
    assert experimental["required_model_calls"] == old["required_model_calls"] - SCHOOL_NORMAL_CALLS
    assert experimental["reserved_output_tokens"] == old["reserved_output_tokens"] - SCHOOL_NORMAL_CALLS * RESEARCH_OUTPUT_CAP
    assert experimental["required_tool_calls"] == old["required_tool_calls"] - 2 * 2 * 8
    assert experimental["tool_breakdown"]["research_publication_reads"] == 16
    assert experimental["tool_breakdown"]["school_initial_and_final_reads"] == 0
    assert experimental["shared_revision_model_calls"] == old["shared_revision_model_calls"]
    outline = reviewed_outline_capacity(task(), remaining_revisions=2)
    assert outline["school_normal_model_calls_reserved"] == outline["school_normal_output_tokens_reserved"] == 0


def test_delivery_renders_every_experiment_field_and_preserves_unknowns_and_citations():
    source = reviewed()
    original = copy.deepcopy(source)
    result = experiment_result(source)
    assert source == original
    assert result["status"] == result["completion_status"] == "experiment_design_ready"
    assert result["publication_status"] == "validated_ai_review" and result["performed_experiments"] is False
    assert result["citations"] == source["citations"] and result["school_format_ready"] is False
    experiment = source["plan"]["experiments"][0]
    assert {field for field, _ in EXPERIMENT_FIELDS} | {"id", "research_question_id", "evidence_ref_ids", "status"} == set(Experiment.model_fields)
    for field, _ in EXPERIMENT_FIELDS:
        assert experiment[field] in result["text"]
    for value in ("E1", "RQ1", "synthetic:source:1", "proposed_not_executed", "用户尚不确定数据来源。",
                  "原始合成引文。", "数据访问待确认。", "仍有证据缺口。", "指定条件下。"):
        assert value in result["text"]


def test_delivery_rejects_unreviewed_or_mismatched_brief():
    source = reviewed()
    source["publication_status"] = "not_evaluated"
    with pytest.raises(AgentError):
        experiment_result(source)
    source = reviewed()
    source["context_manifest"]["brief_revision"] = 2
    with pytest.raises(AgentError):
        experiment_result(source)
