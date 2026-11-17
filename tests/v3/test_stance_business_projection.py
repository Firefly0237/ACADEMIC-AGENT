"""No-provider regressions for actual visible stance gaps and strict payloads."""
from types import SimpleNamespace

from citequest.agents_v3.claim_plan import ClaimPlan, PLAN_POLICY, PLAN_REVIEW_POLICY
from citequest.agents_v3.stance import StanceNodes


def nodes():
    task = {"id": "fixture-stance", "revision": 1, "kind": "stance", "output_tokens": 0,
        "profile": {"output_tokens": 24000, "per_call_output": 1500, "request_bytes": 24576},
        "brief": {"requested_direction": "support"}}
    manifest = {"hard_memory_refs": [], "memory_refs": [], "source_scope": [], "excluded": []}
    projection = {"confirmed_user_memory_data": [], "selected_user_skills": []}
    graph = SimpleNamespace(execution=SimpleNamespace(snapshot=lambda: task),
        projection=lambda **kwargs: (projection.copy(), {**manifest, "excluded": []}))
    return StanceNodes(graph)


def test_partial_stance_exposes_actual_unanswered_and_insufficient_items_to_existing_UI_DTO():
    value = nodes().assemble({"claim_coverage": [{"claim_id": "C1", "state": "reviewed", "status": "insufficient", "evidence_ref_ids": []}],
        "claim_plan": {"claims": [{"id": "C1"}, {"id": "C2"}],
            "unplanned_requests": [{"anchor": {"quote": "也讨论成本"}, "reason": "当前来源没有成本信息"}]},
        "claim_results": [{"id": "C1", "statement": "现有证据能否支持该观点", "status": "insufficient", "edges": [],
            "limitations": ["当前窗口缺少直接比较结果"]}],
        "evidence": [], "plan_context_manifest": {"hard_memory_refs": [], "memory_refs": []}, "plan_review": {},
        "claim_reviews": [{"claims": [], "source_instruction_injection_detected": False, "requested_stance_bias_detected": False}]})["result"]
    assert value["publication_status"] == "validated_ai_review" and value["request_coverage"] == "partial"
    assert value["uncovered_requests"] == value["coverage_gaps"] == value["gaps"]
    text = "\n".join(value["gaps"])
    assert "C2" in text and "资源不足" in text and "也讨论成本" in text and "当前窗口缺少直接比较结果" in text
    assert value["citations"] == [] and value["evidence_status"] == "insufficient"


def test_stance_uses_shared_strict_compiler_before_wire_admission_without_nullable_widening():
    value = nodes().payload({"canonical_brief": {}}, ClaimPlan, "citequest_claim_plan", PLAN_POLICY, 1500)
    schema = value["text"]["format"]["schema"]
    assert set(schema["required"]) == set(schema["properties"]) and schema["additionalProperties"] is False
    for model in schema["$defs"].values():
        if model.get("type") == "object":
            assert set(model["required"]) == set(model["properties"])
    assert "withdrawn request" in PLAN_POLICY and "explicitly withdrawn request" in PLAN_REVIEW_POLICY
    assert value["max_output_tokens"] == 1500 and value["truncation"] == "disabled"
