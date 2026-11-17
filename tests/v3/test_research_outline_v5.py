"""One pure v5 capacity/identity/input-boundary control; no provider or database."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import canonical, digest
from citequest.agents_v3.research_capacity import research_v5_capacity
from citequest.agents_v3.research_outline import ResearchOutlineFlow
from citequest.agents_v3.research_workflow import ResearchWorkflow


def test_reviewed_outline_reserves_complete_school_and_never_asks_for_known_facts():
    task = {"model_calls": 0, "tool_calls": 0, "output_tokens": 0,
        "brief": {"kind": "research", "selected_papers": [{"kind": "native_representation", "id": f"{n:064x}"}
            for n in range(1, 9)], "research_draft_files": []},
        "profile": {"profile_id": "research-v5", "model_calls": 64, "tool_calls": 80,
            "output_tokens": 131072, "per_call_output": 2048}}
    capacity = research_v5_capacity(task, 2, 4, outline_calls=2, remaining_revisions=2)
    assert capacity["admitted"] and capacity["required_model_calls"] == 51
    assert capacity["required_tool_calls"] == 66 and capacity["reserved_output_tokens"] == 104448
    assert capacity["school_normal_model_calls"] == 30
    assert not research_v5_capacity({**task, "profile": {**task["profile"], "tool_calls": 64}},
        2, 4, outline_calls=2)["admitted"]

    boundary = ResearchWorkflow.__new__(ResearchWorkflow)
    boundary.strict_identity_binding = boundary.review_outline = True
    boundary.host = SimpleNamespace(execution=SimpleNamespace(snapshot=lambda: task, content_revision_count=lambda: 0))
    flow = ResearchOutlineFlow.__new__(ResearchOutlineFlow)
    flow.boundary, flow.host = boundary, boundary.host
    candidate = {"brief_sha256": "a" * 64, "questions": [{"id": "RQ1", "question": "拟验证方法是否可行？",
        "retrieval_query": "research methods", "experiment_ids": ["E1"], "user_fields": ["topic", "resources"],
        "expected_outputs": ["未执行的实验方案"]}], "innovation_hypotheses": [], "uncovered_user_requests": []}
    state = {"intake": {"brief_sha256": "a" * 64, "user_confirmed": {"topic": "研究主题", "resources": "公开数据"},
        "explicit_unknowns": {}}, "outline_control": {"candidate": candidate, "source_key": "explicit-settled-review",
            "capacity": {}, "review_raw": "", "accepted": False}}
    review = {"brief_sha256": "a" * 64, "content_sha256": digest(candidate),
        "question_opinions": {"RQ1": {"verdict": "passed", "reason": "Supplied objectives are represented."}},
        "field_coverage": {field: {"state": "represented", "question_ids": ["RQ1"], "reason": "Uses the confirmed field."}
            for field in ("topic", "resources")}, "missing_user_fields": [],
        "unsupported_user_fact_assumptions": False, "decision": "accepted", "reason": "Source-free intent only."}
    model = flow.review_model(state)
    with pytest.raises(ValidationError):
        model.model_validate({**review, "content_sha256": "b" * 64}, strict=True)
    state["outline_control"]["review_raw"] = canonical(review).decode()
    assert flow.validate_review(state)["outline_control"]["accepted"]
    state["outline_control"]["review_raw"] = canonical({**review, "decision": "needs_input",
        "missing_user_fields": ["resources"]}).decode()
    rejected = flow.validate_review(state)["outline_control"]
    assert rejected["route"] == "reserve" and rejected["target"] == "review"
    assert rejected["feedback"]["error_codes"] == ["research_outline_review_fact_or_identity_invalid"]
