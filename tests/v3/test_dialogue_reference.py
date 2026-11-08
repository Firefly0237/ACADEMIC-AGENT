"""Atomic prior-assistant references; fixtures are synthetic and never evidence."""
import copy
import json
from types import SimpleNamespace
from uuid import uuid4

from citequest.agents_v3.context import fit_context, wire_bytes
from citequest.agents_v3.contracts import canonical
from citequest.agents_v3.dialogue_reference import VERSION, add_reference, collect_reference, record_reference_admission


def test_reference_unpublished_or_no_current_authorizer_does_not_expose_a_body():
    graph = SimpleNamespace()
    assert collect_reference(graph, {"history": []}, {})["status"] == "no_previous_published_answer"
    hidden = {"history": [{"answer": {"publication_status": "rejected", "claims": [{"statement": "PRIVATE"}]}}]}
    assert "PRIVATE" not in json.dumps(collect_reference(graph, hidden, {}))
    hidden["history"][0]["answer"]["publication_status"] = "validated_ai_review"
    assert collect_reference(graph, hidden, {})["reason"] == "current_reference_authorizer_not_configured"


def test_whole_reference_target_uses_existing_wire_budget_without_truncation():
    statement = "Synthetic previous assistant statement. " * 50
    reference = {"version": VERSION, "status": "currently_authorized", "target": {
        "id": "assistant-answer:" + "a" * 64, "answer_sha256": "a" * 64,
        "claims": [{"claim_id": "C1", "statement": statement}], "factual_evidence": False,
        "active_user_instructions": False}}
    prompt = add_reference({"context_manifest": {}, "canonical_brief": {"question": "Which previous claim?"}}, reference)
    request = {"input": [{"role": "user", "content": canonical(prompt).decode()}], "instructions": "fixture"}
    large = record_reference_admission(fit_context(request, 16000))
    included = json.loads(large["input"][0]["content"])
    assert included["prior_turns"][0]["assistant_reference_target"]["claims"][0]["statement"] == statement
    assert included["context_manifest"]["dialogue_reference"]["wire_status"] == "included_complete"
    # Determine actual hard request bytes with the production encoder/builder.
    mandatory = copy.deepcopy(prompt)
    mandatory["prior_turns"] = []
    baseline = {**request, "input": [{"role": "user", "content": canonical(mandatory).decode()}]}
    budget = wire_bytes(fit_context(baseline, 16000)) + 120
    fitted = fit_context(request, budget)
    small = record_reference_admission(fitted)
    excluded = json.loads(small["input"][0]["content"])
    assert excluded["prior_turns"] == []
    assert excluded["context_manifest"]["dialogue_reference"]["wire_status"] == "excluded_by_wire_budget"
    assert wire_bytes(small) <= wire_bytes(fitted) <= budget
    assert excluded["context_manifest"]["context_excluded"][0]["kind"] == "prior_answer"


def test_collect_reuses_actual_current_result_projection_and_denies_revoked_evidence():
    # Requires the ordinary product dependency environment, but no SQL/network.
    from citequest.agents_v3.catalog_scope import POLICY, graph_version
    from citequest.agents_v3.contracts import PROFILES, ResearchBrief, RuntimeLimits

    owner, project, task_id = uuid4(), uuid4(), uuid4()
    previous = {"kind": "qa", "publication_status": "validated_ai_review", "completion_status": "bounded_complete",
        "claims": [{"id": "C1", "statement": "Synthetic published statement"}],
        "text": "private prose must not be copied", "citations": [{"ref_id": "E1", "quote": "private quote"}],
        "context_manifest": {"memory_refs": [], "skill_refs": [], "evidence_refs": [{"ref_id": "E1"}]}}
    task = {"id": task_id, "owner_id": owner, "project_id": project, "kind": "qa", "revision": 2, "state": "running",
        "brief": ResearchBrief(kind="qa", question="Which earlier claim?").model_dump(mode="json"),
        "profile": {**PROFILES["qa-v6"].model_dump(), "catalog_scope_policy": POLICY},
        "graph_version": graph_version("qa", "qa-v6", POLICY), "history": [{"answer": previous, "brief_revision": 1}],
        "result": None, "error_code": None, "event_seq": 1, "cancellation_requested": False,
        **dict.fromkeys(("model_calls", "tool_calls", "input_tokens", "output_tokens", "known_ncny", "reserved_ncny", "active_seconds"), 0)}
    called = []

    def allow(actor, identifier, refs):
        called.append((actor.owner_id, identifier, refs))

    graph = SimpleNamespace(execution=SimpleNamespace(database=SimpleNamespace(catalog_scope_policy=POLICY), limits=RuntimeLimits(mode="bounded")),
        memory=SimpleNamespace(current=lambda *_: [], validate_result_context=lambda *_: None), authorize_dialogue_refs=allow)
    visible = collect_reference(graph, task, {})
    assert called == [(owner, task_id, ["E1"])]
    assert visible["target"]["claims"] == [{"claim_id": "C1", "statement": "Synthetic published statement"}]
    assert "private" not in json.dumps(visible)

    def deny(*_):
        raise RuntimeError("revoked")

    graph.authorize_dialogue_refs = deny
    denied = collect_reference(graph, task, {})
    assert denied == {"version": VERSION, "status": "unavailable", "reason": "evidence_unavailable_or_access_revoked"}
