"""Offline scheduling/receipt contracts; synthetic capacities, no model tokens."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import ModuleType
from uuid import UUID

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError
from citequest.research_v3.decisionrag import DecisionReceipt, Document, Passage, navigate
from citequest.research_v3.decisionrag_laya import MODEL_ID, REVISION, SDK_VERSION, WEIGHTS_SHA256, WHEEL_SHA256
from citequest.research_v3 import decisionrag_task as task_module
from citequest.research_v3.decisionrag_task import DecisionRAGTaskConfig, DecisionRAGTaskConfigV2, validate_task_receipt


MODEL_BINDING = {"provider": "local_author_Laya_SystemOne", "model_id": MODEL_ID,
    "revision": REVISION, "weights_sha256": WEIGHTS_SHA256, "sdk_version": SDK_VERSION,
    "sdk_wheel_sha256": WHEEL_SHA256, "device": "cpu", "backend": "eager",
    "max_len": 8192, "head_max_len": 256, "paper_Jev_1_13_reproduction": False,
    "remote_code": False, "automatic_device_fallback": False,
    "native_default_max_len": 1024, "encoder_max_position_embeddings": 8192,
    "request_context_protocol": "laya2"}


class CapacityFixture:
    """Deterministic unit counts only; deployment uses the real SDK tokenizer."""
    def __init__(self, *, limit=8192, fail_at=None):
        self.limit, self.fail_at = limit, fail_at
        self.preflight_tokens, self.checks, self.calls = {}, {}, []

    def preflight(self, request):
        units = 40 + len(request.query.split()) + sum(len(text.split()) + 4 for _, text in request.candidates)
        if len(request.candidates) > 8:
            raise AgentError("laya_shared_question_capacity")
        if units > self.limit:
            raise AgentError("laya_complete_context_exceeds_token_budget")
        self.checks[request.fingerprint()] = {"N" + str(i): {"sequence_tokens": units,
            "truncated": False, "state_tokens_dropped": 0} for i in range(1, len(request.candidates) + 1)}
        self.preflight_tokens[request.fingerprint()] = units * len(request.candidates)

    def score(self, request):
        self.calls.append(request)
        if len(self.calls) == self.fail_at:
            raise AgentError("decisionrag_gateway_transport_failed")
        ids = [identifier for identifier, _ in request.candidates]
        if request.stage == "documents" and "D1" in ids and "D9" in ids:
            probabilities = {identifier: 0.99 if identifier == "D9" else 0.98 if identifier == "D10" else 0.1 for identifier in ids}
        elif request.stage == "documents" and "D1" in ids:
            probabilities = {identifier: 1 - int(identifier[1:]) / 100 for identifier in ids}
        else:
            probabilities = {identifier: (index + 1) / (len(ids) + 1) for index, identifier in enumerate(ids)}
        return DecisionReceipt(request.fingerprint(), probabilities, copy.deepcopy(MODEL_BINDING),
            {"observed_forward_calls": 1, "truncated": False, "state_tokens_dropped": 0,
             "input_tokens": self.preflight_tokens[request.fingerprint()], "external_API_cash_ncny": 0,
             "exact_token_preflight": self.checks[request.fingerprint()]})


def documents(count=2, words=512):
    return tuple(Document("D" + str(i), "word " * words, "v1",
        (Passage("P" + str(i), "word " * words, "D" + str(i), "v1", "P" + str(i)),))
        for i in range(1, count + 1))


def configuration(**updates):
    return DecisionRAGTaskConfigV2.model_validate({"version": "citequest-decisionrag-task/2",
        "judge": {"version": "citequest-decisionrag-remote/2", "base_url": "http://127.0.0.1:18032",
            "bearer_token_file": Path("unused-token")}, "document_keep": 2, "passage_keep": 1, **updates})


def run_fixture(monkeypatch, docs, judge, config=None):
    config = config or configuration()
    module = ModuleType("citequest.research_v3.decisionrag_evidence")

    def bridge(query, ref_ids, passed_judge, *, current_task_scope, before_decision, limits, navigator, **_):
        binding = current_task_scope()
        return navigator(query, docs, passed_judge, scope_binding=binding, check_scope=current_task_scope,
            before_decision=before_decision, limits=limits)

    module.navigate_issued_evidence = bridge
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(task_module, "RemoteBinaryJudge", lambda *_args, **_kwargs: judge)
    context = {"id": UUID(int=1), "owner_id": UUID(int=2), "project_id": UUID(int=3),
        "epoch": 1, "revision": 1, "kind": "qa", "brief": {"question": "which evidence"},
        "profile": {"profile_id": "qa-v9", "qa_content_protocol": "isolated1",
            "evidence_navigation_protocol": "laya2"}}
    refs = [passage.id for doc in docs for passage in doc.passages]
    body = task_module.navigate_task_evidence(config, "which evidence", refs, access=None,
        context=lambda: context, remaining_ms=25000)
    validate_task_receipt(config, body, refs)
    return body, refs, config


def test_512_unit_candidates_have_new_capacity_path_and_legacy_stays_fixed(monkeypatch):
    docs = documents()
    old = DecisionRAGTaskConfig.model_validate({"version": "citequest-decisionrag-task/1",
        "judge": {"version": "citequest-decisionrag-remote/1", "base_url": "http://127.0.0.1:18031",
            "bearer_token_file": Path("unused-token")}})
    assert old.max_forward_calls == 2 and old.judge.sequence_tokens == 1024
    assert "max_forward_calls" not in old.model_dump() and "navigation_protocol" not in old.model_dump()
    legacy_judge = CapacityFixture(limit=1024)
    with pytest.raises(AgentError, match="laya_complete_context_exceeds_token_budget"):
        navigate("which evidence", docs, legacy_judge, scope_binding="scope", check_scope=lambda: "scope",
            before_decision=lambda _: None, limits=old.limits(25))
    assert legacy_judge.calls == []
    body, _, _ = run_fixture(monkeypatch, docs, CapacityFixture())
    assert body["status"] == "ok" and body["decision_calls"] == 2
    assert body["local_usage"]["input_tokens"] == body["local_usage"]["known_input_tokens"]


def test_cross_batch_probabilities_do_not_determine_final_rank(monkeypatch):
    body, _, _ = run_fixture(monkeypatch, documents(10, 10), CapacityFixture())
    assert body["status"] == "ok" and body["decision_calls"] == 4
    assert body["document_ids_kept"] == ["D9", "D10"]
    assert [row["phase"] for row in body["trace"]] == ["screen", "screen", "final", "final"]
    assert sorted(value for group in body["selection_steps"][0]["screening_batches"] for value in group) == sorted("D" + str(i) for i in range(1, 11))


def test_oversized_single_candidate_is_known_zero_inference(monkeypatch):
    judge = CapacityFixture()
    body, _, _ = run_fixture(monkeypatch, documents(1, 9000), judge)
    assert body["status"] == "error" and body["error_code"] == "decisionrag_single_candidate_capacity_exceeded"
    assert body["unknown_outcome"] is False and body["local_usage"]["input_tokens"] == 0
    assert judge.calls == []


def test_oversized_final_shared_context_preserves_completed_usage(monkeypatch):
    body, _, _ = run_fixture(monkeypatch, documents(4, 3000), CapacityFixture())
    assert body["error_code"] == "decisionrag_shared_final_capacity_exceeded"
    assert body["unknown_outcome"] is False and body["decision_calls"] == 2
    assert body["local_usage"]["input_tokens"] > 0 and not body["evidence_ref_ids"]


def test_unknown_score_has_null_totals_and_retains_known_prefix(monkeypatch):
    body, refs, config = run_fixture(monkeypatch, documents(), CapacityFixture(fail_at=2))
    assert body["status"] == "error" and body["unknown_outcome"] is True
    assert body["local_usage"]["observed_forward_calls"] is None and body["local_usage"]["input_tokens"] is None
    assert body["local_usage"]["completed_forward_calls"] == 1 and body["local_usage"]["known_input_tokens"] > 0
    changed = copy.deepcopy(body)
    changed["local_usage"]["input_tokens"] = 0
    with pytest.raises(AgentError, match="decisionrag_MCP_receipt_contract_invalid"):
        validate_task_receipt(config, changed, refs)


@pytest.mark.parametrize("updates", [{"maximum_forward_calls": 2}, {"local_input_token_limit": 1}])
def test_resource_gate_stops_before_forward(monkeypatch, updates):
    judge = CapacityFixture()
    body, _, _ = run_fixture(monkeypatch, documents(10, 10), judge, configuration(**updates))
    assert body["status"] == "error" and body["unknown_outcome"] is False and not judge.calls


def test_protocol_mismatch_and_forged_final_selection_are_rejected(monkeypatch):
    with pytest.raises(ValidationError):
        DecisionRAGTaskConfig.model_validate({"version": "citequest-decisionrag-task/1",
            "judge": {"version": "citequest-decisionrag-remote/2", "base_url": "http://127.0.0.1:18032",
                "bearer_token_file": Path("unused-token")}})
    body, refs, config = run_fixture(monkeypatch, documents(), CapacityFixture())
    body["evidence_ref_ids"] = ["P999"]
    with pytest.raises(AgentError, match="decisionrag_MCP_receipt_contract_invalid"):
        validate_task_receipt(config, body, refs)
