"""Pure artifact/receipt binding contracts; no model or ledger is contacted."""
import copy

import pytest

from citequest.agents_v3.contracts import AgentError, digest
from citequest.research_v3.handoff_contracts import bind_handoff, collect_handoff_audit, source_keys

COMMAND = "synthetic-command"
IDENTITY = {"task_id": "synthetic-task", "command_id": COMMAND,
    "brief_sha256": "b" * 64, "profile_sha256": "c" * 64}


def receipt(key, schema):
    return {"attempt_id": "synthetic-attempt:" + key[-1], "logical_key": key,
        "request_sha256": "d" * 64, "response_sha256": "e" * 64,
        "schema_name": schema, "usage": {"input": 12, "cached": 2, "output": 5}}


def entry():
    artifact = {"question": "Explicit synthetic question."}
    sources = {COMMAND + ":outline:author:0": "citequest_research_outline_planner",
        COMMAND + ":outline:review:1": "citequest_research_outline_reviewer"}
    return bind_handoff({"role": "planner", "sha256": digest(artifact), "artifact": artifact},
        sources, {"brief": IDENTITY["brief_sha256"]})


def test_actual_author_and_reviewer_keys_remain_distinct_across_revision_rounds():
    model = entry()
    receipts = {key: receipt(key, schema) for key, schema in model["audit_binding"]["sources"].items()}
    user = {"role": "user", "artifact": {"confirmed": "synthetic fact"}}
    user["sha256"] = digest(user["artifact"])
    audit = collect_handoff_audit([user, model], receipts, **IDENTITY)
    assert source_keys([user, model]) == list(receipts)
    assert audit["scope"] == "artifact_and_receipt_integrity"
    assert audit["handoffs"][0]["authority"] == "versioned_user_brief"
    sources = audit["handoffs"][1]["sources"]
    assert [item["logical_key"] for item in sources] == list(receipts)
    assert sources[0]["logical_key"].endswith(":author:0")
    assert sources[1]["logical_key"].endswith(":review:1")
    assert sources[0]["usage"] == receipts[sources[0]["logical_key"]]["usage"]


def test_changed_artifacts_missing_receipts_and_wrong_source_identity_are_rejected():
    model = entry()
    receipts = {key: receipt(key, schema) for key, schema in model["audit_binding"]["sources"].items()}
    changed = copy.deepcopy(model)
    changed["artifact"]["question"] = "Changed after hashing."
    with pytest.raises(AgentError):
        collect_handoff_audit([changed], receipts, **IDENTITY)
    with pytest.raises(AgentError):
        collect_handoff_audit([model], {}, **IDENTITY)
    wrong = copy.deepcopy(receipts)
    wrong[next(iter(wrong))]["schema_name"] = "citequest_research_method"
    with pytest.raises(AgentError):
        collect_handoff_audit([model], wrong, **IDENTITY)
    wrong = copy.deepcopy(receipts)
    wrong[next(iter(wrong))]["logical_key"] = "other-command:author:0"
    with pytest.raises(AgentError):
        collect_handoff_audit([model], wrong, **IDENTITY)


def test_target_revision_keeps_explicit_before_after_unaffected_and_conflict_scope():
    artifact = {"target_id": "E1", "replacement": {"id": "E1", "hypothesis": "Proposed changed hypothesis."},
        "resolved_conflict_ids": ["3" * 64 + ":X1"]}
    key = COMMAND + ":E1:repair:1"
    revision = {"target_id": "E1", "before_sha256": "1" * 64,
        "after_sha256": digest(artifact["replacement"]), "unaffected_sha256": "2" * 64,
        "resolved_conflict_refs": ["3" * 64 + ":X1"]}
    model = bind_handoff({"role": "reviser", "sha256": digest(artifact), "artifact": artifact},
        {key: "citequest_research_target_repair"}, {"current_target": "1" * 64}, revision=revision)
    audit = collect_handoff_audit([model], {key: receipt(key, "citequest_research_target_repair")}, **IDENTITY)
    assert audit["handoffs"][0]["revision"] == revision
    assert audit["handoffs"][0]["parent_artifact_sha256s"] == {"current_target": "1" * 64}
    model["audit_binding"]["revision"]["after_sha256"] = "9" * 64
    with pytest.raises(AgentError):
        collect_handoff_audit([model], {key: receipt(key, "citequest_research_target_repair")}, **IDENTITY)
