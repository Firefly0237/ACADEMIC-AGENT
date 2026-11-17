"""Pure admission controls for the separately versioned QA Pro profile."""
import json
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.catalog_scope import POLICY, graph_version, graph_versions
from citequest.agents_v3.contracts import PROFILES, AgentError, TaskCreate, digest
from citequest.agents_v3.model_contracts import model_contract, validated_model_alias
from citequest.agents_v3.product_profiles import NewTaskProfile, TaskProfilePolicy, public_profile_policy
from citequest.agents_v3.qa_v7 import QA7
from citequest.agents_v3.worker import WorkerConfig


def test_registered_pro_preserves_resource_caps_and_conservative_cash_admission():
    previous = PROFILES["qa-v6"].model_dump()
    current = PROFILES["qa-v7"].model_dump()
    model_fields = set(model_contract("deepseek-v4-pro"))
    resource_fields = set(previous) - model_fields - {"profile_id"}
    assert {key: current[key] for key in resource_fields} == {key: previous[key] for key in resource_fields}
    assert validated_model_alias(previous) == "deepseek-flash"
    assert validated_model_alias(current) == "deepseek-v4-pro"
    aggregate = current["input_tokens"] * current["input_miss_ncny"] + current["output_tokens"] * current["output_ncny"]
    request_reserve = current["context_window"] * current["input_miss_ncny"] + current["per_call_output"] * current["output_ncny"]
    assert aggregate == 5446656000 and request_reserve == 9547776000
    assert aggregate + request_reserve == 14994432000 < current["cash_ncny"]


def test_normal_create_policy_graph_and_node_bind_the_same_verified_pro_profile():
    with pytest.raises(ValidationError):
        NewTaskProfile(kind="qa", profile_id="qa-v7", validation_state="development")
    selected = NewTaskProfile(kind="qa", profile_id="qa-v7", validation_state="development",
        review_transport_protocol="function1", qa_execution_protocol="grounded3", review_api_protocol="chat-strict2")
    policy = TaskProfilePolicy(release_id="pure-pro-control", profiles=[selected])
    request = TaskCreate.model_validate_json(json.dumps({
        "project_id": str(uuid4()), "operation_id": str(uuid4()), "profile_id": "qa-v7", "kind": "qa",
        "brief": {"kind": "qa", "catalog": "application_papers", "question": "What is supported?",
            "objective": "Answer using the selected evidence."},
    }))
    policy.admit(request)
    bound = graph_version("qa", "qa-v7", POLICY, "function1", "grounded3", "chat-strict2")
    old = graph_version("qa", "qa-v6", POLICY, "function1", "grounded3", "chat-strict2")
    assert bound in graph_versions(POLICY) and old in graph_versions(POLICY) and bound != old
    assert bound.startswith("citequest-agent-graph/29-") and old.startswith("citequest-agent-graph/28-")
    assert QA7.profile_id == request.profile_id
    assert public_profile_policy(policy)["profile_hashes"]["qa-v7"] == digest(PROFILES["qa-v7"].model_dump(mode="json"))


def test_worker_receipt_map_is_registered_and_omitted_from_legacy_serialization():
    root = Path(__file__).resolve().parents[2]
    value = json.loads((root / "runtime/s08/worker.example.json").read_text(encoding="utf-8"))
    old = WorkerConfig.model_validate_json(json.dumps(value))
    assert "model_metadata_receipt_files" not in old.model_dump(mode="json")
    value["model_metadata_receipt_files"] = {
        "deepseek-flash": "/run/config/flash.json", "deepseek-v4-pro": "/run/config/pro.json",
    }
    current = WorkerConfig.model_validate_json(json.dumps(value))
    assert current.model_metadata_receipt_files["deepseek-v4-pro"] == Path("/run/config/pro.json")
    value["model_metadata_receipt_files"] = {"unregistered": "/run/config/other.json"}
    with pytest.raises(AgentError, match="provider_model_not_registered"):
        WorkerConfig.model_validate_json(json.dumps(value))
