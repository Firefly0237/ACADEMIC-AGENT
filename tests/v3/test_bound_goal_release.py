"""Small pure registration/campaign controls; synthetic policy proofs, no IO service."""
import copy
import json

import pytest
from pydantic import ValidationError
from tests.v3.test_development_campaign import campaign_capsule

from citequest.agents_v3.chunk_tool_policy import TYPED_CHUNK_TASKS
from citequest.agents_v3.contracts import PROFILES, digest, graph_version_for_kind
from citequest.agents_v3.development_campaign import ALLOWED_PAIRS, ALLOWED_PAIRS_V2, VERSION_V2, DevelopmentCampaign
from citequest.agents_v3.product_profiles import FUNCTION_REVIEW_PAIRS, TaskProfilePolicy, public_profile_policy


def test_new_resources_and_graph_versions_are_explicit_without_old_profile_changes():
    for old, new in (("qa-v5", "qa-v6"), ("stance-v2", "stance-v3"), ("workset-v6", "workset-v7"), ("research-v3", "research-v4")):
        expected = PROFILES[old].model_dump()
        expected["profile_id"] = new
        if new == "qa-v6":
            assert expected["active_seconds"] == 120
            expected["active_seconds"] = 450
        assert PROFILES[new].model_dump() == expected
    labels = [graph_version_for_kind(kind, profile) for kind, profile in sorted(ALLOWED_PAIRS_V2)]
    assert len(set(labels)) == 4  # Summary/compare intentionally share one graph.
    for pair in (("qa", "qa-v6"), ("stance", "stance-v3")):
        assert pair in TYPED_CHUNK_TASKS and pair in FUNCTION_REVIEW_PAIRS


def test_campaign_v2_has_exact_scoped_queue_and_v1_serialization_stays_identical(tmp_path):
    _, _, raw = campaign_capsule(tmp_path / "original")
    old = DevelopmentCampaign.model_validate_json(json.dumps(raw), strict=True)
    expected_legacy = {**raw, "catalog_scope_policy": "task-catalog-snapshot/1",
        "semantics": "known_active_client_HTTP_1_unknown_remote_concurrency_not_asserted"}
    assert old.model_dump(mode="json") == expected_legacy
    changed = copy.deepcopy(raw)
    changed["version"] = VERSION_V2
    changed["task_profile_policy"]["profiles"] = [{"kind": kind, "profile_id": profile,
        "validation_state": "development", **({"review_transport_protocol": "function1"} if kind in {"qa", "stance"} else {})}
        for kind, profile in sorted(ALLOWED_PAIRS_V2)]
    profile_policy = TaskProfilePolicy.model_validate_json(json.dumps(changed["task_profile_policy"]), strict=True)
    changed["task_profile_policy_sha256"] = public_profile_policy(profile_policy)["policy_sha256"]
    new = DevelopmentCampaign.model_validate_json(json.dumps(changed), strict=True)
    new._file_sha256 = "a" * 64
    sql, params = new.queue_predicate()
    assert new.task_binding()["version"] == VERSION_V2
    assert params[3:] == tuple(value for pair in sorted(ALLOWED_PAIRS_V2) for value in pair)
    assert sql.count("t.kind=%s") == 5 and "LIMIT" not in sql
    assert not any(profile in params for kind, profile in ALLOWED_PAIRS if (kind, profile) not in ALLOWED_PAIRS_V2)
    changed["task_profile_policy"]["profiles"][0]["profile_id"] = "qa-v5"
    with pytest.raises(ValidationError):
        DevelopmentCampaign.model_validate_json(json.dumps(changed), strict=True)
    assert digest(old.model_dump(mode="json")) == digest(expected_legacy)
