"""Explicit structural-navigation configuration, identity and new-task admission."""
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3 import database as database_module
from citequest.agents_v3.catalog_scope import POLICY, SUFFIX, current_runtime, graph_version, graph_versions
from citequest.agents_v3.contracts import AgentError, QA_SEPARATED_GRAPH_VERSION, RuntimeLimits, TaskCreate, digest
from citequest.agents_v3.database import TaskService
from citequest.agents_v3.product_profiles import (
    NewTaskProfile,
    QA8_PROTOCOL_METADATA,
    TaskProfilePolicy,
    evidence_navigation_protocol,
    require_navigation_source_selection,
)
from citequest.research_v3.decisionrag_task import (
    DecisionRAGTaskConfig,
    DecisionRAGTaskConfigV2,
    DecisionRAGTaskConfigV3,
    DecisionRAGTaskConfigV4,
    load_task_config,
    navigate_task_evidence,
    task_requests_navigation,
    validate_task_receipt,
)


def profile(protocol="structure1"):
    return {"kind": "qa", "profile_id": "qa-v9", "validation_state": "development",
        **QA8_PROTOCOL_METADATA, "qa_content_protocol": "isolated1", "evidence_navigation_protocol": protocol}


def config_data():
    data = json.loads(Path("runtime/decisionrag/task-structure1.example.json").read_text())
    data["judge"]["bearer_token_file"] = Path(data["judge"]["bearer_token_file"])
    data["resource_profile_sha256"] = "a" * 64
    return data


def request(count=1):
    return TaskCreate(project_id=uuid4(), operation_id=uuid4(), kind="qa", profile_id="qa-v9",
        brief={"question": "What do the selected sources establish?", "selected_papers": [
            {"kind": "native_representation", "id": f"{index + 1:064x}"} for index in range(count)]})


def test_structure_protocol_is_server_selected_only_for_isolated_qa9():
    policy = TaskProfilePolicy.model_validate_json(Path("runtime/s08/structure1-policy.example.json").read_text(), strict=True)
    metadata = policy.runtime_metadata("qa", "qa-v9")
    assert metadata == {**QA8_PROTOCOL_METADATA, "qa_content_protocol": "isolated1", "evidence_navigation_protocol": "structure1"}
    task = {"kind": "qa", "profile": {"profile_id": "qa-v9", **metadata},
            "brief": request().brief.model_dump(mode="json")}
    assert evidence_navigation_protocol(task) == "structure1" and task_requests_navigation(task)
    assert task_requests_navigation({"kind": "qa", "profile": {}}) is False
    for changes in ({"profile_id": "qa-v8"}, {"qa_content_protocol": "online1"}, {"qa_content_protocol": None}):
        with pytest.raises(ValidationError):
            NewTaskProfile.model_validate({**profile(), **changes})


def test_new_graph_identity_is_explicit_and_old_graph_bytes_stay_exact():
    common = ("qa", "qa-v9", POLICY, "json1", "grounded3", "chat-json2", "chat-json2", "jsonpatch1", "references1", "isolated1")
    for protocol in (None, "laya1", "laya2"):
        old = QA_SEPARATED_GRAPH_VERSION + "+review-json1+review-chatjson2" + ("+evidence-" + protocol if protocol else "") + SUFFIX
        assert graph_version(*common, protocol) == old
    before = {version for version in graph_versions(POLICY) if "+evidence-structure1" not in version}
    for rerank in (None, "tei-cross-encoder1"):
        for retrieval in (None, "adaptive1"):
            version = graph_version(*common, "structure1", rerank, "a" * 64 if rerank else None, retrieval)
            assert "+evidence-structure1" in version and version.endswith(SUFFIX)
            assert version in graph_versions(POLICY) and version not in before
    assert not any("+evidence-structure1" in version for version in graph_versions(None))
    metadata = {key: value for key, value in profile().items() if key not in {"kind", "validation_state"}}
    task = {"kind": "qa", "profile": {**metadata, "catalog_scope_policy": POLICY}, "graph_version": graph_version(*common, "structure1")}
    assert current_runtime(task, POLICY)
    task["profile"]["evidence_navigation_protocol"] = "laya2"
    assert not current_runtime(task, POLICY)


def test_v3_trusted_loader_and_resource_defaults_are_explicit():
    from citequest.research_v3.navigation_resources import NavigationResourceProfile
    path = Path("runtime/decisionrag/task-structure1.example.json")
    config = load_task_config(path, hashlib.sha256(path.read_bytes()).hexdigest())
    assert type(config) is DecisionRAGTaskConfigV3 and isinstance(config, DecisionRAGTaskConfigV2)
    assert config.navigation_protocol == "structure1" and config.judge.version == "citequest-decisionrag-remote/2"
    assert (config.max_documents, config.max_passages, config.document_keep, config.section_keep, config.passage_keep) == (20, 20, 2, 4, 5)
    assert (config.shortlist_count, config.max_batches_per_level, config.max_forward_calls) == (8, 8, 24)
    assert (config.active_seconds, config.local_input_token_limit, config.max_shared_bytes, config.maximum_leaf_reads, config.hybrid_keep) == (120, 262144, 32768, 8, 5)
    assert config.minimum_route_probability == .5
    resources = NavigationResourceProfile.model_validate_json(Path("runtime/decisionrag/resources-cpu-8.example.json").read_text(), strict=True)
    assert config.resource_profile_sha256 == resources.fingerprint()
    with pytest.raises(AgentError, match="decisionrag_configuration_changed"):
        load_task_config(path, "0" * 64)
    for name, kind in (("task.example.json", DecisionRAGTaskConfig), ("task-laya2.example.json", DecisionRAGTaskConfigV2)):
        old_path = path.parent / name
        loaded = load_task_config(old_path, hashlib.sha256(old_path.read_bytes()).hexdigest())
        assert type(loaded) is kind
        assert loaded.model_dump(mode="json") == json.loads(old_path.read_text())
        assert "resource_profile_sha256" not in loaded.model_dump(mode="json")


def test_v3_requires_a_bound_resource_profile_without_a_default():
    data = config_data()
    data.pop("resource_profile_sha256")
    with pytest.raises(ValidationError):
        DecisionRAGTaskConfigV3.model_validate(data)


def test_ranked_protocol_is_versioned_and_cannot_restore_an_absolute_gate():
    path = Path("runtime/decisionrag/task-structure2.example.json")
    config = load_task_config(path,hashlib.sha256(path.read_bytes()).hexdigest())
    assert type(config) is DecisionRAGTaskConfigV4 and config.navigation_protocol=="structure2"
    assert config.routing_policy=="rank_then_read/1" and config.minimum_route_probability==0
    with pytest.raises(ValidationError):
        DecisionRAGTaskConfigV4.model_validate({**config.model_dump(),"minimum_route_probability":.5})
    common = ('qa','qa-v9',POLICY,'json1','grounded3','chat-json2','chat-json2','jsonpatch1','references1','isolated1')
    old,new = (graph_version(*common,protocol) for protocol in ('structure1','structure2'))
    assert old!=new and old in graph_versions(POLICY) and new in graph_versions(POLICY)
    policy = TaskProfilePolicy.model_validate_json(Path('runtime/s08/structure2-policy.example.json').read_bytes())
    assert policy.runtime_metadata('qa','qa-v9')['evidence_navigation_protocol']=='structure2'
    require_navigation_source_selection(policy.runtime_metadata('qa','qa-v9'),request().brief.model_dump(mode='json'))


def test_worker_claim_lane_does_not_take_an_incompatible_navigation_task():
    from citequest.agents_v3.worker import configured_graph_versions
    for protocol in ('structure1','structure2'):
        path = Path(f'runtime/decisionrag/task-{protocol}.example.json')
        admitted = configured_graph_versions(SimpleNamespace(catalog_scope_policy=POLICY,
            decisionrag_config_file=path,decisionrag_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        assert any('+evidence-'+protocol+'+' in version for version in admitted)
        other = 'structure2' if protocol=='structure1' else 'structure1'
        assert not any('+evidence-'+other+'+' in version for version in admitted)
        assert any('+evidence-' not in version for version in admitted)


@pytest.mark.parametrize("value", [None, "", "a" * 63, "A" * 64, "g" * 64, True, 123])
def test_v3_rejects_invalid_resource_profile_fingerprints(value):
    with pytest.raises(ValidationError):
        DecisionRAGTaskConfigV3.model_validate({**config_data(), "resource_profile_sha256": value})


@pytest.mark.parametrize("changes", [
    {"judge": {"version": "citequest-decisionrag-remote/1", "base_url": "http://127.0.0.1:18031", "bearer_token_file": Path("/run/secrets/token")}},
    {"active_seconds": 121}, {"maximum_forward_calls": 25}, {"max_batches_per_level": 9},
    {"local_input_token_limit": 262145}, {"maximum_leaf_reads": 9}, {"minimum_route_probability": -.1},
    {"minimum_route_probability": 1.1}, {"minimum_route_probability": float("nan")},
    {"shortlist_count": 3}, {"hybrid_keep": 21}, {"maximum_leaf_reads": 4}, {"hybrid_keep": 13},
])
def test_v3_rejects_unsafe_or_incompatible_configuration(changes):
    DecisionRAGTaskConfigV3.model_validate(config_data())
    with pytest.raises(ValidationError):
        DecisionRAGTaskConfigV3.model_validate({**config_data(), **changes})


def test_v3_initial_and_expanded_refs_fit_the_existing_twenty_ref_contract():
    config = DecisionRAGTaskConfigV3.model_validate({**config_data(), "hybrid_keep": 12})
    assert config.hybrid_keep + config.maximum_leaf_reads == 20


def test_worker_accepts_trusted_v3_only_with_live_pinned_search():
    from citequest.agents_v3.worker import WorkerConfig
    path = Path("runtime/decisionrag/task-structure1.example.json")
    values = {"limits": RuntimeLimits(mode="bounded"),
        "business_dsn_file": Path("unused-business"), "checkpoint_dsn_file": Path("unused-checkpoint"),
        "memory_dsn_file": Path("unused-memory"), "skill_root": Path("unused-skills"),
        "temporary_directory": Path("unused-temporary"), "mode": "live", "tool_mode": "search",
        "catalog_scope_policy": POLICY, "decisionrag_config_file": path,
        "decisionrag_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    config = WorkerConfig(**values)
    assert config.decisionrag_config_sha256 == values["decisionrag_config_sha256"]
    for changes in ({"mode": "fixture"}, {"tool_mode": "fixture"}, {"catalog_scope_policy": None},
                    {"decisionrag_config_sha256": None}):
        with pytest.raises(ValidationError):
            WorkerConfig(**{**values, **changes})


def test_v3_cannot_silently_run_or_validate_the_legacy_tool():
    config = DecisionRAGTaskConfigV3.model_validate(config_data())
    def forbidden():
        raise AssertionError("legacy navigation must not inspect context or call a model")
    with pytest.raises(AgentError, match="structure_navigation_requires_structure_tool"):
        navigate_task_evidence(config, "A question", ["ref"], access=None, context=forbidden, remaining_ms=1000)
    with pytest.raises(AgentError, match="structure_navigation_requires_structure_tool"):
        config.limits(1)
    with pytest.raises(AgentError, match="decisionrag_MCP_receipt_contract_invalid"):
        validate_task_receipt(config, {"navigation_protocol": "structure1"}, ["ref"])


class AdmissionDatabase:
    catalog_scope_policy = POLICY

    def __init__(self, existing=None):
        self.existing = existing
        self.queries = []

    @contextmanager
    def connect(self):
        yield self

    def execute(self, sql, params=()):
        self.queries.append(sql)
        if sql.startswith("SELECT 1 FROM cq_app.projects"):
            row = {"owned_project": True}
        elif sql.startswith("SELECT * FROM cq_app.agent_tasks WHERE owner_id="):
            row = self.existing
        else:
            raise AssertionError("Unexpected admission SQL: " + sql)
        return SimpleNamespace(fetchone=lambda: row)


def admission_service(monkeypatch, existing=None):
    monkeypatch.setattr(database_module, "wallet_locks", lambda *_args: None)
    db = AdmissionDatabase(existing)
    policy = TaskProfilePolicy(release_id="structure-scope-test", profiles=[profile()])
    return TaskService(db, RuntimeLimits(mode="bounded"), new_task_profile_policy=policy), db


def test_empty_selection_fails_actual_create_before_any_task_or_wallet_admission(monkeypatch):
    service, db = admission_service(monkeypatch)
    with pytest.raises(AgentError, match="structure_navigation_requires_selected_sources") as error:
        service.create(uuid4(), request(0))
    assert error.value.status == 422
    assert len(db.queries) == 2 and not any(sql.startswith(("INSERT", "UPDATE")) for sql in db.queries)


@pytest.mark.parametrize("count", [1, 20])
def test_explicit_selected_scope_passes_create_scope_gate(monkeypatch, count):
    service, _ = admission_service(monkeypatch)
    def after_scope(*_args):
        raise RuntimeError("reached_existing_schema_gate")
    monkeypatch.setattr(database_module.catalog_scope, "require_schema", after_scope)
    with pytest.raises(RuntimeError, match="reached_existing_schema_gate"):
        service.create(uuid4(), request(count))


def test_existing_idempotent_request_is_not_readmitted_under_new_selected_scope_rule(monkeypatch):
    body = request(0)
    existing = {"id": uuid4(), "create_hash": digest(body.model_dump(mode="json"))}
    service, db = admission_service(monkeypatch, existing)
    monkeypatch.setattr(service, "public", lambda value, **_kwargs: copy.deepcopy(value))
    assert service.create(uuid4(), body) == existing
    assert len(db.queries) == 2


def test_runtime_rechecks_selected_scope_and_old_navigation_keeps_optional_scope():
    for selected in ([], [None] * 21):
        with pytest.raises(AgentError, match="structure_navigation_requires_selected_sources"):
            require_navigation_source_selection({"evidence_navigation_protocol": "structure1"}, {"selected_papers": selected})
    task = {"kind": "qa", "profile": {"profile_id": "qa-v9", "qa_content_protocol": "isolated1", "evidence_navigation_protocol": "structure1"}, "brief": {}}
    with pytest.raises(AgentError, match="structure_navigation_requires_selected_sources"):
        task_requests_navigation(task)
    require_navigation_source_selection({}, {})
    task["profile"]["evidence_navigation_protocol"] = "laya2"
    assert task_requests_navigation(task)
