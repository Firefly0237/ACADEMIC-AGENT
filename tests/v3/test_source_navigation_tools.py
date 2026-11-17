"""Real helper/projection code with explicit in-process boundary doubles.

Settled Execution rows, the session, source authorization and fixed-model
admission are fixtures. No transport, PostgreSQL, model or quality result is
implied by these checks.
"""
from __future__ import annotations

import asyncio
import copy
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from citequest.agents_v3.contracts import AgentError
from citequest.agents_v3.product_profiles import QA8_PROTOCOL_METADATA
from citequest.agents_v3.qa_separated import QASeparated
from citequest.agents_v3.result_projection import authorize_result_dependencies, project_task_result
from citequest.agents_v3.tools import MCPTools
from citequest.research_v3.decisionrag import DecisionRequest
from citequest.research_v3.decisionrag_remote import PROTOCOL_V2, RemoteBinaryJudge, RemoteJudgeConfig
from citequest.v2.domain import Actor


OWNER = UUID("11111111-1111-4111-8111-111111111111")
TASK = UUID("22222222-2222-4222-8222-222222222222")
PROJECT = UUID("33333333-3333-4333-8333-333333333333")
SCOPE = "structure-scope:" + "a" * 64
OLD = {"ref_id": "initial", "source_sha256": "b" * 64, "text": "Existing authorized fixture text."}
NEW = {"ref_id": "expanded", "source_sha256": "c" * 64, "text": "New authorized fixture text."}


def body(*, partial=False):
    evidence = [OLD] if partial else [OLD, NEW]
    return {"protocol": "structure1", "navigation_protocol": "structure1", "status": "partial" if partial else "ok",
        "input_frontier_ref_ids": [OLD["ref_id"]], "evidence_ref_ids": [row["ref_id"] for row in evidence],
        "evidence": copy.deepcopy(evidence), "unknown_outcome": False,
        "structure": {"version": "citequest-source-navigation/1", "scope_ref_id": SCOPE,
            "supplemental_evidence_ref_ids": [] if partial else [NEW["ref_id"]],
            "nodes": [{"id": "fixture-node", "kind": "passage", "read_state": "source_text_returned",
                       "evidence_ref_ids": [evidence[-1]["ref_id"]]}]}}


class ReplayExecution:
    """Explicit durable-row fixture, not an emulation of PostgreSQL reserve."""
    def __init__(self, navigation, verification):
        self.responses = {"navigate_source_structure": navigation, "verify_evidence_batch": verification}
        self.events, self.requests = [], []

    def snapshot(self):
        # Planning-receipt admission is a separate contract; this fixture exercises
        # the settled tool branch without fabricating a planner database row.
        return {"kind": "research", "profile": {"profile_id": "research-v1"}}

    def reserve(self, logical_key, request, *, kind):
        self.events.append("reserve:" + request["name"])
        self.requests.append((logical_key, copy.deepcopy(request), kind))
        return {"id": UUID(int=5), "status": "settled", "response": copy.deepcopy(self.responses[request["name"]])}


class NeverCalledSession:
    async def call_tool(self, *args, **kwargs):
        raise AssertionError("settled helper control must not claim a transport call")


class ReplayTools(MCPTools):
    """Uses real call/navigate_sources, stubbing only admission/lifetime boundaries."""
    def __init__(self, navigation=None, verification=None, *, fail_scope=None):
        navigation = body() if navigation is None else navigation
        verification = verification if verification is not None else {"valid": True,
            "pins": [{"ref_id": row["ref_id"], "source_sha256": row["source_sha256"]} for row in navigation["evidence"]]}
        self.execution = ReplayExecution(navigation, verification)
        self.session, self.scope_checks, self.fail_scope = NeverCalledSession(), 0, fail_scope
        self.navigation_config = SimpleNamespace(navigation_protocol="structure1", max_forward_calls=10,
            active_seconds=60, hybrid_keep=1)
        self.navigation_config_sha256 = "d" * 64
        self.catalog = {name: {"type": "object", "properties": {}, "additionalProperties": True}
                        for name in ("navigate_source_structure", "verify_evidence_batch")}

    @property
    def decisionrag_enabled(self):
        return True

    @property
    def rerank_enabled(self):
        return False

    async def open(self):
        return self

    async def check_scope(self):
        self.scope_checks += 1
        self.execution.events.append("scope:" + str(self.scope_checks))
        if self.scope_checks == self.fail_scope:
            raise AgentError("fixture_current_scope_revoked")


class NavigationHelperTests(unittest.IsolatedAsyncioTestCase):
    async def test_settled_replay_checks_scope_before_after_and_reverifies_expanded_hashes(self):
        tools = ReplayTools()
        result = await tools.navigate_sources("Fixture query", ["initial"], logical_key="fixture:structure")
        self.assertEqual(result["evidence_ref_ids"], ["initial", "expanded"])
        self.assertEqual(tools.execution.events, ["scope:1", "reserve:navigate_source_structure",
            "reserve:verify_evidence_batch", "scope:2"])
        verification = tools.execution.requests[1]
        self.assertEqual(verification[0], "fixture:structure:verify-expanded")
        self.assertEqual(verification[1]["arguments"]["ref_ids"], ["initial", "expanded"])

    async def test_scope_revocation_before_replay_prevents_reservation_and_after_replay_prevents_delivery(self):
        for phase in (1, 2):
            tools = ReplayTools(fail_scope=phase)
            with self.subTest(phase=phase), self.assertRaisesRegex(AgentError, "current_scope_revoked"):
                await tools.navigate_sources("Fixture query", ["initial"], logical_key="fixture:structure")
            self.assertEqual(len(tools.execution.requests), 0 if phase == 1 else 2)

    async def test_a_ref_absent_from_task_registry_is_rejected_even_after_settled_navigation(self):
        tools = ReplayTools(verification={"status": "error", "error_code": "evidence_not_issued_to_task"})
        with self.assertRaisesRegex(AgentError, "evidence_not_issued_to_task"):
            await tools.navigate_sources("Fixture query", ["initial"], logical_key="fixture:structure")
        self.assertEqual([request[1]["name"] for request in tools.execution.requests],
            ["navigate_source_structure", "verify_evidence_batch"])

    async def test_missing_or_changed_literal_hash_cannot_be_treated_as_verified(self):
        for verification in (
            {"valid": False, "pins": []},
            {"valid": True, "pins": [{"ref_id": "initial", "source_sha256": OLD["source_sha256"]}]},
            {"valid": True, "pins": [{"ref_id": "initial", "source_sha256": OLD["source_sha256"]},
                                      {"ref_id": "expanded", "source_sha256": "0" * 64}]},
        ):
            tools = ReplayTools(verification=verification)
            with self.subTest(verification=verification), self.assertRaisesRegex(AgentError, "issued_evidence_changed"):
                await tools.navigate_sources("Fixture query", ["initial"], logical_key="fixture:structure")

    async def test_partial_navigation_still_reauthorizes_retained_retrieval(self):
        tools = ReplayTools(navigation=body(partial=True))
        result = await tools.navigate_sources("Fixture query", ["initial"], logical_key="fixture:structure")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(tools.execution.requests[1][1]["arguments"], {"ref_ids": ["initial"]})


class QASeparatedStructureTests(unittest.TestCase):
    def current(self, response):
        calls = []
        class Tools:
            decisionrag_enabled = True
            navigation_config = SimpleNamespace(hybrid_keep=1)
            async def navigate_sources(self, query, refs, **kwargs):
                calls.append((query, refs, kwargs))
                return copy.deepcopy(response)
        current = QASeparated.__new__(QASeparated)
        task = {"kind": "qa", "profile": {"profile_id": "qa-v9", **QA8_PROTOCOL_METADATA,
            "qa_content_protocol": "isolated1", "evidence_navigation_protocol": "structure1"}}
        current.graph = SimpleNamespace(tools=Tools(), execution=SimpleNamespace(snapshot=lambda: task), wait=asyncio.run)
        current.key = lambda _data, role: "fixture:" + role
        state = {"qa5": {"plan": {"queries": [{"query": "Explicit fixture query"}]},
            "turn": {"text": "Fixture user question"}, "binding": {"fixture": "planner boundary double"}},
            "evidence": [copy.deepcopy(OLD)], "answer": {"previous": "fixture"}}
        return current, state, calls

    def test_structure_branch_accepts_new_authorized_refs_outside_original_frontier(self):
        current, state, calls = self.current(body())
        before = copy.deepcopy(state)
        result = current.navigate_retrieved(state)
        self.assertEqual([row["ref_id"] for row in result["evidence"]], ["initial", "expanded"])
        self.assertEqual(calls[0][1], ["initial"])
        self.assertIsNone(result["answer"])
        self.assertNotIn("evidence", result["qa5"]["evidence_navigation"])
        self.assertFalse(result["qa5"]["evidence_navigation"]["full_corpus_or_full_document_coverage"])
        self.assertEqual(state, before)

    def test_partial_response_preserves_preexisting_retrieval_without_claiming_complete(self):
        current, state, _ = self.current(body(partial=True))
        result = current.navigate_retrieved(state)
        self.assertEqual(result["evidence"], state["evidence"])
        self.assertEqual(result["qa5"]["evidence_navigation"]["status"], "partial")

    def test_error_wrong_protocol_duplicate_or_mismatched_refs_are_rejected(self):
        invalid = []
        value = body(); value["status"] = "error"; invalid.append(value)
        value = body(); value["navigation_protocol"] = "laya1"; invalid.append(value)
        value = body(); value["protocol"] = "laya1"; invalid.append(value)
        value = body(); value["evidence_ref_ids"] = ["initial", "initial"]; value["evidence"] = [OLD, OLD]; invalid.append(value)
        value = body(); value["evidence_ref_ids"] = [123]; value["evidence"] = [{"ref_id": 123}]; invalid.append(value)
        value = body(); value["evidence_ref_ids"] = [{}]; value["evidence"] = [{"ref_id": {}}]; invalid.append(value)
        value = body(); value["evidence"] = [NEW, OLD]; invalid.append(value)
        for response in invalid:
            current, state, _ = self.current(response)
            with self.subTest(status=response["status"], refs=response["evidence_ref_ids"], protocol=response["protocol"]), self.assertRaises(AgentError):
                current.navigate_retrieved(state)


class ResultProjectionStructureTests(unittest.TestCase):
    def fixture(self):
        memory = SimpleNamespace(current=lambda *_: [], validate_result_context=lambda *_: None)
        service = SimpleNamespace(database=None)
        result = {"publication_status": "validated_ai_review", "citations": [{"ref_id": "citation"}],
            "context_manifest": {"evidence_refs": [{"ref_id": "context"}]}, "evidence_navigation": {
                "protocol": "structure1", "input_frontier_ref_ids": ["initial"], "evidence_ref_ids": ["expanded"],
                "structure": {"scope_ref_id": SCOPE, "supplemental_evidence_ref_ids": ["supplement"],
                    "nodes": [{"id": "returned-first", "label": "Returned fixture", "read_state": "source_text_returned",
                               "evidence_ref_ids": ["node-only"]},
                              {"id": "unread-directory", "label": "UNREAD_DIRECTORY_FIXTURE", "read_state": "not_read",
                               "evidence_ref_ids": []},
                              {"id": "returned-last", "label": "Last returned fixture", "read_state": "source_text_returned",
                               "evidence_ref_ids": ["last-node-only"]}]}}}
        task = {"kind": "qa", "project_id": PROJECT, "profile": {"profile_id": "qa-v9"}, "result": result}
        return service, memory, task, result

    def test_projection_authorizes_scope_and_every_dependency_including_node_only_refs(self):
        service, memory, task, result = self.fixture()
        observed = []
        with patch("citequest.agents_v3.user_skills.validate_task_skills", lambda *_: None):
            authorize_result_dependencies(service, memory, lambda _a, _t, refs: observed.extend(refs),
                Actor(OWNER), TASK, task, result)
        self.assertEqual(set(observed), {"citation", "context", "initial", "expanded", "supplement", "node-only", "last-node-only", SCOPE})

    def test_large_dependency_set_is_completely_authorized_in_bounded_batches(self):
        service, memory, task, result = self.fixture()
        result["citations"] = [{"ref_id": f"citation-{index}"} for index in range(70)]
        result["evidence_navigation"]["structure"]["nodes"] = [
            {"id": f"node-{index}", "evidence_ref_ids": [f"node-ref-{index}"]} for index in range(200)]
        batches = []
        with patch("citequest.agents_v3.user_skills.validate_task_skills", lambda *_: None):
            authorize_result_dependencies(service, memory, lambda _a, _t, refs: batches.append(list(refs)),
                Actor(OWNER), TASK, task, result)
        self.assertTrue(all(0 < len(batch) <= 256 for batch in batches))
        self.assertEqual(len(batches), 2)
        actual = {ref for batch in batches for ref in batch}
        self.assertTrue({f"node-ref-{index}" for index in range(200)} <= actual)
        self.assertTrue({f"citation-{index}" for index in range(70)} <= actual)
        self.assertIn(SCOPE, actual)

    def test_retracting_an_unread_scope_source_hides_the_whole_directory_projection(self):
        service, memory, task, _ = self.fixture()
        observed = []
        def authorize(_actor, _task, refs):
            observed.extend(refs)
            if SCOPE in refs:
                raise AgentError("fixture_unread_source_revoked")
        with patch("citequest.agents_v3.user_skills.validate_task_skills", lambda *_: None):
            projected = project_task_result(service, memory, authorize, Actor(OWNER), TASK, task)
        self.assertIn(SCOPE, observed)
        self.assertIsNone(projected["result"])
        self.assertEqual(projected["result_availability"], "evidence_unavailable_or_access_revoked")
        self.assertNotIn("UNREAD_DIRECTORY_FIXTURE", str(projected))

    def test_missing_scope_authority_cannot_publish_navigation_metadata(self):
        service, memory, task, result = self.fixture()
        del result["evidence_navigation"]["structure"]["scope_ref_id"]
        with patch("citequest.agents_v3.user_skills.validate_task_skills", lambda *_: None):
            projected = project_task_result(service, memory, lambda *_: None, Actor(OWNER), TASK, task)
        self.assertIsNone(projected["result"])


class ResourceBindingControls(unittest.TestCase):
    def remote(self, payload):
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def read(self, maximum):
                return json.dumps(payload).encode()
        judge = RemoteBinaryJudge.__new__(RemoteBinaryJudge)
        judge.config = RemoteJudgeConfig(version="citequest-decisionrag-remote/2", base_url="http://127.0.0.1:8700",
            bearer_token_file=Path("UNUSED_FIXTURE_PATH"))
        judge.deadline, judge.expected_resource_profile_sha256 = time.monotonic() + 60, "e" * 64
        judge.token = "EXPLICIT_NONLIVE_FIXTURE"
        judge.opener = SimpleNamespace(open=lambda *_args, **_kwargs: Response())
        return judge

    def payload(self, request, *, resources=True):
        value = {"protocol": PROTOCOL_V2, "operation": "preflight", "request_sha256": request.fingerprint(),
            "status": "ok", "model_binding": {"provider": "explicit_fixed_model_admission_double"}}
        if resources:
            value["model_binding"]["resource_profile_sha256"] = "e" * 64
            value["resource_admission"] = {"profile_sha256": "e" * 64, "admitted": True}
        return value

    def test_old_gateway_without_resources_is_rejected_by_new_bound_runtime(self):
        request = DecisionRequest("Fixture query", "documents", (("D", "Fixture source description"),), "f" * 64)
        for change in ("missing", "different", "not_admitted"):
            value = self.payload(request, resources=change != "missing")
            if change == "different":
                value["resource_admission"]["profile_sha256"] = "0" * 64
            elif change == "not_admitted":
                value["resource_admission"]["admitted"] = False
            with patch("citequest.research_v3.decisionrag_remote.require_model_binding", lambda *_args, **_kwargs: None):
                with self.subTest(change=change), self.assertRaisesRegex(AgentError, "resource_profile_binding_mismatch"):
                    self.remote(value).call("preflight", request)

    def test_matching_resource_profile_is_preserved_without_a_network_or_model_call(self):
        request = DecisionRequest("Fixture query", "documents", (("D", "Fixture source description"),), "f" * 64)
        with patch("citequest.research_v3.decisionrag_remote.require_model_binding", lambda *_args, **_kwargs: None):
            result = self.remote(self.payload(request)).call("preflight", request)
        self.assertEqual(result["resource_admission"]["profile_sha256"], "e" * 64)


if __name__ == "__main__":
    unittest.main()
