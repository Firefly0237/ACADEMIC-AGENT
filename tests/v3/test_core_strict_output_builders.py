"""New-core request exits, old-profile compatibility and known-receipt controls."""
from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.qa_v5 import QA5
from citequest.agents_v3.research_workflow import MethodHandoff, Outline
from citequest.agents_v3.structured_output import strict_output_schema
from citequest.agents_v3.workset_v5_payloads import payload
from citequest.research_v3.qa_v5_contracts import SelectorAnswerV5
from citequest.research_v3.school_contracts import CrossReview, RequestCoverage, SchoolSection, SectionReview

sys.path.insert(0, str(Path(__file__).parent))
import test_qa_v5 as qa_fixtures
import test_research_literature_flow as research_fixtures
import test_workset_v5_graph as workset_fixtures
from test_strict_output_schema import assert_strict_objects


class CoreSchemaBuilderControls(unittest.TestCase):
    def test_all_workset_roles_keep_dynamic_schema_values_and_compile_required(self):
        schema = SelectorAnswerV5.model_json_schema()
        schema["$defs"]["SelectorEdgeV5"]["properties"]["quote_id"]["enum"] = ["Q1", "Q37"]
        original = copy.deepcopy(schema)
        base = {"canonical_brief":{"question":"EXPLICIT CONTROL"}, "context_manifest":{},
            "required_user_turns":[{"turn_id":"U1", "text":"EXPLICIT CONTROL"}],
            "confirmed_user_memory_data":[], "selected_user_skills":[]}
        for role in ("map", "reduce", "document_review", "plan", "plan_review", "atomic_author", "atomic_review", "global_coverage"):
            with self.subTest(role=role):
                request = payload(base, role=role, schema=schema, task_data={})
                actual = request["text"]["format"]["schema"]
                self.assertEqual(actual, strict_output_schema(schema))
                assert_strict_objects(self, actual)
        self.assertEqual(schema, original)

    def test_research_v3_and_school_exit_compile_actual_output_models(self):
        boundary, anchor, _ = research_fixtures.fixture()
        for role, model in (("planner", Outline), ("methods", MethodHandoff),
                ("school_writer", SchoolSection), ("school_independent_reviewer", SectionReview),
                ("school_cross_reviewer", CrossReview), ("school_request_auditor", RequestCoverage)):
            with self.subTest(role=role):
                request = boundary.payload(anchor, role, model, "explicit_control", "EXPLICIT CONTROL", {})
                actual = request["text"]["format"]["schema"]
                self.assertEqual(actual, strict_output_schema(model.model_json_schema()))
                assert_strict_objects(self, actual)

    def test_research_v2_schema_is_unchanged(self):
        boundary, anchor, _ = research_fixtures.fixture()
        boundary.host.execution.value["profile"]["profile_id"] = "research-v2"
        original = SelectorAnswerV5.model_json_schema()
        request = boundary.payload(anchor, "literature", original, "explicit_control", "EXPLICIT CONTROL", {})
        self.assertEqual(request["text"]["format"]["schema"], original)
        self.assertNotIn("kind", request["text"]["format"]["schema"]["required"])

    def test_research_v3_quote_grid_and_known_replay_keep_original_hash(self):
        boundary, anchor, provider = research_fixtures.fixture()
        outcome = research_fixtures.run(boundary, anchor)
        self.assertTrue(outcome["accepted"])
        for item in provider.requests.values():
            assert_strict_objects(self, item["payload"]["text"]["format"]["schema"])
        research_fixtures.test_settled_author_replay_reconstructs_original_cap_after_output_budget_exhausted()

    def test_old_uncompiled_qa5_receipt_fails_closed_in_new_runtime(self):
        execution = qa_fixtures.Execution()
        builder = object.__new__(QA5)
        builder.graph = SimpleNamespace(execution=execution)
        state = {"qa5":{"revision":0}}
        schema = SelectorAnswerV5.model_json_schema()
        prompt = {"context_manifest":{}, "canonical_brief":{"question":"EXPLICIT CONTROL"}}
        compiled = builder.request(state, "author", schema, prompt)
        old = copy.deepcopy(compiled)
        old["text"]["format"]["schema"] = schema
        self.assertNotEqual(digest(old), digest(compiled))
        execution.known[builder.key(state["qa5"], "author")] = {"request":old}
        with self.assertRaises(AgentError) as raised:
            builder.request(state, "author", schema, prompt)
        self.assertEqual(raised.exception.code, "qa5_settled_request_changed")


class CoreGraphSchemaControls(unittest.IsolatedAsyncioTestCase):
    async def test_actual_workset_graph_roles_and_known_cold_replay(self):
        execution, _, provider, _, _ = await workset_fixtures.run_case("compare", 3, [2, 2, 2])
        self.assertTrue(provider.calls)
        for item in execution.known.values():
            assert_strict_objects(self, item["request"]["text"]["format"]["schema"])
        await workset_fixtures.Workset5GraphTests().test_cold_graph_after_settlement_reuses_one_known_model_attempt()
