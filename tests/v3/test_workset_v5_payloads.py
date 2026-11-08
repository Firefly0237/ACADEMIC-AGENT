"""Role isolation and complete SDK wire admission, without model/PG/network."""
import json
import unittest

from citequest.agents_v3.context import wire_bytes
from citequest.agents_v3.contracts import AgentError
from citequest.agents_v3.workset_v5_payloads import payload, role_cap


class PayloadTests(unittest.TestCase):
    def base(self):
        return {"canonical_brief": {"question": "Original question"}, "context_manifest": {},
            "required_user_turns": [{"turn_id": "U1", "text": "Original question"}], "confirmed_user_memory_data": [], "selected_user_skills": []}

    def test_all_registered_roles_import_and_encode_full_requests(self):
        for role in ("map", "reduce", "document_review", "plan", "plan_review", "atomic_author", "atomic_review", "global_coverage"):
            with self.subTest(role=role):
                request = payload(self.base(), role=role, schema={"type": "object"}, task_data={"source_transport": {"sha256": "a" * 64}})
                self.assertGreater(wire_bytes(request), 100)
                self.assertEqual(request["max_output_tokens"], role_cap(role))
                self.assertEqual(request["truncation"], "disabled")
                self.assertIn("context_only is background", request["instructions"])

    def test_required_current_questions_cannot_be_optionally_discarded(self):
        base = self.base()
        base["required_user_turns"][0]["text"] = "长" * 3000
        with self.assertRaisesRegex(AgentError, "required_context_exceeds_wire_limit"):
            payload(base, role="map", schema={}, task_data={}, maximum=2048)
        self.assertEqual(len(base["required_user_turns"][0]["text"]), 3000)

    def test_reviewer_never_sees_generator_skill_body(self):
        base = self.base()
        base["selected_user_skills"] = [{"body": "Generator-specific instructions"}]
        with self.assertRaisesRegex(AgentError, "reviewer_skill_body_forbidden"):
            payload(base, role="atomic_review", schema={}, task_data={})
        payload(base, role="atomic_author", schema={}, task_data={})

    def test_task_data_cannot_replace_current_requirement_state(self):
        with self.assertRaisesRegex(AgentError, "overrides_authority"):
            payload(self.base(), role="map", schema={}, task_data={"canonical_brief": {"question": "replacement"}})

    def test_role_selection_is_trusted_and_unknown_role_refused(self):
        with self.assertRaisesRegex(AgentError, "role_not_registered"):
            payload(self.base(), role="user-supplied-role", schema={}, task_data={})

    def test_independent_role_exclusion_is_explicit_in_manifest(self):
        request = payload(self.base(), role="document_review", schema={}, task_data={})
        prompt = json.loads(request["input"][0]["content"])
        self.assertIn("generator_skill_body_excluded", prompt["context_manifest"]["excluded"])


if __name__ == "__main__":
    unittest.main()
