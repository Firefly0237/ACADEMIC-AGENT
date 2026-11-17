"""Local schema/repair boundaries, not provider semantic quality or PG credits."""
import copy
import json
import unittest

from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError
from citequest.research_v3.quote_spans import SelectorSummary
from citequest.research_v3.typed_feedback import RepairLock, claim_objects, strict_object, validation_feedback
from citequest.research_v3.workset_v5_contracts import SelectorSummaryV5


def candidate():
    return {"claims": [{"id": "C" + str(i), "statement": "Fixture proposition " + str(i), "status": "supported",
        "edges": [{"quote_id": "Q" + str(i), "relation": "supports", "rationale": "Fixture", "conditions": []}],
        "limitations": []} for i in range(1, 8)], "gaps": [], "evidence_status": "supported"}


def error_for(value):
    try:
        SelectorSummaryV5.model_validate(value, strict=True)
    except ValidationError as error:
        return error
    raise AssertionError("expected local validation failure")


class TypedFeedbackTests(unittest.TestCase):
    def test_actual_summary_failure_shape_gets_exact_rule_and_claim_ids(self):
        value = candidate()
        for i in (0, 6):
            value["claims"][i]["edges"][0]["relation"] = "context_only"
        feedback = validation_feedback(error_for(value), value)
        self.assertTrue(feedback["repairable"])
        self.assertEqual(feedback["affected_claim_ids"], ["C1", "C7"])
        self.assertEqual([x["code"] for x in feedback["issues"]], ["claim_supported_requires_supports"] * 2)
        self.assertIn("context_only is background", feedback["issues"][0]["rule"])
        self.assertEqual(feedback["issues"][1]["location"], ["claims", 6])

    def test_unaffected_objects_frozen_across_directed_repair(self):
        value = candidate()
        lock = RepairLock.bind(value, ["C1", "C7"])
        fixed = copy.deepcopy(value)
        fixed["claims"][0]["statement"] = "Revised actual proposition"
        lock.verify(fixed)
        fixed["claims"][3]["statement"] = "Unrelated object was rewritten"
        with self.assertRaisesRegex(AgentError, "typed_feedback_unaffected_claim_changed"):
            lock.verify(fixed)

    def test_deleting_failed_claim_never_manufactures_success(self):
        value = candidate()
        lock = RepairLock.bind(value, ["C7"])
        value["claims"].pop()
        with self.assertRaisesRegex(AgentError, "typed_feedback_required_claim_set_changed"):
            lock.verify(value)

    def test_duplicate_initial_identity_is_not_repairable(self):
        value = candidate()
        value["claims"][6]["id"] = "C1"
        with self.assertRaisesRegex(AgentError, "typed_feedback_claim_identity_ambiguous"):
            RepairLock.bind(value, ["C1"])

    def test_bad_initial_identity_is_not_repairable(self):
        value = candidate()
        value["claims"][0]["id"] = "secret payload"
        with self.assertRaisesRegex(AgentError, "typed_feedback_claim_identity_ambiguous"):
            claim_objects(value)

    def test_enum_feedback_does_not_echo_bad_input(self):
        value = candidate()
        value["claims"][1]["status"] = "private token not to echo"
        feedback = validation_feedback(error_for(value), value)
        self.assertEqual(feedback["issues"][0]["code"], "literal_error")
        self.assertNotIn("private token", json.dumps(feedback))
        self.assertEqual(feedback["affected_claim_ids"], ["C2"])

    def test_extra_property_name_is_not_echoed(self):
        value = candidate()
        value["claims"][2]["sensitive_field_name"] = "sensitive body"
        feedback = validation_feedback(error_for(value), value)
        text = json.dumps(feedback)
        self.assertNotIn("sensitive", text)
        self.assertIn("unrecognized_field", text)
        self.assertEqual(feedback["affected_claim_ids"], ["C3"])

    def test_mixed_requires_both_real_relations(self):
        value = candidate()
        value["claims"][0]["status"] = "mixed"
        self.assertEqual(validation_feedback(error_for(value), value)["issues"][0]["code"], "claim_mixed_requires_both_relations")
        value["claims"][0]["edges"].append({**value["claims"][0]["edges"][0], "relation": "contradicts"})
        SelectorSummaryV5.model_validate(value, strict=True)

    def test_contradicted_requires_contradiction(self):
        value = candidate()
        value["claims"][0]["status"] = "contradicted"
        self.assertEqual(validation_feedback(error_for(value), value)["issues"][0]["code"], "claim_contradicted_requires_contradicts")

    def test_insufficient_requires_truthful_gap_without_forced_support(self):
        value = candidate()
        value["claims"][0].update(status="insufficient", edges=[])
        self.assertEqual(validation_feedback(error_for(value), value)["issues"][0]["code"], "claim_insufficient_requires_gap")
        value["claims"][0]["limitations"] = ["The supplied source does not establish the proposition."]
        SelectorSummaryV5.model_validate(value, strict=True)

    def test_legacy_rule_is_unchanged_and_not_automatically_repaired(self):
        value = candidate()
        value["claims"][0]["edges"][0]["relation"] = "context_only"
        with self.assertRaises(ValidationError) as captured:
            SelectorSummary.model_validate(value, strict=True)
        self.assertEqual(captured.exception.errors()[0]["type"], "value_error")
        feedback = validation_feedback(captured.exception, value)
        self.assertFalse(feedback["repairable"])
        self.assertEqual(feedback["issues"][0]["code"], "unregistered_validation_rule")

    def test_no_transport_or_acl_failure_is_accepted_as_content_feedback(self):
        with self.assertRaises(TypeError):
            validation_feedback(AgentError("source_permission_revoked"), candidate())

    def test_duplicate_json_and_nonfinite_never_lose_identity(self):
        for raw in ('{"claims":[],"claims":[]}', '{"value":NaN}', 'not JSON', '[]'):
            with self.subTest(raw=raw), self.assertRaises(AgentError):
                strict_object(raw)

    def test_repair_lock_is_immutable_and_returns_hashes(self):
        value = candidate()
        lock = RepairLock.bind(value, ["C1"])
        original = lock.contract()
        value["claims"][2]["statement"] = "External mutation"
        self.assertEqual(lock.contract(), original)
        self.assertNotIn("Fixture proposition", json.dumps(original))
        self.assertEqual(len(original["unaffected_claim_sha256"]), 6)

    def test_new_bounded_conditions_prevent_unbounded_note_fanout(self):
        value = candidate()
        value["claims"][0]["edges"][0]["conditions"] = ["x" * 501]
        self.assertEqual(validation_feedback(error_for(value), value)["issues"][0]["code"], "string_too_long")

    def test_local_schema_is_not_semantic_entailment(self):
        value = candidate()
        # Valid labels and IDs still say nothing about whether Q1 really proves
        # the arbitrary proposition. The separate source reviewer is required.
        SelectorSummaryV5.model_validate(value, strict=True)
        self.assertEqual(value["claims"][0]["statement"], "Fixture proposition 1")


if __name__ == "__main__":
    unittest.main()
