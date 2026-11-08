"""Pure mechanism/authority boundary checks; the judge is an explicit test double.

No model, provider, database, index, or product quality result is implied.
"""
import unittest
from dataclasses import replace
from unittest.mock import patch

from citequest.agents_v3.contracts import AgentError
from citequest.research_v3.decisionrag import DecisionReceipt, Document, NavigationLimits, Passage, navigate


def documents():
    return tuple(Document(f"D{i}", f"Actual fixture document {i}", f"V{i}",
        tuple(Passage(f"D{i}P{j}", f"Fixture passage {i}/{j}.", f"D{i}", f"V{i}", f"ref:{i}:{j}")
              for j in range(1, 3))) for i in range(1, 4))


class FixtureJudge:
    def __init__(self, *, fault=None):
        self.fault = fault
        self.calls = []
        self.preflights = []
        self.binding = {"provider":"explicit_test_double", "version":"1"}

    def preflight(self, request):
        self.preflights.append(request)
        if self.fault == "capacity" and len(request.candidates) > 4:
            raise AgentError("fixture_capacity")

    def score(self, request):
        self.calls.append(request)
        if self.fault == "mutated_model" and len(self.calls) == 2:
            self.binding["version"] = "2"
        scores = {identifier:(index + 1)/len(request.candidates)
                  for index,(identifier,_) in enumerate(request.candidates)}
        if self.fault == "extra_id":
            scores["invented"] = 1.0
        if self.fault == "nan":
            scores[next(iter(scores))] = float("nan")
        return DecisionReceipt(request.fingerprint(), scores, self.binding, {} if self.fault == "empty_usage" else {"fixture":True})


class DecisionNavigation(unittest.TestCase):
    def run_navigation(self, judge, **changes):
        args = {"scope_binding":"server-scope-v1", "check_scope":lambda:"server-scope-v1",
                "before_decision":lambda request:None,
                "limits":NavigationLimits(document_keep=2, passage_keep=3)}
        args.update(changes)
        return navigate("Which method?", documents(), judge, **args)

    def test_actual_parent_children_shared_frontier_stable_prune_and_original_refs(self):
        judge = FixtureJudge()
        admissions = []
        result = self.run_navigation(judge, before_decision=lambda request:admissions.append(request.fingerprint()))
        self.assertEqual(result["document_ids_kept"], ["D3", "D2"])
        self.assertEqual(result["pruned_document_ids"], ["D1"])
        self.assertEqual(result["evidence_ref_ids"], ["ref:2:2", "ref:2:1", "ref:3:2"])
        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(admissions, [request.fingerprint() for request in judge.calls])
        self.assertEqual([len(request.candidates) for request in judge.calls], [3, 4])
        self.assertEqual(len(judge.preflights[1].candidates), 6)  # All leaves checked before pruning.
        self.assertFalse(result["paper_benchmark_reproduction"])
        state, questions, aliases = judge.calls[1].wire_data()
        self.assertEqual(len(state["candidates"]), len(questions))
        self.assertEqual(set(aliases.values()), {"D3P1", "D3P2", "D2P1", "D2P2"})

    def test_full_unpruned_frontier_capacity_refuses_before_first_inference(self):
        judge = FixtureJudge(fault="capacity")
        with self.assertRaisesRegex(AgentError, "fixture_capacity"):
            self.run_navigation(judge)
        self.assertEqual(judge.calls, [])

    def test_current_scope_revocation_after_inference_stops_delivery(self):
        judge = FixtureJudge()
        def current_scope():
            return "revoked" if judge.calls else "server-scope-v1"
        with self.assertRaisesRegex(AgentError, "scope_or_version_changed"):
            self.run_navigation(judge, check_scope=current_scope)
        self.assertEqual(len(judge.calls), 1)

    def test_active_deadline_includes_actual_scope_io(self):
        judge, clock = FixtureJudge(), [0]
        def slow_scope():
            clock[0] = 31
            return "server-scope-v1"
        with patch("citequest.research_v3.decisionrag.time.monotonic",side_effect=lambda:clock[0]), \
                self.assertRaisesRegex(AgentError,"active_budget_exhausted"):
            self.run_navigation(judge,check_scope=slow_scope)
        self.assertEqual(judge.calls, [])

    def test_exact_probabilities_and_immutable_model_binding_required(self):
        for fault, error in (("extra_id", "identity_mismatch"), ("nan", "invalid_probability"),
                             ("mutated_model", "judge_changed_between_levels"), ("empty_usage", "usage_binding_missing")):
            with self.subTest(fault=fault), self.assertRaisesRegex(AgentError, error):
                self.run_navigation(FixtureJudge(fault=fault))

    def test_malformed_hierarchy_and_input_fail_before_model(self):
        original = documents()
        invalid = (replace(original[0], passages=(object(),)),)
        judge = FixtureJudge()
        with self.assertRaisesRegex(AgentError, "passage_identity"):
            navigate("Question", invalid, judge, scope_binding="scope", check_scope=lambda:"scope",
                     before_decision=lambda _:None)
        wrong_version = (replace(original[0], passages=(replace(original[0].passages[0], source_version="V9"),)),)
        with self.assertRaisesRegex(AgentError, "actual_source_hierarchy"):
            navigate("Question", wrong_version, judge, scope_binding="scope", check_scope=lambda:"scope",
                     before_decision=lambda _:None)
        with self.assertRaisesRegex(AgentError, "bounded_server_scope"):
            navigate(None, original, judge, scope_binding="scope", check_scope=lambda:"scope",
                     before_decision=lambda _:None)
        self.assertEqual(judge.calls, [])


if __name__ == "__main__":
    unittest.main()
