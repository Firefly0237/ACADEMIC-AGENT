"""Pure fixed-identity controls with explicit synthetic semantic opinions.

The scope is a local shape substitute. No source truth, model, SQL, HTTP or
product acceptance is simulated by these contract checks.
"""
from __future__ import annotations

import copy
import unittest

from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.qa_answer_audit import AnswerAudit
from citequest.research_v3.current_requirements import RequirementsCandidate, initial_state
from citequest.research_v3.qa_v5_contracts import bound_combined_review_model
from citequest.research_v3.review_grid import ReviewGrid


class SyntheticScope:
    """Only the two shape inputs used by the real ReviewGrid constructor."""

    binding = {"binding_sha256": "c" * 64}

    def compact_candidate(self, candidate, *, track_review=False):
        return copy.deepcopy(candidate)


def fixture(*, count=2, long_requirement=False):
    texts = ["Synthetic requested item " + str(index + 1) for index in range(count)]
    if long_requirement:
        texts = ["x" * 4500]
    text = "\n".join(texts)
    turn_sha = digest(text)
    requirements = []
    offset = 0
    for index, instruction in enumerate(texts):
        spans = [
            {"turn_id": "U1", "start": offset + start, "end": offset + min(start + 1500, len(instruction)),
             "text": instruction[start:start + 1500], "turn_sha256": turn_sha}
            for start in range(0, len(instruction), 1500)
        ]
        requirements.append({
            "id": "R" + str(index + 1), "category": "question", "spans": spans,
            "state": "active", "introduced_turn_id": "U1", "retired_turn_id": None, "replacement_ids": [],
        })
        offset += len(instruction) + 1
    candidate = RequirementsCandidate.model_validate({
        "base_sha256": digest(initial_state("a" * 64).model_dump()), "delta_sha256": "b" * 64,
        "user_turn": {"turn_id": "U1", "text": text},
        "projected": {"authority_sha256": "a" * 64, "revision": 1,
                      "turns": [{"turn_id": "U1", "text_sha256": turn_sha}],
                      "requirements": requirements, "accepted": False},
        "required_review_ids": [item["id"] for item in requirements], "clarification": None,
    }, strict=True)
    answer = {
        "claims": [{"id": "C1", "statement": "Explicit synthetic claim.", "edges": [{"quote_id": "Q1"}]}],
        "gaps": ["Explicit synthetic evidence limit."],
        "presentation": {"format": "prose", "paragraphs": [["C1"]]},
    }
    grid = ReviewGrid(SyntheticScope(), answer, request_coverage=False)
    base = bound_combined_review_model(grid, digest(candidate.model_dump()), turn_sha)
    audit = AnswerAudit(base, grid, candidate)
    value = {
        "candidate_sha256": digest(candidate.model_dump()), "answer_candidate_sha256": grid.binding["candidate_sha256"],
        "user_turn_sha256": turn_sha,
        "opinions": [{"requirement_id": item.id, "verdict": "correct", "rationale": "Synthetic interpretation."}
                     for item in candidate.projected.requirements],
        "current_turn_coverage": "complete", "decision": "accepted",
        "source_instruction_injection_detected": False, "requested_stance_bias_detected": False,
        "claims": {"C1": {"verdict": "entailed", "conditions_preserved": True, "explanation": "Synthetic opinion.",
                          "quotes": {"Q1": {"relation": "matches_declared_relations", "conditions_preserved": True,
                                             "explanation": "Synthetic opinion."}}}},
        "request_coverage": "complete", "uncovered_requests": [],
        "requirement_coverage": {item.id: {"status": "fulfilled", "locations": ["C1"], "explanation": "Synthetic coverage."}
                                 for item in candidate.projected.requirements},
        "gap_review": {"G1": {"verdict": "faithful", "explanation": "Synthetic gap opinion."}},
    }
    return audit, value, candidate, grid, base


class AnswerAuditControls(unittest.TestCase):
    def test_fixed_properties_locations_and_hashes_reject_invented_or_missing_identity(self):
        audit, value, _, _, _ = fixture()
        schema = audit.model.model_json_schema()

        def closed_objects(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node["additionalProperties"], False)
                    self.assertEqual(set(node.get("required", [])), set(node.get("properties", {})))
                for item in node.values():
                    closed_objects(item)
            elif isinstance(node, list):
                for item in node:
                    closed_objects(item)

        closed_objects(schema)
        parsed_coverage = audit.model.model_validate(value, strict=True).requirement_coverage
        self.assertEqual(set(type(parsed_coverage).model_fields), {"R1", "R2"})
        for fault in ("missing_R", "extra_R", "extra_G", "foreign_location", "wrong_goal", "wrong_answer"):
            wrong = copy.deepcopy(value)
            if fault == "missing_R":
                wrong["requirement_coverage"].pop("R1")
            elif fault == "extra_R":
                wrong["requirement_coverage"]["R3"] = copy.deepcopy(wrong["requirement_coverage"]["R1"])
            elif fault == "extra_G":
                wrong["gap_review"]["G2"] = copy.deepcopy(wrong["gap_review"]["G1"])
            elif fault == "foreign_location":
                wrong["requirement_coverage"]["R1"]["locations"] = ["C99"]
            elif fault == "wrong_goal":
                wrong["candidate_sha256"] = "0" * 64
            else:
                wrong["answer_candidate_sha256"] = "0" * 64
            with self.subTest(fault=fault), self.assertRaises(ValidationError):
                audit.model.model_validate(wrong, strict=True)

    def test_global_coverage_and_wire_rids_agree_without_free_text_or_implicit_acceptance(self):
        audit, value, _, _, _ = fixture()
        value["requirement_coverage"]["R2"].update(status="bounded_evidence_gap", locations=["G1"])
        value.update(request_coverage="partial", uncovered_requests=["R2"])
        result = audit.assess(value)
        self.assertEqual(result["wire_uncovered_requirement_ids"], ["R2"])
        self.assertEqual(result["request_coverage"], "partial")
        self.assertEqual(result["repair_claim_ids"], [])
        for field, wrong_value in (("request_coverage", "complete"), ("uncovered_requests", []),
                                   ("uncovered_requests", ["Invent another question"]),
                                   ("uncovered_requests", ["R2", "R2"])):
            with self.subTest(field=field, value=wrong_value), self.assertRaises(ValidationError):
                audit.model.model_validate({**value, field: wrong_value}, strict=True)
        empty_location = copy.deepcopy(value)
        empty_location["requirement_coverage"]["R2"]["locations"] = []
        with self.assertRaises(ValidationError):
            audit.model.model_validate(empty_location, strict=True)

    def test_negative_gap_opinions_and_repair_targets_remain_independent(self):
        audit, value, candidate, _, _ = fixture()
        value["requirement_coverage"]["R1"].update(status="format_mismatch", locations=["C1", "presentation"])
        value["requirement_coverage"]["R2"].update(status="bounded_evidence_gap", locations=["G1"])
        value.update(request_coverage="partial", uncovered_requests=["R1", "R2"])
        for verdict in ("contradictory", "unsubstantiated"):
            value["gap_review"]["G1"]["verdict"] = verdict
            result = audit.assess(value)
            self.assertEqual(result["bad_gap_ids"], ["G1"])
            self.assertEqual(result["gap_review"]["G1"]["verdict"], verdict)
            self.assertEqual(result["repair_claim_ids"], ["C1"])
            self.assertEqual(result["incomplete_requirement_ids"], ["R1", "R2"])
            self.assertIn(candidate.projected.requirements[0].spans[0].text, result["uncovered_requests"][0])
        value["requirement_coverage"]["R1"].update(status="missing_response", locations=[])
        self.assertEqual(audit.assess(value)["repair_claim_ids"], [])

    def test_full_server_projection_capacity_and_old_grid_contract_are_preserved(self):
        for kwargs in ({"count": 24}, {"count": 1, "long_requirement": True}):
            audit, value, candidate, grid, base = fixture(**kwargs)
            before = base.model_json_schema()
            for row in value["requirement_coverage"].values():
                row.update(status="bounded_evidence_gap", locations=["G1"])
            value.update(request_coverage="partial", uncovered_requests=list(audit.requirement_ids))
            result = audit.assess(value)
            self.assertEqual(len(result["uncovered_requests"]), len(candidate.projected.requirements))
            for requirement, projected in zip(candidate.projected.requirements, result["uncovered_requests"]):
                self.assertIn("\n".join(span.text for span in requirement.spans), projected)
            AnswerAudit(base, grid, candidate)
            self.assertEqual(base.model_json_schema(), before)
            self.assertNotIn("requirement_coverage", before["properties"])
            legacy_grid = ReviewGrid(SyntheticScope(), grid.candidate)
            legacy_base = bound_combined_review_model(legacy_grid, digest(candidate.model_dump()), digest(candidate.user_turn.text))
            with self.assertRaises(AgentError) as error:
                AnswerAudit(legacy_base, legacy_grid, candidate)
            self.assertEqual(error.exception.code, "qa_answer_audit_facts_only_base_required")


if __name__ == "__main__":
    unittest.main()
