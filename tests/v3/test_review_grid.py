"""Independent fixed-property grid constraints; opinions are explicit synthetic fixtures."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent))
from test_quote_spans_v3 import source, wire

from citequest.agents_v3.workset import as_answer
from citequest.research_v3.contracts import validate_review
from citequest.research_v3.quote_spans import DocumentSummaryV3, QuoteRegistry, SelectorAnswer, SelectorSummary
from citequest.research_v3.review_grid import GridProtocolError, ReviewGrid


def case(*, forty=False):
    evidence = [source("".join(chr(65 + i % 26) * 900 for i in range(40 if forty else 8)))]
    first = QuoteRegistry(evidence).scope(evidence)
    value = wire("Q1")
    choices = [["Q4", "Q5"], ["Q5", "Q6", "Q7"], ["Q7"], ["Q7"]]
    if forty:
        choices = [["Q" + str(j + 1) for j in range(i * 5, (i + 1) * 5)] for i in range(8)]
    value["claims"] = [
        {
            **copy.deepcopy(value["claims"][0]),
            "id": "C" + str(i + 1),
            "edges": [{**copy.deepcopy(value["claims"][0]["edges"][0]), "quote_id": q} for q in ids],
        }
        for i, ids in enumerate(choices)
    ]
    answer = first.expand_summary(SelectorSummary.model_validate(value, strict=True))
    canonical = answer.model_dump()
    scope = QuoteRegistry(evidence).scope(evidence, candidates=[canonical])
    return answer, scope, ReviewGrid(scope, canonical)


def opinion(grid):
    return {
        "claims": {
            identifier: {
                "verdict": "entailed",
                "conditions_preserved": True,
                "explanation": "Explicit fixture only.",
                "quotes": {
                    q: {
                        "relation": "matches_declared_relations",
                        "conditions_preserved": True,
                        "explanation": "Explicit relation fixture.",
                    }
                    for q in ids
                },
            }
            for identifier, ids in grid.identities.items()
        },
        "source_instruction_injection_detected": False,
        "requested_stance_bias_detected": False,
        "request_coverage": "complete",
        "uncovered_requests": [],
    }


class GridControls(unittest.TestCase):
    def setUp(self):
        self.answer, self.scope, self.grid = case()
        self.value = opinion(self.grid)

    def test_real_failure_shape_C4_cannot_borrow_issued_Q5(self):
        self.assertEqual(self.grid.identities["C4"], ["Q7"])
        self.assertIn("Q5", self.scope.allowed)
        self.value["claims"]["C4"]["quotes"]["Q5"] = copy.deepcopy(self.value["claims"]["C1"]["quotes"]["Q5"])
        self.assertFalse(Draft202012Validator(self.grid.schema()).is_valid(self.value))
        with self.assertRaises(ValidationError):
            self.grid.parse(json.dumps(self.value))

    def test_every_object_has_fixed_required_properties_and_no_extra(self):
        def check(node):
            if isinstance(node, dict):
                if node.get("type") == "object":
                    self.assertIs(node["additionalProperties"], False)
                    self.assertEqual(set(node.get("required", [])), set(node.get("properties", {})))
                for item in node.values():
                    check(item)
            elif isinstance(node, list):
                for item in node:
                    check(item)

        check(self.grid.schema())
        self.assertNotIn('"claim_id"', json.dumps(self.grid.schema()))
        self.assertNotIn('"edge_ref_ids"', json.dumps(self.grid.schema()))

    def test_valid_grid_preserves_claims_refs_and_semantic_gate(self):
        parsed = self.grid.parse(json.dumps(self.value))
        review = self.grid.canonical(parsed)
        validate_review(as_answer(self.answer), review)
        self.assertEqual([row.claim_id for row in review.claims], [claim.id for claim in self.answer.claims])
        self.assertEqual(len(review.claims[1].edge_ref_ids), 3)
        self.assertEqual(review.request_coverage, "complete")

    def test_missing_claim_span_or_opinion_is_never_defaulted(self):
        for level in ("claim", "span", "opinion"):
            value = copy.deepcopy(self.value)
            if level == "claim":
                value["claims"].pop("C4")
            elif level == "span":
                value["claims"]["C4"]["quotes"].pop("Q7")
            else:
                value["claims"]["C4"]["quotes"]["Q7"].pop("conditions_preserved")
            with self.subTest(level=level), self.assertRaises(ValidationError):
                self.grid.parse(json.dumps(value))

    def test_unknown_claim_unknown_span_and_legacy_array_rejected(self):
        for level in ("claim", "span", "legacy"):
            value = copy.deepcopy(self.value)
            if level == "claim":
                value["claims"]["C9"] = copy.deepcopy(value["claims"]["C4"])
            elif level == "span":
                value["claims"]["C4"]["quotes"]["Q99999"] = copy.deepcopy(value["claims"]["C4"]["quotes"]["Q7"])
            else:
                value["claims"] = list(value["claims"].values())
            with self.subTest(level=level), self.assertRaises(ValidationError):
                self.grid.parse(json.dumps(value))

    def test_duplicate_JSON_keys_are_rejected_instead_of_overwritten(self):
        raw = json.dumps(self.value)
        raw = raw.replace('"C4": {', '"C4": {}, "C4": {', 1)
        with self.assertRaises(GridProtocolError) as caught:
            self.grid.parse(raw)
        self.assertEqual(caught.exception.code, "review_grid_duplicate_property")

    def test_single_span_relation_failure_overrides_entailed_claim(self):
        for verdict in ("does_not_match", "insufficient"):
            value = copy.deepcopy(self.value)
            value["claims"]["C4"]["quotes"]["Q7"]["relation"] = verdict
            review = self.grid.canonical(self.grid.parse(json.dumps(value)))
            with self.subTest(verdict=verdict), self.assertRaises(ValueError):
                validate_review(as_answer(self.answer), review)

    def test_single_span_lost_numeric_condition_rejects(self):
        self.value["claims"]["C4"]["quotes"]["Q7"]["conditions_preserved"] = False
        review = self.grid.canonical(self.grid.parse(json.dumps(self.value)))
        self.assertFalse(review.claims[-1].numeric_and_conditions_preserved)
        with self.assertRaises(ValueError):
            validate_review(as_answer(self.answer), review)

    def test_insufficient_claim_cannot_hide_unverified_selected_quote_relation(self):
        candidate = self.answer.model_dump()
        candidate["claims"][0]["status"] = "insufficient"
        grid = self.scope.review_grid(candidate)
        value = opinion(grid)
        value["claims"]["C1"]["verdict"] = "insufficient"
        value["claims"]["C1"]["quotes"]["Q4"]["relation"] = "insufficient"
        review = grid.canonical(grid.parse(json.dumps(value)))
        self.assertEqual(review.claims[0].verdict, "invalid")
        with self.assertRaises(ValueError):
            validate_review(as_answer(DocumentSummaryV3.model_validate(candidate, strict=True)), review)

    def test_empty_scope_explicit_gap_has_empty_fixed_quotes_and_no_semantic_fill(self):
        scope = QuoteRegistry([], allow_empty=True).scope([])
        candidate = {"kind": "qa", "requested_direction": "neutral", "claims": [{"id": "C1",
            "statement": "The bounded source scope contains no evidence for this question.", "status": "insufficient",
            "edges": [], "limitations": ["No assertion that relevant literature does not exist."]}],
            "gaps": ["Source evidence is missing."], "evidence_status": "insufficient", "retrieval_scope": "Empty bounded fixture."}
        answer = scope.expand_answer(SelectorAnswer.model_validate(candidate, strict=True))
        grid = scope.review_grid(answer.model_dump())
        value = opinion(grid)
        value["claims"]["C1"]["verdict"] = "insufficient"
        self.assertEqual(value["claims"]["C1"]["quotes"], {})
        validate_review(answer, grid.canonical(grid.parse(json.dumps(value))))

    def test_injection_and_bias_flags_not_dropped(self):
        for flag in ("source_instruction_injection_detected", "requested_stance_bias_detected"):
            value = copy.deepcopy(self.value)
            value[flag] = True
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                validate_review(as_answer(self.answer), self.grid.canonical(self.grid.parse(json.dumps(value))))

    def test_partial_request_is_explicit_and_complete_cannot_keep_omissions(self):
        self.value.update(
            request_coverage="partial", uncovered_requests=["Evaluation metric was not explicitly answered."]
        )
        review = self.grid.canonical(self.grid.parse(json.dumps(self.value)))
        validate_review(as_answer(self.answer), review)
        self.assertEqual(review.request_coverage, "partial")
        self.value["request_coverage"] = "complete"
        with self.assertRaises(ValidationError):
            self.grid.parse(json.dumps(self.value))

    def test_empty_omissions_cannot_disguise_partial(self):
        self.value["request_coverage"] = "partial"
        with self.assertRaises(ValidationError):
            self.grid.parse(json.dumps(self.value))

    def test_different_scope_parsed_model_is_not_rebound(self):
        candidate = self.answer.model_dump()
        candidate["claims"][0]["statement"] += " Different requested comparison."
        other = ReviewGrid(self.scope, candidate)
        with self.assertRaises(GridProtocolError):
            self.grid.canonical(other.parse(json.dumps(opinion(other))))

    def test_eight_claims_forty_selected_spans_schema_is_bounded(self):
        answer, _, grid = case(forty=True)
        self.assertEqual(grid.statistics()["selected_quote_cells"], 40)
        self.assertLess(grid.statistics()["schema_utf8_bytes"], 16000)
        validate_review(as_answer(answer), grid.canonical(grid.parse(json.dumps(opinion(grid)))))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(GridControls)
    )
    args.out.write_text(
        json.dumps(
            {
                "status": "PASS" if result.wasSuccessful() else "FAIL",
                "controls": result.testsRun,
                "scope": "dynamic Pydantic/JSONSchema real validation, explicit synthetic opinions; no model/DB",
                "capacity": case(forty=True)[2].statistics(),
                "model_requests_sent": 0,
            },
            indent=2,
        )
        + "\n"
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
