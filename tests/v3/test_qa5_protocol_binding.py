"""AST-loaded pure fixture contracts, not a provider/runtime import test.

The host has the locked Pydantic but not every Linux Agent dependency. Load the
actual narrow definitions without importing a DB driver or provider; no model,
SQL, network, schema repair or test substitute for semantic evidence is used.
"""
from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, create_model

from citequest.agents_v3.contracts import AgentError, digest
from citequest.research_v3.current_requirements import RequirementOpinion, RequirementReview
from citequest.research_v3.review_grid import ReviewGrid

ROOT = Path(__file__).resolve().parents[2]


def load_definitions(path: str, names: set[str], namespace: dict) -> None:
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in nodes} == names
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)


class SyntheticScope:
    """Explicit transport fixture; these Q IDs are never product evidence refs."""
    def __init__(self) -> None:
        self.binding = {"binding_sha256":digest("EXPLICIT_SYNTHETIC_PROTOCOL_ONLY")}
        self.units = {}

    def compact_candidate(self, value: dict, track_review: bool = False) -> dict:
        return copy.deepcopy(value)

def synthetic_grid() -> tuple[dict, ReviewGrid]:
    scope, candidate = SyntheticScope(), {"claims":[]}
    for index in range(1, 9):
        selectors = ["Q" + str(index * 3 + offset) for offset in range(3 if index <= 4 else 2)]
        scope.units.update({key:SimpleNamespace(reference="EXPLICIT_SYNTHETIC:" + key) for key in selectors})
        candidate["claims"].append({"id":"C" + str(index), "statement":"Synthetic protocol cell only; no scientific assertion.",
            "status":"insufficient", "limitations":["No academic evidence is supplied in this protocol fixture."],
            "edges":[{"quote_id":key, "relation":"context_only", "rationale":"Synthetic transport fixture only.", "conditions":[]} for key in selectors]})
    grid = ReviewGrid(scope, candidate)
    assert len(grid.identities) == 8 and sum(map(len, grid.identities.values())) == 20
    return candidate, grid

def synthetic_contract(candidate: dict, grid: ReviewGrid, user_text: str) -> tuple[type, dict]:
    """Bind fixture identities in the advertised schema, not after generation.

    These hashes identify a transport-only candidate, not an accepted production
    RequirementsCandidate. With no requirement IDs issued by this fixture, an
    opinion would be an invented identity. Review decisions remain model opinions
    and never become an academic acceptance claim.
    """
    expected = {"candidate_sha256":digest(candidate), "user_turn_sha256":digest(user_text),
        "required_requirement_review_ids":[]}
    model = create_model("QA5SyntheticBoundFlatReview_" + grid.binding["binding_sha256"][:16],
        __base__=combined_review_model(grid),
        candidate_sha256=(Literal[expected["candidate_sha256"]], ...),
        user_turn_sha256=(Literal[expected["user_turn_sha256"]], ...),
        opinions=(list[RequirementOpinion], Field(min_length=0, max_length=0)))
    return model, expected


def fixture():
    namespace = {"__name__":__name__, "copy":copy, "digest":digest, "SimpleNamespace":SimpleNamespace,
        "Literal":Literal, "Field":Field, "create_model":create_model, "BaseModel":BaseModel,
        "AgentError":AgentError, "RequirementOpinion":RequirementOpinion,
        "RequirementReview":RequirementReview, "ReviewGrid":ReviewGrid}
    load_definitions("src/citequest/research_v3/qa_v5_contracts.py", {"combined_review_model"}, namespace)
    load_definitions("tests/v3/test_qa5_protocol_binding.py",
        {"SyntheticScope", "synthetic_grid", "synthetic_contract"}, namespace)
    candidate, grid = namespace["synthetic_grid"]()
    model, expected = namespace["synthetic_contract"](candidate, grid, "Explicit protocol-only user turn")
    value = {"candidate_sha256":expected["candidate_sha256"], "user_turn_sha256":expected["user_turn_sha256"],
        "opinions":[], "current_turn_coverage":"complete", "decision":"accepted",
        "claims":{cid:{"verdict":"insufficient", "conditions_preserved":True, "explanation":"Synthetic formatting cell.",
            "quotes":{qid:{"relation":"matches_declared_relations", "conditions_preserved":True,
                "explanation":"Synthetic formatting cell."} for qid in qids}}
            for cid,qids in grid.identities.items()},
        "source_instruction_injection_detected":False, "requested_stance_bias_detected":False,
        "request_coverage":"complete", "uncovered_requests":[]}
    return model, grid, expected, value


class ProtocolBindingControls(unittest.TestCase):
    def test_full_eight_claim_twenty_span_contract_retains_ten_root_fields(self):
        model, grid, expected, value = fixture()
        self.assertEqual(len(model.model_fields), 10)
        self.assertEqual(len(grid.identities), 8)
        self.assertEqual(sum(map(len, grid.identities.values())), 20)
        self.assertEqual(model.model_validate(value, strict=True).candidate_sha256, expected["candidate_sha256"])
        schema = model.model_json_schema()
        self.assertEqual(schema["properties"]["candidate_sha256"]["const"], expected["candidate_sha256"])
        self.assertEqual(schema["properties"]["opinions"]["maxItems"], 0)

    def test_both_wrong_goal_hashes_fail_before_receipt_acceptance(self):
        model, _, _, value = fixture()
        for field in ("candidate_sha256", "user_turn_sha256"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                model.model_validate({**value, field:"0"*64}, strict=True)

    def test_invented_requirement_opinion_is_rejected(self):
        model, _, _, value = fixture()
        with self.assertRaises(ValidationError):
            model.model_validate({**value, "opinions":[{"requirement_id":"R1", "verdict":"correct",
                "rationale":"Invented fixture requirement."}]}, strict=True)

    def test_missing_claim_quote_and_extra_root_never_become_a_protocol_pass(self):
        model, grid, _, value = fixture()
        first = next(iter(grid.identities))
        for fault in ("claim", "quote", "extra"):
            broken = copy.deepcopy(value)
            if fault == "claim":
                broken["claims"].pop(first)
            elif fault == "quote":
                broken["claims"][first]["quotes"].pop(grid.identities[first][0])
            else:
                broken["unadvertised"] = True
            with self.subTest(fault=fault), self.assertRaises(ValidationError):
                model.model_validate(broken, strict=True)

    def test_transport_check_does_not_force_positive_semantic_decisions(self):
        model, _, _, value = fixture()
        parsed = model.model_validate({**value, "decision":"needs_clarification",
            "current_turn_coverage":"ambiguous"}, strict=True)
        self.assertEqual(parsed.decision, "needs_clarification")


if __name__ == "__main__":
    unittest.main()
