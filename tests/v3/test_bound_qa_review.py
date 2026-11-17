"""Pure actual Pydantic definitions; no provider, graph, SQL or model call.

AST loading avoids host's absent Linux dependencies; this is not a runtime
import or product acceptance result. The complete real ReviewGrid is retained.
"""
from __future__ import annotations

import copy
import re
import sys
import unittest
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ValidationError, create_model

from citequest.agents_v3.contracts import AgentError, digest
from citequest.research_v3.current_requirements import RequirementReview

sys.path.insert(0,str(Path(__file__).parent))
from test_qa5_protocol_binding import fixture, load_definitions


def models():
    _,grid,_,value = fixture()
    ns = {"__name__":__name__,"re":re,"Literal":Literal,"BaseModel":BaseModel,"create_model":create_model,
          "RequirementReview":RequirementReview,"AgentError":AgentError,"digest":digest}
    load_definitions("src/citequest/research_v3/qa_v5_contracts.py",{"combined_review_model","bound_combined_review_model"},ns)
    legacy = ns["combined_review_model"](grid)
    bound = ns["bound_combined_review_model"](grid,"a"*64,"b"*64)
    value.update(candidate_sha256="a"*64,user_turn_sha256="b"*64)
    return ns,grid,legacy,bound,value


class BoundReviewControls(unittest.TestCase):
    def test_two_server_identities_are_literal_and_answer_grid_hash_is_rejected(self):
        _,grid,_,bound,value = models()
        schema = bound.model_json_schema()
        self.assertEqual(schema["properties"]["candidate_sha256"]["const"],"a"*64)
        self.assertEqual(schema["properties"]["user_turn_sha256"]["const"],"b"*64)
        for field,wrong in (("candidate_sha256",grid.binding["candidate_sha256"]),("user_turn_sha256","c"*64)):
            with self.subTest(field=field),self.assertRaises(ValidationError):
                bound.model_validate({**value,field:wrong},strict=True)

    def test_all_eight_claims_twenty_cells_and_independent_negative_opinions_retained(self):
        _,grid,legacy,bound,value = models()
        self.assertEqual(set(bound.model_fields),set(legacy.model_fields))
        self.assertEqual(len(grid.identities),8)
        self.assertEqual(sum(map(len,grid.identities.values())),20)
        value.update(opinions=[{"requirement_id":"R1","verdict":"incorrect","rationale":"Explicit synthetic disagreement."}],
                     decision="needs_revision",current_turn_coverage="incomplete")
        parsed = bound.model_validate(value,strict=True)
        self.assertEqual(parsed.opinions[0].verdict,"incorrect")
        self.assertEqual(parsed.decision,"needs_revision")
        wrong = copy.deepcopy(value)
        wrong["claims"]["C1"]["quotes"].pop(next(iter(wrong["claims"]["C1"]["quotes"])))
        with self.assertRaises(ValidationError):
            bound.model_validate(wrong,strict=True)

    def test_legacy_schema_unchanged_and_invalid_control_input_fails_before_schema(self):
        ns,grid,legacy,_,_ = models()
        before = legacy.model_json_schema()
        ns["bound_combined_review_model"](grid,"c"*64,"d"*64)
        self.assertEqual(ns["combined_review_model"](grid).model_json_schema(),before)
        self.assertNotIn("const",before["properties"]["candidate_sha256"])
        for wrong in (None,"not a SHA","A"*64):
            with self.subTest(wrong=wrong),self.assertRaises(AgentError):
                ns["bound_combined_review_model"](grid,wrong,"b"*64)


if __name__ == "__main__":
    unittest.main()
