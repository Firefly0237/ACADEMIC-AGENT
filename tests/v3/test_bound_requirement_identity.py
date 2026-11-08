"""Pure real Pydantic controls; no SDK, graph execution, DB, network or model.

Research definitions are AST-loaded because the host lacks the locked Linux
Agent dependencies. This is schema/validator verification, not product proof.
"""
from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints, ValidationError, create_model, model_validator

from citequest.agents_v3.contracts import AgentError, Strict, digest
from citequest.research_v3.brief import BriefField
from citequest.research_v3.current_requirements import RequirementDelta, RequirementReview, UserTurn, initial_state
from citequest.research_v3.requirements_identity import (
    bound_identity_model,
    bound_requirement_plan_model,
    bound_requirement_review_model,
)
from citequest.research_v3.research_plan import Experiment, QuestionPlan, ResearchPlanReview
from citequest.research_v3.school_contracts import CrossReview, RequestCoverage, SchoolSection, SectionReview

ROOT = Path(__file__).resolve().parents[2]


class PlanFixture(Strict):
    requirement_delta: RequirementDelta
    marker: Literal["retained"]

    @model_validator(mode="after")
    def fixture_validator(self):
        if self.requirement_delta.retain_ids == ["R9"]:
            raise ValueError("synthetic parent validator remains active")
        return self


def actual_research_definitions(source_root: Path = ROOT):
    ns = {"__name__": __name__, "Annotated": Annotated, "Literal": Literal, "Field": Field,
        "StringConstraints": StringConstraints, "model_validator": model_validator, "BaseModel": BaseModel,
        "create_model": create_model, "Strict": Strict, "digest": digest, "AgentError": AgentError,
        "BriefField": BriefField, "Experiment": Experiment, "QuestionPlan": QuestionPlan,
        "ResearchPlanReview": ResearchPlanReview,
        "Text": Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)],
        "SHA": Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]}
    groups = {
        "research_workflow.py": {"OutlineQuestion", "Outline", "MethodHandoff", "ConflictPosition", "RoleConflict",
            "ReviewEnvelope", "TargetRepair", "ResearchWorkflow"},
        "research_help.py": {"Suggestion", "Suggestions", "SuggestionReview"},
        "research_intake.py": {"DraftFieldReview", "DraftReview"},
    }
    for filename, names in groups.items():
        path = source_root / "src/citequest/agents_v3" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        selected = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
        assert {node.name for node in selected} == names
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)
    for name in set().union(*groups.values()):
        model = ns[name]
        if isinstance(model, type) and issubclass(model, BaseModel):
            model.model_rebuild(_types_namespace=ns)
    return ns


class BoundIdentityControls(unittest.TestCase):
    def test_all_four_delta_identities_and_both_inherited_validators(self):
        state, turn = initial_state("a"*64), UserTurn(turn_id="U2", text="Use a concise summary.")
        before = PlanFixture.model_json_schema()
        bound = bound_requirement_plan_model(PlanFixture, state, turn)
        expected = {"parent_sha256": digest(state.model_dump(mode="json")), "authority_sha256": "a"*64,
            "turn_id": "U2", "turn_sha256": digest(turn.text)}
        delta = {**expected, "retain_ids": [], "operations": [], "intent": "clear", "clarification": None}
        value = {"requirement_delta": delta, "marker": "retained"}
        bound.model_validate(value, strict=True)
        schema = bound.model_json_schema()
        child = schema["$defs"][schema["properties"]["requirement_delta"]["$ref"].split("/")[-1]]
        for field, exact in expected.items():
            self.assertEqual(child["properties"][field]["const"], exact)
            with self.subTest(field=field), self.assertRaises(ValidationError):
                bound.model_validate({**value, "requirement_delta": {**delta, field: "U3" if field == "turn_id" else "f"*64}}, strict=True)
        for malformed in ({"intent": "needs_clarification"}, {"retain_ids": ["R9"]}):
            with self.assertRaises(ValidationError):
                bound.model_validate({**value, "requirement_delta": {**delta, **malformed}}, strict=True)
        self.assertEqual(PlanFixture.model_json_schema(), before)

    def test_review_identity_does_not_fix_negative_semantic_opinions(self):
        before = RequirementReview.model_json_schema()
        bound = bound_requirement_review_model("b"*64, "c"*64)
        value = {"candidate_sha256": "b"*64, "user_turn_sha256": "c"*64,
            "opinions": [{"requirement_id": "R1", "verdict": "incorrect", "rationale": "Synthetic semantic disagreement."}],
            "current_turn_coverage": "incomplete", "decision": "needs_revision"}
        self.assertEqual(bound.model_validate(value, strict=True).decision, "needs_revision")
        for field in ("candidate_sha256", "user_turn_sha256"):
            with self.assertRaises(ValidationError):
                bound.model_validate({**value, field: "d"*64}, strict=True)
        self.assertEqual(RequirementReview.model_json_schema(), before)

    def test_allowlist_rejects_conflict_choice_invalid_hash_and_missing_field(self):
        ns = actual_research_definitions()
        for base, fields in ((ns["ConflictPosition"], {"handoff_sha256": "a"*64}),
                (RequirementReview, {"candidate_sha256": "A"*64}),
                (RequirementReview, {"brief_sha256": "a"*64}), (RequirementReview, {})):
            with self.subTest(fields=fields), self.assertRaises(AgentError):
                bound_identity_model(base, **fields)

    def test_research_default_model_identity_and_complete_unique_hash_coverage(self):
        ns = actual_research_definitions()
        workflow = object.__new__(ns["ResearchWorkflow"])
        workflow.strict_identity_binding = False
        models = [(ns[name], "brief_sha256") for name in ("Outline", "MethodHandoff", "TargetRepair", "Suggestions")]
        models += [(model, "content_sha256") for model in (SchoolSection, SectionReview, CrossReview, RequestCoverage,
            ns["SuggestionReview"], ns["DraftReview"])]
        models[4] = (SchoolSection, "brief_sha256")
        originals = {model: copy.deepcopy(model.model_json_schema()) for model, _ in models}
        for model, field in models:
            self.assertIs(workflow.identity_model(model, **{field: "a"*64}), model)
        self.assertIs(workflow.plan_review_model("a"*64, "b"*64), ns["ReviewEnvelope"])
        workflow.strict_identity_binding = True
        for model, field in models:
            with self.subTest(model=model.__name__):
                bound = workflow.identity_model(model, **{field: "a"*64})
                self.assertEqual(bound.model_json_schema()["properties"][field]["const"], "a"*64)
                self.assertEqual(set(bound.model_fields), set(model.model_fields))
                self.assertEqual(model.model_json_schema(), originals[model])
        review = workflow.plan_review_model("a"*64, "b"*64)
        schema = review.model_json_schema()
        self.assertEqual(schema["properties"]["content_sha256"]["const"], "b"*64)
        nested = schema["$defs"][schema["properties"]["review"]["$ref"].split("/")[-1]]
        self.assertEqual(nested["properties"]["brief_sha256"]["const"], "a"*64)
        # Conflict handoff is a model-selected alternative, never one fixed hash.
        self.assertNotIn("const", schema["$defs"]["ConflictPosition"]["properties"]["handoff_sha256"])
        value = {"content_sha256": "b"*64, "scope": "cross_plan_ledger", "review": {
            "brief_sha256": "a"*64, "items": [{"target_id": identifier, "verdict": "needs_revision", "reasons": ["Unresolved."]}
                for identifier in ("RQ1", "E1")], "fabricated_completed_work_detected": False,
            "unapproved_resource_or_data_access_detected": False, "source_instruction_injection_detected": False},
            "request_coverage": "partial", "uncovered_user_requests": ["Unresolved constraint."],
            "feasibility": "needs_input", "conflicts": []}
        self.assertEqual(review.model_validate(value, strict=True).review.items[0].verdict, "needs_revision")
        value["review"]["brief_sha256"] = "c"*64
        with self.assertRaises(ValidationError):
            review.model_validate(value, strict=True)
        value["review"]["brief_sha256"] = "a"*64
        value["uncovered_user_requests"] = []
        with self.assertRaises(ValidationError):
            review.model_validate(value, strict=True)


if __name__ == "__main__":
    unittest.main()
