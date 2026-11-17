"""Per-criterion quantitative opinions on the existing grounded answer audit.

Identities, applicability and publication disposition are structural checks.
Metric, unit and setting judgments remain explicitly identified AI opinions.
"""
from __future__ import annotations

import copy
from typing import Literal

from jsonpointer import JsonPointerException, resolve_pointer
from pydantic import BaseModel, Field, create_model, model_validator

from citequest.research_v3.contracts import Strict

from .contracts import AgentError, digest
from .qa_answer_audit import AnswerAudit

VERSION = "citequest-qa-quantitative-audit/1"
REFERENCE_VERSION = "citequest-quantitative-reference-input/1"
CRITERIA = ("metric", "unit", "setting")

QUANTITATIVE_POLICY = """
<quantitative_audit>
Independently classify every current C. Use quantitative_result for a claim
reporting a numerical measurement, effect, range or count. Bibliographic dates,
identifiers and method labels alone remain other. Preserve the supplied C keys.
For each quantitative result judge metric, unit and setting separately against
the actual published body and that C's original selected quotes and source scope.
Use explicit only when the body unambiguously communicates the criterion for
this claim and the selected evidence supports its meaning. Source context can
establish what a criterion means; a source label by itself supplies no missing
wording in the published answer. Shared wording in the body can apply to several
claims when that applicability is unambiguous.
Metric identifies the measured quantity or score. Unit preserves the scale and
distinctions among percentages, relative changes, percentage points and counts.
Setting preserves the applicable data/population, method, comparison baseline
and relevant conditions. A pipeline's highest reported improvement needs its
evidenced method scope. Keep source wording and uncertainty when the original
measure or table alignment remains ambiguous. Explain each criterion briefly.
Mark missing for an omitted, ambiguous or unsupported necessary criterion.
Quantitative results require an actual explicit/missing judgment for all three
criteria. Other claims require not_applicable for all three. Use genuine negative
opinions whenever the criterion is missing. The server routes affected claims to
the existing bounded content-repair process and independently reviews them again.
Keep the original user question and selected evidence boundary throughout.
These judgments are AI criterion opinions with a deterministic publication gate;
they provide no general guarantee of numerical truth.
</quantitative_audit>
"""


class QuantitativeCriterion(Strict):
    status: Literal["explicit", "missing", "not_applicable"]
    reason: str = Field(min_length=1, max_length=120)

    @model_validator(mode="after")
    def reason_has_text(self):
        if not self.reason.strip():
            raise ValueError("quantitative criterion reason must contain text")
        return self


class QuantitativeClaimReview(Strict):
    kind: Literal["quantitative_result", "other"]
    metric: QuantitativeCriterion
    unit: QuantitativeCriterion
    setting: QuantitativeCriterion

    @model_validator(mode="after")
    def consistent_applicability(self):
        states = [getattr(self, criterion).status for criterion in CRITERIA]
        if self.kind == "quantitative_result" and "not_applicable" in states:
            raise ValueError("every quantitative result needs all three actual criterion judgments")
        if self.kind == "other" and any(state != "not_applicable" for state in states):
            raise ValueError("other claims require all three criteria to be not_applicable")
        return self


def quantitative_input(grid) -> dict:
    """Reconstruct the original complete projection from the authorized grid."""
    evidence = []
    for claim in grid.candidate["claims"]:
        selected = []
        for edge in claim["edges"]:
            selector = edge["quote_id"]
            if selector not in grid.scope.units:
                raise AgentError("qa_quantitative_quote_binding_invalid")
            unit = grid.scope.units[selector]
            source = grid.scope.registry.sources[unit.reference]
            selected.append({
                "declared_edge": copy.deepcopy(edge),
                "original_quote": unit.text,
                "quote_locator": grid.scope.locator(unit),
                "source_window": {key: copy.deepcopy(source[key]) for key in (
                    "source_ref", "source_sha256", "unit_text_sha256", "span_start", "span_end",
                    "coordinate_system", "coverage", "coverage_kind", "work_version_id",
                    "file_id", "file_revision", "material_version_id", "representation_id",
                ) if key in source},
            })
        evidence.append({"claim_id": claim["id"], "selected_quote_evidence": selected})
    return {
        "body": {"claim_statements": [{"id": claim["id"], "statement": claim["statement"]}
                                        for claim in grid.candidate["claims"]],
                 "presentation": copy.deepcopy(grid.candidate.get("presentation"))},
        "claim_limitations": {claim["id"]: copy.deepcopy(claim.get("limitations", []))
                              for claim in grid.candidate["claims"]},
        "claim_evidence": evidence,
        "source_contexts_location": "untrusted_evidence_data in this same review request",
    }


def quote_source_provenance(scope) -> dict:
    """Expose one document/version identity per source and attribute every Q."""
    selectors = [unit["quote_id"] for context in scope.contexts for unit in context["quote_units"]]
    if len(selectors) != len(set(selectors)) or not set(selectors) <= scope.units.keys():
        raise AgentError("qa_quantitative_reference_identity_ambiguous")
    provenance, source_keys, quote_sources = {}, {}, {}
    for selector in sorted(selectors):
        source = scope.registry.sources[scope.units[selector].reference]
        identity = {key: copy.deepcopy(source[key]) for key in (
            "source_ref", "work_version_id", "file_id", "file_revision", "material_version_id",
            "representation_id", "source_version_sha256", "coverage", "coverage_kind", "coordinate_system",
        ) if key in source}
        identity_hash = digest(identity)
        if identity_hash not in source_keys:
            source_keys[identity_hash] = "S" + str(len(provenance) + 1)
            provenance[source_keys[identity_hash]] = identity
        quote_sources[selector] = source_keys[identity_hash]
    return {"source_provenance": provenance, "quote_sources": quote_sources}


def _reference_projection(grid, expanded: dict) -> dict:
    quote_paths = {}
    for context_index, context in enumerate(grid.scope.contexts):
        for unit_index, unit in enumerate(context["quote_units"]):
            selector = unit["quote_id"]
            if selector in quote_paths:
                raise AgentError("qa_quantitative_reference_identity_ambiguous")
            quote_paths[selector] = f"/untrusted_evidence_data/{context_index}/quote_units/{unit_index}"
    selected = {edge["quote_id"] for claim in grid.candidate["claims"] for edge in claim["edges"]}
    if not selected <= quote_paths.keys():
        raise AgentError("qa_quantitative_reference_quote_not_in_context")
    return {"protocol": REFERENCE_VERSION,
        "candidate_sha256": digest(grid.candidate),
        "quote_scope_binding_sha256": grid.scope.binding["binding_sha256"],
        "expanded_input_sha256": digest(expanded),
        "claim_pointers": {claim["id"]: f"/candidate/claims/{index}"
            for index, claim in enumerate(grid.candidate["claims"])},
        "quote_pointers": {selector: quote_paths[selector] for selector in sorted(selected)},
        **quote_source_provenance(grid.scope),
        "candidate_pointer": "/candidate", "source_contexts_pointer": "/untrusted_evidence_data",
        "resolution": "Resolve JSON Pointers within this same request. Claims contain statements, limitations and declared edges. Q units contain original text and spans. quote_sources binds Q to the document/version in source_provenance. The server verifies full original locators, window hashes and permissions. Evaluate the complete original evidence and all original criteria."}


def expand_quantitative_reference_input(grid, references: dict) -> dict:
    """Check fixed pointers against the live scope and restore its full audit input."""
    expanded = quantitative_input(grid)
    if references != _reference_projection(grid, expanded):
        raise AgentError("qa_quantitative_reference_binding_changed")
    root = {"candidate": grid.candidate, "untrusted_evidence_data": grid.scope.contexts}
    try:
        for identifier, pointer in references["claim_pointers"].items():
            if resolve_pointer(root, pointer).get("id") != identifier:
                raise AgentError("qa_quantitative_reference_claim_changed")
        for selector, pointer in references["quote_pointers"].items():
            view = resolve_pointer(root, pointer)
            if view != grid.scope.view(selector):
                raise AgentError("qa_quantitative_reference_quote_changed")
            original = (view["text"] if "text" in view else
                "".join(grid.scope.units[key].text for key in view["covered_quote_units"]))
            if original != grid.scope.units[selector].text:
                raise AgentError("qa_quantitative_reference_text_changed")
    except (JsonPointerException, KeyError, AttributeError, TypeError) as error:
        raise AgentError("qa_quantitative_reference_resolution_invalid") from error
    return expanded


def quantitative_reference_input(grid, expanded: dict) -> dict:
    references = _reference_projection(grid, expanded)
    if digest(expand_quantitative_reference_input(grid, references)) != digest(expanded):
        raise AgentError("qa_quantitative_reference_reconstruction_changed")
    return references


class QuantitativeAnswerAudit(AnswerAudit):
    """Add fixed C opinions while retaining every original R/G/CQ validator."""

    def __init__(self, base_review_model, grid, candidate, *, reference_input: bool = False) -> None:
        super().__init__(base_review_model, grid, candidate)
        expanded_input = quantitative_input(grid)
        input_projection = (quantitative_reference_input(grid, expanded_input)
            if reference_input else expanded_input)
        self.binding.pop("binding_sha256")
        self.binding.update(
            quantitative_audit_protocol=VERSION,
            quantitative_claim_ids=list(self.claim_ids),
            quantitative_input_sha256=digest(expanded_input),
            quantitative_policy_sha256=digest(QUANTITATIVE_POLICY),
        )
        if reference_input:
            self.binding.update(quantitative_input_protocol=REFERENCE_VERSION,
                quantitative_reference_input_sha256=digest(input_projection))
        self.binding["binding_sha256"] = digest(self.binding)
        suffix = self.binding["binding_sha256"][:16]
        rows = create_model(
            "CurrentQuantitativeClaims_" + suffix,
            __base__=Strict,
            **{identifier: (QuantitativeClaimReview, ...) for identifier in self.claim_ids},
        )
        self.model = create_model(
            "QuantitativelyAuditedAnswer_" + suffix,
            __base__=self.model,
            quantitative_review=(rows, ...),
        )
        self.projection.update(
            binding=copy.deepcopy(self.binding),
            untrusted_quantitative_input=input_projection,
            quantitative_criteria=list(CRITERIA),
            quantitative_assessment_authority="independent AI criterion opinions; numerical truth is not guaranteed",
        )

    def assess(self, parsed: BaseModel | dict) -> dict:
        raw = parsed.model_dump() if isinstance(parsed, BaseModel) else parsed
        value = self.model.model_validate(raw, strict=True).model_dump()
        result = super().assess(value)
        opinions = value["quantitative_review"]
        rejected = [identifier for identifier in self.claim_ids
                    if opinions[identifier]["kind"] == "quantitative_result"
                    and any(opinions[identifier][criterion]["status"] == "missing" for criterion in CRITERIA)]
        result.update(
            quantitative_review=opinions,
            quantitative_rejected_claim_ids=rejected,
            quantitative_assessment_authority="independent AI criterion opinions; numerical truth is not guaranteed",
        )
        return result
