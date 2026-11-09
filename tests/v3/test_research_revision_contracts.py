"""Pure explicit-selection, unchanged-section and resolution contracts."""
import copy

import pytest

from citequest.agents_v3.contracts import AgentError, digest
from citequest.research_v3.revision_contracts import RevisionResolution, select_revision_findings, validate_resolution, validate_revision_scope
from citequest.research_v3.review_contracts import ReviewReport


def report():
    sections = {f"section_{index}": "a" * 64 for index in range(14)}
    roles = []
    for role, identifier, target in (("evidence", "E1", "section_0"), ("methods", "M1", "section_1"), ("structure", "S1", "section_2")):
        roles.append({"role": role, "document_sha256": "d" * 64,
            "sections": [{"section_id": section, "section_sha256": sha,
                "status": "needs_revision" if section == target else "no_issue_identified",
                "finding_ids": [identifier] if section == target else [], "reason": "Public reason."}
                for section, sha in sections.items()],
            "findings": [{"id": identifier, "section_ids": [target], "severity": "major", "kind": "methodology",
                "manuscript_quote": "Original text.", "description": "Known finding.", "recommendation": "Proposed correction.",
                "claim_ids": [], "citations": []}], "assessment": "revision_recommended"})
    return ReviewReport(source_document_sha256="d" * 64, source_binding_sha256="b" * 64,
        section_sha256s=sections, role_reports=roles, aggregation={"document_sha256": "d" * 64,
            "role_report_sha256s": {item["role"]: digest(item) for item in roles},
            "findings": [{"finding_key": key, "disposition": "retained", "reason": "Preserve the concern."}
                for key in ("evidence:E1", "methods:M1", "structure:S1")],
            "conflicts": [], "actions": [], "decision": "revise", "reason": "Explicit findings."}).model_dump(mode="json")


def manuscript():
    return [{"section_id": f"section_{index}", "blocks": [{"text": "Original text."}]} for index in range(14)]


def test_selection_is_explicit_known_unique_and_at_most_two_distinct_sections():
    value = report()
    selected, targets = select_revision_findings(value, ["methods:M1", "evidence:E1"], list(value["section_sha256s"]))
    assert [item["finding_key"] for item in selected] == ["methods:M1", "evidence:E1"]
    assert targets == ["section_0", "section_1"]
    for keys in ([], ["evidence:E1", "evidence:E1"], ["evidence:E8"], ["evidence:E1", "methods:M1", "structure:S1"]):
        with pytest.raises(AgentError):
            select_revision_findings(value, keys, list(value["section_sha256s"]))


def test_unselected_sections_remain_exact_and_order_cannot_change():
    original = manuscript()
    revised = copy.deepcopy(original)
    revised[0]["blocks"][0]["text"] = "Revised exact passage."
    unchanged = validate_revision_scope(original, revised, ["section_0", "section_1"])
    assert len(unchanged) == 12 and unchanged["section_13"] == digest(original[13])
    bad = copy.deepcopy(revised)
    bad[13]["blocks"][0]["text"] = "Unapproved change."
    with pytest.raises(AgentError):
        validate_revision_scope(original, bad, ["section_0", "section_1"])
    with pytest.raises(AgentError):
        validate_revision_scope(original, list(reversed(revised)), ["section_0", "section_1"])


def test_resolution_binds_real_finding_and_passage_but_unresolved_is_not_success():
    selected, _ = select_revision_findings(report(), ["evidence:E1"], [f"section_{index}" for index in range(14)])
    sections = manuscript()
    sections[0]["blocks"][0]["text"] = "Revised exact passage."
    resolution = RevisionResolution(source_document_sha256="d" * 64, review_report_sha256="b" * 64,
        candidate_content_sha256=digest(sections), items=[{"finding_key": "evidence:E1", "verdict": "unresolved",
            "section_ids": ["section_0"], "revised_quote": "exact passage", "explanation": "The original concern remains.",
            "citations": [{"ref_id": "source:1", "quote": "Literal source"}]}])
    args = {"source_sha": "d" * 64, "report_sha": "b" * 64, "sections": sections,
        "selected_findings": selected, "evidence": [{"ref_id": "source:1", "text": "Literal source quotation."}]}
    validate_resolution(resolution, **args)
    assert resolution.items[0].verdict == "unresolved"
    bad = resolution.model_dump(mode="json")
    bad["items"][0]["revised_quote"] = "Invented revised passage."
    with pytest.raises(AgentError):
        validate_resolution(RevisionResolution.model_validate(bad, strict=True), **args)
