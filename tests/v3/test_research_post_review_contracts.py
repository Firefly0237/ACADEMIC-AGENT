"""Pure report identity, literal citation and dissent contracts; no inference."""
import copy

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import AgentError, digest
from citequest.research_v3.review_contracts import (
    Aggregation, ReviewDiagnostics, ReviewReport, RoleReport, bound_role_model,
    render_review, validate_aggregation, validate_role_report,
)

DOCUMENT = "d" * 64
SECTIONS = {f"section_{index}": "a" * 64 for index in range(14)}
TEXTS = {key: "The manuscript contains this exact sentence." for key in SECTIONS}
QUOTES = {"source:1": ["An original quote under a precise condition."]}
CLAIM_QUOTES = {"RQ1:C1": QUOTES}


def role_report(role):
    findings = []
    if role in {"evidence", "methods"}:
        finding = {"id": "E1" if role == "evidence" else "M1", "section_ids": ["section_0"],
            "severity": "major", "kind": "source_support" if role == "evidence" else "methodology",
            "manuscript_quote": "this exact sentence", "description": "A concise independent negative finding.",
            "recommendation": "Consider a targeted revision.", "claim_ids": ["RQ1:C1"] if role == "evidence" else [],
            "citations": [{"ref_id": "source:1", "quote": "original quote"}] if role == "evidence" else []}
        findings.append(finding)
    return RoleReport.model_validate({"role": role, "document_sha256": DOCUMENT,
        "sections": [{"section_id": key, "section_sha256": sha,
            "status": "needs_revision" if key == "section_0" and findings else "no_issue_identified",
            "finding_ids": [findings[0]["id"]] if key == "section_0" and findings else [], "reason": "Concise public reason."}
            for key, sha in SECTIONS.items()], "findings": findings,
        "assessment": "revision_recommended" if findings else "no_blocking_findings"}, strict=True)


def reports_and_aggregation():
    reports = [role_report(role) for role in ("evidence", "methods", "structure")]
    aggregate = Aggregation(document_sha256=DOCUMENT,
        role_report_sha256s={item.role: digest(item.model_dump(mode="json")) for item in reports},
        findings=[{"finding_key": key, "disposition": "disputed", "reason": "Both positions remain visible."}
            for key in ("evidence:E1", "methods:M1")],
        conflicts=[{"id": "X1", "finding_keys": ["evidence:E1", "methods:M1"], "explanation": "A retained disagreement."}],
        actions=[{"section_ids": ["section_0"], "finding_keys": ["evidence:E1", "methods:M1"],
            "recommendation": "Evaluate both concerns before revision."}], decision="revise", reason="Negative findings remain.")
    return reports, aggregate


def validate(report):
    return validate_role_report(report, role=report.role, document_sha256=DOCUMENT,
        section_sha256s=SECTIONS, section_texts=TEXTS, claim_ids={"RQ1:C1"},
        quotes=QUOTES, claim_quotes=CLAIM_QUOTES)


def test_complete_negative_reports_and_dissent_are_valid_and_visible():
    reports, aggregate = reports_and_aggregation()
    for report in reports:
        validate(report)
    validate_aggregation(aggregate, reports, document_sha256=DOCUMENT, section_sha256s=SECTIONS)
    report = ReviewReport(source_document_sha256=DOCUMENT, source_binding_sha256="b" * 64,
        section_sha256s=SECTIONS, role_reports=reports, aggregation=aggregate)
    text = render_review(report)
    for value in ("evidence:E1", "methods:M1", "original quote", "A retained disagreement.", "section_13", DOCUMENT):
        assert value in text


def test_citations_sections_and_aggregation_cannot_silently_change_or_hide_findings():
    value = role_report("evidence").model_dump(mode="json")
    bad = copy.deepcopy(value)
    bad["findings"][0]["citations"][0]["quote"] = "Invented quotation."
    with pytest.raises(AgentError):
        validate(RoleReport.model_validate(bad, strict=True))
    bad = copy.deepcopy(value)
    bad["sections"][0]["section_sha256"] = "f" * 64
    with pytest.raises(AgentError):
        validate(RoleReport.model_validate(bad, strict=True))
    with pytest.raises(AgentError):
        validate_role_report(RoleReport.model_validate(value, strict=True), role="evidence", document_sha256=DOCUMENT,
            section_sha256s=SECTIONS, section_texts=TEXTS, claim_ids={"RQ1:C1"}, quotes=QUOTES,
            claim_quotes={"RQ1:C1": {"unrelated:source": QUOTES["source:1"]}})
    reports, aggregate = reports_and_aggregation()
    for changes in ({"findings": aggregate.findings[:1]}, {"decision": "no_blocking_findings"}):
        with pytest.raises(AgentError):
            validate_aggregation(aggregate.model_copy(update=changes), reports,
                document_sha256=DOCUMENT, section_sha256s=SECTIONS)


def test_protocol_identity_and_public_partial_diagnostics_are_strict():
    model = bound_role_model("evidence", DOCUMENT)
    changed = role_report("evidence").model_dump(mode="json")
    changed["document_sha256"] = "f" * 64
    with pytest.raises(ValidationError):
        model.model_validate(changed, strict=True)
    diagnostic = {"phase": "evidence", "error_codes": ["research_review_output_schema_invalid"],
        "completed_roles": [], "remaining_roles": ["evidence", "methods", "structure", "aggregate"]}
    assert ReviewDiagnostics.model_validate(diagnostic, strict=True).model_dump(mode="json") == diagnostic
    with pytest.raises(ValidationError):
        ReviewDiagnostics.model_validate({**diagnostic, "raw_model_response": "unreviewed candidate"}, strict=True)
