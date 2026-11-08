"""Two pure lossless-reference controls over explicit synthetic scope doubles."""
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from citequest.agents_v3.contracts import AgentError, canonical, digest
from citequest.agents_v3.qa_quantitative_audit import (
    REFERENCE_VERSION, QuantitativeAnswerAudit, expand_quantitative_reference_input,
    quantitative_input, quantitative_reference_input, quote_source_provenance,
)

sys.path.insert(0, str(Path(__file__).parent))
from test_qa_quantitative_audit import audited_fixture


def reference_fixture():
    original, opinions, candidate, grid, base = audited_fixture()
    unit = grid.scope.units["Q1"]
    view = {"quote_id": "Q1", "text": unit.text,
        "source_span_start": unit.start, "source_span_end": unit.end}
    grid.scope.contexts = [{"context_id": "CTX1", "quote_units": [copy.deepcopy(view)]}]
    grid.scope.view = lambda selector: copy.deepcopy(view) if selector == "Q1" else None
    return original, opinions, candidate, grid, base


def test_reference_input_round_trip_preserves_full_original_input_and_all_review_criteria():
    original, opinions, candidate, grid, base = reference_fixture()
    before = copy.deepcopy((grid.candidate, grid.scope.contexts))
    full = original.projection["untrusted_quantitative_input"]
    assert quantitative_input(grid) == full
    references = quantitative_reference_input(grid, full)
    assert references["protocol"] == REFERENCE_VERSION
    assert references["claim_pointers"] == {"C1": "/candidate/claims/0", "C2": "/candidate/claims/1"}
    assert references["quote_pointers"] == {"Q1": "/untrusted_evidence_data/0/quote_units/0"}
    assert references["expanded_input_sha256"] == digest(full)
    assert expand_quantitative_reference_input(grid, references) == full
    assert "claim_evidence" not in references and "original_quote" not in references
    revised = QuantitativeAnswerAudit(base, grid, candidate, reference_input=True)
    assert revised.projection["untrusted_quantitative_input"] == references
    assert revised.binding["quantitative_input_sha256"] == original.binding["quantitative_input_sha256"]
    assert revised.binding["quantitative_reference_input_sha256"] == digest(references)
    assert revised.assess(opinions)["quantitative_review"] == original.assess(opinions)["quantitative_review"]
    assert set(revised.model.model_fields) == set(original.model.model_fields)
    default = QuantitativeAnswerAudit(base, grid, candidate, reference_input=False)
    assert canonical(default.projection) == canonical(original.projection)
    assert canonical(default.binding) == canonical(original.binding)
    assert canonical(default.model.model_json_schema()) == canonical(original.model.model_json_schema())
    assert (grid.candidate, grid.scope.contexts) == before


def test_reference_input_rejects_changed_claim_quote_identity_text_pointer_and_original_hash():
    # Equal offsets and null display IDs still require visible document/version
    # attribution, including two distinct versions of the same document.
    _original, _opinions, _candidate, multi, _base = reference_fixture()
    source_a = multi.scope.registry.sources["synthetic-ref"]
    source_a["work_version_id"] = "version-A"
    views = {"Q1": copy.deepcopy(multi.scope.contexts[0]["quote_units"][0])}
    views["Q1"]["document_id"] = None
    for selector, ref, document, version in (
        ("Q2", "second-document", "e" * 64, "version-B"),
        ("Q3", "new-version", "f" * 64, "version-C"),
    ):
        multi.scope.registry.sources[ref] = {**copy.deepcopy(source_a),
            "source_ref": {"kind": "native_representation", "id": document}, "work_version_id": version}
        unit = multi.scope.units["Q1"]
        multi.scope.units[selector] = SimpleNamespace(reference=ref, start=unit.start, end=unit.end, text=unit.text)
        views[selector] = {**copy.deepcopy(views["Q1"]), "quote_id": selector}
    multi.candidate["claims"][1]["edges"] = [{"quote_id": "Q2"}]
    multi.candidate["claims"].append({"id": "C3", "statement": "A distinct version.", "edges": [{"quote_id": "Q3"}]})
    multi.scope.contexts = [{"context_id": "CTX1", "document_id": None, "quote_units": list(views.values())}]
    multi.scope.view = lambda selector: copy.deepcopy(views[selector])
    # The shared author/reviewer bridge also attributes a Q absent from claims.
    unit = multi.scope.units["Q1"]
    multi.scope.units["Q4"] = SimpleNamespace(reference=unit.reference, start=unit.start, end=unit.end, text=unit.text)
    views["Q4"] = {"quote_id": "Q4", "document_id": None, "covered_quote_units": ["Q1"]}
    multi.scope.contexts[0]["quote_units"].append(views["Q4"])
    original_input = quantitative_input(multi)
    attributed = quantitative_reference_input(multi, original_input)
    assert len(attributed["source_provenance"]) == len(set(attributed["quote_sources"].values())) == 3
    assert attributed["quote_sources"]["Q4"] == attributed["quote_sources"]["Q1"]
    assert "Q4" not in attributed["quote_pointers"]
    assert quote_source_provenance(multi.scope)["quote_sources"] == attributed["quote_sources"]
    for selector, version in (("Q1", "version-A"), ("Q2", "version-B"), ("Q3", "version-C")):
        assert attributed["source_provenance"][attributed["quote_sources"][selector]]["work_version_id"] == version
    assert expand_quantitative_reference_input(multi, attributed) == original_input
    changed = copy.deepcopy(attributed)
    changed["quote_sources"]["Q1"] = attributed["quote_sources"]["Q2"]
    with pytest.raises(AgentError):
        expand_quantitative_reference_input(multi, changed)

    _original, _opinions, _candidate, grid, _base = reference_fixture()
    references = quantitative_reference_input(grid, quantitative_input(grid))
    for section, key, bad in (
        ("claim_pointers", "C1", "/candidate/claims/1"),
        ("quote_pointers", "Q1", "/untrusted_evidence_data/0/quote_units/4"),
        ("quote_pointers", "Q2", "/untrusted_evidence_data/0/quote_units/0"),
    ):
        changed = copy.deepcopy(references)
        changed[section][key] = bad
        with pytest.raises(AgentError):
            expand_quantitative_reference_input(grid, changed)
    changed = {**references, "expanded_input_sha256": "0" * 64}
    with pytest.raises(AgentError):
        expand_quantitative_reference_input(grid, changed)
    grid.scope.contexts[0]["quote_units"][0]["text"] = "Changed synthetic source text."
    with pytest.raises(AgentError, match="qa_quantitative_reference_quote_changed"):
        expand_quantitative_reference_input(grid, references)
