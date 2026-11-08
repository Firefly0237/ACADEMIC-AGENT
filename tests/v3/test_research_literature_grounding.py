"""Shared citation protocol controls; source/semantic opinions are fixtures."""
import copy
import hashlib
import json

import pytest
from pydantic import ValidationError

from citequest.research_v3.literature_grounding import LiteratureGrounding
from citequest.research_v3.quote_spans import QuoteProtocolError


def evidence():
    text = "EXPLICIT SOURCE FIXTURE: a hypothesis is unverified. Formula glyph \uf8ff stays unchanged."
    return [{"ref_id": "EXPLICIT-TASK-ISSUED:chunk-1", "source_ref": {"kind":"native_representation","id":"a"*64},
        "text":text,"span_start":17,"span_end":17+len(text),"coordinate_system":"native_literal_unicode_codepoints",
        "source_sha256":hashlib.sha256(text.encode()).hexdigest(),"unit_text_sha256":"b"*64}]


def candidate(scope):
    return {"kind":"qa","requested_direction":"neutral","claims":[{"id":"C1","statement":"该假设尚未经验证。",
        "status":"supported","edges":[{"quote_id":sorted(scope.scope.allowed)[0],"relation":"supports",
        "rationale":"原文明确标为未验证。","conditions":[]}],"limitations":[]}],"gaps":[],
        "retrieval_scope":"one explicit fixture excerpt, not a full-document claim","evidence_status":"supported"}


def test_server_expands_true_ref_and_PUA_without_model_copy_or_paper_mapping():
    items = evidence()
    scope = LiteratureGrounding(items)
    answer = scope.parse_author(json.dumps(candidate(scope), ensure_ascii=False))
    edge = answer.claims[0].edges[0]
    assert edge.ref_id == items[0]["ref_id"] and edge.ref_id != items[0]["source_ref"]["id"]
    assert edge.quote == items[0]["text"] and "\uf8ff" in edge.quote
    assert edge.quote_locator.span_start == 17
    assert "document_id" not in items[0]


def test_old_paper_ID_in_ref_field_is_rejected_instead_of_silently_mapped():
    scope = LiteratureGrounding(evidence())
    value = candidate(scope)
    edge = value["claims"][0]["edges"][0]
    edge.pop("quote_id")
    edge.update(ref_id="a"*64,quote="a guessed source excerpt")
    with pytest.raises(ValidationError):
        scope.parse_author(json.dumps(value))


def test_unknown_selector_and_supported_context_only_are_not_accepted():
    scope = LiteratureGrounding(evidence())
    value = candidate(scope)
    value["claims"][0]["edges"][0]["quote_id"] = "Q99999"
    with pytest.raises(QuoteProtocolError):
        scope.parse_author(json.dumps(value))
    value = candidate(scope)
    value["claims"][0]["edges"][0]["relation"] = "context_only"
    with pytest.raises(ValidationError) as error:
        scope.parse_author(json.dumps(value))
    assert error.value.errors()[0]["type"] == "claim_supported_requires_supports"


def test_review_uses_fixed_claim_and_each_selected_quote_identity():
    scope = LiteratureGrounding(evidence())
    answer = scope.parse_author(json.dumps(candidate(scope)))
    grid = scope.review_grid(answer)
    quote = next(iter(grid.identities["C1"]))
    value = {"claims":{"C1":{"verdict":"entailed","conditions_preserved":True,"explanation":"明确合成语义意见。",
        "quotes":{quote:{"relation":"matches_declared_relations","conditions_preserved":True,"explanation":"明确合成意见。"}}}},
        "source_instruction_injection_detected":False,"requested_stance_bias_detected":False,
        "request_coverage":"complete","uncovered_requests":[]}
    assert scope.parse_review(json.dumps(value), answer).claims[0].edge_ref_ids == [evidence()[0]["ref_id"]]
    missing = copy.deepcopy(value)
    missing["claims"]["C1"]["quotes"] = {}
    with pytest.raises(ValidationError):
        scope.parse_review(json.dumps(missing), answer)
