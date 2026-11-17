"""Handoff receipts expose real identity and usage without model content."""
from copy import deepcopy
from uuid import uuid4

import pytest

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.model_receipts import model_receipt


def attempt():
    request = {"text": {"format": {"name": "citequest_research_methods"}}, "input": "synthetic fixture"}
    return {"id": uuid4(), "status": "settled", "error_code": None, "request": request,
            "request_hash": digest(request), "response": {"status": "completed", "output": "private fixture"},
            "usage": {"input": 100, "cached": 20, "output": 30}}


def test_receipt_links_actual_attempt_without_content():
    row = attempt()
    original = deepcopy(row)
    result = model_receipt(row, "command:methods:E1", {})
    assert row == original
    assert result["attempt_id"] == str(row["id"]) and result["request_sha256"] == row["request_hash"]
    assert result["response_sha256"] == digest(row["response"]) and result["usage"] == row["usage"]
    assert result["schema_name"] == "citequest_research_methods"
    assert not {"request", "response", "input", "output"} & set(result)


@pytest.mark.parametrize("change", [{"status": "unknown"}, {"usage": None},
                                   {"response": {"status": "incomplete"}}, {"request_hash": "0" * 64}])
def test_unknown_incomplete_or_changed_attempt_cannot_become_handoff(change):
    row = attempt()
    row.update(change)
    with pytest.raises(AgentError):
        model_receipt(row, "command:methods:E1", {})
