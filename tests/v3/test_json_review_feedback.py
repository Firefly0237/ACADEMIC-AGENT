"""One pure control for actionable, data-minimizing JSON review format feedback."""
from types import SimpleNamespace

import pytest
from pydantic import Field, ValidationError

from citequest.agents_v3.qa_v7 import QA7
from citequest.research_v3.contracts import Strict


class ReviewShape(Strict):
    explanation: str = Field(max_length=120)


def test_json_review_feedback_identifies_bounds_without_echoing_untrusted_values_or_keys():
    workflow = object.__new__(QA7)
    workflow.graph = SimpleNamespace(execution=SimpleNamespace(snapshot=lambda: {
        "profile": {"review_api_protocol": "chat-json1"},
    }))
    with pytest.raises(ValidationError) as failure:
        ReviewShape.model_validate({"explanation": "x" * 121, "private-untrusted-key": "private-value"}, strict=True)
    details = workflow.review_validation_details(failure.value, ReviewShape)
    assert details["locations"] == [
        {"type": "string_too_long", "loc": ["explanation"]},
        {"type": "extra_forbidden", "loc": ["unrecognized_field"]},
    ]
    assert "private" not in str(details) and "x" * 121 not in str(details)
    workflow.graph.execution.snapshot = lambda: {"profile": {"review_api_protocol": "chat-strict2"}}
    assert workflow.review_validation_details(failure.value, ReviewShape) == {"type": "ValidationError"}
