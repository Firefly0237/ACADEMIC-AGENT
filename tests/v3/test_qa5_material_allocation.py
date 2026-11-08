"""Private/native source contracts are admitted explicitly, never clipped."""
import pytest
from pydantic import ValidationError

from citequest.research_v3.qa_v5_contracts import QAPlanV5,result_limits


def test_private_three_result_limit_and_native_twenty_are_explicit():
    material = {"kind":"material_file","id":"server-owned-id"}
    native = {"kind":"native_representation","id":"server-owned-id"}
    assert result_limits([material]) == [1,2,3]
    assert result_limits([native]) == list(range(1,21))
    assert result_limits([native,material]) == list(range(2,8))
    assert result_limits([material,native]) == list(range(2,7))
    assert result_limits([material]*20) == [20]


def test_invalid_second_query_rejects_the_whole_typed_plan_before_execution():
    value = {"requirement_delta":{"parent_sha256":"a"*64,"authority_sha256":"b"*64,"turn_id":"U1","turn_sha256":"c"*64,
        "retain_ids":[],"operations":[],"intent":"clear","clarification":None},"action":"retrieve",
        "queries":[{"query":"valid query","limit":1},{"query":"bad\x00query","limit":1}],"existing_ref_ids":[]}
    with pytest.raises(ValidationError):
        QAPlanV5.model_validate(value,strict=True)
