"""Pure input boundary: no Task, model budget or tool execution is created."""
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import TaskCreate


def request(profile="stance-v2", **brief):
    return {"project_id": str(uuid4()), "operation_id": str(uuid4()), "kind": "stance", "profile_id": profile,
        "brief": {"kind": "stance", "question": "用户的待考证命题", **brief}}


def test_existing_Brief_direction_guard_covers_both_stance_profiles_before_admission():
    for profile in ("stance-v1-r2", "stance-v2"):
        for fields in ({}, {"requested_direction": "neutral"}):
            with pytest.raises(ValidationError, match="stance requires explicit support/refute"):
                TaskCreate.model_validate(request(profile, **fields))
        for direction in ("support", "refute"):
            value = TaskCreate.model_validate(request(profile, requested_direction=direction))
            assert value.brief.requested_direction == direction
