"""Synthetic input controls only; these records never enter corpus counts."""

import json
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.data_v2.application_contract import ApplicationFreezeInput, original_inventory
from citequest.v2.domain import DomainError


def synthetic_inventory():
    return [{"rank": rank, "native_id": rank + 1, "synthetic": True} for rank in range(3000)]


def encode(rows):
    return b"\n".join(json.dumps(row).encode() for row in rows)


def test_strict_internal_freeze_input_accepts_digest_preconditions_only():
    request = ApplicationFreezeInput(
        client_operation_id=uuid4(), expected_profile_sha256="a" * 64, expected_qualification_sha256="b" * 64
    )
    assert request.expected_qualification_sha256 == "b" * 64
    with pytest.raises(ValidationError):
        ApplicationFreezeInput(**request.model_dump(), qualified=692)


@pytest.mark.parametrize("mutation", ["truncate", "duplicate_rank", "duplicate_identity", "reorder", "coerce_native"])
def test_original_candidate_denominator_and_order_are_not_caller_adjustable(mutation):
    values = synthetic_inventory()
    if mutation == "truncate":
        values.pop()
    elif mutation == "duplicate_rank":
        values[1]["rank"] = 0
    elif mutation == "duplicate_identity":
        values[1]["native_id"] = values[0]["native_id"]
    elif mutation == "coerce_native":
        values[1]["native_id"] = "2"
    else:
        values[0], values[1] = values[1], values[0]
    with pytest.raises(DomainError, match="original_candidate_order_changed"):
        original_inventory(encode(values))


def test_complete_synthetic_inventory_validation_does_not_qualify_it():
    values = original_inventory(encode(synthetic_inventory()))
    assert len(values) == 3000
    assert all("qualified" not in value for value in values)
