"""Pure native-index rejection controls, not server/TEI success evidence."""

from __future__ import annotations

import asyncio
import copy
import time
from contextvars import copy_context
from types import SimpleNamespace
from uuid import UUID

import grpc
import pytest

from citequest.index_v2.native import CountedDetails, RPCAudit, RPCBudgetError, schema_signature, verify_readback
from citequest.index_v2.rows import compile_row
from citequest.index_v2.schema import build_schema


def row():
    return compile_row(
        snapshot_id=UUID("7e1ae64c-a3a0-4fd0-947b-b0730c1167f1"),
        projection_id=UUID("0a790f6b-70ce-4fc6-a8db-764315fc42b8"),
        native_id="raw-native-001",
        security_domain="public",
        source_namespace="synthetic_control",
        source_kind="standard_title_abstract",
        projection_sha256="a" * 64,
        source_pointer_sha256="b" * 64,
        tokenizer_profile_sha256="c" * 64,
        full_unit_text="Actual immutable text.",
        start=0,
        end=22,
        ordinal=1,
        exact_token_count=lambda text: 4,
    )


def actual():
    return {**row(), "dense": [0.125] * 1024}


def test_sdk_schema_normalization_preserves_complete_native_functions():
    schema, _ = build_schema("http://tei-strict:8080")
    source = schema.to_dict()
    server = copy.deepcopy(source)
    for fn in server["functions"]:
        fn["params"] = {key: str(value) for key, value in fn["params"].items()}
    for field in server["fields"]:
        for key in ["enable_analyzer", "analyzer_params"]:
            if key in field.get("params", {}) and not isinstance(field["params"][key], str):
                import json

                field["params"][key] = json.dumps(field["params"][key])
    assert schema_signature(source) == schema_signature(server)
    server["functions"][1]["params"]["truncate"] = "true"
    assert schema_signature(source) != schema_signature(server)


def test_readback_all_scalars_null_and_1024_real_values_are_required():
    proof = verify_readback([row()], [actual()], complete=True)
    assert proof["complete"] and proof["matched"] == 1 and proof["missing_ids"] == []
    assert verify_readback([row()], [], complete=False)["missing_ids"] == [row()["chunk_id"]]
    with pytest.raises(ValueError, match="missing"):
        verify_readback([row()], [], complete=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "text",
        "year0",
        "profile",
        "extra",
        "dense_dim",
        "nan",
        "inf",
        "zero",
        "bool",
        "sparse_nan",
        "sparse_key",
        "unknown",
        "duplicate",
    ],
)
def test_no_bad_native_readback_can_be_verified(mutation):
    value = actual()
    rows = [value]
    if mutation == "text":
        value["text"] += "!"
    if mutation == "year0":
        value["year"] = 0
    if mutation == "profile":
        value["profile_sha256"] = "d" * 64
    if mutation == "extra":
        value["hidden"] = True
    if mutation == "dense_dim":
        value["dense"].pop()
    if mutation == "nan":
        value["dense"][0] = float("nan")
    if mutation == "inf":
        value["dense"][0] = float("inf")
    if mutation == "zero":
        value["dense"] = [0.0] * 1024
    if mutation == "bool":
        value["dense"][0] = True
    if mutation == "sparse_nan":
        value["sparse"] = {1: float("nan")}
    if mutation == "sparse_key":
        value["sparse"] = {"1": 0.5}
    if mutation == "unknown":
        value["chunk_id"] = "f" * 64
    if mutation == "duplicate":
        rows.append(copy.deepcopy(value))
    with pytest.raises(ValueError):
        verify_readback([row()], rows, complete=True)


def request():
    return SimpleNamespace(SerializeToString=lambda: b"synthetic request bytes")


def details():
    return CountedDetails("/milvus.proto.milvus.MilvusService/Upsert", 60, None, None, False, None)


class Completed:
    def add_done_callback(self, callback):
        callback(self)

    def code(self):
        return grpc.StatusCode.OK


def test_rpc_retry_invocations_are_separate_bounded_and_timeout_capped():
    audit = RPCAudit()
    calls = []

    def continuation(actual_details, message):
        calls.append(actual_details)
        assert 0 < actual_details.timeout <= 10
        return Completed()

    with audit.window("upsert", 10, 2) as window:
        audit.intercept_unary_unary(continuation, details(), request())
        audit.intercept_unary_unary(continuation, details(), request())
        with pytest.raises(RPCBudgetError):
            audit.intercept_unary_unary(continuation, details(), request())
        assert len(window.calls) == len(calls) == 2
    assert len([event for event in audit.events if event["event"] == "rpc_start"]) == 2


def test_expired_missing_and_lifetime_copied_rpc_scope_are_refused():
    audit = RPCAudit()

    def forbidden(*args):
        raise AssertionError("must reject before RPC dispatch")

    with pytest.raises(RPCBudgetError):
        audit.intercept_unary_unary(forbidden, details(), request())
    with audit.window("upsert", 10) as window:
        captured = copy_context()
        window.deadline = time.monotonic() - 1
        with pytest.raises(RPCBudgetError):
            audit.intercept_unary_unary(forbidden, details(), request())
    with pytest.raises(RPCBudgetError):
        captured.run(audit.intercept_unary_unary, forbidden, details(), request())


def test_inherited_async_task_cannot_spend_parent_rpc_window():
    audit = RPCAudit()

    async def run():
        async def child():
            with pytest.raises(RPCBudgetError):
                audit.intercept_unary_unary(lambda *_: Completed(), details(), request())

        with audit.window("upsert", 10):
            await asyncio.create_task(child())

    asyncio.run(run())
