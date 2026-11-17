"""Official SDK and gRPC hooks, with bounded, observable SDK RPC retries.

An intercepted RPC invocation is observable; TCP packet retransmissions and a
server's internal Function retries are separate, not claimed as zero/exactly-once.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import re
import threading
import time
from collections import namedtuple
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import grpc
import httpx

from .rows import canonical_sha
from .schema import IndexProfile, build_schema, checked_collection_name, checked_endpoint


class RPCBudgetError(RuntimeError):
    pass


class MilvusDatabaseScopeError(ValueError):
    """A local namespace admission failure, never an SDK-retryable RPC error."""


def checked_database(value: str) -> str:
    # Trusted deployment input, never a model/user-supplied collection selector.
    if type(value) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value) is None:
        raise MilvusDatabaseScopeError("canonical_milvus_database_required")
    return value


def checked_database_call(method: str, kwargs: dict) -> None:
    if (method in {"use_database", "using_database", "rename_collection"}
            or "database" in method or method.startswith(("create_user", "drop_user", "update_password", "grant_", "revoke_"))
            or {"db_name", "database", "context", "metadata", "_context", "new_db_name"} & kwargs.keys()):
        raise MilvusDatabaseScopeError("immutable_milvus_database_scope_required")


class MilvusDatabaseScope(grpc.UnaryUnaryClientInterceptor):
    """Verify the official SDK's actual database metadata on every scoped RPC.

    PyMilvus 2.6.17 routes via CallContext's ``dbname`` gRPC metadata; many
    protobuf db_name fields are intentionally empty. We validate, never inject
    or rewrite that metadata. Startup Connect is global; all admitted data,
    collection, index and alias RPCs must carry exactly one matching header.
    Unknown RPC methods fail closed in a non-default namespace.
    """

    SCOPED = frozenset({
        "CreateCollection", "DropCollection", "DescribeCollection", "HasCollection", "ShowCollections",
        "LoadCollection", "ReleaseCollection", "GetLoadingProgress", "GetLoadState", "GetCollectionStatistics",
        "CreateIndex", "DropIndex", "DescribeIndex", "GetIndexState", "GetIndexBuildProgress",
        "Insert", "Upsert", "Delete", "Query", "Search", "HybridSearch", "Flush", "GetFlushState",
        "CreateAlias", "AlterAlias", "DropAlias", "ListAliases", "DescribeAlias",
        "CreatePartition", "DropPartition", "HasPartition", "ShowPartitions", "GetPartitionStatistics",
        "LoadPartitions", "ReleasePartitions", "GetQuerySegmentInfo", "GetPersistentSegmentInfo", "GetReplicas",
        "DescribeDatabase", "AllocTimestamp",
    })
    GLOBAL = frozenset({"Connect", "GetVersion", "CheckHealth"})

    def __init__(self, database: str, audit: RPCAudit) -> None:
        self.database, self.audit = checked_database(database), audit

    def intercept_unary_unary(self, continuation: Any, client_call_details: Any, request: Any) -> Any:
        method = str(client_call_details.method).rsplit("/", 1)[-1]
        values = [value for key, value in client_call_details.metadata or () if key.lower() == "dbname"]
        reason = None
        if method not in self.SCOPED | self.GLOBAL and self.database != "default":
            reason = "unsupported_namespace_RPC"
        elif method in self.SCOPED and (len(values) != 1 or (values[0] or "default") != self.database):
            reason = "missing_duplicate_or_mismatched_dbname_metadata"
        elif values and (len(values) != 1 or (values[0] or "default") != self.database):
            reason = "mismatched_global_dbname_metadata"
        fields = getattr(getattr(request, "DESCRIPTOR", None), "fields_by_name", {})
        for field_name in ("db_name", "dbName", "new_db_name", "newDBName"):
            if field_name in fields and getattr(request, field_name) not in {"", self.database}:
                reason = "protobuf_database_scope_mismatch"
        if method == "CreateCollection" and "schema" in fields:
            from pymilvus.grpc_gen import schema_pb2
            schema = schema_pb2.CollectionSchema()
            schema.ParseFromString(request.schema)
            for field_name in ("db_name", "dbName"):
                if field_name in schema.DESCRIPTOR.fields_by_name and getattr(schema, field_name) not in {"", self.database}:
                    reason = "embedded_schema_database_scope_mismatch"
        self.audit.record({"event": "database_scope_check", "method": method,
                           "milvus_database": self.database, "metadata_count": len(values),
                           "accepted": reason is None, "rejection": reason})
        if reason is not None:
            raise MilvusDatabaseScopeError(reason)
        return continuation(client_call_details, request)


class IncompleteReadbackError(ValueError):
    """A fully observed iterator result is missing known IDs, not hash-invalid."""

    def __init__(self, collection: str, expected_ids: set[str], found_ids: set[str], page_proofs: list[dict]) -> None:
        super().__init__("full_readback_missing_expected_IDs")
        self.observation = {
            "collection": collection,
            "expected_count": len(expected_ids),
            "found_count": len(found_ids),
            "missing_count": len(expected_ids - found_ids),
            "expected_IDs_sha256": canonical_sha({"ids": sorted(expected_ids)}),
            "found_IDs_sha256": canonical_sha({"ids": sorted(found_ids)}),
            "missing_ids": sorted(expected_ids - found_ids),
            "pages": len(page_proofs),
            "page_proofs_sha256": canonical_sha({"pages": page_proofs}),
            "complete": False,
        }


def _owner() -> tuple[int, object | None]:
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    return threading.get_ident(), task


@dataclass
class RPCWindow:
    logical_id: str
    operation: str
    deadline: float
    maximum_calls: int
    owner: tuple[int, object | None] = field(default_factory=_owner)
    active: bool = True
    calls: list[dict] = field(default_factory=list)


Details = namedtuple("Details", "method timeout metadata credentials wait_for_ready compression")


class CountedDetails(Details, grpc.ClientCallDetails):
    pass


class RPCAudit(grpc.UnaryUnaryClientInterceptor):
    def __init__(self, path: Path | None = None) -> None:
        self.path, self.lock = path, threading.Lock()
        self.current: ContextVar[RPCWindow | None] = ContextVar("s05_rpc_window", default=None)
        self.events: list[dict] = []

    def record(self, event: dict) -> None:
        value = {"at": datetime.now(UTC).isoformat(), **event}
        with self.lock:
            self.events.append(value)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()

    @contextmanager
    def window(self, operation: str, seconds: float, maximum_calls: int = 8) -> Iterator[RPCWindow]:
        if (
            type(seconds) not in {float, int}
            or not 0 < seconds <= 600
            or type(maximum_calls) is not int
            or not 1 <= maximum_calls <= 1000
        ):
            raise ValueError("bounded_rpc_window_required")
        if self.current.get() is not None:
            raise RPCBudgetError("nested_rpc_window_forbidden")
        window = RPCWindow(str(uuid4()), operation, time.monotonic() + seconds, maximum_calls)
        token = self.current.set(window)
        self.record(
            {
                "event": "logical_start",
                "logical_id": window.logical_id,
                "operation": operation,
                "maximum_rpc_invocations": maximum_calls,
                "window_seconds": seconds,
            }
        )
        try:
            yield window
        finally:
            window.active = False
            self.current.reset(token)
            self.record(
                {
                    "event": "logical_end",
                    "logical_id": window.logical_id,
                    "operation": operation,
                    "rpc_invocations": len(window.calls),
                    "window_expired": time.monotonic() >= window.deadline,
                }
            )

    def intercept_unary_unary(self, continuation: Any, client_call_details: Any, request: Any) -> Any:
        window = self.current.get()
        if window is None or not window.active or window.owner != _owner():
            raise RPCBudgetError("active_owned_rpc_window_required")
        remaining = window.deadline - time.monotonic()
        if remaining <= 0 or len(window.calls) >= window.maximum_calls:
            # Not grpc.RpcError: the SDK retry decorator must not treat our
            # exhausted local admission window as a retryable network failure.
            raise RPCBudgetError("rpc_window_or_invocation_budget_exhausted")
        requested = client_call_details.timeout
        timeout = min(remaining, requested) if requested is not None else remaining
        details = CountedDetails(
            client_call_details.method,
            timeout,
            client_call_details.metadata,
            client_call_details.credentials,
            client_call_details.wait_for_ready,
            client_call_details.compression,
        )
        event = {
            "event": "rpc_start",
            "logical_id": window.logical_id,
            "operation": window.operation,
            "method": client_call_details.method,
            "sequence": len(window.calls) + 1,
            "request_sha256": hashlib.sha256(request.SerializeToString()).hexdigest(),
            "timeout_seconds": timeout,
        }
        query_diagnostic = str(client_call_details.method).endswith("/Query") and hasattr(request, "collection_name")
        if query_diagnostic:
            event["query_context"] = {
                "collection_name": request.collection_name,
                "guarantee_timestamp": request.guarantee_timestamp,
                "travel_timestamp": request.travel_timestamp,
                "consistency_level": request.consistency_level,
                "use_default_consistency": request.use_default_consistency,
            }
        window.calls.append(event)
        self.record(event)
        start = time.monotonic()
        response = continuation(details, request)

        def done(call: Any) -> None:
            diagnostic = {}
            if query_diagnostic and call.code() == grpc.StatusCode.OK:
                try:
                    reply = call.result()
                    diagnostic = {
                        "query_response_context": {
                            "collection_name": reply.collection_name,
                            "session_ts": reply.session_ts,
                        }
                    }
                except Exception as exc:
                    diagnostic = {"query_diagnostic_error": type(exc).__name__}
            self.record(
                {
                    **diagnostic,
                    "event": "rpc_end",
                    "logical_id": window.logical_id,
                    "sequence": event["sequence"],
                    "method": event["method"],
                    "status": str(call.code()),
                    "elapsed_seconds": time.monotonic() - start,
                }
            )

        response.add_done_callback(done)
        return response


def _params(params: dict) -> dict:
    result = {}
    for key, value in params.items():
        if key in {"enable_analyzer", "analyzer_params"} and isinstance(value, str):
            value = json.loads(value)
        result[key] = value
    return result


def schema_signature(value: dict) -> dict:
    fields = [
        {
            "name": field["name"],
            "type": int(field["type"]),
            "is_primary": field.get("is_primary", False),
            "auto_id": field.get("auto_id", False),
            "nullable": field.get("nullable", False),
            "params": _params(field.get("params", {})),
            "is_partition_key": field.get("is_partition_key", False),
        }
        for field in value["fields"]
    ]
    functions = [
        {
            "name": fn["name"],
            "type": int(fn["type"]),
            "input_field_names": list(fn["input_field_names"]),
            "output_field_names": list(fn["output_field_names"]),
            "params": {key: str(item) for key, item in fn["params"].items()},
        }
        for fn in value.get("functions", [])
    ]
    return {
        "auto_id": value.get("auto_id", False),
        "enable_dynamic_field": value["enable_dynamic_field"],
        "fields": sorted(fields, key=lambda item: item["name"]),
        "functions": sorted(functions, key=lambda item: item["name"]),
    }


def verify_readback(expected: list[dict], actual: list[dict], *, complete: bool) -> dict:
    expected_map = {row["chunk_id"]: row for row in expected}
    if not expected or len(expected_map) != len(expected):
        raise ValueError("nonempty_unique_expected_rows_required")
    found: dict[str, str] = {}
    vector_hashes = {}
    for row in actual:
        identifier = row.get("chunk_id")
        if identifier not in expected_map or identifier in found:
            raise ValueError("unknown_or_duplicate_actual_chunk_ID")
        original = expected_map[identifier]
        if set(row) != {*original, "dense"}:
            raise ValueError("actual_readback_field_set_mismatch")
        scalars = {key: row[key] for key in original}
        if canonical_sha(scalars) != canonical_sha(original):
            raise ValueError("actual_scalar_text_hash_or_config_mismatch")
        dense = row["dense"].tolist() if hasattr(row["dense"], "tolist") else row["dense"]
        if (
            not isinstance(dense, list)
            or len(dense) != 1024
            or any(type(value) not in {float, int} or not math.isfinite(value) for value in dense)
            or sum(value * value for value in dense) <= 0
        ):
            raise ValueError("actual_dense_1024_finite_nonzero_required")
        found[identifier] = canonical_sha(scalars)
        vector_hashes[identifier] = canonical_sha({"dense": dense})
    missing = sorted(set(expected_map) - set(found))
    if complete and missing:
        raise ValueError("actual_readback_missing_expected_IDs")
    return {
        "matched": len(found),
        "expected": len(expected_map),
        "missing_ids": missing,
        "complete": not missing,
        "rows_sha256": canonical_sha(dict(sorted(found.items()))),
        "vector_values_sha256": canonical_sha(dict(sorted(vector_hashes.items()))),
    }


class NativeMilvus:
    """Owned loopback service; only Milvus Functions create vector fields."""

    def __init__(
        self, *, audit: RPCAudit, uri: str = "http://127.0.0.1:19534", endpoint: str = "http://tei-strict:8080",
        database: str = "default"
    ) -> None:
        from pymilvus import MilvusClient

        if uri != "http://127.0.0.1:19534" or importlib.metadata.version("pymilvus") != "2.6.17":
            raise ValueError("owned_exact_milvus_runtime_required")
        self.audit, self.endpoint = audit, checked_endpoint(endpoint)
        self.database = checked_database(database)
        self.absolute_deadline: float | None = None
        self.channel = grpc.intercept_channel(
            grpc.insecure_channel(
                "127.0.0.1:19534",
                options=[
                    ("grpc.enable_retries", 0),
                    ("grpc.max_send_message_length", 64 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ],
            ),
            MilvusDatabaseScope(self.database, audit), audit,
        )
        with audit.window("connect", 15, 12):
            # channel is a supported GrpcHandler argument forwarded by the
            # pinned SDK ConnectionConfig; no private stub monkey-patching.
            self.client = MilvusClient(uri=uri, channel=self.channel, dedicated=True, timeout=10,
                                       db_name="" if self.database == "default" else self.database)

    def call(self, method: str, *, seconds: float = 30, maximum_calls: int = 8, **kwargs: Any) -> Any:
        checked_database_call(method, kwargs)
        if self.absolute_deadline is not None:
            seconds = min(seconds, self.absolute_deadline - time.monotonic())
            if seconds <= 0:
                raise RPCBudgetError("controller_window_exhausted_before_rpc")
        with self.audit.window(method, seconds, maximum_calls):
            return getattr(self.client, method)(timeout=seconds, **kwargs)

    def collection_proof(self, name: str) -> dict:
        name = checked_collection_name(name)
        version = self.call("get_server_version")
        if version not in {"2.6.0", "v2.6.0"}:
            raise ValueError("actual_server_version_mismatch")
        schema, _ = build_schema(self.endpoint)
        actual = self.call("describe_collection", collection_name=name)
        if schema_signature(actual) != schema_signature(schema.to_dict()):
            raise ValueError("actual_native_schema_functions_or_config_mismatch")
        descriptions = [
            self.call("describe_index", collection_name=name, index_name=index)
            for index in self.call("list_indexes", collection_name=name)
        ]
        indexed = {item["field_name"]: item for item in descriptions}
        profile = IndexProfile()
        signatures = {}
        for field_name, settings in {
            "dense": {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "M": str(profile.hnsw_m),
                "efConstruction": str(profile.hnsw_ef_construction),
            },
            "sparse": {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": str(profile.bm25_k1),
                "bm25_b": str(profile.bm25_b),
            },
        }.items():
            item = indexed.get(field_name, {})
            flat = {**item, **item.get("params", {})}
            if any(str(flat.get(key)) != value for key, value in settings.items()):
                raise ValueError("actual_index_parameters_mismatch")
            signatures[field_name] = {key: str(flat[key]) for key in settings}
        remaining = 5 if self.absolute_deadline is None else min(5, self.absolute_deadline - time.monotonic())
        if remaining <= 0:
            raise RPCBudgetError("controller_window_exhausted_before_TEI_info")
        with httpx.Client(trust_env=False, timeout=remaining) as http:
            response = http.get(getattr(self, "tei_info_url", "http://127.0.0.1:18088/info"))
        if response.status_code != 200 or len(response.content) > 200000:
            raise ValueError("actual_strict_TEI_info_unavailable")
        tei = response.json()
        required = {
            "model_id": profile.encoder,
            "model_sha": profile.encoder_revision,
            "model_dtype": "float32",
            "max_input_length": profile.maximum_tokens,
            "max_client_batch_size": profile.maximum_client_batch_size,
        }
        if any(tei.get(key) != value for key, value in required.items()):
            raise ValueError("actual_TEI_encoder_revision_dtype_or_limit_mismatch")
        configuration = {
            "server_version": version,
            "sdk_version": "2.6.17",
            "schema_signature": schema_signature(actual),
            "indexes": signatures,
            "profile_sha256": profile.fingerprint(),
            "tei_identity": required,
        }
        if self.database != "default":
            configuration["milvus_database"] = self.database
        return {
            **configuration,
            "configuration_sha256": canonical_sha(configuration),
            "observed_indexes": descriptions,
            "observed_tei_info": tei,
        }

    def create(self, name: str) -> dict:
        name = checked_collection_name(name)
        if not self.call("has_collection", collection_name=name):
            schema, indexes = build_schema(self.endpoint)
            self.call(
                "create_collection",
                collection_name=name,
                schema=schema,
                index_params=indexes,
                consistency_level="Strong",
                seconds=120,
                maximum_calls=30,
            )
        return self.collection_proof(name)

    def read_ids(self, name: str, rows: list[dict], *, seconds: float = 45) -> tuple[list[dict], dict]:
        checked_collection_name(name)
        identifiers = [row["chunk_id"] for row in rows]
        if len(identifiers) > 100 or not identifiers:
            raise ValueError("bounded_readback_batch_required")
        actual = self.call(
            "get",
            collection_name=name,
            ids=identifiers,
            output_fields=[*rows[0], "dense"],
            consistency_level="Strong",
            seconds=seconds,
        )
        return actual, verify_readback(rows, actual, complete=False)

    def upsert_and_read(self, name: str, rows: list[dict], *, seconds: float = 120) -> dict:
        if not rows or len(rows) > 16 or any({"dense", "sparse"} & row.keys() for row in rows):
            raise ValueError("bounded_vector_free_function_input_required")
        error = None
        deadline = time.monotonic() + seconds
        try:
            self.call(
                "upsert",
                collection_name=checked_collection_name(name),
                data=rows,
                seconds=min(90, seconds * 0.75),
                maximum_calls=8,
            )
        except Exception as exc:
            error = {"type": type(exc).__name__, "message": str(exc)[:600], "write_outcome": "unknown_until_readback"}
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RPCBudgetError("delivery_total_window_exhausted_before_readback")
            _, proof = self.read_ids(name, rows, seconds=min(45, remaining))
        except Exception as exc:
            return {
                "state": "uncertain",
                "write_error": error,
                "readback_error": {"type": type(exc).__name__, "message": str(exc)[:600]},
            }
        return {
            "state": "verified" if proof["complete"] else "uncertain",
            "write_error": error,
            "readback": proof,
            "application_automatic_retries": 0,
            "sdk_retries": "bounded_and_rpc_intercepted",
        }

    def scan_all(self, name: str, rows: list[dict], *, seconds: float = 300) -> dict:
        expected = {row["chunk_id"]: row for row in rows}
        if not rows or len(expected) != len(rows):
            raise ValueError("nonempty_unique_full_expectation_required")
        found: set[str] = set()
        page_proofs = []
        with self.audit.window("full_readback", seconds, 1000):
            iterator = self.client.query_iterator(
                collection_name=checked_collection_name(name),
                filter='chunk_id != ""',
                batch_size=100,
                output_fields=[*rows[0], "dense"],
                consistency_level="Strong",
                timeout=seconds,
            )
            try:
                while batch := iterator.next():
                    identifiers = [row.get("chunk_id") for row in batch]
                    if len(set(identifiers)) != len(identifiers) or any(
                        identifier not in expected or identifier in found for identifier in identifiers
                    ):
                        raise ValueError("unknown_or_duplicate_full_readback_ID")
                    page_proofs.append(
                        verify_readback([expected[identifier] for identifier in identifiers], batch, complete=True)
                    )
                    found.update(identifiers)
            finally:
                iterator.close()
        if found != set(expected):
            raise IncompleteReadbackError(name, set(expected), found, page_proofs)
        return {
            "matched": len(found),
            "expected": len(expected),
            "complete": True,
            "all_IDs_and_scalar_hashes_verified": True,
            "all_dense_1024_finite_nonzero_verified": True,
            "rows_sha256": canonical_sha(
                {identifier: canonical_sha(expected[identifier]) for identifier in sorted(found)}
            ),
            "page_proofs_sha256": canonical_sha({"pages": page_proofs}),
            "pages": len(page_proofs),
            "native_sparse_raw_output": "Milvus 2.6 forbids BM25 Function raw output; schema/index and native text search are separately verified",
        }

    def close(self) -> None:
        self.client.close()
        self.channel.close()
