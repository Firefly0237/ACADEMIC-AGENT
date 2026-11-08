"""Milvus 2.6 native BM25/TEI schema; no client vector construction."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from hashlib import sha256
from typing import Any
from urllib.parse import urlsplit

NAME = re.compile(r"cq_v2_[a-z][a-z0-9_]{0,99}\Z")


@dataclass(frozen=True)
class IndexProfile:
    """Explicit settings; compatibility still requires a real server canary."""

    profile_version: str = "citequest-native-index/2-server-truncation-enum"
    server_version: str = "2.6.0"
    sdk_version: str = "2.6.17"
    encoder: str = "BAAI/bge-m3"
    encoder_revision: str = "5617a9f61b028005a4858fdac845db406aefb181"
    dense_dim: int = 1024
    maximum_tokens: int = 2048
    maximum_text_bytes: int = 65535
    maximum_native_id_bytes: int = 512
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    maximum_client_batch_size: int = 2
    analyzer: str = "english"
    truncation_direction: str = "Right"

    def __post_init__(self) -> None:
        # Version 2 adds the actual 2.6.0 server's case-sensitive enum. The
        # preparation v1 never reached a successful server schema admission.
        if any(
            type(getattr(self, field.name)) is not type(field.default) or getattr(self, field.name) != field.default
            for field in fields(self)
        ):
            raise ValueError("profile_v1_does_not_accept_implicit_overrides")

    def fingerprint(self) -> str:
        return sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_collection_name(name: str) -> str:
    if not NAME.fullmatch(name):
        raise ValueError("owned_collection_name_required")
    return name


def checked_endpoint(value: str) -> str:
    """Endpoint is host configuration, never taken from an end-user request."""
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.hostname not in {"tei-strict", "127.0.0.1", "localhost"}
    ):
        raise ValueError("explicit_owned_strict_TEI_endpoint_required")
    # A DNS name here is only the operator-owned compose service, not a fetch API.
    return value.rstrip("/")


def build_schema(endpoint: str, profile: IndexProfile | None = None) -> tuple[Any, Any]:
    """Build actual SDK objects lazily, so the pure profile can be inspected alone."""
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    config = profile or IndexProfile()
    endpoint = checked_endpoint(endpoint)
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    for name, length in (
        ("chunk_id", 64),
        ("row_key", 64),
        ("snapshot_id", 36),
        ("native_id", config.maximum_native_id_bytes),
        ("security_domain", 100),
        ("projection_sha256", 64),
        ("source_text_sha256", 64),
        ("unit_text_sha256", 64),
        ("source_pointer_sha256", 64),
        ("tokenizer_profile_sha256", 64),
        ("profile_sha256", 64),
        ("source_kind", 32),
        ("source_namespace", 100),
    ):
        schema.add_field(name, DataType.VARCHAR, max_length=length, is_primary=name == "chunk_id")
    schema.add_field("ordinal", DataType.INT64)
    schema.add_field("span_start", DataType.INT64)
    schema.add_field("span_end", DataType.INT64)
    schema.add_field("unit_codepoints", DataType.INT64)
    schema.add_field("token_count", DataType.INT32)
    schema.add_field("year", DataType.INT16, nullable=True)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=config.maximum_text_bytes,
        enable_analyzer=True,
        analyzer_params={"type": config.analyzer},
    )
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=config.dense_dim)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(
        Function(
            name="text_bm25", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"]
        )
    )
    schema.add_function(
        Function(
            name="text_dense",
            function_type=FunctionType.TEXTEMBEDDING,
            input_field_names=["text"],
            output_field_names=["dense"],
            params={
                "provider": "tei",
                "endpoint": endpoint,
                "truncate": "false",
                "truncation_direction": config.truncation_direction,
                "max_client_batch_size": config.maximum_client_batch_size,
            },
        )
    )
    indexes = MilvusClient.prepare_index_params()
    indexes.add_index(
        field_name="dense",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": config.hnsw_m, "efConstruction": config.hnsw_ef_construction},
    )
    # BM25 parameters belong on the sparse index, never on the BM25 Function.
    indexes.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": config.bm25_k1, "bm25_b": config.bm25_b},
    )
    return schema, indexes
