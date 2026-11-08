"""Official native Functions with the explicitly different GPU backend binding."""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

import grpc
import httpx

from citequest.index_v2.native import MilvusDatabaseScope, NativeMilvus, RPCAudit, checked_database, schema_signature
from citequest.index_v2.rows import canonical_sha
from citequest.index_v2.schema import IndexProfile, checked_collection_name

from .backend import ENDPOINT, INFO_URL, read_profile


def build_schema() -> tuple[Any, Any]:
    from pymilvus import DataType, Function, FunctionType, MilvusClient

    config = IndexProfile()
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    for name, length in (
        ("chunk_id", 64),
        ("row_key", 64),
        ("snapshot_id", 36),
        ("native_id", 512),
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
    for name in ["ordinal", "span_start", "span_end", "unit_codepoints"]:
        schema.add_field(name, DataType.INT64)
    schema.add_field("token_count", DataType.INT32)
    schema.add_field("year", DataType.INT16, nullable=True)
    schema.add_field(
        "text",
        DataType.VARCHAR,
        max_length=config.maximum_text_bytes,
        enable_analyzer=True,
        analyzer_params={"type": config.analyzer},
    )
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1024)
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
                "endpoint": ENDPOINT,
                "truncate": "false",
                "truncation_direction": "Right",
                "max_client_batch_size": 2,
            },
        )
    )
    indexes = MilvusClient.prepare_index_params()
    indexes.add_index(
        field_name="dense", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200}
    )
    indexes.add_index(
        field_name="sparse",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    return schema, indexes


class GPUCandidateNative(NativeMilvus):
    def __init__(self, *, audit: RPCAudit, profile_path: Path, environment: str = "windows-dev",
                 database: str = "default") -> None:
        self.backend_profile, self.backend_sha256, _ = read_profile(profile_path)
        self.profile_path = profile_path
        if environment == "windows-dev":
            self.info_url = INFO_URL
            super().__init__(audit=audit, database=database)
        elif environment == "docker-linux":
            from pymilvus import MilvusClient

            if importlib.metadata.version("pymilvus") != "2.6.17":
                raise ValueError("pinned_official_SDK_required")
            self.info_url = "http://tei-strict-gpu:8080/info"
            self.audit, self.endpoint, self.absolute_deadline = audit, ENDPOINT, None
            self.database = checked_database(database)
            self.channel = grpc.intercept_channel(
                grpc.insecure_channel(
                    "citequest-s05-index-milvus-1:19530",
                    options=[
                        ("grpc.enable_retries", 0),
                        ("grpc.max_send_message_length", 64 * 1024 * 1024),
                        ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                    ],
                ),
                MilvusDatabaseScope(self.database, audit), audit,
            )
            with audit.window("connect", 15, 12):
                self.client = MilvusClient(
                    uri="http://citequest-s05-index-milvus-1:19530", channel=self.channel, dedicated=True, timeout=10,
                    db_name="" if self.database == "default" else self.database
                )
        else:
            raise ValueError("explicit_owned_execution_environment_required")

    def create(self, name: str) -> dict:
        name = checked_collection_name(name)
        if not self.call("has_collection", collection_name=name):
            schema, indexes = build_schema()
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

    def collection_proof(self, name: str) -> dict:
        _, current_sha, _ = read_profile(self.profile_path)
        if current_sha != self.backend_sha256:
            raise ValueError("GPU_backend_profile_changed")
        version = self.call("get_server_version")
        if version not in {"2.6.0", "v2.6.0"}:
            raise ValueError("actual_server_version_mismatch")
        actual = self.call("describe_collection", collection_name=checked_collection_name(name))
        schema, _ = build_schema()
        if schema_signature(actual) != schema_signature(schema.to_dict()):
            raise ValueError("actual_GPU_native_schema_functions_or_config_mismatch")
        descriptions = [
            self.call("describe_index", collection_name=name, index_name=key)
            for key in self.call("list_indexes", collection_name=name)
        ]
        indexed = {row["field_name"]: {**row, **row.get("params", {})} for row in descriptions}
        signatures = {
            "dense": {"index_type": "HNSW", "metric_type": "COSINE", "M": "16", "efConstruction": "200"},
            "sparse": {
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "BM25",
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": "1.2",
                "bm25_b": "0.75",
            },
        }
        if any(
            str(indexed.get(field, {}).get(key)) != expected
            for field, values in signatures.items()
            for key, expected in values.items()
        ):
            raise ValueError("actual_GPU_native_index_parameters_mismatch")
        with httpx.Client(trust_env=False, timeout=5) as http:
            response = http.get(self.info_url)
            response.raise_for_status()
            info = response.json()
        identity = {
            "model_id": "BAAI/bge-m3",
            "model_sha": "5617a9f61b028005a4858fdac845db406aefb181",
            "model_dtype": "float32",
            "max_input_length": 2048,
            "max_client_batch_size": 2,
            "max_batch_requests": 2,
            "version": "1.9.4",
        }
        if any(info.get(key) != value for key, value in identity.items()):
            raise ValueError("actual_GPU_F32_revision_or_limit_mismatch")
        configuration = {
            "server_version": version,
            "sdk_version": "2.6.17",
            "schema_signature": schema_signature(actual),
            "indexes": signatures,
            "profile_sha256": IndexProfile().fingerprint(),
            "tei_identity": identity,
            "backend_profile_sha256": self.backend_sha256,
            "backend_profile": self.backend_profile,
        }
        if self.database != "default":
            configuration["milvus_database"] = self.database
        return {
            **configuration,
            "configuration_sha256": canonical_sha(configuration),
            "observed_tei_info": info,
            "observed_indexes": descriptions,
        }
