"""Pinned official Milvus 2.6.17 retrieval, native BM25/TEI and standard RRF."""
from __future__ import annotations
import importlib.metadata
import math
import re
from dataclasses import dataclass
from threading import Lock
from urllib.parse import urlsplit

import grpc
import httpx
from citequest.index_v2.native import MilvusDatabaseScope, RPCAudit, checked_database, checked_database_call, schema_signature
from citequest.index_v2.schema import IndexProfile, checked_collection_name
from citequest.index_v2.rows_v2 import canonical_sha
from citequest.v2.domain import DomainError
from .schemas import SearchRequest


@dataclass(frozen=True)
class NativeHit:
    chunk_id: str
    row_key: str
    score: float


def filter_expression(request: SearchRequest, domain: str, excluded: list[str], included: list[str] | None = None) -> tuple[str, dict]:
    # Only server-authored grammar enters an expression; user values are typed
    # template parameters. Query text is never interpolated into this grammar.
    if len(excluded) > 5000 or any(re.fullmatch(r"[a-f0-9]{64}", value) is None for value in excluded):
        raise DomainError("visibility_prefilter_exceeds_bounded_native_plan", 503)
    expression, values = "security_domain == {domain}", {"domain": domain}
    if included is not None:
        if not 1 <= len(included) <= 20 or any(re.fullmatch(r"[a-f0-9]{64}", value) is None for value in included):
            raise DomainError("bounded_selected_paper_scope_required", 422)
        expression += " and row_key in {included}"
        values["included"] = included
    if excluded:
        expression += " and row_key not in {excluded}"
        values["excluded"] = excluded
    if request.filters.year_from is not None:
        expression += " and year >= {year_from}"
        values["year_from"] = request.filters.year_from
    if request.filters.year_to is not None:
        expression += " and year <= {year_to}"
        values["year_to"] = request.filters.year_to
    return expression, values


class SearchNative:
    """Trusted runtime configuration only. No endpoint, vectors or expr from users.

    The lazy official SDK channel has retries disabled and each operation has an
    intercepted actual-RPC budget. The remote TEI identity is checked for every
    dense request; no fallback model or client embedding is substituted.
    """
    def __init__(self, *, uri: str, tei_info_url: str, audit: RPCAudit | None = None,
                 token: str = "", tls_ca: bytes | None = None, backend_profile_sha256: str | None = None,
                 function_endpoint: str = "http://tei-strict:8080", database: str = "default"):
        parsed = urlsplit(uri)
        info = urlsplit(tei_info_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password
                or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or info.scheme not in {"http", "https"} or not info.hostname or info.username or info.password
                or info.query or info.fragment):
            raise ValueError("trusted_explicit_native_endpoints_required")
        if backend_profile_sha256 is not None and re.fullmatch(r"[a-f0-9]{64}", backend_profile_sha256) is None:
            raise ValueError("explicit_native_backend_sha256_required")
        self.uri, self.tei_info_url, self.token, self.tls_ca = uri, tei_info_url, token, tls_ca
        self.backend_profile_sha256, self.function_endpoint = backend_profile_sha256, function_endpoint
        self.database = checked_database(database)
        self.audit, self._lock = audit or RPCAudit(), Lock()
        self.client, self.channel = None, None

    def _connect(self):
        with self._lock:
            if self.client is not None:
                return
            if importlib.metadata.version("pymilvus") != "2.6.17":
                raise DomainError("pinned_native_sdk_required", 503)
            from pymilvus import MilvusClient
            parsed = urlsplit(self.uri)
            target = f"{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 19530)}"
            options = [("grpc.enable_retries", 0), ("grpc.max_receive_message_length", 8 * 1024 * 1024)]
            raw = (grpc.secure_channel(target, grpc.ssl_channel_credentials(root_certificates=self.tls_ca), options=options)
                   if parsed.scheme == "https" else grpc.insecure_channel(target, options=options))
            self.channel = grpc.intercept_channel(raw, MilvusDatabaseScope(self.database, self.audit), self.audit)
            try:
                with self.audit.window("search_connect", 10, 12):
                    self.client = MilvusClient(uri=self.uri, token=self.token, channel=self.channel, dedicated=True, timeout=8,
                                               db_name="" if self.database == "default" else self.database)
            except Exception:
                self.channel.close()
                self.channel = None
                raise

    def call(self, method: str, *, seconds: float = 15, maximum_calls: int = 8, **kwargs):
        checked_database_call(method, kwargs)
        if not 0 < seconds <= 120 or type(maximum_calls) is not int or not 1 <= maximum_calls <= 40:
            raise ValueError("bounded_explicit_native_RPC_profile_required")
        self._connect()
        with self.audit.window("search_" + method, seconds, maximum_calls):
            return getattr(self.client, method)(timeout=min(12, seconds), **kwargs)

    def close(self):
        if self.client is not None:
            self.client.close()
        if self.channel is not None:
            self.channel.close()

    def check_binding(self, expected_proof: dict):
        """Pure configuration admission; callable before spending a model token."""
        if expected_proof.get("milvus_database", "default") != self.database:
            raise DomainError("native_runtime_database_binding_mismatch", 503)
        if expected_proof.get("backend_profile_sha256") != self.backend_profile_sha256:
            raise DomainError("native_runtime_backend_binding_mismatch", 503)
        functions = expected_proof.get("schema_signature", {}).get("functions", [])
        endpoints = [function["params"].get("endpoint") for function in functions if function["name"] == "text_dense"]
        if endpoints != [self.function_endpoint]:
            raise DomainError("native_runtime_function_endpoint_binding_mismatch", 503)
        if self.backend_profile_sha256 is not None:
            backend = expected_proof.get("backend_profile")
            if not isinstance(backend, dict) or canonical_sha(backend) != self.backend_profile_sha256:
                raise DomainError("native_canonical_backend_proof_mismatch", 503)
            identity = expected_proof.get("tei_identity", {})
            bindings = {"model_id": "model_id", "revision": "model_sha", "dtype": "model_dtype",
                        "maximum_input_tokens": "max_input_length", "maximum_client_batch_size": "max_client_batch_size", "tei_version": "version"}
            if (backend.get("function_endpoint") != self.function_endpoint
                    or backend.get("semantic_index_profile_sha256") != IndexProfile().fingerprint()
                    or any(identity.get(target) != backend.get(source) for source, target in bindings.items())):
                raise DomainError("native_backend_provider_identity_binding_mismatch", 503)

    def verify(self, collection: str, expected_proof: dict, *, dense: bool):
        self.check_binding(expected_proof)
        if self.call("get_server_version") != expected_proof.get("server_version"):
            raise DomainError("native_server_version_drift", 503)
        actual = self.call("describe_collection", collection_name=checked_collection_name(collection))
        if schema_signature(actual) != expected_proof.get("schema_signature"):
            raise DomainError("native_collection_schema_or_function_drift", 503)
        if expected_proof.get("profile_sha256") != IndexProfile().fingerprint():
            raise DomainError("native_generation_profile_mismatch", 503)
        if dense:
            with httpx.Client(trust_env=False, timeout=5, follow_redirects=False) as client:
                with client.stream("GET", self.tei_info_url) as response:
                    response.raise_for_status()
                    parts, length = [], 0
                    for part in response.iter_bytes():
                        length += len(part)
                        if length > 200000:
                            raise DomainError("embedding_identity_response_oversized", 503)
                        parts.append(part)
            import json
            identity = json.loads(b"".join(parts))
            required = expected_proof.get("tei_identity")
            if not required or any(identity.get(key) != value for key, value in required.items()):
                raise DomainError("embedding_revision_dtype_or_limit_drift", 503)

    def search(self, *, collection: str, request: SearchRequest, security_domain: str,
               excluded_row_keys: list[str], expected_proof: dict, included_row_keys: list[str] | None = None) -> list[NativeHit]:
        return self._search(collection=collection, request=request, security_domain=security_domain,
            excluded_row_keys=excluded_row_keys, expected_proof=expected_proof, included_row_keys=included_row_keys,
            group_papers=True)

    def search_chunks(self, *, collection: str, request: SearchRequest, security_domain: str,
                      excluded_row_keys: list[str], expected_proof: dict,
                      included_row_keys: list[str] | None = None) -> list[NativeHit]:
        """RAG chunks may share a paper; public paper search remains grouped.

        Same official dense/BM25 Functions, RRF, pinned schema/database/proof
        and server-side current-access filters. This is not a ranking change
        to the paper search benchmark or an unscoped collection endpoint.
        """
        if request.field != "text" or request.sort != "relevance":
            raise DomainError("chunk_evidence_requires_text_relevance", 422)
        return self._search(collection=collection, request=request, security_domain=security_domain,
            excluded_row_keys=excluded_row_keys, expected_proof=expected_proof, included_row_keys=included_row_keys,
            group_papers=False)

    def _search(self, *, collection: str, request: SearchRequest, security_domain: str,
                excluded_row_keys: list[str], expected_proof: dict,
                included_row_keys: list[str] | None, group_papers: bool) -> list[NativeHit]:
        from pymilvus import AnnSearchRequest, RRFRanker
        self.verify(collection, expected_proof, dense=request.mode != "bm25")
        expression, parameters = filter_expression(request, security_domain, excluded_row_keys, included_row_keys)
        common = dict(collection_name=checked_collection_name(collection), limit=request.candidate_limit,
                      output_fields=["chunk_id", "row_key"], consistency_level="Strong")
        if group_papers:
            common.update(group_by_field="row_key", group_size=1, strict_group_size=False)
        if request.mode == "hybrid":
            branches = [AnnSearchRequest(data=[request.q], anns_field=field, param=params,
                                        limit=min(1000, max(100, request.candidate_limit * 5)),
                                        expr=expression, expr_params=parameters)
                        for field, params in [("dense", {"metric_type": "COSINE", "params": {"ef": 1000}}),
                                              ("sparse", {"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}})]]
            result = self.call("hybrid_search", reqs=branches, ranker=RRFRanker(k=60), **common)
        else:
            field = "sparse" if request.mode == "bm25" else "dense"
            params = ({"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}} if field == "sparse"
                      else {"metric_type": "COSINE", "params": {"ef": 1000}})
            result = self.call("search", data=[request.q], anns_field=field, search_params=params,
                               filter=expression, filter_params=parameters, **common)
        if not isinstance(result, list) or len(result) != 1 or len(result[0]) > request.candidate_limit:
            raise DomainError("invalid_native_search_shape", 503)
        hits, seen = [], set()
        for row in result[0]:
            entity, identifier, score = row.get("entity", {}), row.get("chunk_id", row.get("id")), row.get("distance")
            row_key = entity.get("row_key")
            if (not isinstance(identifier, str) or re.fullmatch(r"[a-f0-9]{64}", identifier) is None
                    or not isinstance(row_key, str) or re.fullmatch(r"[a-f0-9]{64}", row_key) is None
                    or entity.get("chunk_id", identifier) != identifier
                    or type(score) not in {int, float} or not math.isfinite(score)
                    or (row_key if group_papers else identifier) in seen):
                raise DomainError("native_result_identity_or_grouping_invalid", 503)
            seen.add(row_key if group_papers else identifier)
            hits.append(NativeHit(identifier, row_key, float(score)))
        return hits
