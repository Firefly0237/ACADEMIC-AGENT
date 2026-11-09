"""Pinned self-hosted TEI cross-encoder ranking of complete authorized evidence.

The caller owns source authorization and the enclosing durable tool attempt.
This adapter only permutes its input; it does not retrieve, crop, judge claims,
download a model, retry requests, or fall back to a different ranking method.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx2
from pydantic import Field, model_validator

from citequest.agents_v3.contracts import AgentError, Strict, canonical, digest, validate_user_text

PROTOCOL = "tei-cross-encoder1"
RECEIPT_VERSION = "citequest-tei-rerank-receipt/1"


class RerankConfig(Strict):
    version: Literal["citequest-tei-rerank/1"]
    deployment_kind: Literal["self_hosted"]
    base_url: str = Field(min_length=8, max_length=2048)
    expected_model_id: str = Field(min_length=1, max_length=256)
    expected_model_sha: str = Field(pattern=r"^[a-f0-9]{40}$")
    expected_tei_version: str = Field(min_length=1, max_length=64)
    minimum_max_input_length: int = Field(default=8192, ge=1, le=131072)
    maximum_candidates: int = Field(default=20, ge=1, le=20)
    maximum_request_bytes: int = Field(default=131072, ge=1024, le=131072)
    timeout_seconds: int = Field(default=8, ge=1, le=25)
    truncate: Literal[False] = False
    raw_scores: Literal[True] = True
    return_text: Literal[False] = False

    @model_validator(mode="after")
    def trusted_endpoint(self):
        value = urlsplit(self.base_url)
        if (value.scheme not in {"http", "https"} or not value.hostname or value.username is not None
                or value.password is not None or value.query or value.fragment or value.path not in {"", "/"}):
            raise ValueError("reranker requires one trusted service origin without credentials or query")
        return self


def load_config(path: Path, expected_sha256: str) -> RerankConfig:
    if (path.is_symlink() or not path.is_file() or path.stat().st_size > 16384
            or not isinstance(expected_sha256, str) or len(expected_sha256) != 64):
        raise AgentError("rerank_trusted_configuration_required", 503)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise AgentError("rerank_configuration_changed", 503)
    return RerankConfig.model_validate_json(raw, strict=True)


def _pairs(items: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate JSON property")
        result[key] = value
    return result


def _constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _header_integer(headers: httpx2.Headers, name: str) -> int | None:
    value = headers.get(name)
    if value is None:
        return None
    if not value.isascii() or not value.isdigit():
        raise AgentError("rerank_metadata_header_invalid", 503)
    return int(value)


class TEICrossEncoder:
    def __init__(self, config: RerankConfig, configuration_sha256: str, *, transport=None) -> None:
        if not isinstance(configuration_sha256, str) or re.fullmatch(r"[a-f0-9]{64}", configuration_sha256) is None:
            raise AgentError("rerank_configuration_sha256_required", 503)
        self.config, self.configuration_sha256, self.transport = config, configuration_sha256, transport

    def _info(self, value: object) -> dict:
        config = self.config
        if (not isinstance(value, dict) or value.get("model_id") != config.expected_model_id
                or value.get("model_sha") != config.expected_model_sha
                or value.get("version") != config.expected_tei_version
                or not isinstance(value.get("model_type"), dict) or set(value["model_type"]) != {"reranker"}
                or not isinstance(value["model_type"]["reranker"], dict)
                or value.get("auto_truncate") is not False
                or type(value.get("max_input_length")) is not int
                or value["max_input_length"] < config.minimum_max_input_length
                or type(value.get("max_client_batch_size")) is not int
                or value["max_client_batch_size"] < config.maximum_candidates):
            raise AgentError("rerank_service_identity_or_capacity_mismatch", 503)
        return {key: value[key] for key in (
            "model_id", "model_sha", "version", "model_type", "auto_truncate", "max_input_length", "max_client_batch_size")}

    def rerank(self, query: str, evidence: list[dict], *, remaining_ms: int,
               check_scope: Callable[[], object]) -> tuple[list[dict], dict]:
        config = self.config
        if (not isinstance(query, str) or not 1 <= len(query) <= 2000 or type(evidence) is not list
                or len(evidence) > config.maximum_candidates or type(remaining_ms) is not int
                or not 1 <= remaining_ms <= 30000):
            raise AgentError("rerank_input_or_deadline_invalid", 400)
        try:
            validate_user_text(query)
            refs = [item["ref_id"] for item in evidence]
            texts = [item["text"] for item in evidence]
            if (any(not isinstance(ref, str) or not ref for ref in refs) or len(set(refs)) != len(refs)
                    or any(not isinstance(text, str) or not text.strip() for text in texts)):
                raise ValueError("invalid evidence")
        except (KeyError, TypeError, ValueError):
            raise AgentError("rerank_complete_evidence_required", 400) from None
        request = {"query": query, "texts": texts, "truncate": False, "raw_scores": True, "return_text": False}
        raw = canonical(request)
        if len(raw) > config.maximum_request_bytes:
            raise AgentError("rerank_complete_request_exceeds_capacity", 413)
        started = time.monotonic()
        deadline = started + min(remaining_ms / 1000, config.timeout_seconds)
        receipt = {"version": RECEIPT_VERSION, "protocol": PROTOCOL,
            "configuration_sha256": self.configuration_sha256, "query_sha256": digest(query),
            "input_evidence_sha256": digest(evidence), "input_ref_ids": refs,
            "input_text_sha256": [hashlib.sha256(text.encode()).hexdigest() for text in texts],
            "input_candidates": len(evidence), "output_candidates": len(evidence),
            "observed_model": None, "identity_observation_sha256": [], "ranks": [],
            "output_ref_ids": [], "source_tokens_dropped": 0, "source_bytes_modified": False,
            "external_API_cash_ncny": 0, "observed_rerank_HTTP_requests": 0,
            "observed_pair_scores": 0, "input_tokens": None,
            "server_total_ms": None, "server_inference_ms": None,
            "automatic_retry": False, "automatic_fallback": False,
            "identity_basis": "service_info_before_and_after_not_per_response_attestation"}
        check_scope()
        if not evidence:
            return evidence, {**receipt, "status": "empty_no_inference", "client_elapsed_ms": 0}

        def request_json(client, method: str, path: str, content: bytes | None = None):
            check_scope()
            left = deadline - time.monotonic()
            if left <= 0:
                raise AgentError("rerank_active_budget_exhausted", 503)
            with client.stream(method, config.base_url.rstrip("/") + path, content=content,
                               headers={"Content-Type": "application/json"}, timeout=left) as response:
                if response.status_code != 200:
                    raise AgentError("rerank_http_" + str(response.status_code), 503)
                body = bytearray()
                for chunk in response.iter_bytes():
                    body.extend(chunk)
                    if len(body) > 65536:
                        raise AgentError("rerank_response_byte_limit", 503)
                    if time.monotonic() >= deadline:
                        raise AgentError("rerank_active_budget_exhausted", 503)
                try:
                    parsed = json.loads(bytes(body), object_pairs_hook=_pairs, parse_constant=_constant)
                except (ValueError, UnicodeDecodeError):
                    raise AgentError("rerank_response_json_invalid", 503) from None
                return parsed, response.headers

        try:
            with httpx2.Client(transport=self.transport, trust_env=False, follow_redirects=False) as client:
                before, _ = request_json(client, "GET", "/info")
                observed = self._info(before)
                rows, headers = request_json(client, "POST", "/rerank", raw)
                receipt["observed_rerank_HTTP_requests"] = 1
                after, _ = request_json(client, "GET", "/info")
                if self._info(after) != observed:
                    raise AgentError("rerank_service_changed_during_request", 503)
        except httpx2.HTTPError:
            raise AgentError("rerank_transport_failed", 503) from None
        check_scope()
        if time.monotonic() >= deadline:
            raise AgentError("rerank_active_budget_exhausted", 503)
        if (not isinstance(rows, list) or len(rows) != len(evidence)
                or any(not isinstance(row, dict) or set(row) != {"index", "score"}
                       or type(row["index"]) is not int or not 0 <= row["index"] < len(evidence)
                       or type(row["score"]) not in {int, float} or not math.isfinite(row["score"]) for row in rows)
                or {row["index"] for row in rows} != set(range(len(evidence)))):
            raise AgentError("rerank_rank_permutation_invalid", 503)
        ranked = sorted(rows, key=lambda row: (-row["score"], row["index"]))
        result = [evidence[row["index"]] for row in ranked]
        receipt.update(status="ranked", observed_model=observed,
            identity_observation_sha256=[digest(observed), digest(self._info(after))],
            ranks=[{**row, "ref_id": refs[row["index"]]} for row in ranked],
            output_ref_ids=[item["ref_id"] for item in result], observed_pair_scores=len(rows),
            input_tokens=_header_integer(headers, "x-compute-tokens"),
            server_total_ms=_header_integer(headers, "x-total-time"),
            server_inference_ms=_header_integer(headers, "x-inference-time"),
            client_elapsed_ms=round((time.monotonic() - started) * 1000, 3))
        return result, receipt


def validate_receipt(result: dict, query: str, config: RerankConfig, expected_sha256: str) -> None:
    """The durable MCP client verifies a complete, unchanged evidence permutation."""
    try:
        receipt, evidence = result["rerank_receipt"], result["evidence"]
        if (receipt["version"] != RECEIPT_VERSION or receipt["protocol"] != PROTOCOL
                or receipt["configuration_sha256"] != expected_sha256 or receipt["query_sha256"] != digest(query)
                or receipt["source_tokens_dropped"] != 0 or receipt["source_bytes_modified"] is not False
                or receipt["external_API_cash_ncny"] != 0 or receipt["automatic_retry"] is not False
                or receipt["automatic_fallback"] is not False or type(evidence) is not list
                or not 0 <= len(evidence) <= config.maximum_candidates
                or any(type(receipt[key]) is not int for key in (
                    "input_candidates", "output_candidates", "source_tokens_dropped", "external_API_cash_ncny",
                    "observed_rerank_HTTP_requests", "observed_pair_scores"))):
            raise ValueError("receipt contract")
        by_ref = {item["ref_id"]: item for item in evidence}
        refs = receipt["input_ref_ids"]
        if (len(by_ref) != len(evidence) or len(refs) != len(evidence) or len(set(refs)) != len(refs)
                or set(refs) != set(by_ref) or receipt["input_candidates"] != len(evidence)
                or receipt["output_candidates"] != len(evidence)):
            raise ValueError("candidate permutation")
        original = [by_ref[ref] for ref in refs]
        if (digest(original) != receipt["input_evidence_sha256"]
                or [hashlib.sha256(item["text"].encode()).hexdigest() for item in original] != receipt["input_text_sha256"]
                or [item["ref_id"] for item in evidence] != receipt["output_ref_ids"]):
            raise ValueError("source bytes changed")
        if not evidence:
            if (receipt["status"] != "empty_no_inference" or receipt["ranks"] or receipt["observed_rerank_HTTP_requests"] != 0
                    or receipt["observed_pair_scores"] != 0 or receipt["input_tokens"] is not None):
                raise ValueError("empty inference")
            return
        observed = TEICrossEncoder(config, expected_sha256)._info(receipt["observed_model"])
        ranks = receipt["ranks"]
        if (receipt["status"] != "ranked" or receipt["observed_rerank_HTTP_requests"] != 1
                or receipt["observed_pair_scores"] != len(evidence) or len(ranks) != len(evidence)
                or receipt["identity_observation_sha256"] != [digest(observed), digest(observed)]
                or {row["index"] for row in ranks} != set(range(len(evidence)))
                or any(type(row["index"]) is not int or type(row["score"]) not in {int, float}
                    or not math.isfinite(row["score"]) or row["ref_id"] != refs[row["index"]] for row in ranks)
                or ranks != sorted(ranks, key=lambda row: (-row["score"], row["index"]))
                or [row["ref_id"] for row in ranks] != receipt["output_ref_ids"]):
            raise ValueError("score permutation")
        for key in ("input_tokens", "server_total_ms", "server_inference_ms"):
            if receipt[key] is not None and (type(receipt[key]) is not int or receipt[key] < 0):
                raise ValueError("invented or malformed usage")
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        raise AgentError("rerank_MCP_receipt_contract_invalid", 503) from None
