"""Private fixed-model probability transport, with no retrieval authority."""
from __future__ import annotations

import ipaddress
import json
import time
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from pydantic import Field, model_validator

from citequest.agents_v3.contracts import AgentError, Strict, canonical
from citequest.research_v3.typed_feedback import strict_object

from .decisionrag import DecisionReceipt, DecisionRequest, validate_receipt
from .decisionrag_laya import MODEL_ID, REVISION, SDK_VERSION, WEIGHTS_SHA256, WHEEL_SHA256

PROTOCOL = "citequest-decisionrag-probability/1"
PROTOCOL_V2 = "citequest-decisionrag-probability/2"
REQUEST_BYTES = 32768
RESPONSE_BYTES = 131072


class RemoteJudgeConfig(Strict):
    version: Literal["citequest-decisionrag-remote/1", "citequest-decisionrag-remote/2"]
    base_url: str = Field(min_length=1, max_length=500)
    bearer_token_file: Path

    @property
    def protocol(self) -> str:
        return PROTOCOL_V2 if self.version.endswith("/2") else PROTOCOL

    @property
    def sequence_tokens(self) -> int:
        return 8192 if self.version.endswith("/2") else 1024

    @model_validator(mode="after")
    def private_endpoint(self):
        parts = urlsplit(self.base_url)
        if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password
                or parts.query or parts.fragment or parts.path not in {"", "/"}):
            raise ValueError("a fixed private probability origin is required")
        if parts.hostname not in {"localhost", "host.docker.internal"}:
            try:
                address = ipaddress.ip_address(parts.hostname)
            except ValueError:
                raise ValueError("explicit private address required") from None
            if (not address.is_private or address.is_unspecified or address.is_multicast
                    or address.is_link_local):
                raise ValueError("explicit private address required")
        return self


def token_from_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file() or not 32 <= path.stat().st_size <= 512:
        raise AgentError("decisionrag_private_token_file_required")
    token = path.read_text(encoding="ascii").strip()
    if not 32 <= len(token) <= 256 or any(not (char.isalnum() or char in "-_=") for char in token):
        raise AgentError("decisionrag_private_token_invalid")
    return token


def require_model_binding(binding: dict, *, sequence_tokens: int = 1024) -> None:
    expected = {"provider": "local_author_Laya_SystemOne", "model_id": MODEL_ID,
        "revision": REVISION, "weights_sha256": WEIGHTS_SHA256, "sdk_version": SDK_VERSION,
        "sdk_wheel_sha256": WHEEL_SHA256, "device": "cpu", "backend": "eager",
        "max_len": sequence_tokens, "head_max_len": 256, "paper_Jev_1_13_reproduction": False,
        "remote_code": False, "automatic_device_fallback": False}
    if sequence_tokens == 8192:
        expected.update(native_default_max_len=1024, encoder_max_position_embeddings=8192,
            request_context_protocol="laya2")
    elif sequence_tokens != 1024:
        raise AgentError("decisionrag_remote_fixed_context_mismatch")
    if not isinstance(binding, dict) or any(binding.get(key) != value for key, value in expected.items()):
        raise AgentError("decisionrag_remote_fixed_model_mismatch")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, target):
        raise AgentError("decisionrag_probability_redirect_rejected")


class RemoteBinaryJudge:
    """The owning MCP invocation supplies one absolute monotonic deadline."""
    def __init__(self, config: RemoteJudgeConfig, *, deadline: float):
        self.config, self.deadline = config, deadline
        self.token = token_from_file(config.bearer_token_file)
        self.opener = build_opener(ProxyHandler({}), NoRedirect())
        self.preflight_tokens: dict[str, int] = {}

    def call(self, operation: str, request: DecisionRequest) -> dict:
        if not 1 <= len(request.candidates) <= 8:
            raise AgentError("laya_shared_question_capacity")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise AgentError("decisionrag_active_budget_exhausted")
        payload = {"protocol": self.config.protocol, "operation": operation,
            "remaining_ms": max(1, min(25000, int(remaining * 1000))),
            "request": {"query": request.query, "stage": request.stage,
                "candidates": [list(item) for item in request.candidates],
                "scope_binding": request.scope_binding}, "request_sha256": request.fingerprint()}
        body = canonical(payload)
        if len(body) > REQUEST_BYTES:
            raise AgentError("decisionrag_remote_request_byte_limit")
        wire = Request(self.config.base_url.rstrip("/") + "/v1/decision", data=body, method="POST",
            headers={"Authorization": "Bearer " + self.token, "Content-Type": "application/json"})
        try:
            with self.opener.open(wire, timeout=remaining) as response:
                raw = response.read(RESPONSE_BYTES + 1)
                if response.status != 200 or len(raw) > RESPONSE_BYTES:
                    raise AgentError("decisionrag_remote_response_byte_or_status")
        except HTTPError as error:
            # Do not echo a URL, service body or credentials into public errors.
            code = {401: "decisionrag_gateway_unauthorized", 413: "decisionrag_remote_request_byte_limit",
                429: "decisionrag_gateway_busy", 503: "decisionrag_gateway_unhealthy"}.get(error.code,
                    "decisionrag_gateway_request_rejected")
            raise AgentError(code) from None
        except (URLError, TimeoutError, OSError):
            raise AgentError("decisionrag_gateway_transport_failed") from None
        if time.monotonic() >= self.deadline:
            raise AgentError("decisionrag_active_budget_exhausted")
        value = strict_object(raw.decode("utf-8"))
        if (value.get("protocol") != self.config.protocol or value.get("operation") != operation
                or value.get("request_sha256") != request.fingerprint()):
            raise AgentError("decisionrag_remote_response_identity_mismatch")
        if value.get("status") != "ok":
            code = value.get("error_code")
            if code not in {"laya_complete_context_exceeds_token_budget", "laya_shared_question_capacity",
                "laya_reserved_mask_token_in_source", "decisionrag_physical_deadline_exceeded"}:
                code = "decisionrag_local_probability_failed"
            raise AgentError(code)
        require_model_binding(value.get("model_binding"), sequence_tokens=self.config.sequence_tokens)
        return value

    def preflight(self, request: DecisionRequest) -> None:
        value = self.call("preflight", request)
        if value.get("forwards") != 0 or value.get("source_tokens_dropped") != 0:
            raise AgentError("decisionrag_remote_preflight_performed_inference")
        stats = value.get("exact_token_preflight")
        expected = {"N" + str(index) for index in range(1, len(request.candidates) + 1)}
        if (not isinstance(stats, dict) or set(stats) != expected
                or any(not isinstance(row, dict) or type(row.get("sequence_tokens")) is not int
                    or not 1 <= row["sequence_tokens"] <= self.config.sequence_tokens
                    or row.get("truncated") is not False or row.get("state_tokens_dropped") != 0
                    for row in stats.values())):
            raise AgentError("decisionrag_complete_token_preflight_missing")
        self.preflight_tokens[request.fingerprint()] = sum(row["sequence_tokens"] for row in stats.values())

    def score(self, request: DecisionRequest) -> DecisionReceipt:
        value = self.call("score", request)
        receipt = DecisionReceipt(value["request_sha256"], value.get("probabilities"),
            value["model_binding"], value.get("usage"))
        validate_receipt(request, receipt)
        if (receipt.usage.get("observed_forward_calls") != 1 or receipt.usage.get("truncated") is not False
                or receipt.usage.get("state_tokens_dropped") != 0
                or receipt.usage.get("external_API_cash_ncny") != 0):
            raise AgentError("decisionrag_remote_usage_or_forward_mismatch")
        return receipt
