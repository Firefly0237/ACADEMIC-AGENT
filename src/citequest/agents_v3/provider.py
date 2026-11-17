"""Pinned official SDK; billing guard runs at the actual physical wire boundary."""
from __future__ import annotations

import asyncio
import contextvars
import hashlib
import importlib.metadata
import json
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx2
import openai

from .contracts import AgentError, canonical, digest, redact
from .database import Execution
from .function_output import validate_profile_transport
from .model_contracts import model_contract, validated_model_alias
from .provider_contract import validate_reply_model, wire_contract_violation
from .strict_chat import compile_review_request, is_strict_review, normalize_chat_response, recorded_chat_response, usage_counts_chat
from .typed_submission import VERSION as STRUCTURED_VERSION
from .typed_submission import compile_submission, normalize_submission, recorded_submission, submission_role
from .json_review import VERSION as JSON_REVIEW_VERSION
from .json_review import JSON2_VERSION

MODEL = "deepseek-flash"
SDK_VERSION = "3.45.0"
LIVE_ORIGIN = "https://api.deepseek.com"
FIXTURE_ORIGIN = "https://s08-provider.invalid"
POLICY_MARKER = "CITEQUEST_PRODUCT_AGENT_POLICY_V1"
CALL: contextvars.ContextVar[dict | None] = contextvars.ContextVar("s08_physical_call", default=None)


def usage_counts(response: dict) -> dict:
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise AgentError("provider_usage_missing")
    values = [usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")]
    if any(type(x) is not int or x < 0 for x in values) or values[0] + values[1] != values[2]:
        raise AgentError("provider_usage_malformed")
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens", 0)
    if type(cached) is not int or not 0 <= cached <= values[0] or any((usage.get("server_tool_use") or {}).values()):
        raise AgentError("provider_usage_unrecognized")
    return {"input": values[0], "cached": cached, "output": values[1]}


def private_http_receipt(body: bytes, status: int, authorization: str | None) -> tuple[object, dict, dict]:
    """Separate untrusted response data from safe transport observations.

    JSON is retained privately after structured redaction. Non-JSON is not
    retained as text: no structural rule can safely remove arbitrary secrets
    or reasoning from a proxy's HTML/plain error. Its digest remains auditable.
    """
    def reject_constant(_value: str) -> None:
        raise ValueError("provider_nonfinite_JSON")

    try:
        result = json.loads(body, parse_constant=reject_constant)
    except (UnicodeError, ValueError, RecursionError):
        result, kind = None, "invalid_json"
    else:
        kind = ("null" if result is None else "boolean" if type(result) is bool else "object" if isinstance(result, dict)
            else "array" if isinstance(result, list) else "string" if isinstance(result, str) else "number")
    usage_state = ("not_applicable" if not isinstance(result, dict) else "missing" if "usage" not in result
        else "null" if result["usage"] is None else "object" if isinstance(result["usage"], dict) else "non_object")
    metadata = {"http_status": status, "body_sha256": hashlib.sha256(body).hexdigest(), "body_bytes": len(body),
        "json_type": kind, "usage_field_state": usage_state}
    private = (result if isinstance(result, dict) else {"_cq_received_json_value": result}) if kind != "invalid_json" else {"_cq_non_json_body_recorded": False}
    private = redact(private)
    # Structured redaction cannot remove a credential echoed inside an error
    # string. Remove the actual request credential as well, without recording
    # that credential, any headers, or an error message in public metadata.
    secrets = [authorization] if authorization else []
    if authorization and authorization.startswith("Bearer "):
        secrets.append(authorization[7:])
    secrets = [value for value in secrets if len(value) >= 8]

    def scrub(value: object) -> object:
        if isinstance(value, dict):
            return {key: scrub(child) for key, child in value.items() if not any(secret in key for secret in secrets)}
        if isinstance(value, list):
            return [scrub(child) for child in value]
        if isinstance(value, str):
            for secret in secrets:
                value = value.replace(secret, "[redacted]")
        return value

    return result, scrub(private), metadata


def validate_payload(payload: dict, *, expected_model: str = MODEL, expected_reasoning: str = "none") -> None:
    expected_alias = model_contract(expected_model)["model_alias"]
    if (expected_reasoning not in {"none", "low"} or payload.get("model") != expected_alias
            or payload.get("reasoning") != {"effort": expected_reasoning}):
        raise AgentError("provider_model_policy_changed")
    if POLICY_MARKER not in payload.get("instructions", ""):
        raise AgentError("provider_policy_missing")
    if any(payload.get(k) for k in ("stream", "background", "conversation", "previous_response_id", "store")):
        raise AgentError("provider_stateless_nonstream_required")
    if payload.get("truncation") != "disabled" or any(x.get("role") == "developer" for x in payload.get("input", []) if isinstance(x, dict)):
        raise AgentError("provider_unsupported_history_policy")
    if any(tool.get("type") != "function" for tool in payload.get("tools", [])):
        raise AgentError("provider_server_tools_forbidden")


class BillingClient(openai.DefaultAsyncHttpxClient):
    def __init__(self, execution: Execution, *, origin: str, transport: httpx2.MockTransport | None) -> None:
        super().__init__(transport=transport, trust_env=False, follow_redirects=False, timeout=30)
        self.execution, self.origin = execution, origin

    async def send(self, request: httpx2.Request, **kwargs: Any) -> httpx2.Response:
        try:
            return await self._send_owned(request, **kwargs)
        except BaseException as error:
            context = CALL.get()
            cancel = getattr(self.execution, "cancel_model_waiter", None)
            if context and context.get("execution") is self.execution and cancel is not None:
                # A stale/lost owner cannot change a successor's ticket;
                # cleanup failure must not replace the original wire error.
                with suppress(Exception):
                    cancel("provider_" + (error.code if isinstance(error, AgentError) else "execution_aborted"))
            raise

    async def _send_owned(self, request: httpx2.Request, **kwargs: Any) -> httpx2.Response:
        context = CALL.get()
        if not context or context["execution"] is not self.execution:
            raise AgentError("provider_execution_context_missing")
        chat_review = context.get("chat_review")
        endpoint = (chat_review.receipt["endpoint"] if chat_review is not None else self.origin + "/responses")
        if str(request.url) != endpoint or request.method != "POST":
            raise AgentError("provider_endpoint_denied")
        if request.headers.get("x-stainless-retry-count") not in (None, "0"):
            raise AgentError("provider_implicit_retry_denied")
        wire_payload = json.loads(request.content)
        payload = context["payload"] if chat_review is not None else wire_payload
        task = self.execution.snapshot()
        role = submission_role(task["profile"], payload, context["logical_key"])
        effort = "low" if task["profile"].get("structured_output_protocol") == JSON2_VERSION and role == "review" else "none"
        validate_payload(payload, expected_model=validated_model_alias(task["profile"]), expected_reasoning=effort)
        expected_wire = chat_review.wire if chat_review is not None else context["payload"]
        if wire_payload != expected_wire:
            raise AgentError("provider_sdk_changed_payload")
        if (role is not None or is_strict_review(task["profile"], payload)) != (chat_review is not None):
            raise AgentError("provider_review_api_protocol_changed")
        if len(request.content) > task["profile"]["request_bytes"]:
            raise AgentError("provider_wire_byte_limit")
        waiting = False
        while True:
            # Reauthorize at the physical boundary, including after waiting for another Task.
            guard = context.get("before_dispatch")
            if guard is not None:
                await guard()
            payload_guard = context.get("before_payload_dispatch")
            if payload_guard is not None:
                await payload_guard(payload)
            try:
                attempt = self.execution.reserve(context["logical_key"], payload)
                break
            except AgentError as error:
                if error.code != "global_model_dispatch_busy":
                    raise
                if not waiting:
                    self.execution.progress("model_dispatch_slot", "waiting")
                    waiting = True
                self.execution.snapshot()  # finite Task active deadline; cancellation/owner loss aborts
                await asyncio.sleep(0.25)
        if attempt["status"] == "settled":
            return httpx2.Response(200, content=canonical(attempt["response"]), request=request)
        response = None
        settled = False
        wire_started = False
        try:
            if chat_review is not None:
                observe_wire = getattr(self.execution, "observe_model_wire", None)
                if observe_wire is None:
                    raise AgentError("chat_strict_durable_wire_binding_required")
                observe_wire(attempt["id"], {**chat_review.receipt,
                    "actual_wire_sha256": hashlib.sha256(request.content).hexdigest(),
                    "actual_wire_bytes": len(request.content)})
            # Anchor before the synchronous database call, not after it returns.
            # Its full lock/commit time is conservatively deducted from the send window.
            loop = asyncio.get_running_loop()
            before_mark = loop.time()
            seconds = self.execution.mark_sent(attempt["id"])
            deadline = before_mark + seconds
            if loop.time() >= deadline:
                raise AgentError("active_budget_exhausted_before_wire")
            async with asyncio.timeout_at(deadline):
                # reserve and mark_sent may both wait for PG locks. Recheck after
                # those waits, immediately before the only physical HTTP entry.
                if guard is not None:
                    await guard()
                if payload_guard is not None:
                    await payload_guard(payload)
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise AgentError("active_budget_exhausted_before_wire")
                read_seconds = 30
                if chat_review is not None and chat_review.receipt["protocol"] == JSON2_VERSION and role == "review":
                    read_seconds = task["profile"].get("review_read_seconds")
                    expected_read_seconds = 300 if task["profile"].get("profile_id") == "qa-v9" else 180
                    if type(read_seconds) is not int or read_seconds != expected_read_seconds:
                        raise AgentError("provider_review_read_timeout_contract_changed")
                timeouts = {"connect": 5, "pool": 5, "read": read_seconds, "write": 30}
                request.extensions["timeout"] = {key: min(seconds, remaining) for key, seconds in timeouts.items()}
                wire_started = True
                response = await super().send(request, **{**kwargs, "stream": True})
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > 4 * 1024 * 1024:
                        raise AgentError("provider_response_byte_limit")
                    body.extend(chunk)
            result, private_response, observation = private_http_receipt(bytes(body), response.status_code, request.headers.get("authorization"))
            if chat_review is not None and isinstance(result, dict):
                if chat_review.receipt["protocol"] in {STRUCTURED_VERSION, JSON_REVIEW_VERSION, JSON2_VERSION}:
                    private_response = recorded_submission(task["profile"], payload, context["logical_key"],
                        result, private_response, response.status_code)
                else:
                    private_response = recorded_chat_response(result, private_response, response.status_code,
                        protocol=chat_review.receipt["protocol"])
            try:
                # This observation changes no usage, cost, reserve or attempt
                # outcome. A missing receipt must never be treated as a free
                # request or advance the graph, including a successful HTTP200.
                self.execution.observe_model_http_receipt(attempt["id"], private_response, observation)
            except Exception as error:
                raise AgentError("provider_http_receipt_persistence_failed") from error
            http_error = response.status_code != 200
            if observation["json_type"] == "invalid_json":
                raise AgentError("provider_http_error_non_json_response" if http_error else "provider_non_json_response")
            if not isinstance(result, dict):
                raise AgentError("provider_http_error_response_not_object" if http_error else "provider_response_not_object")
            if not http_error:
                # A successful reply with a different new-contract model does
                # not establish its billing identity. Preserve its receipt and
                # full reservation through the existing unknown path.
                validate_reply_model(payload, result)
            try:
                usage = usage_counts_chat(result) if chat_review is not None else usage_counts(result)
            except AgentError as error:
                if http_error:
                    raise AgentError("provider_http_error_" + error.code.removeprefix("provider_")) from error
                raise
            error = self.execution.settle(attempt["id"], private_response, usage)
            settled = True
            if error:
                raise AgentError(error)
            if http_error:
                raise AgentError("provider_http_error_with_known_usage")
            if chat_review is None and result.get("status") != "completed":
                raise AgentError("provider_response_not_completed")
            # A receipt can be accounted after ownership loss; it must never advance the stale graph.
            self.execution.snapshot()
            headers = {k: v for k, v in response.headers.items() if k.lower() not in {"content-length", "content-encoding"}}
            # Match known replay: neither the SDK result nor the private ledger
            # needs response headers/credentials/reasoning echoed in JSON.
            return httpx2.Response(response.status_code, headers=headers, content=canonical(private_response), request=request)
        except BaseException as exc:
            if not settled:
                try:
                    if wire_started:
                        code = exc.code if isinstance(exc, AgentError) else type(exc).__name__
                        self.execution.unknown(attempt["id"], code)
                    else:
                        code = exc.code if isinstance(exc, AgentError) else type(exc).__name__
                        self.execution.not_sent(attempt["id"], code, before_wire=True)
                except psycopg_error_types():
                    # The durable sent row/reservation remains; a later owner reconciles, never retries blindly.
                    pass
            raise
        finally:
            if response is not None:
                await response.aclose()


def psycopg_error_types() -> tuple[type[BaseException], ...]:
    import psycopg
    return (psycopg.Error, AgentError)


class ResponsesProvider:
    def __init__(self, execution: Execution, *, api_key_file: Path | None = None,
                 metadata_receipt_file: Path | None = None, transport: httpx2.MockTransport | None = None, fixture: bool = False) -> None:
        if importlib.metadata.version("openai") != SDK_VERSION:
            raise AgentError("provider_sdk_lock_mismatch")
        if fixture != (type(transport) is httpx2.MockTransport):
            raise AgentError("fixture_transport_must_be_explicit")
        if fixture and api_key_file is not None:
            raise AgentError("fixture_must_not_receive_credentials")
        if not fixture and (api_key_file is None or transport is not None):
            raise AgentError("live_provider_credential_reference_required")
        self.metadata_sha256 = None
        if not fixture:
            if metadata_receipt_file is None:
                raise AgentError("live_provider_frozen_metadata_receipt_required")
            raw = metadata_receipt_file.read_bytes()
            metadata = json.loads(raw)
            profile = execution.snapshot()["profile"]
            alias = validated_model_alias(profile)
            age = datetime.now(UTC) - datetime.fromisoformat(metadata["at"])
            if (metadata.get("status") != "PASS" or not timedelta(0) <= age <= timedelta(hours=24)
                or metadata.get("model", {}).get("id") != alias
                or metadata.get("model", {}).get("context_window") != profile["context_window"]
                or type(metadata.get("model", {}).get("max_output_tokens")) is not int
                or metadata["model"]["max_output_tokens"] < profile["per_call_output"]
                or metadata.get("model_version_from_pricing") != profile["model_version"]
                or metadata.get("rates_ncny_per_token") != {"input_miss": profile["input_miss_ncny"], "input_cached": profile["input_cached_ncny"], "output": profile["output_ncny"]}):
                raise AgentError("live_provider_metadata_expired_or_changed")
            if profile.get("profile_id") in {"qa-v8", "qa-v9"}:
                effort = metadata["model"].get("effort")
                levels = effort.get("supported_levels") if isinstance(effort, dict) else None
                if not isinstance(levels, list) or "low" not in levels:
                    raise AgentError("live_provider_review_low_effort_not_supported")
            self.metadata_sha256 = digest(raw)
        execution.provider_metadata_sha256 = self.metadata_sha256
        key = "fixture-no-credential" if fixture else api_key_file.read_text(encoding="utf-8").strip()
        if not key:
            raise AgentError("provider_credential_empty")
        self.execution = execution
        self.fixture = fixture
        self.before_dispatch = None
        self.before_payload_dispatch = None
        self.http = BillingClient(execution, origin=FIXTURE_ORIGIN if fixture else LIVE_ORIGIN, transport=transport)
        self.sdk = openai.AsyncOpenAI(api_key=key, base_url=FIXTURE_ORIGIN if fixture else LIVE_ORIGIN,
            max_retries=0, http_client=self.http)
        # Share the same guarded HTTP client and wallet. The Beta backend is
        # selected per immutable review profile, never as an automatic retry.
        self.chat_sdk = self.sdk.with_options(base_url=(FIXTURE_ORIGIN if fixture else LIVE_ORIGIN) + "/beta")

    async def complete(self, payload: dict, logical_key: str) -> dict:
        profile = self.execution.snapshot()["profile"]
        validate_profile_transport(profile,payload)
        role = submission_role(profile, payload, logical_key)
        effort = "low" if profile.get("structured_output_protocol") == JSON2_VERSION and role == "review" else "none"
        validate_payload(payload, expected_model=validated_model_alias(profile), expected_reasoning=effort)
        if role is not None:
            chat_review = compile_submission(profile, payload, logical_key, fixture=self.fixture)
        else:
            chat_review = (compile_review_request(payload, fixture=self.fixture, protocol=profile["review_api_protocol"])
                if is_strict_review(profile, payload) else None)
        if self.before_dispatch is None and not self.fixture:
            raise AgentError("live_scope_guard_missing")
        if self.before_payload_dispatch is None and not self.fixture:
            raise AgentError("live_context_guard_missing")
        raw = await self._complete_once(payload, logical_key, chat_review=chat_review)
        validate_reply_model(payload, raw)
        if chat_review is None:
            return raw
        protocol = profile["review_api_protocol"]
        response = normalize_submission(profile, payload, logical_key, raw)
        if profile.get("qa_execution_protocol") != "grounded3":
            return response
        violation = wire_contract_violation(payload, raw, protocol=protocol, profile=profile, logical_key=logical_key)
        if violation is None:
            return response
        # Only the durable harness capability can assign the one child identity.
        # Its Task lock checks the settled parent, exact request and usage, plus
        # per-parent and Task recovery limits. Replays reuse that same child.
        child_key = self.execution.reserve_protocol_recovery(logical_key, payload, violation)
        child = await self._complete_once(payload, child_key, chat_review=chat_review)
        validate_reply_model(payload, child)
        result = normalize_submission(profile, payload, child_key, child)
        if wire_contract_violation(payload, child, protocol=protocol, profile=profile, logical_key=child_key) is not None:
            raise AgentError("provider_contract_violation_after_recovery")
        return result

    async def _complete_once(self, payload: dict, logical_key: str, *, chat_review=None) -> dict:
        token = CALL.set({"execution": self.execution, "payload": payload, "logical_key": logical_key,
                          "chat_review": chat_review,
                          "before_dispatch": self.before_dispatch,
                          "before_payload_dispatch": self.before_payload_dispatch})
        try:
            if chat_review is not None:
                sdk = self.sdk if chat_review.receipt["protocol"] in {JSON_REVIEW_VERSION, JSON2_VERSION} else self.chat_sdk
                result = await sdk.chat.completions.create(**chat_review.sdk_arguments())
                return result.model_dump(mode="json", exclude_none=True)
            result = await self.sdk.responses.create(**payload)
            return result.model_dump(mode="json", exclude_none=True)
        finally:
            CALL.reset(token)

    async def close(self) -> None:
        await self.sdk.close()


def output_items(response: dict) -> list[dict]:
    """Stateless supported history, without storing or replaying private reasoning."""
    result = []
    for item in response.get("output", []):
        if item.get("type") == "function_call":
            result.append({key: item[key] for key in ("type", "name", "call_id", "arguments")})
        elif item.get("type") == "message":
            result.append({"type": "message", "role": "assistant", "content": [
                {"type": "output_text", "text": content["text"]}
                for content in item.get("content", []) if content.get("type") == "output_text"]})
        elif item.get("type") not in {"reasoning", "reasoning_text"}:
            raise AgentError("provider_output_item_unsupported")
    return result


def output_text(response: dict) -> str:
    return "".join(content["text"] for item in output_items(response) if item["type"] == "message"
        for content in item["content"])
