"""Pinned official SDK Mock controls; fake ledger, no network or model fees."""
from __future__ import annotations

import copy
import json
import unittest
from uuid import uuid4

import httpx2

from citequest.agents_v3.contracts import PROFILES, AgentError, redact
from citequest.agents_v3.provider import MODEL, POLICY_MARKER, ResponsesProvider, private_http_receipt


def payload() -> dict:
    return {"model": MODEL, "reasoning": {"effort": "none"}, "truncation": "disabled", "max_output_tokens": 100,
        "instructions": POLICY_MARKER + "\nEXPLICIT SDK test only.", "input": [{"role": "user", "content": "EXPLICIT fixture"}]}


def completed() -> dict:
    return {"id": "explicit-http-control", "created_at": 0, "object": "response", "status": "completed", "model": MODEL,
        "output": [{"id": "explicit-message", "type": "message", "status": "completed", "role": "assistant",
            "content": [{"type": "output_text", "text": "Explicit fixture.", "annotations": []}]}],
        "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12, "input_tokens_details": {"cached_tokens": 0}}}


class Ledger:
    def __init__(self, *, fail_receipt: bool = False) -> None:
        self.task_id, self.run_id = uuid4(), uuid4()
        self.command = {"id": uuid4()}
        self.row = {"id": uuid4(), "status": "reserved", "reserved_ncny": 12345, "cost_ncny": None, "usage": None, "response": None}
        self.events: list[dict] = []
        self.fail_receipt = fail_receipt
        self.order: list[str] = []
        self.original_request = None

    def snapshot(self) -> dict:
        return {"profile": PROFILES["qa-v5"].model_dump(mode="json")}

    def reserve(self, _key: str, request: dict) -> dict:
        if self.original_request is not None:
            assert request == self.original_request
        self.original_request = copy.deepcopy(request)
        return copy.deepcopy(self.row)

    def mark_sent(self, _attempt: object) -> float:
        self.row["status"] = "sent"
        self.order.append("sent")
        return 120.0

    def observe_model_http_receipt(self, _attempt: object, response: dict, metadata: dict) -> None:
        if self.fail_receipt:
            raise RuntimeError("EXPLICIT receipt transaction failed")
        assert self.row["status"] == "sent" and self.row["usage"] is None
        self.row["response"] = redact(response)
        self.events.append(copy.deepcopy(metadata))
        self.order.append("HTTP_receipt")

    def settle(self, _attempt: object, response: dict, usage: dict) -> None:
        assert self.row["response"] == response
        self.row.update(status="settled", usage=usage, cost_ncny=100, reserved_ncny=0)
        self.order.append("settled")

    def unknown(self, _attempt: object, code: str) -> None:
        self.row.update(status="unknown", error_code=code)
        self.order.append("unknown")

    def not_sent(self, *_args: object, **_kwargs: object) -> None:
        raise AssertionError("an HTTP response cannot become not_sent")


class HTTPReceiptControls(unittest.IsolatedAsyncioTestCase):
    async def execute_case(self, status: int, body: dict | bytes, *, fail_receipt: bool = False) -> tuple[Ledger, list, Exception | None, dict | None]:
        ledger, requests = Ledger(fail_receipt=fail_receipt), []
        raw = body if isinstance(body, bytes) else json.dumps(body).encode()

        async def respond(request: httpx2.Request) -> httpx2.Response:
            requests.append({"retry": request.headers.get("x-stainless-retry-count")})
            return httpx2.Response(status, content=raw, headers={"Content-Type": "application/json"})

        provider = ResponsesProvider(ledger, transport=httpx2.MockTransport(respond), fixture=True)
        error, result = None, None
        try:
            try:
                result = await provider.complete(payload(), str(ledger.command["id"]) + ":explicit")
            except Exception as raised:
                error = raised
        finally:
            await provider.close()
        self.assertEqual(requests, [{"retry": "0"}])
        return ledger, requests, error, result

    async def test_400_and_429_are_classified_before_missing_usage_without_refund_or_retry(self) -> None:
        for status in (400, 429):
            with self.subTest(status=status):
                body = {"error": {"message": "PRIVATE EXPLICIT error echo Bearer fixture-no-credential", "api_key": "private-key", "headers": {"secret": True}},
                    "reasoning_content": "PRIVATE reasoning", "output": [{"type": "reasoning", "content": "PRIVATE reasoning"}]}
                ledger, _, error, _ = await self.execute_case(status, body)
                self.assertIsInstance(error, AgentError)
                self.assertEqual(error.code, "provider_http_error_usage_missing")
                self.assertEqual(ledger.row["status"], "unknown")
                self.assertIsNone(ledger.row["usage"])
                self.assertIsNone(ledger.row["cost_ncny"])
                self.assertEqual(ledger.row["reserved_ncny"], 12345)
                self.assertEqual(ledger.order, ["sent", "HTTP_receipt", "unknown"])
                self.assertEqual(ledger.events[0]["http_status"], status)
                self.assertEqual(set(ledger.events[0]), {"http_status", "body_sha256", "body_bytes", "json_type", "usage_field_state"})
                private = json.dumps(ledger.row["response"])
                for secret in ("fixture-no-credential", "private-key", "PRIVATE reasoning", '"headers"', '"api_key"'):
                    self.assertNotIn(secret, private)
                self.assertNotIn("PRIVATE EXPLICIT", json.dumps(ledger.events))

    async def test_HTTP200_without_usage_stays_unknown_with_actual_response_saved(self) -> None:
        body = completed()
        body.pop("usage")
        ledger, _, error, _ = await self.execute_case(200, body)
        self.assertEqual(error.code, "provider_usage_missing")
        self.assertEqual(ledger.events[0]["http_status"], 200)
        self.assertEqual(ledger.events[0]["usage_field_state"], "missing")
        self.assertIsNotNone(ledger.row["response"])
        self.assertEqual(ledger.row["status"], "unknown")
        self.assertEqual(ledger.row["reserved_ncny"], 12345)

    async def test_malformed_usage_is_unknown_not_zero_and_has_distinct_diagnosis(self) -> None:
        body = completed()
        body["usage"]["input_tokens"] = "10"
        ledger, _, error, _ = await self.execute_case(200, body)
        self.assertEqual(error.code, "provider_usage_malformed")
        self.assertEqual(ledger.events[0]["usage_field_state"], "object")
        self.assertIsNone(ledger.row["usage"])
        self.assertEqual(ledger.row["status"], "unknown")
        self.assertEqual(ledger.row["reserved_ncny"], 12345)

    async def test_completed_HTTP200_observes_before_known_usage_settlement(self) -> None:
        ledger, _, error, result = await self.execute_case(200, completed())
        self.assertIsNone(error)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(ledger.order, ["sent", "HTTP_receipt", "settled"])
        self.assertEqual(ledger.row["usage"], {"input": 10, "cached": 0, "output": 2})
        self.assertEqual(ledger.row["reserved_ncny"], 0)

    async def test_HTTP_error_with_authentic_usage_accounts_usage_then_rejects_content(self) -> None:
        ledger, _, error, _ = await self.execute_case(429, completed())
        self.assertEqual(error.code, "provider_http_error_with_known_usage")
        self.assertEqual(ledger.row["status"], "settled")
        self.assertEqual(ledger.row["reserved_ncny"], 0)
        self.assertEqual(ledger.row["usage"]["output"], 2)

    async def test_receipt_write_failure_cannot_settle_or_leak_or_retry(self) -> None:
        ledger, _, error, _ = await self.execute_case(200, completed(), fail_receipt=True)
        self.assertEqual(error.code, "provider_http_receipt_persistence_failed")
        self.assertEqual(ledger.row["status"], "unknown")
        self.assertIsNone(ledger.row["response"])
        self.assertIsNone(ledger.row["usage"])
        self.assertEqual(ledger.row["reserved_ncny"], 12345)

    async def test_non_JSON_retains_only_safe_digest_and_actual_HTTP_status(self) -> None:
        ledger, _, error, _ = await self.execute_case(503, b"PRIVATE proxy page headers=secret reasoning=private")
        self.assertEqual(error.code, "provider_http_error_non_json_response")
        self.assertEqual(ledger.events[0]["json_type"], "invalid_json")
        self.assertEqual(ledger.row["response"], {"_cq_non_json_body_recorded": False})
        self.assertNotIn("PRIVATE", json.dumps(ledger.row["response"]))


class ReceiptShapeControls(unittest.TestCase):
    def test_all_JSON_shapes_and_usage_states_have_safe_fixed_metadata(self) -> None:
        for value, expected in ((None, "null"), (False, "boolean"), (1, "number"), ("text", "string"), ([], "array")):
            with self.subTest(kind=expected):
                _parsed, private, metadata = private_http_receipt(json.dumps(value).encode(), 200, None)
                self.assertEqual(metadata["json_type"], expected)
                self.assertEqual(metadata["usage_field_state"], "not_applicable")
                self.assertEqual(private, {"_cq_received_json_value": value})
        for value, expected in (({}, "missing"), ({"usage": None}, "null"), ({"usage": []}, "non_object"), ({"usage": {}}, "object")):
            self.assertEqual(private_http_receipt(json.dumps(value).encode(), 200, None)[2]["usage_field_state"], expected)


if __name__ == "__main__":
    unittest.main()
