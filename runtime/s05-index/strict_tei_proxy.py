# S05 compatibility shim copied from frozen S02; no model or embedding code.
import hashlib
import json
import os
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path


def stamp() -> str:
    return datetime.now(UTC).isoformat()


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def strict_tei_proxy() -> None:
    """HTTP compatibility layer only: preserve inputs and force explicit false.

    Milvus 2.6.0 omits its false boolean due to Go JSON omitempty. This layer
    performs no tokenization, embedding, normalization, retry, or fallback.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.error import HTTPError, URLError
    from urllib.request import ProxyHandler, Request, build_opener

    audit_path = Path(os.environ.get("S05_PROXY_AUDIT", "/evidence/strict-proxy.jsonl"))
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    upstream = "http://tei:80"
    opener = build_opener(ProxyHandler({}))

    def audit(event: dict) -> None:
        with lock, audit_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"at": stamp(), **event}, ensure_ascii=False) + "\n")
            stream.flush()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *values: object) -> None:
            del format, values

        def respond(self, status: int, body: bytes, content_type: str = "application/json") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/health":
                self.respond(200, b'{"service":"s05-strict-tei","upstream_health_not_implied":true}')
                return
            if self.path != "/info":
                self.respond(404, b'{"error":"unsupported path"}')
                return
            try:
                with opener.open(upstream + "/info", timeout=10) as response:
                    self.respond(
                        response.status, response.read(), response.headers.get("Content-Type", "application/json")
                    )
            except (HTTPError, URLError, TimeoutError, OSError):
                self.respond(502, b'{"error":"upstream unavailable"}')

        def do_POST(self) -> None:
            call_id = uuid.uuid4().hex
            if self.path != "/embed":
                self.respond(404, b'{"error":"unsupported path"}')
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 2_000_000:
                    raise ValueError("invalid body length")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("object required")
                inputs = payload.get("inputs")
                if not isinstance(inputs, str) and not (
                    isinstance(inputs, list) and all(isinstance(x, str) for x in inputs)
                ):
                    raise ValueError("inputs must be string or ordered list of strings")
                if "truncate" in payload and payload["truncate"] is not False:
                    raise ValueError("only truncate=false is allowed")
                before = digest_text(json.dumps(inputs, ensure_ascii=False, separators=(",", ":")))
                payload["truncate"] = False
                after = digest_text(json.dumps(payload["inputs"], ensure_ascii=False, separators=(",", ":")))
                assert before == after
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                audit(
                    {
                        "event": "forward",
                        "call_id": call_id,
                        "inputs_sha256": before,
                        "upstream_inputs_sha256": after,
                        "inputs_unchanged": True,
                        "input_count": 1 if isinstance(inputs, str) else len(inputs),
                        "input_sha256s": [digest_text(value) for value in ([inputs] if isinstance(inputs, str) else inputs)],
                        "truncate": False,
                    }
                )
            except (ValueError, TypeError, AssertionError) as exc:
                audit({"event": "policy_reject", "call_id": call_id, "reason": str(exc)})
                self.respond(400, b'{"error":"strict TEI policy rejected request","error_type":"ProxyPolicyViolation"}')
                return
            try:
                request = Request(
                    upstream + "/embed", data=body, headers={"Content-Type": "application/json"}, method="POST"
                )
                with opener.open(request, timeout=120) as response:
                    result = response.read()
                    status = response.status
                    content_type = response.headers.get("Content-Type", "application/json")
            except HTTPError as exc:
                status, result = exc.code, exc.read()
                content_type = exc.headers.get("Content-Type", "application/json")
            except (URLError, TimeoutError, OSError) as exc:
                status = 502
                result = json.dumps({"error": "upstream unavailable", "error_type": type(exc).__name__}).encode()
                content_type = "application/json"
            audit(
                {
                    "event": "upstream_result",
                    "call_id": call_id,
                    "upstream_status": status,
                    "response_sha256": hashlib.sha256(result).hexdigest(),
                }
            )
            self.respond(status, result, content_type)

    audit(
        {
            "event": "started",
            "compatibility": "Milvus false boolean omitempty; force explicit false only",
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
    )
    server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    strict_tei_proxy()
