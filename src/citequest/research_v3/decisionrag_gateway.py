"""Single-model private ASGI gateway with a physically bounded CPU child.

The gateway has no database, source-reader or generative-model credentials.
Only the admitted MCP bridge selects and reauthorizes literal evidence.
"""
from __future__ import annotations

import asyncio
import hmac
import importlib.metadata
import ipaddress
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import Field, ValidationError, model_validator
from starlette.concurrency import run_in_threadpool

from citequest.agents_v3.contracts import AgentError, Strict, canonical

from .decisionrag import DecisionRequest
from .decisionrag_remote import PROTOCOL, PROTOCOL_V2, REQUEST_BYTES, RESPONSE_BYTES, token_from_file


class GatewayConfig(Strict):
    version: Literal["citequest-decisionrag-Windows-CPU/1", "citequest-decisionrag-Linux-CPU/1", "citequest-decisionrag-Linux-CPU/2"]
    host: str
    port: int = Field(ge=1024, le=65535)
    bearer_token_file: Path
    assets: Path
    asset_receipt_file: Path
    asset_receipt_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    cache_root: Path

    @property
    def protocol(self) -> str:
        return PROTOCOL_V2 if self.version.endswith("/2") else PROTOCOL

    @property
    def sequence_tokens(self) -> int:
        return 8192 if self.version.endswith("/2") else 1024

    @model_validator(mode="after")
    def fixed_private_binding(self):
        address = ipaddress.ip_address(self.host)
        if not address.is_private or address.is_unspecified or address.is_multicast or address.is_link_local:
            raise ValueError("explicit private listen address required")
        return self


class WireRequest(Strict):
    query: str = Field(min_length=1, max_length=8000)
    stage: Literal["documents", "passages"]
    candidates: list[tuple[str, str]] = Field(min_length=1, max_length=8)
    scope_binding: str = Field(pattern=r"^[a-f0-9]{64}$")

    def request(self) -> DecisionRequest:
        if (len({item[0] for item in self.candidates}) != len(self.candidates)
                or any(not identifier or not text.strip() for identifier, text in self.candidates)):
            raise AgentError("decisionrag_gateway_candidate_identity_invalid")
        return DecisionRequest(self.query, self.stage, tuple(self.candidates), self.scope_binding)


class GatewayCall(Strict):
    protocol: Literal["citequest-decisionrag-probability/1", "citequest-decisionrag-probability/2"]
    operation: Literal["preflight", "score"]
    remaining_ms: int = Field(ge=1, le=25000)
    request: WireRequest
    request_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


def _model_process(connection, config_value: dict, parent_pid: int) -> None:
    """Spawn entry waits for the parent to apply the exact Windows job first."""
    judge = None
    try:
        config = GatewayConfig.model_validate(config_value, strict=True)
        if sys.platform == "linux" and config.version in {"citequest-decisionrag-Linux-CPU/1", "citequest-decisionrag-Linux-CPU/2"}:
            from .decisionrag_linux import apply_child_limits
            apply_child_limits(parent_pid)
        elif os.name != "nt" or config.version != "citequest-decisionrag-Windows-CPU/1":
            raise AgentError("decisionrag_explicit_process_platform_required")
        if connection.recv_bytes(32) != b"job_applied":
            return
        root = config.cache_root.resolve(strict=True)
        for key in ("HF_HOME", "TORCH_HOME", "XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR", "TEMP", "TMP"):
            os.environ[key] = str(root)
        os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = "2"
        expected = {"torch": "2.15.0" if os.name == "nt" else "2.15.0+cpu", "transformers": "5.22.1", "huggingface-hub": "1.33.0",
            "tokenizers": "0.23.3", "safetensors": "0.8.0", "numpy": "2.5.4"}
        if any(importlib.metadata.version(name) != version for name, version in expected.items()):
            raise AgentError("decisionrag_existing_CPU_runtime_mismatch")
        # All wheel imports occur after the same stdlib artifact integrity gate
        # used by the already verified Linux entry.
        from .decisionrag_laya import verify_before_import
        wheel = verify_before_import(config.assets, config.asset_receipt_file, config.asset_receipt_sha256)
        sys.path.insert(0, str(wheel))
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        if str(torch.__version__) != "2.15.0+cpu" or torch.version.cuda is not None:
            raise AgentError("decisionrag_explicit_CPU_runtime_required")
        from .decisionrag_laya import LayaLocalConfig, LayaLocalJudge
        judge = LayaLocalJudge(LayaLocalConfig(config.assets / "model", wheel,
            config.asset_receipt_file, config.asset_receipt_sha256, root,
            max_len=config.sequence_tokens, navigation_protocol="laya2" if config.version.endswith("/2") else "laya1"))
        connection.send_bytes(canonical({"status": "ready", "model_binding": judge.model_binding,
            "model_load_seconds": judge.load_seconds, "forwards": 0}))
        while True:
            raw = connection.recv_bytes(REQUEST_BYTES)
            call = GatewayCall.model_validate_json(raw, strict=True)
            request = call.request.request()
            result = {"protocol": config.protocol, "operation": call.operation,
                "request_sha256": request.fingerprint(), "model_binding": judge.model_binding}
            try:
                if call.protocol != config.protocol or call.request_sha256 != request.fingerprint():
                    raise AgentError("decisionrag_remote_request_identity_mismatch")
                if call.operation == "preflight":
                    checks = judge.preflight(request)
                    result.update(status="ok", forwards=0, source_tokens_dropped=0, exact_token_preflight=checks)
                else:
                    result.update(status="ok", **asdict(judge.score(request)))
            except AgentError as error:
                result.update(status="error", error_code=error.code)
            connection.send_bytes(canonical(result))
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        try:
            connection.send_bytes(canonical({"status": "error", "error_code":
                error.code if isinstance(error, AgentError) else "decisionrag_local_model_process_failed"}))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if judge is not None:
            judge.close()
        connection.close()


class ModelProcess:
    """One hot child, zero queued inference, no automatic timeout restart."""
    def __init__(self, config: GatewayConfig):
        if ((os.name == "nt" and config.version != "citequest-decisionrag-Windows-CPU/1")
                or (sys.platform == "linux" and config.version not in {"citequest-decisionrag-Linux-CPU/1", "citequest-decisionrag-Linux-CPU/2"})
                or (os.name != "nt" and sys.platform != "linux")):
            raise AgentError("decisionrag_explicit_process_platform_required")
        self.process, self.connection, self.job = None, None, None
        self.failure = None
        self.gate = threading.Lock()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self.connection = parent
        self.process = context.Process(target=_model_process,
            args=(child, config.model_dump(), os.getpid()), daemon=True, name="citequest-decisionrag-CPU")
        try:
            self.process.start()
            child.close()
            if os.name == "nt":
                from .decisionrag_windows import WindowsProcessJob
                self.job = WindowsProcessJob(self.process.pid)
            else:
                from .decisionrag_linux import LinuxProcessHandle
                self.job = LinuxProcessHandle(self.process.pid)
            self.connection.send_bytes(b"job_applied")
            self.startup = self.receive(time.monotonic() + 60)
            if self.startup.get("status") != "ready":
                raise AgentError("decisionrag_model_startup_failed")
            self.job.observation()
        except BaseException:
            self.close()
            raise

    def receive(self, deadline: float) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self.connection.poll(remaining):
            self.failure = "decisionrag_physical_deadline_exceeded"
            self.close()
            raise AgentError(self.failure)
        try:
            raw = self.connection.recv_bytes(RESPONSE_BYTES)
        except (EOFError, OSError):
            self.failure = "decisionrag_probability_process_lost"
            self.close()
            raise AgentError(self.failure) from None
        return json.loads(raw)

    def roundtrip(self, raw: bytes, deadline: float) -> dict:
        if self.failure or not self.process or not self.process.is_alive():
            raise AgentError("decisionrag_gateway_unhealthy")
        if time.monotonic() >= deadline:
            raise AgentError("decisionrag_physical_deadline_exceeded")
        self.connection.send_bytes(raw)
        value = self.receive(deadline)
        if time.monotonic() >= deadline:
            self.failure = "decisionrag_physical_deadline_exceeded"
            self.close()
            raise AgentError(self.failure)
        value["process_resources"] = self.job.observation()
        return value

    def close(self) -> None:
        if self.process and self.process.pid:
            if self.process.is_alive():
                self._signal_child(signal.SIGTERM)
            self.process.join(2)
            if self.process.is_alive():
                self._signal_child(getattr(signal, "SIGKILL", 9))
                self.process.join(2)
            if self.process.is_alive():
                raise AgentError("decisionrag_probability_process_stop_unverified")
        if self.connection:
            self.connection.close()
        if self.job:
            self.job.close()

    def _signal_child(self, sig: int) -> None:
        try:
            if sys.platform == "linux" and os.getpgid(self.process.pid) == self.process.pid:
                os.killpg(self.process.pid, sig)
            elif sig == signal.SIGTERM:
                self.process.terminate()
            else:
                self.process.kill()
        except ProcessLookupError:
            pass  # The subsequent join/is_alive check still verifies cleanup.


def create_app(config: GatewayConfig, *, process_factory=ModelProcess) -> FastAPI:
    token = token_from_file(config.bearer_token_file)
    config.cache_root.mkdir(parents=True, exist_ok=True)
    state = {}

    @asynccontextmanager
    async def lifespan(_app):
        # On Linux PR_SET_PDEATHSIG follows the spawning thread. Start from the
        # durable ASGI main thread, never an expiring worker-pool thread.
        state["process"] = process_factory(config)
        try:
            yield
        finally:
            await run_in_threadpool(state["process"].close)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authorized(request: Request) -> bool:
        return hmac.compare_digest(request.headers.get("authorization", ""), "Bearer " + token)

    @app.get("/healthz")
    async def health(request: Request):
        if not authorized(request):
            return JSONResponse({"error_code": "decisionrag_gateway_unauthorized"}, status_code=401)
        process = state["process"]
        ready = not process.failure and process.process.is_alive()
        return JSONResponse({"ready": ready, "protocol": config.protocol,
            "model_binding": process.startup.get("model_binding"), "concurrency": 1,
            "model_load_seconds": process.startup.get("model_load_seconds"),
            "physical_deadline_enforced_by_parent": True,
            "process_supervisor": config.version,
            "input_limits": {"candidates": 8, "sequence_tokens": config.sequence_tokens,
                "head_tokens": 256, "complete_token_preflight_required": True}}, status_code=200 if ready else 503)

    @app.post("/v1/decision")
    async def decision(request: Request):
        started = time.monotonic()
        if not authorized(request):
            return JSONResponse({"error_code": "decisionrag_gateway_unauthorized"}, status_code=401)
        raw = bytearray()
        try:
            async with asyncio.timeout(3):
                async for chunk in request.stream():
                    raw.extend(chunk)
                    if len(raw) > REQUEST_BYTES:
                        return JSONResponse({"error_code": "decisionrag_remote_request_byte_limit"}, status_code=413)
        except TimeoutError:
            return JSONResponse({"error_code": "decisionrag_request_body_timeout"}, status_code=408)
        try:
            call = GatewayCall.model_validate_json(bytes(raw), strict=True)
            value = call.request.request()
            if call.protocol != config.protocol or value.fingerprint() != call.request_sha256:
                raise AgentError("decisionrag_remote_request_identity_mismatch")
        except (ValidationError, AgentError):
            return JSONResponse({"error_code": "decisionrag_gateway_request_invalid"}, status_code=422)
        process = state["process"]
        if process.failure or not process.process.is_alive():
            return JSONResponse({"error_code": "decisionrag_gateway_unhealthy"}, status_code=503)
        if not process.gate.acquire(blocking=False):
            return JSONResponse({"error_code": "decisionrag_gateway_busy"}, status_code=429)
        try:
            result = await run_in_threadpool(process.roundtrip, bytes(raw), started + call.remaining_ms / 1000)
            return JSONResponse(result)
        except AgentError:
            return JSONResponse({"error_code": "decisionrag_gateway_unhealthy"}, status_code=503)
        finally:
            process.gate.release()

    return app
