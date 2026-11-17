"""Real file IPC/lifecycle with an explicitly simulated host and storage lease.

No Docker/parser/PG/GCS runs here; late worker_stopped is a protocol fixture.
Actual Linux termination proof remains an independent integration requirement.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.materials_v2.text import MaterialError
from citequest.materials_v3.private_spool_client import PrivateSpoolBrokerParser
from citequest.materials_v3.private_spool_recovery import discover
from citequest.v2.storage import StoredBlob

IMAGE = "sha256:" + "1" * 64


@pytest.fixture
def ipc(tmp_path):
    ingestion, execution_id, nonce = uuid4(), uuid4(), uuid4().hex
    lease = tmp_path / "data/runtime-s04-linux/artifacts" / str(ingestion) / str(execution_id)
    exchange = tmp_path / "data/runtime-s04-linux/exchanges" / nonce
    exchange.mkdir(parents=True)
    profile = tmp_path / "runtime/explicit-fixture-parser.json"
    profile.parent.mkdir()
    profile.write_text(json.dumps({"platform": "linux", "deadline_seconds": 2}))
    sha = hashlib.sha256(profile.read_bytes()).hexdigest()
    source = b"EXPLICIT simulated storage lease bytes"
    blob = StoredBlob("owner:" + str(uuid4()), hashlib.sha256(source).hexdigest(), len(source))

    class ExecutionFixture:
        active = False
        closed = threading.Event()

        @contextmanager
        def handoff(self, _blob, *, root, ingestion_id):
            assert root == tmp_path and ingestion_id == ingestion
            path = lease / "input/source.bin"
            path.parent.mkdir(parents=True)
            path.write_bytes(source)
            self.active = True
            try:
                yield path
            finally:
                path.unlink()
                path.parent.rmdir()
                self.active = False
                self.closed.set()

    execution = ExecutionFixture()
    client = PrivateSpoolBrokerParser(root=tmp_path, exchange=exchange, nonce=nonce, parser_image_id=IMAGE)
    result = SimpleNamespace(root=tmp_path, lease=lease, exchange=exchange, profile=profile, sha=sha,
        ingestion=ingestion, blob=blob, execution=execution, client=client, thread=None,
        release=threading.Event(), received=threading.Event(), wrote_receipt=threading.Event())
    yield result
    result.release.set()
    if result.thread is not None:
        result.thread.join(timeout=5)


def host_fixture(ipc, *, delay=0.0, withhold_response=False):
    def run():
        target = ipc.exchange / "request-1.json"
        until = time.monotonic() + 3
        while not target.exists() and time.monotonic() < until:
            time.sleep(0.005)
        if not target.exists():
            return
        raw = target.read_bytes()
        message = json.loads(raw)
        assert message["nonce"] == ipc.client.nonce
        ipc.received.set()
        time.sleep(delay)
        output = ipc.lease / "parse"
        output.mkdir()
        receipt = {"result": {"state": "ready", "fixture": "not_a_real_parser"}, "image_id": IMAGE,
            "profile_sha256": ipc.sha, "worker_stopped": True}
        receipt_raw = json.dumps(receipt).encode()
        (output / "receipt.json").write_bytes(receipt_raw)
        ipc.wrote_receipt.set()
        if withhold_response:
            assert ipc.release.wait(4)
        response = {"nonce": ipc.client.nonce, "request_sha256": hashlib.sha256(raw).hexdigest(),
            "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest()}
        pending = ipc.exchange / "response-1.pending"
        pending.write_text(json.dumps(response))
        pending.replace(ipc.exchange / "response-1.json")

    ipc.thread = threading.Thread(target=run, daemon=True)
    ipc.thread.start()


def invoke(ipc, source, *, seconds=1.5, bad_profile=False):
    return ipc.client({"source_path": str(source), "source_sha256": ipc.blob.sha256,
        "media_type": "text/plain", "expected_profile_sha256": "0" * 64 if bad_profile else ipc.sha},
        root=ipc.root, profile_path=ipc.profile, output=ipc.lease / "parse", python=Path("unused"), deadline_seconds=seconds)


def test_normal_receipt_releases_input_only_after_fixed_stop_protocol(ipc):
    host_fixture(ipc)
    with ipc.client.source_context(ipc.execution, ipc.blob, ingestion_id=ipc.ingestion) as source:
        result = invoke(ipc, source)
        assert result["worker_stopped"] and ipc.execution.active and source.exists()
    assert ipc.execution.closed.is_set() and not ipc.execution.active and not source.exists()
    record = json.loads((ipc.lease / "parser-request-state-1.json").read_bytes())
    assert record["state"] == "confirmed_stopped" and record["publication_permitted"] is False
    assert record["request_sha256"] == hashlib.sha256((ipc.exchange / "request-1.json").read_bytes()).hexdigest()


def test_late_stop_inside_cleanup_preserves_original_timeout(ipc):
    host_fixture(ipc, delay=0.2)
    with pytest.raises(MaterialError, match="PARSER_DEADLINE_EXCEEDED"), ipc.client.source_context(
            ipc.execution, ipc.blob, ingestion_id=ipc.ingestion) as source:
        invoke(ipc, source, seconds=0.05)
    assert ipc.execution.closed.is_set() and not source.exists()
    record = json.loads((ipc.lease / "parser-request-state-1.json").read_bytes())
    assert record["state"] == "failed_but_stopped" and record["termination_confirmed"]
    assert record["publication_permitted"] is False


def test_lost_ack_quarantines_until_late_fixed_response_without_dispatch(ipc):
    host_fixture(ipc, delay=0.1, withhold_response=True)
    with pytest.raises(MaterialError, match="PARSER_TERMINATION_UNCONFIRMED"), ipc.client.source_context(
            ipc.execution, ipc.blob, ingestion_id=ipc.ingestion) as source:
        invoke(ipc, source, seconds=0.05)
    assert ipc.wrote_receipt.is_set() and source.exists() and ipc.execution.active
    assert not ipc.execution.closed.is_set() and ipc.client.quarantined
    inventory = discover(ipc.root)
    assert inventory["deletions"] == inventory["dispatches"] == 0
    assert len(inventory["items"]) == 1 and not inventory["items"][0]["stop_evidence_observed"]
    with pytest.raises(MaterialError, match="INSTANCE_QUARANTINED"):
        invoke(ipc, source)
    assert not (ipc.exchange / "request-2.json").exists()
    ipc.release.set()
    assert ipc.execution.closed.wait(2)
    until = time.monotonic() + 1
    record = None
    while time.monotonic() < until:
        try:
            record = json.loads((ipc.lease / "parser-request-state-1.json").read_bytes())
        except OSError:
            # Windows may transiently deny opening a path during atomic
            # replacement. Product discovery already retains on OSError.
            time.sleep(0.01)
            continue
        if record["state"] == "late_stop_cleanup_only":
            break
        time.sleep(0.01)
    assert record["state"] == "late_stop_cleanup_only" and not source.exists()
    assert record["publication_permitted"] is False


@pytest.mark.parametrize("corrupt", ["nonce", "receipt_hash"])
def test_unbound_stop_cannot_release_unknown_input(ipc, corrupt):
    host_fixture(ipc, delay=0.1, withhold_response=True)
    with pytest.raises(MaterialError, match="PARSER_TERMINATION_UNCONFIRMED"), ipc.client.source_context(
            ipc.execution, ipc.blob, ingestion_id=ipc.ingestion) as source:
        invoke(ipc, source, seconds=0.05)
    raw = (ipc.exchange / "request-1.json").read_bytes()
    receipt = (ipc.lease / "parse/receipt.json").read_bytes()
    wrong = {"nonce": "wrong" if corrupt == "nonce" else ipc.client.nonce,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "receipt_sha256": "0" * 64 if corrupt == "receipt_hash" else hashlib.sha256(receipt).hexdigest()}
    (ipc.exchange / "response-1.json").write_text(json.dumps(wrong))
    time.sleep(0.15)
    assert source.exists() and ipc.execution.active and not ipc.execution.closed.is_set()
    inventory = discover(ipc.root)
    assert inventory["items"][0]["stop_evidence_observed"] is False
    ipc.release.set()
    assert ipc.execution.closed.wait(2)
    assert not source.exists()


def test_pre_dispatch_client_rejection_releases_input_with_zero_request(ipc):
    with pytest.raises(MaterialError, match="BROKER_PROFILE_BINDING_MISMATCH"), ipc.client.source_context(
            ipc.execution, ipc.blob, ingestion_id=ipc.ingestion) as source:
        invoke(ipc, source, bad_profile=True)
    assert ipc.execution.closed.is_set() and not source.exists()
    assert not (ipc.exchange / "request-1.json").exists()
