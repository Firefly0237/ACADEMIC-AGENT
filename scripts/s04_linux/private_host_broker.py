"""Versioned secret-free host adapter; reuses the fixed parser execution helper.

No PG/GCS/model credentials are loaded. The original launcher/broker/profile
remain untouched. Restart is observation-only; consumed requests never replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import host_broker as fixed_transport  # noqa: E402
from host_broker import ROOT as HELPER_ROOT  # noqa: E402
from host_broker import atomic, command, execute  # noqa: E402

from citequest.materials_v3.broker_protocol import rooted, strict_object  # noqa: E402
from citequest.materials_v3.host_identity import current_process_identity, host_identity  # noqa: E402
from citequest.materials_v3.private_spool_protocol import TRANSPORT, admit  # noqa: E402


def write_operation_state(path: Path, value: dict) -> None:
    """Durable process-death journal; not a proof across a host/boot change."""
    pending = path.with_suffix(".pending")
    with pending.open("wb") as stream:
        stream.write(json.dumps(value, sort_keys=True).encode())
        stream.flush()
        os.fsync(stream.fileno())
    pending.replace(path)
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    args = parser.parse_args()
    raw = args.profile.read_bytes()
    if len(raw) > 1024**2 or hashlib.sha256(raw).hexdigest() != args.profile_sha256:
        raise ValueError("exact_bounded_private_parent_profile_required")
    profile = json.loads(raw, object_pairs_hook=strict_object)
    if (profile.get("schema_version") not in {"cq.linux-material-parent-execution/2", "cq.linux-material-parent-execution/3"}
            or profile.get("source_transport") != TRANSPORT
            or profile.get("product_api_docker_access") is not False or HELPER_ROOT != ROOT):
        raise ValueError("explicit_private_parent_execution_required")
    for relative, field in (
        ("scripts/s04_linux/private_host_broker.py", "host_private_broker_sha256"),
        ("scripts/s04_linux/host_broker.py", "host_execution_helper_sha256"),
        ("src/citequest/materials_v3/private_spool_protocol.py", "host_private_protocol_sha256"),
        ("src/citequest/materials_v3/host_identity.py", "host_identity_sha256"),
    ):
        if hashlib.sha256(rooted(ROOT, relative).read_bytes()).hexdigest() != profile[field]:
            raise ValueError("exact_private_host_source_required")
    release_path = rooted(ROOT, profile["parser_release_file"])
    if hashlib.sha256(release_path.read_bytes()).hexdigest() != profile["parser_release_sha256"]:
        raise ValueError("exact_parser_release_required")
    release = json.loads(release_path.read_bytes(), object_pairs_hook=strict_object)
    if (profile["image_id"] != release["image_id"] or profile["parser_profile_relative"] != release["parser_profile_relative"]
            or profile["parser_profile_sha256"] != release["parser_profile_sha256"]):
        raise ValueError("parser_image_and_policy_release_mismatch")
    if command(["image", "inspect", profile["image_id"], "--format", "{{.Id}}"], timeout=5) != profile["image_id"]:
        raise ValueError("existing_fixed_parser_image_required_no_implicit_pull")
    exchange = rooted(ROOT, profile["exchange_relative"])
    if not exchange.is_relative_to(ROOT / "data/runtime-s04-linux/exchanges"):
        raise ValueError("owned_exchange_required")
    exchange.mkdir(parents=True, exist_ok=True)
    # A restarted host must not consume an old request a second time. Discovery
    # and physical-stop reconciliation are separate read-only/operator actions.
    if (exchange / "broker-started.json").exists() or any(exchange.glob("consumed-*.json")):
        raise ValueError("existing_exchange_requires_observation_not_dispatch_restart")
    (exchange / "broker-started.json").write_text(json.dumps({"profile_sha256": args.profile_sha256,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "transport": TRANSPORT,
        "host": host_identity(ROOT), "process": current_process_identity(),
        "docker_write_protocol": "private-broker-Docker-write/1",
        "docker_daemon_id": command(["info", "--format", "{{.ID}}"], timeout=5)}))
    start, sequence, lease = time.monotonic(), 1, None
    physical_command = fixed_transport.command
    write_ordinal = 0

    def journaled_command(arguments: list[str], *, timeout: float = 5) -> str:
        nonlocal write_ordinal
        if arguments[0] not in {"create", "start", "kill"}:
            return physical_command(arguments, timeout=timeout)
        write_ordinal += 1
        path = exchange / f"docker-write-{write_ordinal}.json"
        if path.exists():
            raise ValueError("existing_write_intent_requires_reconciliation")
        target = (arguments[arguments.index("--name") + 1] if arguments[0] == "create" else arguments[1])
        assert target == "cq-s04-linux-" + profile["nonce"] + "-" + str(sequence)
        value = {"version":"private-broker-Docker-write/1", "profile_sha256":args.profile_sha256,
            "nonce":profile["nonce"], "sequence":sequence, "ordinal":write_ordinal,
            "operation":arguments[0], "target":target, "image_id":profile["image_id"],
            "argv_sha256":hashlib.sha256(json.dumps(arguments).encode()).hexdigest(), "state":"intent"}
        write_operation_state(path, value)
        value["state"] = "sent"
        write_operation_state(path, value)
        try:
            reply = physical_command(arguments, timeout=timeout)
        except BaseException:
            value["state"] = "unknown"
            write_operation_state(path, value)
            raise
        value["state"] = "acknowledged"
        if arguments[0] == "create":
            value["container_id"] = reply
        write_operation_state(path, value)
        return reply

    fixed_transport.command = journaled_command
    outcome = "idle_window_finished"
    try:
        while time.monotonic() - start < 240:
            if (exchange / "parent-done.json").exists() or (exchange / "STOP").exists():
                outcome = "parent_done_or_stopped"
                break
            request = exchange / f"request-{sequence}.json"
            if not request.exists():
                time.sleep(0.02)
                continue
            raw_request = request.read_bytes()
            value = admit(raw_request, root=ROOT, profile=profile, sequence=sequence, lease_root=lease)
            with (exchange / f"consumed-{sequence}.json").open("x", encoding="utf-8") as stream:
                json.dump({"request_sha256": value["request_sha256"], "transport": TRANSPORT}, stream)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name == "posix":
                descriptor = os.open(exchange, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            if (exchange / f"response-{sequence}.json").exists():
                raise ValueError("stale_response_cannot_dispatch")
            # Existing standard Docker CLI/resource/secretless/no-network
            # enforcement; the child sees only this one read-only source.
            response = execute(value, profile, exchange)
            atomic(exchange / f"response-{sequence}.json", response)
            receipt = json.loads((value["output"] / "receipt.json").read_bytes())
            if receipt.get("worker_stopped") is not True:
                outcome = "physical_stop_unconfirmed_retain_no_more_dispatch"
                break
            lease = value["output"].parent
            sequence += 1
            if sequence > 2:
                outcome = "two_fixed_operations_finished"
                break
    finally:
        atomic(exchange / "broker-finished.json", {"profile_sha256": args.profile_sha256,
            "outcome": outcome, "parent_done": (exchange / "parent-done.json").exists(),
            "dispatch_restart_permitted": False, "no_model_credentials": True})


if __name__ == "__main__":
    main()
