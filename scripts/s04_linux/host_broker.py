"""Fixed host CLI launcher. No PG/model secrets, sockets or arbitrary parameters.

Runs only the approved immutable image and the two operations for one File/lease.
The PG worker is nonroot and never receives Docker access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from citequest.materials_v3.broker_protocol import admit, rooted, strict_object  # noqa:E402


def command(args: list[str], *, timeout: float = 5) -> str:
    result = subprocess.run(
        ["docker", *args], capture_output=True, timeout=max(0.1, timeout), encoding="utf-8", errors="replace"
    )
    if result.returncode:
        raise RuntimeError("Docker control failed: " + result.stderr[:500])
    return result.stdout.strip()


def atomic(path: Path, value: dict) -> None:
    pending = path.with_suffix(".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    pending.rename(path)


def execute(value: dict, profile: dict, exchange: Path) -> dict:
    start = time.monotonic()
    item, output = value["message"], value["output"]
    worker = value["profile"]
    seq = item["sequence"]
    output.mkdir(parents=True, exist_ok=False)
    control = {
        **item["request"],
        "root": "/app",
        "source_path": "/app/input/material",
        "output": "/app/output",
        "profile": "/app/" + item["profile_relative"],
        "profile_sha256": value["profile_sha256"],
        "start_token": profile["nonce"] + str(seq),
        "orphan_deadline_seconds": item["deadline_seconds"],
    }
    control_path = output.parent / f"control-{seq}.json"
    control_path.write_text(json.dumps(control, ensure_ascii=False), encoding="utf-8")
    name = "cq-s04-linux-" + profile["nonce"] + "-" + str(seq)
    args = [
        "create",
        "--name",
        name,
        "--label",
        "citequest.owner=S04-linux-host-broker",
        "--network",
        "none",
        "--read-only",
        "--user",
        "10001:10001",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--memory",
        str(worker["memory_limit_bytes"]),
        "--memory-swap",
        str(worker["memory_limit_bytes"]),
        "--cpus",
        "2",
        "--pids-limit",
        "128",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,noexec,size=128m",
        "--log-opt",
        "max-size=2m",
        "--log-driver",
        "json-file",
        "--log-opt",
        "compress=false",
        "--log-opt",
        "max-file=1",
        "--env",
        "HOME=/tmp",
        "--env",
        "TMPDIR=/tmp",
    ]
    for source, target, readonly in [
        (value["source"], "/app/input/material", True),
        (control_path, "/app/input/request.json", True),
        (output, "/app/output", False),
        (ROOT / "data/runtime-v2/docling-models", "/app/data/runtime-v2/docling-models", True),
    ]:
        args += ["--mount", f"type=bind,source={source},target={target}" + (",readonly" if readonly else "")]
    args += ["--entrypoint", "python", profile["image_id"], "-S", "scripts/s04_linux/worker_entry.py"]
    name_id = command(args)
    violations = []
    logs = {"stdout": bytearray(), "stderr": bytearray()}
    phase_observed = False
    command(["start", name])
    log_process = subprocess.Popen(["docker", "logs", "--follow", name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def drain(label: str) -> None:
        pipe = getattr(log_process, label)
        while data := pipe.read(8192):
            remaining = worker["max_log_bytes_each"] - len(logs[label])
            logs[label].extend(data[: max(0, remaining)])
            if len(data) > remaining:
                violations.append("WORKER_LOG_LIMIT_EXCEEDED")
                break

    threads = [threading.Thread(target=drain, args=(label,), daemon=True) for label in logs]
    for thread in threads:
        thread.start()
    state = {}
    try:
        while True:
            remaining = item["deadline_seconds"] - (time.monotonic() - start)
            if remaining <= 0:
                violations.append("PARSER_DEADLINE_EXCEEDED")
                break
            state = json.loads(command(["inspect", name, "--format", "{{json .State}}"], timeout=min(5, remaining)))
            files = list(output.rglob("*"))
            if any(p.is_symlink() for p in files):
                violations.append("WORKER_ARTIFACT_SYMLINK")
            if sum(p.stat().st_size for p in files if p.is_file()) > worker["max_output_artifact_bytes"]:
                violations.append("WORKER_ARTIFACT_LIMIT_EXCEEDED")
            phase = output / "phase.json"
            if phase.exists() and not phase_observed:
                expected = {
                    "start_token": control["start_token"],
                    "phase": "parsing",
                    "profile_sha256": value["profile_sha256"],
                    "source_sha256": item["request"]["source_sha256"],
                }
                if phase.stat().st_size > 4096 or json.loads(phase.read_bytes()) != expected:
                    violations.append("WORKER_PHASE_EVENT_INVALID")
                else:
                    phase_observed = True
                    atomic(
                        exchange / f"phase-{seq}.json",
                        {
                            "nonce": profile["nonce"],
                            "request_sha256": value["request_sha256"],
                            "profile_sha256": value["profile_sha256"],
                        },
                    )
            if (exchange / f"cancel-{seq}").exists():
                violations.append("PARSER_CANCELLED")
            if violations or state.get("Status") in {"exited", "dead"}:
                break
            time.sleep(0.05)
    except Exception:
        violations.append("BROKER_DOCKER_CONTROL_FAILED")
        raise
    finally:
        if state.get("Status") not in {"exited", "dead"}:
            try:
                command(["kill", name])
            except Exception:
                violations.append("WORKER_STOP_STATE_UNCERTAIN")
        log_process.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=1)
        for label, data in logs.items():
            (output / (label + ".log")).write_bytes(data)
    state = json.loads(command(["inspect", name, "--format", "{{json .State}}"]))
    files = list(output.rglob("*"))
    total = sum(p.stat().st_size for p in files if p.is_file())
    result_path = output / "result.json"
    result = {"state": "failed", "code": "WORKER_NO_RESULT", "can_qa": False}
    if total > worker["max_output_artifact_bytes"] or any(p.is_symlink() for p in files):
        violations.append("WORKER_ARTIFACT_LIMIT_EXCEEDED")
    elif result_path.is_file():
        result = json.loads(result_path.read_bytes(), object_pairs_hook=strict_object)
        if result.get("profile_sha256") != value["profile_sha256"]:
            violations.append("WORKER_PROFILE_BINDING_MISMATCH")
    if state.get("OOMKilled"):
        violations.append("PARSER_MEMORY_LIMIT")
    elapsed = time.monotonic() - start
    if elapsed > item["deadline_seconds"]:
        violations.append("PARSER_DEADLINE_EXCEEDED")
    if hashlib.sha256(value["source"].read_bytes()).hexdigest() != item["request"]["source_sha256"]:
        violations.append("ORIGINAL_SOURCE_CHANGED_DURING_PARSE")
    if violations:
        result = {
            "state": "failed",
            "code": violations[0],
            "can_qa": False,
            "partial_result_preserved": result_path.exists(),
        }
    receipt = {
        "schema_version": "cq.linux-parser-receipt/1",
        "result": result,
        "validation_phase_observed": phase_observed,
        "profile_sha256": value["profile_sha256"],
        "profile_after_sha256": value["profile_sha256"],
        "elapsed_seconds": elapsed,
        "deadline_seconds": item["deadline_seconds"],
        "worker_stopped": state.get("Status") in {"exited", "dead"},
        "image_id": profile["image_id"],
        "container_id": name_id,
        "container_state": state,
        "violations": violations,
        "artifact_bytes": total,
        "no_credentials_in_worker_environment": True,
        "broker_mode": "fixed-host-cli-no-socket",
    }
    observed_host = json.loads(command(["inspect", name, "--format", "{{json .HostConfig}}"]))
    receipt["observed_host_config"] = {
        k: observed_host.get(k)
        for k in [
            "NetworkMode",
            "ReadonlyRootfs",
            "Memory",
            "MemorySwap",
            "NanoCpus",
            "PidsLimit",
            "CapDrop",
            "SecurityOpt",
        ]
    }
    receipt["observed_user"] = command(["inspect", name, "--format", "{{.Config.User}}"])
    raw = json.dumps(receipt, ensure_ascii=False, allow_nan=False, indent=2).encode()
    if len(raw) + total > worker["max_output_artifact_bytes"]:
        receipt["result"] = {"state": "failed", "code": "WORKER_ARTIFACT_LIMIT_EXCEEDED", "can_qa": False}
        raw = json.dumps(receipt, indent=2).encode()
    (output / "receipt.json").write_bytes(raw)
    return {
        "nonce": profile["nonce"],
        "request_sha256": value["request_sha256"],
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    args = parser.parse_args()
    raw = args.profile.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.profile_sha256:
        raise ValueError("Exact execution profile required")
    profile = json.loads(raw, object_pairs_hook=strict_object)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != profile["host_broker_sha256"]:
        raise ValueError("Exact fixed host launcher required")
    if (
        hashlib.sha256((ROOT / "src/citequest/materials_v3/broker_protocol.py").read_bytes()).hexdigest()
        != profile["host_protocol_sha256"]
    ):
        raise ValueError("Exact fixed host protocol required")
    if profile["broker_mode"] != "fixed-host-cli-no-socket" or profile["product_api_docker_access"] is not False:
        raise ValueError("Restricted broker required")
    if not profile["image_id"].startswith("sha256:") or len(profile["image_id"]) != 71:
        raise ValueError("Immutable image required")
    exchange = rooted(ROOT, profile["exchange_relative"])
    if not exchange.is_relative_to(ROOT / "data/runtime-s04-linux/exchanges"):
        raise ValueError("Owned exchange required")
    exchange.mkdir(parents=True, exist_ok=True)
    with (exchange / "broker-started.json").open("x", encoding="utf8") as f:
        json.dump(
            {
                "profile_sha256": args.profile_sha256,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            f,
        )
    started = time.monotonic()
    sequence = 1
    lease = None
    while time.monotonic() - started < 240:
        if (exchange / "parent-done.json").exists() or (exchange / "STOP").exists():
            break
        request = exchange / f"request-{sequence}.json"
        if request.exists():
            raw = request.read_bytes()
            value = admit(raw, root=ROOT, profile=profile, sequence=sequence, lease_root=lease)
            with (exchange / f"consumed-{sequence}.json").open("x", encoding="utf8") as f:
                json.dump({"request_sha256": value["request_sha256"]}, f)
            if (exchange / f"response-{sequence}.json").exists():
                raise ValueError("Stale receipt rejected")
            request_started = time.monotonic()
            try:
                response = execute(value, profile, exchange)
            except Exception as exc:
                name = "cq-s04-linux-" + profile["nonce"] + "-" + str(sequence)
                try:
                    state = json.loads(command(["inspect", name, "--format", "{{json .State}}"]))
                    if state.get("Running") is True:
                        command(["kill", name])
                        state = json.loads(command(["inspect", name, "--format", "{{json .State}}"]))
                except Exception:
                    state = {"Status": "unknown", "Running": None}
                receipt = {
                    "schema_version": "cq.linux-parser-receipt/1",
                    "result": {"state": "failed", "code": "BROKER_CHILD_CONTROL_FAILED", "can_qa": False},
                    "profile_sha256": value["profile_sha256"],
                    "validation_phase_observed": False,
                    "worker_stopped": state.get("Running") is False,
                    "container_state": state,
                    "elapsed_seconds": time.monotonic() - request_started,
                    "deadline_seconds": value["message"]["deadline_seconds"],
                    "exception": type(exc).__name__,
                    "detail": str(exc)[:1000],
                    "image_id": profile["image_id"],
                }
                value["output"].mkdir(parents=True, exist_ok=True)
                receipt_raw = json.dumps(receipt, indent=2).encode()
                (value["output"] / "receipt.json").write_bytes(receipt_raw)
                response = {
                    "nonce": profile["nonce"],
                    "request_sha256": value["request_sha256"],
                    "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest(),
                }
            atomic(exchange / f"response-{sequence}.json", response)
            lease = value["output"].parent
            sequence += 1
            if sequence > 2:
                break
        time.sleep(0.05)
    atomic(
        exchange / "broker-finished.json",
        {
            "completed_requests": sequence - 1,
            "elapsed_seconds": time.monotonic() - started,
            "parent_done": (exchange / "parent-done.json").exists(),
            "PG_or_model_keys_accessed": False,
        },
    )


if __name__ == "__main__":
    main()
