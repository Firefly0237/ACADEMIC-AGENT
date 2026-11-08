"""Exact local supervisor recovery after parent/broker termination.

This trusted CLI has no HTTP/model exposure. Unavailable host/daemon, changed
fence, live original broker, live parent or unidentified parser means retain.
Only --stop-owned-children may stop a fully identity-checked child. No retry,
source ingestion, late-result adoption, ready overwrite, or cloud claim exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from uuid import UUID

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from citequest.materials_v3.broker_protocol import rooted, strict_object  # noqa: E402
from citequest.materials_v3.host_identity import host_identity, stopped_original_process  # noqa: E402

DOCKER = ["docker", "--config", str(ROOT / "data/runtime-s07/docker-config")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(*args: str, timeout: float = 10) -> bytes:
    reply = subprocess.run([*DOCKER, *args], capture_output=True, timeout=timeout)
    if reply.returncode:
        raise RuntimeError("recovery_control_unavailable_" + args[0])
    return reply.stdout


def inspected(identifier: str) -> dict | None:
    reply = subprocess.run([*DOCKER, "inspect", identifier], capture_output=True, timeout=10)
    if reply.returncode:
        # Docker absence is authoritative only when the SAME live daemon
        # responds to info, before and after. Other errors remain unknown.
        if b"No such object" in reply.stderr or b"No such container" in reply.stderr:
            return None
        raise RuntimeError("container_state_unavailable")
    return json.loads(reply.stdout)[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--parent-id", required=True)
    parser.add_argument("--dsn-file", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stop-owned-children", action="store_true")
    parser.add_argument("--apply-failure", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(ROOT / "reports/stages/v3/S15-cloud-blob")
    out.mkdir(parents=True, exist_ok=False)
    status = {"status":"retain_unknown", "model_calls":0, "main_database":False,
              "publication_permitted":False, "retry_permitted":False}
    try:
        assert sha(args.profile) == args.profile_sha256
        profile = json.loads(args.profile.read_bytes(), object_pairs_hook=strict_object)
        assert profile["schema_version"] == "cq.linux-material-parent-execution/3"
        ingestion, execution = UUID(profile["assigned_ingestion_id"]), UUID(profile["assigned_execution_id"])
        assert profile["database_name"] == "cq_s15_blob_20261106_r2"
        nonce = profile["nonce"]
        exchange = rooted(ROOT, profile["exchange_relative"])
        started = json.loads((exchange / "broker-started.json").read_bytes())
        assert started["profile_sha256"] == args.profile_sha256
        assert started["source_sha256"] == profile["host_private_broker_sha256"]
        assert started.get("docker_write_protocol") == "private-broker-Docker-write/1", "old_broker_without_write_journal_cannot_be_reaped"
        assert started["host"] == host_identity(ROOT), "different_host_or_boot_remains_quarantined"
        assert stopped_original_process(started["process"]), "original_broker_may_still_dispatch"
        daemon = call("info", "--format", "{{.ID}}").decode().strip()
        assert daemon and daemon == started["docker_daemon_id"], "different_or_unavailable_daemon"
        journals = []
        for path in sorted(exchange.glob("docker-write-*.json")):
            assert not path.is_symlink() and path.stat().st_size <= 8192
            operation = json.loads(path.read_bytes(), object_pairs_hook=strict_object)
            assert operation["version"] == "private-broker-Docker-write/1"
            assert operation["profile_sha256"] == args.profile_sha256 and operation["nonce"] == nonce
            assert operation["image_id"] == profile["image_id"]
            assert type(operation["ordinal"]) is int and operation["ordinal"] > 0
            assert path.name == f"docker-write-{operation['ordinal']}.json"
            assert type(operation["sequence"]) is int and operation["sequence"] in {1, 2}
            assert operation["target"] == "cq-s04-linux-" + nonce + "-" + str(operation["sequence"])
            assert operation["operation"] in {"create", "start", "kill"}
            # A stopped client/CLI or an inspect404 cannot prove that a sent
            # create/start will not arrive later at the Docker daemon.
            assert operation["state"] == "acknowledged", "sent_or_unknown_broker_write_retained"
            journals.append({**operation,"record_sha256":sha(path)})
        assert sorted(entry["ordinal"] for entry in journals) == list(range(1, len(journals) + 1))
        parent = inspected(args.parent_id)
        assert parent and parent["Id"] == args.parent_id and parent["Image"] == profile["parent_runtime_image_id"]
        assert parent["Name"] == "/cq-s15-material-parent-" + nonce
        assert parent["Config"]["Labels"].get("cq.scope") == "material-private-parent"
        parent_mounts = {mount["Destination"]:mount for mount in parent["Mounts"]}
        assert Path(parent_mounts["/app/runtime/material-parent/execution-active.json"]["Source"]).resolve() == args.profile.resolve()
        assert not parent_mounts["/app/runtime/material-parent/execution-active.json"]["RW"]
        assert Path(parent_mounts["/run/material-dsn"]["Source"]).resolve() == args.dsn_file.resolve()
        actual_arguments = parent["Config"]["Cmd"]
        assert actual_arguments[actual_arguments.index("--profile-sha256") + 1] == args.profile_sha256
        assert actual_arguments[actual_arguments.index("--file-id") + 1] == profile["assigned_file_id"]
        storage_path = Path(parent_mounts["/run/storage.json"]["Source"]).resolve(strict=True)
        assert sha(storage_path) == profile["storage_config_sha256"]
        assert parent["State"]["Running"] is False and parent["State"]["StartedAt"] != "0001-01-01T00:00:00Z"
        assert parent["HostConfig"]["RestartPolicy"]["Name"] in {"", "no"}
        assert not any(mount["Type"] == "volume" for mount in parent["Mounts"])
        lease = rooted(ROOT, f"data/runtime-s04-linux/artifacts/{ingestion}/{execution}")
        children = []
        for sequence in (1, 2):
            request_path = exchange / f"request-{sequence}.json"
            consumed_path = exchange / f"consumed-{sequence}.json"
            sequence_journals = [entry for entry in journals if entry["sequence"] == sequence]
            if not request_path.exists():
                assert not consumed_path.exists() and not sequence_journals
                continue
            assert not request_path.is_symlink() and request_path.stat().st_size <= 262144
            raw = request_path.read_bytes()
            request_sha256 = hashlib.sha256(raw).hexdigest()
            if consumed_path.exists():
                assert not consumed_path.is_symlink() and consumed_path.stat().st_size <= 8192
                consumed = json.loads(consumed_path.read_bytes(), object_pairs_hook=strict_object)
                assert consumed["request_sha256"] == request_sha256, "consumed_request_fence_changed_no_reap"
                assert consumed["transport"] == profile["source_transport"]
            else:
                assert not sequence_journals, "Docker_write_without_consumed_request_retained"
            request = json.loads(raw, object_pairs_hook=strict_object)
            assert request["nonce"] == nonce and request["sequence"] == sequence
            output = rooted(ROOT, request["output_relative"])
            assert output == lease / ("parse" if sequence == 1 else "chunk")
            source = rooted(ROOT, request["source_relative"])
            assert source.is_relative_to(lease)
            assert request["request"]["source_sha256"] == profile["assigned_source_sha256"]
            assert request["profile_sha256"] == profile["source_bindings"][request["profile_relative"]]
            name = "cq-s04-linux-" + nonce + "-" + str(sequence)
            child = inspected(name)
            observation = {"name":name, "request_sha256":request_sha256,
                           "running":False, "exists":child is not None}
            if child is not None:
                assert child["Image"] == profile["image_id"] and child["Name"] == "/" + name
                assert child["Config"]["Labels"].get("citequest.owner") == "S04-linux-host-broker"
                assert child["HostConfig"]["NetworkMode"] == "none" and child["HostConfig"]["ReadonlyRootfs"]
                assert child["HostConfig"]["RestartPolicy"]["Name"] in {"", "no"}
                mounts = {item["Destination"]: item for item in child["Mounts"]}
                assert all(item["Type"] == "bind" for item in child["Mounts"])
                assert Path(mounts["/app/input/material"]["Source"]).resolve() == source and not mounts["/app/input/material"]["RW"]
                assert Path(mounts["/app/output"]["Source"]).resolve() == output
                identifier = child["Id"]
                create_receipts = [entry for entry in sequence_journals if entry["operation"] == "create"]
                assert len(create_receipts) == 1 and create_receipts[0]["container_id"] == identifier
                if child["State"]["Running"]:
                    assert args.stop_owned_children, "owned_child_still_running_no_implicit_kill"
                    call("kill", "--signal", "KILL", identifier)
                    call("wait", identifier)
                    child = inspected(identifier)
                    assert child and child["Id"] == identifier and child["Image"] == profile["image_id"]
                    observation["explicit_owned_KILL_sent"] = True
                assert child["State"]["Running"] is False
                observation.update(id=identifier, image=child["Image"], exit_code=child["State"]["ExitCode"])
            children.append(observation)
        assert call("info", "--format", "{{.ID}}").decode().strip() == daemon
        proof = {"version":"local-private-parent-stop/1", "profile_sha256":args.profile_sha256,
            "ingestion_id":str(ingestion), "execution_id":str(execution), "source_sha256":profile["assigned_source_sha256"],
            "host":started["host"], "original_broker_process":started["process"], "broker_stopped":True,
            "docker_daemon_id":daemon, "physical_stop_confirmed":True,
            "parent":{"id":parent["Id"],"image":parent["Image"],"running":False,"exit_code":parent["State"]["ExitCode"]},
            "children":children, "acknowledged_broker_writes":journals,
            "late_outputs_adopted":False, "cross_host_or_GCP_proof":False}
        proof_path = out / "stop-proof.json"
        proof_path.write_text(json.dumps(proof, indent=2) + "\n")
        status.update(status="physical_stop_observed_no_failure_mutation",proof_sha256=sha(proof_path))
        if args.apply_failure:
            name = "cq-s15-reap-" + nonce
            command = ["create", "--name", name, "--label", "cq.owner=citequest", "--label", "cq.task=s15-material-reap",
                "--network", "citequest-s08-fixture_default", "--read-only", "--user", "10001:10001",
                "--cpus", "1", "--memory", "1g", "--pids-limit", "64", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=64m"]
            mounts = [(out,"/proof",False),(args.profile.resolve(),"/run/profile.json",True),
                (args.profile.resolve(),"/app/runtime/material-parent/execution-active.json",True),
                (storage_path,"/run/storage.json",True),
                (args.dsn_file.resolve(),"/run/business-dsn",True),
                (ROOT / "scripts/s04_linux/reap_private_parent_db.py","/run/reap.py",True)]
            for actual, target, readonly in mounts:
                command += ["--mount",f"type=bind,source={actual},target={target}" + (",readonly" if readonly else "")]
            command += ["--entrypoint","timeout",profile["parent_runtime_image_id"],"--signal=KILL","20s","python","/run/reap.py",
                "--proof","/proof/stop-proof.json","--proof-sha256",sha(proof_path),"--profile","/run/profile.json",
                "--dsn","/run/business-dsn","--storage","/run/storage.json","--out","/proof/database-receipt.json"]
            identifier = call(*command).decode().strip()
            (out / "reaper-container-id.txt").write_text(identifier + "\n")
            code = None
            try:
                call("start", identifier)
                code = int(call("wait", identifier, timeout=25).decode().strip())
            finally:
                terminal = inspected(identifier)
                assert terminal and terminal["Id"] == identifier and terminal["Image"] == profile["parent_runtime_image_id"]
                assert terminal["Config"]["Labels"].get("cq.task") == "s15-material-reap"
                assert not any(mount["Type"] == "volume" for mount in terminal["Mounts"])
                if terminal["State"]["Running"]:
                    call("kill","--signal","KILL",identifier)
                    call("wait",identifier)
                    terminal = inspected(identifier)
                logs = subprocess.run([*DOCKER,"logs",identifier],capture_output=True,timeout=5)
                (out / "db-helper.log").write_bytes(logs.stdout + logs.stderr)
                (out / "db-helper-state.json").write_text(json.dumps({"id":identifier,"image":terminal["Image"],"state":terminal["State"]},indent=2)+'\n')
                if code == 0 and not terminal["State"]["Running"]:
                    call("rm",identifier)
            assert code == 0, "fenced_SQL_failure_retained_no_retry"
            receipt = json.loads((out / "database-receipt.json").read_bytes())
            assert receipt["publication_permitted"] is False and receipt["late_outputs_adopted"] is False
            source = lease / "input/source.bin"
            if source.exists():
                assert not source.is_symlink() and source.stat().st_size == profile["assigned_size_bytes"]
                assert sha(source) == profile["assigned_source_sha256"]
                source.chmod(0o600)
                source.unlink()
                source.parent.rmdir()
            status.update(status="fenced_failure_and_owned_input_cleanup_complete", input_exists=source.exists(),
                          database_receipt_sha256=sha(out / "database-receipt.json"))
    except BaseException as error:
        status.update(error_type=type(error).__name__,error_code=str(error) if isinstance(error, AssertionError) else "control_failed_retain")
        raise
    finally:
        (out / "verification.json").write_text(json.dumps(status,indent=2) + "\n")


if __name__ == "__main__":
    main()
