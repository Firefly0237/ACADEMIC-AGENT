"""Pure transport controls with authored bytes; no PG/parser/SDK/GCP execution."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import pytest

from citequest.materials_v3.private_spool_protocol import TRANSPORT, admit

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def handoff(tmp_path):
    ingestion, lease = str(uuid4()), str(uuid4())
    source = tmp_path / "data/runtime-s04-linux/artifacts" / ingestion / lease / "input/source.bin"
    source.parent.mkdir(parents=True)
    data = b"EXPLICIT synthetic private source protocol fixture"
    source.write_bytes(data)
    relative = "runtime/s04-linux/worker-profile-cpu-v3.json"
    worker = tmp_path / relative
    worker.parent.mkdir(parents=True)
    worker.write_bytes((ROOT / relative).read_bytes())
    sha = hashlib.sha256(worker.read_bytes()).hexdigest()
    output = source.parent.parent / "parse"
    profile = {"schema_version": "cq.linux-material-parent-execution/2", "source_transport": TRANSPORT,
        "product_api_docker_access": False, "nonce": uuid4().hex, "parser_profile_relative": relative,
        "parser_profile_sha256": sha, "source_bindings": {relative: sha},
        "assigned_ingestion_id": ingestion, "assigned_source_sha256": hashlib.sha256(data).hexdigest(),
        "assigned_size_bytes": len(data), "assigned_media_type": "text/plain"}
    message = {"schema_version": "cq.parser-broker-request/1", "nonce": profile["nonce"], "sequence": 1,
        "source_relative": source.relative_to(tmp_path).as_posix(), "output_relative": output.relative_to(tmp_path).as_posix(),
        "profile_relative": relative, "profile_sha256": sha, "deadline_seconds": 30,
        "request": {"source_sha256": profile["assigned_source_sha256"], "media_type": "text/plain", "expected_profile_sha256": sha}}
    return tmp_path, source, output, profile, message


def call(root, profile, message, sequence=1, lease_root=None):
    return admit(json.dumps(message).encode(), root=root, profile=profile, sequence=sequence, lease_root=lease_root)


def test_exact_private_input_hash_size_lease_and_no_cas_claim(handoff):
    root, source, output, profile, message = handoff
    value = call(root, profile, message)
    assert value["source"] == source and value["output"] == output
    assert value["operation"] == "parse" and value["message"] == message
    assert not (root / "data/runtime-v2-app/blobs").exists()


def test_v3_server_execution_id_is_exact_and_v2_asrun_contract_remains(handoff):
    root, _source, output, original, message = handoff
    assert call(root, original, message)["operation"] == "parse"
    profile = {**original, "schema_version": "cq.linux-material-parent-execution/3",
               "assigned_execution_id": output.parent.name}
    assert call(root, profile, message)["operation"] == "parse"
    with pytest.raises(ValueError, match="server_assigned_execution"):
        call(root, {**profile, "assigned_execution_id": str(uuid4())}, message)
    with pytest.raises(ValueError, match="server_assigned_chunk_execution"):
        call(root, profile, message, sequence=2, lease_root=None)


@pytest.mark.parametrize("case", ["legacy_profile", "wrong_transport", "docker_API", "nonce", "bool_sequence",
    "unapproved_policy", "policy_sha", "nan_deadline", "expanded_deadline", "source_escape", "old_CAS",
    "wrong_ingestion", "wrong_lease_input", "size", "hash", "existing_output", "extra_request", "duplicate_keys"])
def test_private_handoff_negative_boundaries(handoff, case):
    root, source, output, original_profile, original_message = handoff
    profile, message = deepcopy(original_profile), deepcopy(original_message)
    if case == "legacy_profile":
        profile["schema_version"] = "cq.linux-parser-execution/1"
    elif case == "wrong_transport":
        profile["source_transport"] = "local-cas"
    elif case == "docker_API":
        profile["product_api_docker_access"] = True
    elif case == "nonce":
        message["nonce"] = uuid4().hex
    elif case == "bool_sequence":
        message["sequence"] = True
    elif case == "unapproved_policy":
        message["profile_relative"] = "runtime/user-policy.json"
    elif case == "policy_sha":
        message["profile_sha256"] = "0" * 64
    elif case == "nan_deadline":
        message["deadline_seconds"] = float("nan")
    elif case == "expanded_deadline":
        message["deadline_seconds"] = 121
    elif case == "source_escape":
        message["source_relative"] = "../source.bin"
    elif case == "old_CAS":
        path = root / "data/runtime-v2-app/blobs/domain/sha"
        path.parent.mkdir(parents=True)
        path.write_bytes(source.read_bytes())
        message["source_relative"] = path.relative_to(root).as_posix()
    elif case == "wrong_ingestion":
        profile["assigned_ingestion_id"] = str(uuid4())
    elif case == "wrong_lease_input":
        path = source.parent.parent.parent / str(uuid4()) / "input/source.bin"
        path.parent.mkdir(parents=True)
        path.write_bytes(source.read_bytes())
        message["source_relative"] = path.relative_to(root).as_posix()
    elif case == "size":
        profile["assigned_size_bytes"] += 1
    elif case == "hash":
        source.write_bytes(b"x" * source.stat().st_size)
    elif case == "existing_output":
        output.mkdir()
    elif case == "extra_request":
        message["request"]["source_path"] = "/arbitrary"
    if case == "duplicate_keys":
        raw = json.dumps(message)[:-1] + ',"sequence":1}'
        with pytest.raises(ValueError, match="Duplicate"):
            admit(raw.encode(), root=root, profile=profile, sequence=1, lease_root=None)
    else:
        with pytest.raises(ValueError):
            call(root, profile, message)


def test_same_lease_chunk_reuses_actual_parse_output_validator(handoff):
    root, _source, output, profile, message = handoff
    call(root, profile, message)
    output.mkdir()
    # Explicit parser result fixture; this is not an actual parser execution.
    blocks = [{"text": "EXPLICIT parsed block fixture"}]
    (output / "result.json").write_text(json.dumps({"blocks": blocks}))
    chunk_input = output.parent / "chunk-input.json"
    content = {"file_sha256": profile["assigned_source_sha256"], "blocks": blocks}
    chunk_input.write_text(json.dumps(content))
    relative = "runtime/s04-linux/chunk-worker-profile.json"
    path = root / relative
    path.write_bytes((ROOT / relative).read_bytes())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    profile["source_bindings"][relative] = sha
    message.update(sequence=2, source_relative=chunk_input.relative_to(root).as_posix(),
        output_relative=(output.parent / "chunk").relative_to(root).as_posix(),
        profile_relative=relative, profile_sha256=sha)
    message["request"] = {"operation": "chunk", "source_sha256": hashlib.sha256(chunk_input.read_bytes()).hexdigest(),
        "expected_profile_sha256": sha}
    value = call(root, profile, message, sequence=2, lease_root=output.parent)
    assert value["operation"] == "chunk"
    content["blocks"] = [{"text": "substituted"}]
    chunk_input.write_text(json.dumps(content))
    message["request"]["source_sha256"] = hashlib.sha256(chunk_input.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="actual parse output"):
        call(root, profile, message, sequence=2, lease_root=output.parent)
