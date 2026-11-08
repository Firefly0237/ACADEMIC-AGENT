"""Explicit v2 trusted-parent handoff; immutable parser image stays separate.

The old CAS broker validator and release files remain unchanged. A private source
must belong to this exact ingestion/lease, rather than posing as a CAS object.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from uuid import UUID

from .broker_protocol import admit as admit_legacy
from .broker_protocol import rooted, strict_object
from .fixed_profiles import approved_operation_profile

TRANSPORT = "private-execution-spool/1"


def admit(raw: bytes, *, root: Path, profile: dict, sequence: int, lease_root: Path | None) -> dict:
    if (profile.get("schema_version") not in {"cq.linux-material-parent-execution/2", "cq.linux-material-parent-execution/3"}
            or profile.get("source_transport") != TRANSPORT
            or profile.get("product_api_docker_access") is not False):
        raise ValueError("explicit_private_parent_profile_required")
    if profile["schema_version"] == "cq.linux-material-parent-execution/3":
        assigned = profile.get("assigned_execution_id")
        if not isinstance(assigned, str) or str(UUID(assigned)) != assigned:
            raise ValueError("server_assigned_execution_identity_required")
    if sequence == 2:
        if profile["schema_version"] == "cq.linux-material-parent-execution/3":
            expected_lease = rooted(root, "data/runtime-s04-linux/artifacts/" + profile["assigned_ingestion_id"]
                                    + "/" + profile["assigned_execution_id"])
            if lease_root != expected_lease:
                raise ValueError("exact_server_assigned_chunk_execution_required")
        # The existing strict chunk validator already requires exactly the same
        # lease's actual parsed blocks and original File hash, with no CAS use.
        return admit_legacy(raw, root=root, profile=profile, sequence=sequence, lease_root=lease_root)
    if sequence != 1 or lease_root is not None or len(raw) > 262144:
        raise ValueError("first_single_parse_request_required")
    item = json.loads(raw, object_pairs_hook=strict_object)
    if (set(item) != {"schema_version", "nonce", "sequence", "source_relative", "output_relative",
                      "profile_relative", "profile_sha256", "deadline_seconds", "request"}
            or item["schema_version"] != "cq.parser-broker-request/1" or item["nonce"] != profile["nonce"]
            or type(item["sequence"]) is not int or item["sequence"] != 1):
        raise ValueError("exact_private_handoff_envelope_required")
    relative = profile["parser_profile_relative"]
    if not approved_operation_profile("parse", relative) or item["profile_relative"] != relative:
        raise ValueError("fixed_parser_policy_required")
    worker_raw = rooted(root, relative).read_bytes()
    worker_sha = hashlib.sha256(worker_raw).hexdigest()
    if (item["profile_sha256"] != worker_sha or profile["source_bindings"][relative] != worker_sha
            or profile["parser_profile_sha256"] != worker_sha):
        raise ValueError("parser_image_policy_binding_changed")
    worker = json.loads(worker_raw)
    seconds = item["deadline_seconds"]
    if type(seconds) not in {int, float} or not math.isfinite(seconds) or not 0 < seconds <= worker["deadline_seconds"]:
        raise ValueError("remaining_deadline_cannot_expand")
    request = item["request"]
    if (type(request) is not dict or set(request) != {"source_sha256", "media_type", "expected_profile_sha256"}
            or request["expected_profile_sha256"] != worker_sha
            or request["source_sha256"] != profile["assigned_source_sha256"]
            or request["media_type"] != profile["assigned_media_type"]):
        raise ValueError("exact_assigned_source_and_policy_required")
    output = rooted(root, item["output_relative"])
    artifact_root = root / "data/runtime-s04-linux/artifacts"
    if not output.is_relative_to(artifact_root) or output.exists():
        raise ValueError("new_owned_output_required")
    parts = output.relative_to(artifact_root).parts
    if (len(parts) != 3 or parts[-1] != "parse" or any(str(UUID(value)) != value for value in parts[:2])
            or parts[0] != profile["assigned_ingestion_id"]):
        raise ValueError("exact_assigned_ingestion_and_lease_required")
    if profile["schema_version"] == "cq.linux-material-parent-execution/3" and parts[1] != profile["assigned_execution_id"]:
        raise ValueError("exact_server_assigned_execution_required")
    source = rooted(root, item["source_relative"])
    if source != output.parent / "input/source.bin":
        raise ValueError("exact_private_execution_input_required")
    size = profile["assigned_size_bytes"]
    if (type(size) is not int or not 0 < size <= 64 * 1024**2
            or not source.is_file() or source.stat().st_size != size):
        raise ValueError("bounded_private_source_size_required")
    sha = hashlib.sha256()
    with source.open("rb") as stream:
        while part := stream.read(65536):
            sha.update(part)
    if sha.hexdigest() != profile["assigned_source_sha256"]:
        raise ValueError("private_source_hash_changed")
    return {"message": item, "source": source, "output": output, "profile": worker,
            "profile_sha256": worker_sha, "request_sha256": hashlib.sha256(raw).hexdigest(), "operation": "parse"}
