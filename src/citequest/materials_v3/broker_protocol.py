"""Pure strict admission for the fixed developer host launcher."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from uuid import UUID

from .fixed_profiles import CHUNK_PROFILE, DEFAULT_PARSE_PROFILE, approved_operation_profile


def strict_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate broker JSON key")
        result[key] = value
    return result


def rooted(root: Path, relative: str) -> Path:
    if type(relative) is not str or "\\" in relative or ":" in relative:
        raise ValueError("POSIX relative path required")
    parts = PurePosixPath(relative)
    if parts.is_absolute() or ".." in parts.parts or not parts.parts:
        raise ValueError("Parent or absolute path rejected")
    value = root
    for part in parts.parts:
        value = value / part
        if value.is_symlink():
            raise ValueError("Symlink rejected")
    if not value.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escaped owned workspace")
    return value


def admit(raw: bytes, *, root: Path, profile: dict, sequence: int, lease_root: Path | None) -> dict:
    if len(raw) > 262144:
        raise ValueError("Bounded request required")
    item = json.loads(raw, object_pairs_hook=strict_object)
    expected = {
        "schema_version",
        "nonce",
        "sequence",
        "source_relative",
        "output_relative",
        "profile_relative",
        "profile_sha256",
        "deadline_seconds",
        "request",
    }
    if set(item) != expected or item["schema_version"] != "cq.parser-broker-request/1":
        raise ValueError("Exact broker envelope required")
    if item["nonce"] != profile["nonce"] or type(item["sequence"]) is not int or item["sequence"] != sequence:
        raise ValueError("Nonce or sequence mismatch")
    operation = "parse" if sequence == 1 else "chunk" if sequence == 2 else None
    relative = profile.get("parser_profile_relative", DEFAULT_PARSE_PROFILE) if sequence == 1 else CHUNK_PROFILE
    if operation is None or not approved_operation_profile(operation, relative) or item["profile_relative"] != relative:
        raise ValueError("Fixed operation/profile required")
    profile_path = rooted(root, relative)
    worker_raw = profile_path.read_bytes()
    worker_sha = hashlib.sha256(worker_raw).hexdigest()
    if item["profile_sha256"] != worker_sha or profile["source_bindings"][relative] != worker_sha:
        raise ValueError("Approved worker profile mismatch")
    worker = json.loads(worker_raw)
    seconds = item["deadline_seconds"]
    if type(seconds) not in {int, float} or not math.isfinite(seconds) or not 0 < seconds <= worker["deadline_seconds"]:
        raise ValueError("Remaining budget cannot expand")
    request = item["request"]
    allowed = {
        "operation",
        "source_sha256",
        "media_type",
        "expected_profile_sha256",
        "expected_tokenizer_profile_sha256",
    }
    if type(request) is not dict or set(request) - allowed or request.get("operation", "parse") != operation:
        raise ValueError("Fixed request fields required")
    if request.get("expected_profile_sha256") != worker_sha:
        raise ValueError("Request profile mismatch")
    source = rooted(root, item["source_relative"])
    output = rooted(root, item["output_relative"])
    artifact_root = root / "data/runtime-s04-linux/artifacts"
    if not output.is_relative_to(artifact_root) or output.exists():
        raise ValueError("New owned output required")
    parts = output.relative_to(artifact_root).parts
    if len(parts) != 3 or parts[-1] != operation or any(str(UUID(value)) != value for value in parts[:2]):
        raise ValueError("Single canonical ingestion/lease output required")
    if lease_root is not None and output.parent != lease_root:
        raise ValueError("Cross lease rejected")
    if not source.is_file() or source.stat().st_size > 64 * 1024**2:
        raise ValueError("Bounded regular source required")
    source_raw = source.read_bytes()
    if hashlib.sha256(source_raw).hexdigest() != request["source_sha256"]:
        raise ValueError("Source hash mismatch")
    if sequence == 1:
        if (
            not source.is_relative_to(root / "data/runtime-v2-app/blobs")
            or request["source_sha256"] != profile["assigned_source_sha256"]
            or request["media_type"] != profile["assigned_media_type"]
        ):
            raise ValueError("Exact assigned File source required")
    else:
        if source != output.parent / "chunk-input.json":
            raise ValueError("Same lease chunk input required")
        parsed = json.loads((output.parent / "parse/result.json").read_bytes())
        chunk_input = json.loads(source_raw)
        if chunk_input != {"file_sha256": profile["assigned_source_sha256"], "blocks": parsed["blocks"]}:
            raise ValueError("Chunk input must retain actual parse output")
    return {
        "message": item,
        "source": source,
        "output": output,
        "profile": worker,
        "profile_sha256": worker_sha,
        "request_sha256": hashlib.sha256(raw).hexdigest(),
        "operation": operation,
    }
