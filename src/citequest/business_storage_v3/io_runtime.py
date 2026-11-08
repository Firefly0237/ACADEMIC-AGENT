"""Trusted child IO using the existing pinned official SDK; no database/model IO."""

from __future__ import annotations

import json
import math
import os
import resource
import select
from pathlib import Path

from citequest.platform_v3.blob_config import create_blob_store
from citequest.platform_v3.blob_store import BlobIdentity

from .contracts import OperationDeadline, parse_reference, reference_value
from .io_contracts import IOJob


def _limit(kind: int, requested: int) -> None:
    _, hard = resource.getrlimit(kind)
    value = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    resource.setrlimit(kind, (value, value))


def run(job_path: str, parent: int, observations: dict) -> int:
    path = Path(job_path)
    if path.name != "job.json" or path.is_symlink() or not path.is_absolute():
        raise ValueError("fixed_private_io_job_path_required")
    directory = path.parent.resolve(strict=True)
    if directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077:
        raise ValueError("private_owned_io_directory_required")
    if path.stat().st_size > 65536:
        raise ValueError("io_job_bytes_exceeded")
    job = IOJob.model_validate_json(path.read_bytes(), strict=True)
    if job.expected_parent_pid != parent or os.getppid() != parent:
        raise ValueError("io_parent_changed_before_sdk")
    deadline = OperationDeadline(job.expires_at)
    remaining = deadline.remaining()
    if remaining > job.profile.maximum_active_seconds:
        raise ValueError("io_deadline_exceeds_published_maximum")
    _limit(resource.RLIMIT_CORE, 0)
    _limit(resource.RLIMIT_AS, job.profile.address_space_bytes)
    _limit(resource.RLIMIT_CPU, max(1, math.ceil(remaining)))
    _limit(resource.RLIMIT_NOFILE, job.profile.file_descriptor_limit)
    _limit(resource.RLIMIT_FSIZE, job.profile.maximum_object_bytes + 65536)
    if job.storage.spool_dir.resolve() != directory / "sdk-spool":
        raise ValueError("io_sdk_spool_must_stay_inside_job")
    # SDK import itself can probe local sockets (urllib3 HAS_IPV6). Defer the
    # whole SDK loading path, including Client/ADC, until the parent's GO.
    result = {"version": "citequest-blob-io-result/1", "operation": job.operation,
              "parent_pid": parent, "pid": os.getpid(), "supervision": "linux-pdeathsig-kill",
              "profile": job.profile.version, "reference": None,
              "limits": {"address_space": list(resource.getrlimit(resource.RLIMIT_AS)),
                         "cpu": list(resource.getrlimit(resource.RLIMIT_CPU)),
                         "open_files": list(resource.getrlimit(resource.RLIMIT_NOFILE)),
                         "file_size": list(resource.getrlimit(resource.RLIMIT_FSIZE))}}
    ready = {**result, "version": "citequest-blob-io-ready/1", "observations": dict(observations)}
    ready_raw = json.dumps(ready, separators=(",", ":"), allow_nan=False).encode()
    if len(ready_raw) > job.profile.maximum_result_bytes:
        raise ValueError("io_ready_bytes_exceeded")
    with (directory / "ready.pending").open("xb") as output:
        output.write(ready_raw)
        output.flush()
        os.fsync(output.fileno())
    os.replace(directory / "ready.pending", directory / "ready.json")
    readable, _, _ = select.select([0], [], [], deadline.remaining())
    if not readable or os.read(0, 3) != b"GO\n":
        raise ValueError("fixed_parent_GO_required_before_any_network")
    deadline.remaining()
    observations["go"] = True
    if job.operation != "probe":
        store = create_blob_store(job.storage)
        deadline.remaining()
        if job.operation == "put":
            identity = BlobIdentity(**job.identity)
            source = directory / "source.bin"
            if source.is_symlink() or not source.is_file() or source.stat().st_size != identity.size_bytes:
                raise ValueError("io_original_input_path_or_size_invalid")
            with source.open("rb") as stream:
                result["reference"] = reference_value(store.put(identity, stream))
        elif job.operation == "reconcile":
            if job.storage.backend == "filesystem":
                raise ValueError("io_supervised_reconcile_requires_gcs_backend")
            reference = store.reconcile(BlobIdentity(**job.identity))
            result["reference"] = reference_value(reference) if reference else None
        else:
            reference = parse_reference(job.reference)
            with store.open_verified(reference) as verified, (directory / "verified.bin").open("xb") as output:
                size = 0
                while True:
                    deadline.remaining()
                    part = verified.read(65536)
                    if not part:
                        break
                    size += len(part)
                    if size > reference.identity.size_bytes:
                        raise ValueError("io_verified_output_exceeds_reference")
                    output.write(part)
            if size != reference.identity.size_bytes:
                raise ValueError("io_verified_output_size_mismatch")
            result["reference"] = reference_value(reference)
    deadline.remaining()
    result["observations"] = dict(observations)
    raw = json.dumps(result, separators=(",", ":"), allow_nan=False).encode()
    if len(raw) > job.profile.maximum_result_bytes:
        raise ValueError("io_result_bytes_exceeded")
    with (directory / "result.json").open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return 0
