"""Explicit native source-broker profiles; never fall back to a soft lock."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from filelock import UnixFileLock, WindowsFileLock

WINDOWS_PROFILE = "windows-source-gate/1"
LINUX_PROFILE = "linux-source-broker/1"
LEGACY_POLICY = {"id": "cq.shared-source-network/1", "backend": "WindowsFileLock4.2.9+PG-session-advisory",
    "per_hop_cooldown_seconds": {"arxiv": 3.1, "crossref": 1.0}, "unknown_ack": "retain_unknown_fail_closed"}
LINUX_POLICY = {**LEGACY_POLICY, "id": "cq.shared-source-network/2-linux-broker",
    "backend": "UnixFileLock4.2.9+PG-session-advisory", "deployment": "one_source_broker_shared_persistent_POSIX_lock_root"}


def policy(profile: str) -> dict:
    if profile == WINDOWS_PROFILE:
        return LEGACY_POLICY
    if profile == LINUX_PROFILE:
        return LINUX_POLICY
    raise ValueError("explicit_source_gate_profile_required")


def policy_sha(profile: str) -> str:
    return hashlib.sha256(json.dumps(policy(profile), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def native_lock(path: Path, profile: str):
    policy(profile)
    expected = "nt" if profile == WINDOWS_PROFILE else "posix"
    if os.name != expected:
        raise RuntimeError("source_gate_native_OS_profile_mismatch")
    implementation = WindowsFileLock if expected == "nt" else UnixFileLock
    return implementation(str(path), timeout=0, thread_local=False, is_singleton=False)
