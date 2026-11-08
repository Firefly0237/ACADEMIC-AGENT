"""Fixed parser profile names shared by image entry and trusted host admission.

This registry is immutable application code, never uploaded metadata, a Skill,
model argument or an HTTP setting. Each execution still pins exact file bytes.
"""

from __future__ import annotations

DEFAULT_PARSE_PROFILE = "runtime/s04-linux/worker-profile.json"
XYZ_PARSE_PROFILE = "runtime/s04-linux/worker-profile-xyz-v2.json"
CPU_ABI_PARSE_PROFILE = "runtime/s04-linux/worker-profile-cpu-v3.json"
CHUNK_PROFILE = "runtime/s04-linux/chunk-worker-profile.json"
IMAGE_PROFILE_RELATIVES = frozenset({DEFAULT_PARSE_PROFILE, XYZ_PARSE_PROFILE, CPU_ABI_PARSE_PROFILE, CHUNK_PROFILE})


def approved_operation_profile(operation: str, relative: str) -> bool:
    if operation == "parse":
        return relative in {DEFAULT_PARSE_PROFILE, XYZ_PARSE_PROFILE, CPU_ABI_PARSE_PROFILE}
    return operation == "chunk" and relative == CHUNK_PROFILE
