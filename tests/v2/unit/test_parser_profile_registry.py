"""Execution selection, host admission and fixed image entry share one registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest

from citequest.materials_v3.broker_protocol import admit
from citequest.materials_v3.fixed_profiles import (
    CPU_ABI_PARSE_PROFILE,
    DEFAULT_PARSE_PROFILE,
    IMAGE_PROFILE_RELATIVES,
    XYZ_PARSE_PROFILE,
    approved_operation_profile,
)


def control(root: Path, selected: str, declared: str | None) -> tuple[dict, dict]:
    source = root / "data/runtime-v2-app/blobs/public/control"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"explicit test input")
    target = root / selected
    target.parent.mkdir(parents=True)
    target.write_text(json.dumps({"deadline_seconds": 120}), encoding="utf8")
    profile_sha = hashlib.sha256(target.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    profile = {
        "nonce": "explicit",
        "source_bindings": {selected: profile_sha},
        "assigned_source_sha256": source_sha,
        "assigned_media_type": "text/plain",
    }
    if declared is not None:
        profile["parser_profile_relative"] = declared
    request = {
        "schema_version": "cq.parser-broker-request/1",
        "nonce": "explicit",
        "sequence": 1,
        "source_relative": source.relative_to(root).as_posix(),
        "output_relative": f"data/runtime-s04-linux/artifacts/{uuid4()}/{uuid4()}/parse",
        "profile_relative": selected,
        "profile_sha256": profile_sha,
        "deadline_seconds": 120,
        "request": {"source_sha256": source_sha, "media_type": "text/plain", "expected_profile_sha256": profile_sha},
    }
    return profile, request


@pytest.mark.parametrize(
    "selected,declared",
    [
        (DEFAULT_PARSE_PROFILE, None),
        (DEFAULT_PARSE_PROFILE, DEFAULT_PARSE_PROFILE),
        (XYZ_PARSE_PROFILE, XYZ_PARSE_PROFILE),
        (CPU_ABI_PARSE_PROFILE, CPU_ABI_PARSE_PROFILE),
    ],
)
def test_exact_default_or_explicit_xyz_selection_reaches_admission(
    tmp_path: Path, selected: str, declared: str | None
) -> None:
    profile, request = control(tmp_path, selected, declared)
    result = admit(json.dumps(request).encode(), root=tmp_path, profile=profile, sequence=1, lease_root=None)
    assert result["operation"] == "parse" and result["profile_sha256"] == request["profile_sha256"]
    assert selected in IMAGE_PROFILE_RELATIVES and approved_operation_profile("parse", selected)


@pytest.mark.parametrize(
    "selected,declared",
    [
        (XYZ_PARSE_PROFILE, None),
        (CPU_ABI_PARSE_PROFILE, None),
        (DEFAULT_PARSE_PROFILE, XYZ_PARSE_PROFILE),
        ("runtime/s04-linux/unapproved.json", "runtime/s04-linux/unapproved.json"),
    ],
)
def test_no_implicit_upgrade_wrong_selection_or_unapproved_profile(
    tmp_path: Path, selected: str, declared: str | None
) -> None:
    profile, request = control(tmp_path, selected, declared)
    with pytest.raises(ValueError, match="Fixed operation/profile"):
        admit(json.dumps(request).encode(), root=tmp_path, profile=profile, sequence=1, lease_root=None)


def test_shared_registry_is_fixed_and_cannot_expand_unknown_operation() -> None:
    assert (
        frozenset(
            {
                DEFAULT_PARSE_PROFILE,
                XYZ_PARSE_PROFILE,
                CPU_ABI_PARSE_PROFILE,
                "runtime/s04-linux/chunk-worker-profile.json",
            }
        )
        == IMAGE_PROFILE_RELATIVES
    )
    assert not approved_operation_profile("unknown", XYZ_PARSE_PROFILE)
    assert not approved_operation_profile("chunk", XYZ_PARSE_PROFILE)
