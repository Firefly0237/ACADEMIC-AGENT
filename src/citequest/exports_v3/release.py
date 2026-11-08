"""Pure trusted release verification shared by API and offline export worker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field

from citequest.agents_v3.contracts import Strict, digest
from citequest.research_v3.school_contracts import CSL_SHA, SHA, TEMPLATE_SHA

from .service import ExportProfile


class CertifiedRelease(Strict):
    schema_version: Literal["citequest-certified-linux-export-profile/1"]
    status: str = Field(min_length=1, max_length=120)
    id: str = Field(min_length=1, max_length=120)
    image_id: str = Field(pattern=r"^sha256:[a-f0-9]{64}$")
    toolchain_sha256: SHA
    font_profile: str = Field(min_length=1, max_length=120)
    deadline_seconds: Literal[90]
    lease_seconds: Literal[120]
    owner_active_limit: Literal[2]
    global_active_limit: Literal[100]
    capacity_certification_manifest: str = Field(min_length=1, max_length=500)
    capacity_certification_manifest_sha256: SHA
    toolchain_identity: str = Field(min_length=1, max_length=500)
    evidence: dict[str, str]
    not_verified: list[str]
    actual_model_requests: Literal[0]
    host_adapter_sha256: SHA
    host_worker_sha256: SHA

    def profile(self) -> ExportProfile:
        return ExportProfile(self.id, self.font_profile, self.toolchain_sha256)


def asset(workspace: Path, relative: str) -> Path:
    root = workspace.resolve(strict=True)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("released_export_asset_must_be_workspace_relative")
    candidate = root / path
    actual = candidate.resolve(strict=True)
    if not actual.is_relative_to(root) or candidate.is_symlink():
        raise ValueError("released_export_asset_outside_workspace")
    return actual


def load_release(path: Path, expected_sha256: str) -> CertifiedRelease:
    source = path.read_bytes()
    if digest(source) != expected_sha256:
        raise ValueError("trusted_export_release_profile_sha256_mismatch")
    release = CertifiedRelease.model_validate_json(source, strict=True)
    package = Path(__file__).resolve().parent
    if (
        digest((package / "docker_renderer.py").read_bytes()) != release.host_adapter_sha256
        or digest((package / "worker.py").read_bytes()) != release.host_worker_sha256
    ):
        raise ValueError("running_host_export_source_differs_from_certified_release")
    return release


def verify_release_assets(workspace: Path, release: CertifiedRelease) -> dict:
    """No Docker/renderer construction, credential read, network or database IO."""
    identity = json.loads(asset(workspace, release.toolchain_identity).read_bytes())
    proof_bytes = asset(workspace, release.capacity_certification_manifest).read_bytes()
    if len(proof_bytes) > 1024**2 or digest(proof_bytes) != release.capacity_certification_manifest_sha256:
        raise ValueError("capacity_certification_manifest_hash_mismatch")
    proof = json.loads(proof_bytes)
    pages = proof.get("coverage", {}).get("pages")
    if (
        digest(identity) != release.toolchain_sha256
        or identity.get("font_profile") != release.font_profile
        or identity.get("template_sha256") != TEMPLATE_SHA
        or identity.get("csl_sha256") != CSL_SHA
        or proof.get("status") != "passed"
        or proof.get("image_id") != release.image_id
        or proof.get("toolchain_sha256") != release.toolchain_sha256
        or proof.get("font_profile") != release.font_profile
        or proof.get("coverage", {}).get("status") != "passed"
        or proof.get("visual_review", {}).get("status") != "passed"
        or type(pages) is not int
        or pages < 1
        or proof["visual_review"].get("pages_reviewed") != pages
    ):
        raise ValueError("same_profile_certified_capacity_required")
    for relative, expected in (
        ("assets/export/sjtu-proposal.docx", TEMPLATE_SHA),
        ("assets/export/harvard.csl", CSL_SHA),
    ):
        if digest(asset(workspace, relative).read_bytes()) != expected:
            raise ValueError("official_export_asset_pin_mismatch")
    return identity
