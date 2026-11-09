"""Standard-library release assembly from actual CI render evidence.

This module never renders, builds an image or creates a visual acceptance. The
separate, explicitly supplied certification must describe the exact candidate.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(value: object) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            value.update(block)
    return value.hexdigest()


def source_files(workspace: Path) -> dict[str, str]:
    """Files copied into the renderer, independently observed on host and image."""
    root = workspace.resolve(strict=True)
    paths = [*root.glob("src/citequest/**/*.py"), *root.glob("scripts/s13/*.py")]
    paths += [root / "runtime/s13" / name for name in (
        "Dockerfile", "Dockerfile.dockerignore", "requirements.txt", "pyproject.toml", "uv.lock",
        "font-packages.lock.json", "fontconfig.xml",
    )]
    paths += [root / "assets/export" / name for name in ("sjtu-proposal.docx", "harvard.csl")]
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("release_source_files_missing_or_symlinked")
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in sorted(paths)}


def relative_file(root: Path, relative: str) -> Path:
    base = root.resolve(strict=True)
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not relative:
        raise ValueError("release_artifact_relative_path_required")
    candidate = base / path
    actual = candidate.resolve(strict=True)
    if candidate.is_symlink() or not actual.is_relative_to(base) or not actual.is_file():
        raise ValueError("release_artifact_outside_candidate")
    return actual


def read_json(path: Path) -> dict:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024**2:
        raise ValueError("bounded_regular_release_json_required")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("release_json_object_required")
    return value


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def verify_candidate(workspace: Path, directory: Path) -> tuple[dict, dict, dict]:
    candidate = read_json(directory / "candidate.json")
    if (candidate.get("schema_version") != "citequest-export-release-candidate/1"
        or candidate.get("status") != "awaiting_certification"
        or type(candidate.get("actual_model_requests")) is not int or candidate["actual_model_requests"] != 0):
        raise ValueError("export_release_candidate_version_required")
    files = source_files(workspace)
    if candidate.get("source_files_sha256") != files or candidate.get("source_sha256") != sha256(files):
        raise ValueError("export_candidate_current_source_mismatch")
    image_id = candidate.get("image_id")
    if not isinstance(image_id, str) or re.fullmatch(r"sha256:[a-f0-9]{64}", image_id) is None:
        raise ValueError("actual_export_image_id_required")
    artifacts = candidate.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("actual_release_artifacts_required")
    required = {"renderer-image.tar", "toolchain-identity.json", "layout/capacity.json",
        "layout/artifacts/proposal.docx", "layout/artifacts/proposal.html", "layout/pdf/proposal.pdf"}
    if not required.issubset(artifacts):
        raise ValueError("actual_image_and_layout_artifacts_required")
    for relative, expected in artifacts.items():
        if file_sha256(relative_file(directory, relative)) != expected:
            raise ValueError("export_candidate_artifact_sha256_mismatch")
    identity = read_json(relative_file(directory, "toolchain-identity.json"))
    layout = read_json(relative_file(directory, "layout/capacity.json"))
    if (sha256(identity) != candidate.get("toolchain_sha256")
        or identity.get("source_files_sha256") != files
        or identity.get("source_sha256") != candidate["source_sha256"]
        or layout.get("image_id") != image_id
        or layout.get("toolchain_sha256") != candidate["toolchain_sha256"]
        or layout.get("font_profile") != identity.get("font_profile")):
        raise ValueError("export_candidate_image_toolchain_source_mismatch")
    coverage = layout.get("coverage", {})
    pages = coverage.get("pages")
    pngs = layout.get("page_png_sha256")
    if (coverage.get("status") != "passed" or type(pages) is not int or not 1 <= pages <= 100
        or not isinstance(pngs, dict) or len(pngs) != pages
        or coverage.get("docx_sha256") != artifacts.get("layout/artifacts/proposal.docx")
        or coverage.get("pdf_sha256") != artifacts.get("layout/pdf/proposal.pdf")
        or any(artifacts.get("layout/" + name) != value for name, value in pngs.items())):
        raise ValueError("actual_complete_capacity_artifacts_required")
    return candidate, identity, layout


def validate_certification(candidate: dict, identity: dict, layout: dict, certificate: dict) -> None:
    """Require externally supplied acceptance of these exact, generated pages."""
    pages = layout["coverage"]["pages"]
    visual = certificate.get("visual_review", {})
    if (certificate.get("schema_version") != "citequest-capacity-render-certification/2"
        or certificate.get("status") != "passed"
        or certificate.get("image_id") != candidate["image_id"]
        or certificate.get("source_sha256") != candidate["source_sha256"]
        or certificate.get("toolchain_sha256") != candidate["toolchain_sha256"]
        or certificate.get("font_profile") != identity["font_profile"]
        or certificate.get("capacity_artifact_sha256") != candidate["artifacts"]["layout/capacity.json"]
        or certificate.get("coverage") != layout["coverage"]
        or certificate.get("actual_model_requests") != 0
        or visual.get("status") != "passed"
        or type(visual.get("pages_reviewed")) is not int or visual["pages_reviewed"] != pages
        or visual.get("pages_observed_individually") != list(range(1, pages + 1))
        or visual.get("png_sha256") != layout["page_png_sha256"]
        or not isinstance(visual.get("evidence"), str) or not visual["evidence"].strip()):
        raise ValueError("same_candidate_actual_all_page_certification_required")


def certify_release(
    workspace: Path, candidate_dir: Path, certificate_file: Path, certificate_sha256: str, output: Path
) -> dict:
    candidate, identity, layout = verify_candidate(workspace, candidate_dir)
    if file_sha256(certificate_file) != certificate_sha256:
        raise ValueError("explicit_capacity_certification_sha256_mismatch")
    certificate = read_json(certificate_file)
    validate_certification(candidate, identity, layout, certificate)
    if output.exists():
        raise FileExistsError("fresh_release_output_required")
    output.mkdir(parents=True)
    shutil.copyfile(candidate_dir / "toolchain-identity.json", output / "toolchain-identity.json")
    shutil.copyfile(certificate_file, output / "capacity-certification.json")
    release = {
        "schema_version": "citequest-certified-linux-export-profile/1", "status": "certified_ci_release",
        "id": "linux-export-" + candidate["toolchain_sha256"][:16] + "/1",
        "image_id": candidate["image_id"], "toolchain_sha256": candidate["toolchain_sha256"],
        "font_profile": identity["font_profile"], "deadline_seconds": 90, "lease_seconds": 120,
        "owner_active_limit": 2, "global_active_limit": 100,
        "capacity_certification_manifest": "capacity-certification.json",
        "capacity_certification_manifest_sha256": file_sha256(output / "capacity-certification.json"),
        "toolchain_identity": "toolchain-identity.json",
        "evidence": {"source_sha256": candidate["source_sha256"],
            "capacity_artifact_sha256": candidate["artifacts"]["layout/capacity.json"],
            "renderer_archive_sha256": candidate["artifacts"]["renderer-image.tar"]},
        "not_verified": [], "actual_model_requests": 0,
        "host_adapter_sha256": candidate["source_files_sha256"]["src/citequest/exports_v3/docker_renderer.py"],
        "host_worker_sha256": candidate["source_files_sha256"]["src/citequest/exports_v3/worker.py"],
    }
    write_json(output / "release-profile.json", release)
    release_sha = file_sha256(output / "release-profile.json")
    for role in ("api", "worker"):
        value = read_json(workspace / f"runtime/s13/export-{role}.example.json")
        value.update(workspace="/opt/citequest", release_assets_root="/run/export-release",
            release_profile_file="/run/export-release/release-profile.json",
            release_profile_sha256=release_sha)
        write_json(output / f"export-{role}.json", value)
    asset_dir = output / "assets/export"
    asset_dir.mkdir(parents=True)
    for name in ("sjtu-proposal.docx", "harvard.csl"):
        shutil.copyfile(workspace / "assets/export" / name, asset_dir / name)
    write_json(output / "release-receipt.json", {
        "schema_version": "citequest-export-release-receipt/1", "status": "certified_release_assembled",
        "release_profile_sha256": release_sha, "image_id": candidate["image_id"],
        "source_sha256": candidate["source_sha256"], "actual_model_requests": 0,
        "certification_sha256": certificate_sha256,
    })
    return {"status": "certified_release_assembled", "release_profile_sha256": release_sha,
        "image_id": candidate["image_id"]}
