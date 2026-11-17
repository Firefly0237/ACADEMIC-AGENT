"""Pure assembly contracts, without Docker, document conversion or certification."""

import importlib.util
from pathlib import Path

import pytest

from citequest.exports_v3.release_build import source_files, validate_certification


ROOT = Path(__file__).resolve().parents[2]


def test_release_cli_help_has_no_resource_execution(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("export_release_cli", ROOT / "scripts/s13/build_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("CLI help invoked subprocess"))
    monkeypatch.setattr(module.sys, "argv", ["build_release.py", "--help"])
    with pytest.raises(SystemExit) as result:
        module.main()
    assert result.value.code == 0
    assert "certify" in capsys.readouterr().out


def test_missing_visual_proof_never_becomes_certified():
    candidate = {"image_id": "sha256:" + "1" * 64, "source_sha256": "2" * 64,
        "toolchain_sha256": "3" * 64, "artifacts": {"layout/capacity.json": "4" * 64}}
    identity = {"font_profile": "test-only"}
    layout = {"coverage": {"status": "passed", "pages": 1}, "page_png_sha256": {"pages/page-1.png": "5" * 64}}
    certificate = {"schema_version": "citequest-capacity-render-certification/2", "status": "passed",
        "image_id": candidate["image_id"], "source_sha256": candidate["source_sha256"],
        "toolchain_sha256": candidate["toolchain_sha256"], "font_profile": identity["font_profile"],
        "capacity_artifact_sha256": "4" * 64, "coverage": layout["coverage"], "actual_model_requests": 0,
        "visual_review": {"status": "not_provided"}}
    with pytest.raises(ValueError, match="actual_all_page_certification"):
        validate_certification(candidate, identity, layout, certificate)
    # Even a supplied acceptance cannot certify an unobserved page/artifact.
    certificate["visual_review"] = {"status": "passed", "pages_reviewed": 1,
        "pages_observed_individually": [1], "png_sha256": {"pages/page-1.png": "6" * 64},
        "evidence": "in-memory negative test fixture"}
    with pytest.raises(ValueError, match="actual_all_page_certification"):
        validate_certification(candidate, identity, layout, certificate)


def test_release_source_manifest_contains_real_host_and_build_inputs():
    sources = source_files(ROOT)
    for path in ("src/citequest/exports_v3/docker_renderer.py", "src/citequest/exports_v3/worker.py",
        "scripts/s13/build_release.py", "scripts/s13/capacity_entry.py", "runtime/s13/Dockerfile",
        "runtime/s13/requirements.txt", "assets/export/sjtu-proposal.docx"):
        assert len(sources[path]) == 64
    assert "runtime/s13/release-profile.json" not in sources
