"""Pinned Docker Engine adapter for development/Compute Engine queue workers.

The renderer receives no Docker socket, database secret, provider key or network.
This is not an implementation or verification of the Cloud Run Jobs API.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from uuid import uuid4

from citequest.agents_v3.contracts import digest
from citequest.research_v3.school_contracts import CSL_SHA, TEMPLATE_SHA, SchoolDocument

from .contracts import Format
from .service import ExportProfile
from .worker import RenderBundle, RenderFailure


class DockerRenderer:
    def __init__(
        self,
        *,
        docker: Path,
        docker_config: Path,
        image_id: str,
        template: Path,
        csl: Path,
        profile: ExportProfile,
        toolchain_identity: dict,
        certified_capacity_manifest: Path,
        certified_capacity_manifest_sha256: str,
    ) -> None:
        if re.fullmatch(r"sha256:[a-f0-9]{64}", image_id) is None:
            raise ValueError("immutable_export_image_id_required")
        if (
            digest(toolchain_identity) != profile.toolchain_sha256
            or toolchain_identity.get("font_profile") != profile.font_profile
        ):
            raise ValueError("actual_toolchain_identity_must_match_queued_profile")
        if re.fullmatch(r"[a-f0-9]{64}", certified_capacity_manifest_sha256) is None:
            raise ValueError("separate_certified_capacity_fixture_manifest_required")
        if certified_capacity_manifest.is_symlink() or certified_capacity_manifest.stat().st_size > 1024**2:
            raise ValueError("bounded_capacity_certification_manifest_required")
        proof_bytes = certified_capacity_manifest.read_bytes()
        if hashlib.sha256(proof_bytes).hexdigest() != certified_capacity_manifest_sha256:
            raise ValueError("capacity_certification_manifest_hash_mismatch")
        proof = json.loads(proof_bytes)
        if (
            proof.get("status") != "passed"
            or proof.get("image_id") != image_id
            or proof.get("toolchain_sha256") != profile.toolchain_sha256
            or proof.get("font_profile") != profile.font_profile
            or proof.get("coverage", {}).get("status") != "passed"
            or proof.get("visual_review", {}).get("status") != "passed"
            or proof.get("visual_review", {}).get("pages_reviewed") != proof.get("coverage", {}).get("pages")
            or not isinstance(proof.get("coverage", {}).get("pages"), int)
            or proof["coverage"]["pages"] < 1
        ):
            raise ValueError("same_image_toolchain_all_page_capacity_certification_required")
        for path, expected in ((template, TEMPLATE_SHA), (csl, CSL_SHA)):
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("official_export_source_pin_mismatch")
        self.profile = profile
        self.docker, self.config, self.image_id = docker.resolve(), docker_config.resolve(), image_id
        self.template, self.csl = template.resolve(), csl.resolve()
        self.identity, self.certification = toolchain_identity, certified_capacity_manifest_sha256
        self.environment = {
            name: value
            for name, value in os.environ.items()
            if name.upper()
            in {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA"}
        }

    def command(self, *args: str) -> list[str]:
        return [str(self.docker), "--config", str(self.config), *args]

    def render(
        self,
        document: SchoolDocument,
        references: list[dict],
        formats: list[Format],
        output: Path,
        *,
        timeout_seconds: float,
    ) -> RenderBundle:
        if not 0 < timeout_seconds <= 90:
            raise ValueError("bounded_export_job_timeout_required")
        deadline = time.monotonic() + timeout_seconds

        def remaining() -> float:
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise TimeoutError("export_renderer_deadline_exceeded")
            return seconds

        inspected = subprocess.run(
            self.command("image", "inspect", self.image_id, "--format", "{{.Id}}"),
            capture_output=True,
            text=True,
            check=True,
            timeout=min(10, remaining()),
            env=self.environment,
        )
        if inspected.stdout.strip() != self.image_id:
            raise ValueError("actual_docker_image_identity_mismatch")
        inputs, artifacts = output / "job-input", output / "job-output"
        inputs.mkdir(mode=0o755)
        artifacts.mkdir(mode=0o777)
        # Host spool's private parent is mode0700; only these job mounts are
        # visible to the nonroot renderer. No other owner's directory is mounted.
        if os.name != "nt":
            inputs.chmod(0o755)
            artifacts.chmod(0o777)
        job = {
            "document": document.model_dump(mode="json"),
            "references": references,
            "formats": formats,
            "document_sha256": digest(document.model_dump(mode="json")),
            "toolchain_sha256": self.profile.toolchain_sha256,
        }
        path = inputs / "job.json"
        path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf8")
        if os.name != "nt":
            path.chmod(0o444)
        name = "cq-export-" + str(uuid4())
        args = [
            "run",
            "--rm",
            "--name",
            name,
            "--network",
            "none",
            "--read-only",
            "--user",
            "10001:10001",
            "--cpus",
            "2",
            "--memory",
            "2g",
            "--pids-limit",
            "128",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=256m",
        ]
        for source, target, readonly in (
            (inputs, "/inputs", True),
            (artifacts, "/outputs", False),
            (self.template, "/assets/sjtu.docx", True),
            (self.csl, "/assets/harvard.csl", True),
        ):
            if "," in str(source):
                raise ValueError("trusted_export_mount_path_contains_comma")
            args += [
                "--mount",
                "type=bind,source=" + str(source) + ",target=" + target + (",readonly" if readonly else ""),
            ]
        args += ["--entrypoint", "python", self.image_id, "scripts/s13/bundle_entry.py"]
        try:
            result = subprocess.run(
                self.command(*args),
                capture_output=True,
                timeout=remaining(),
                check=False,
                env=self.environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired as error:
            subprocess.run(
                self.command("kill", name), capture_output=True, timeout=10, check=False, env=self.environment
            )
            raise RenderFailure(
                "export_renderer_deadline_exceeded", (error.stdout or b"") + b"\n" + (error.stderr or b"")
            ) from None
        if result.returncode != 0:
            raise RenderFailure("export_renderer_process_failed", result.stdout + b"\n" + result.stderr)
        receipt_path = artifacts / "bundle-receipt.json"
        if not receipt_path.is_file() or receipt_path.stat().st_size > 1024**2:
            raise RenderFailure("export_renderer_receipt_missing", result.stdout + b"\n" + result.stderr)
        receipt = json.loads(receipt_path.read_text(encoding="utf8"))
        receipt["capacity_certification_manifest_sha256"] = self.certification
        receipt["renderer_image_id"] = self.image_id
        files = {
            kind: artifacts / ("pdf/proposal.pdf" if kind == "pdf" else "artifacts/proposal." + kind)
            for kind in formats
        }
        return RenderBundle(files, receipt)
