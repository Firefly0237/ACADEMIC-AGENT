"""Explicit local-release fixtures; no Docker, API, secret or model access."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("cq_current_release_ops", Path(__file__).resolve().parents[2] / "tools/cq.py")
assert SPEC is not None and SPEC.loader is not None
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)
CID = "a" * 64
IMAGE = "sha256:" + "b" * 64


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CurrentReleaseOperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.base = self.root / "runtime/base.yml"
        self.override = self.root / "candidate/compose.override.yml"
        self.config = self.root / "candidate/config/agent-api.json"
        self.manifest = self.root / "candidate/manifest.json"
        for path in (self.base, self.override, self.config):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.base.write_text("services: {web: {image: explicit-baseline-fixture}}\n", encoding="utf-8")
        self.override.write_text("services: {web: {image: explicit-release-fixture}}\n", encoding="utf-8")
        self.config.write_text('{"explicit_nonsecret_fixture":true}\n', encoding="utf-8")
        self.manifest.write_text(json.dumps({"image": {"image_id": IMAGE}, "files": {
            "compose.override.yml": sha(self.override), "config/agent-api.json": sha(self.config)}}), encoding="utf-8")
        self.stack = {"id": "web", "project": "explicit-current-release-fixture", "compose": "runtime/base.yml",
            "compose_overrides": ["candidate/compose.override.yml"], "services": ["web"], "lifecycle": True,
            "active_release": {"image_id": IMAGE, "config_manifest": "candidate/manifest.json",
                "config_manifest_sha256": sha(self.manifest), "compose_sha256": {
                    "runtime/base.yml": sha(self.base), "candidate/compose.override.yml": sha(self.override)}}}
        self.data = {"stacks": [self.stack]}

    def record(self, *, paths: list[Path] | None = None, image: str = IMAGE,
               service: str = "web", oneoff: str = "False", directory: Path | None = None) -> str:
        return json.dumps([CID, str(directory or self.base.parent),
            ",".join(str(path) for path in (paths if paths is not None else [self.base, self.override])), service, oneoff, image])

    def responses(self, run: object, **kwargs: object) -> None:
        run.side_effect = [SimpleNamespace(stdout=CID), SimpleNamespace(stdout=self.record(**kwargs))]

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    @patch.object(ops.subprocess, "run")
    def test_current_release_selects_exact_id_and_never_constructs_recreate(self, run: object, _prefix: object) -> None:
        self.responses(run)
        identifiers = ops.check_ownership(self.data, "web", self.root)
        self.assertEqual(identifiers, [CID])
        command = ops.stack_command(self.data, "web", "start", self.root, identifiers=identifiers)
        self.assertEqual(command, ["docker", "start", CID])
        self.assertFalse(set(command) & {"compose", "up", "build", "pull", "create"})
        self.assertEqual(run.call_count, 2)  # Both mocked calls are ps/inspect only.

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    @patch.object(ops.subprocess, "run")
    def test_missing_override_is_refused(self, run: object, _prefix: object) -> None:
        self.responses(run, paths=[self.base])
        with self.assertRaises(ops.OperationError):
            ops.check_ownership(self.data, "web", self.root)

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    @patch.object(ops.subprocess, "run")
    def test_changed_override_order_is_refused(self, run: object, _prefix: object) -> None:
        self.responses(run, paths=[self.override, self.base])
        with self.assertRaises(ops.OperationError):
            ops.check_ownership(self.data, "web", self.root)

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    @patch.object(ops.subprocess, "run")
    def test_wrong_image_is_refused(self, run: object, _prefix: object) -> None:
        self.responses(run, image="sha256:" + "c" * 64)
        with self.assertRaisesRegex(ops.OperationError, "image differs"):
            ops.check_ownership(self.data, "web", self.root)

    @patch.object(ops.subprocess, "run")
    def test_wrong_manifest_hash_is_refused_before_Docker(self, run: object) -> None:
        self.stack["active_release"]["config_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(ops.OperationError, "missing or changed"):
            ops.check_ownership(self.data, "web", self.root)
        run.assert_not_called()

    @patch.object(ops.subprocess, "run")
    def test_changed_config_bytes_are_refused_before_Docker(self, run: object) -> None:
        self.config.write_text('{"explicit_nonsecret_fixture":false}', encoding="utf-8")
        with self.assertRaisesRegex(ops.OperationError, "missing or changed"):
            ops.check_ownership(self.data, "web", self.root)
        run.assert_not_called()

    @patch.object(ops.subprocess, "run")
    def test_changed_compose_bytes_are_refused_before_Docker(self, run: object) -> None:
        self.override.write_text("services: {}", encoding="utf-8")
        with self.assertRaisesRegex(ops.OperationError, "missing or changed"):
            ops.check_ownership(self.data, "web", self.root)
        run.assert_not_called()

    @patch.object(ops.subprocess, "run")
    def test_missing_local_release_has_no_old_base_fallback(self, run: object) -> None:
        self.manifest.unlink()
        with self.assertRaisesRegex(ops.OperationError, "missing or changed"):
            ops.check_ownership(self.data, "web", self.root)
        run.assert_not_called()

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    @patch.object(ops.subprocess, "run")
    def test_original_service_oneoff_and_directory_constraints_remain(self, run: object, _prefix: object) -> None:
        for values in ({"service": "other"}, {"oneoff": "True"}, {"directory": self.root}):
            with self.subTest(values=values):
                self.responses(run, **values)
                with self.assertRaises(ops.OperationError):
                    ops.check_ownership(self.data, "web", self.root)


if __name__ == "__main__":
    unittest.main()
