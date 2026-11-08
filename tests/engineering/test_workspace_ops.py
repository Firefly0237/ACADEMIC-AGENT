"""Engineering controls run without services, third-party dependencies or model calls."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("cq_ops", Path(__file__).resolve().parents[2] / "tools/cq.py")
assert SPEC is not None and SPEC.loader is not None
ops = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ops)
CONTAINER_ID = "a" * 64


def ownership_record(directory: Path, *, service: str = "web", identifier: str = CONTAINER_ID) -> str:
    return json.dumps([identifier, str(directory), str(ops.ROOT / "deploy/core/compose.yaml"), service, "False"])


def registry_fixture() -> dict:
    """Explicit test registry, independent of a user's local stack inventory."""
    return {"stacks": [{"id": "fixture-web", "project": "citequest-operations-fixture",
        "compose": "deploy/core/compose.yaml", "services": ["web"], "lifecycle": True}],
        "docker_config": "unused-explicit-fixture"}


class WorkspaceOperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_relative_path_cannot_escape(self) -> None:
        for value in ("../outside", ".", "", str(self.root.parent.resolve())):
            with self.subTest(value=value), self.assertRaises(ops.OperationError):
                ops.within_root(self.root, value)

    def test_symlink_cannot_redirect_cleanup(self) -> None:
        (self.root / "other").mkdir()
        try:
            (self.root / ".pytest_cache").symlink_to(self.root / "other", target_is_directory=True)
        except OSError as error:
            self.skipTest(f"OS does not permit symlink creation: {error.__class__.__name__}")
        with self.assertRaises(ops.OperationError):
            ops.clean_cache(self.root, apply=True)
        self.assertTrue((self.root / "other").exists())

    def test_preview_preserves_cache_and_unrelated_files(self) -> None:
        (self.root / ".coverage").write_text("cache", encoding="utf-8")
        (self.root / "data").mkdir()
        result = ops.clean_cache(self.root)
        self.assertFalse(result["applied"])
        self.assertEqual(result["paths"], [".coverage"])
        self.assertTrue((self.root / ".coverage").exists())

    def test_apply_only_removes_allowlisted_caches(self) -> None:
        for name in (".pytest_cache", "data", ".tmp", "runtime", "reports"):
            (self.root / name).mkdir()
            (self.root / name / "preserve").write_text("x", encoding="utf-8")
        result = ops.clean_cache(self.root, apply=True)
        self.assertEqual(result["paths"], [".pytest_cache"])
        self.assertFalse((self.root / ".pytest_cache").exists())
        self.assertTrue(
            all((self.root / name / "preserve").exists() for name in ("data", ".tmp", "runtime", "reports"))
        )

    @patch.object(ops.shutil, "rmtree", side_effect=PermissionError("denied"))
    def test_cache_permission_failure_is_recorded(self, _remove: object) -> None:
        (self.root / ".pytest_cache").mkdir()
        result = ops.clean_cache(self.root, apply=True)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["errors"], [{"path": ".pytest_cache", "error": "PermissionError"}])
        self.assertTrue((self.root / ".pytest_cache").exists())

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    def test_status_filters_exact_project(self, _prefix: object) -> None:
        data = registry_fixture()
        command = ops.stack_command(data, "fixture-web", "status")
        self.assertIn("label=com.docker.compose.project=citequest-operations-fixture", command)
        self.assertNotIn("inspect", command)
        self.assertIn(f"label=com.docker.compose.project.working_dir={ops.ROOT / 'deploy/core'}", command)

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    @patch.object(ops.subprocess, "run")
    def test_project_name_collision_cannot_stop_another_checkout(self, run: object, _prefix: object) -> None:
        run.side_effect = [SimpleNamespace(stdout=CONTAINER_ID), SimpleNamespace(stdout=ownership_record(self.root))]
        with self.assertRaises(ops.OperationError):
            ops.check_ownership(registry_fixture(), "fixture-web")

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    @patch.object(ops.subprocess, "run")
    def test_matching_project_directory_is_accepted(self, run: object, _prefix: object) -> None:
        run.side_effect = [
            SimpleNamespace(stdout=CONTAINER_ID),
            SimpleNamespace(stdout=ownership_record(ops.ROOT / "deploy/core")),
        ]
        self.assertEqual(ops.check_ownership(registry_fixture(), "fixture-web"), [CONTAINER_ID])

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    def test_start_only_existing_normal_services(self, _prefix: object) -> None:
        command = ops.stack_command(registry_fixture(), "fixture-web", "start", identifiers=[CONTAINER_ID])
        self.assertEqual(command[-2:], ["start", CONTAINER_ID])
        self.assertFalse(set(command) & {"compose", "up", "build", "pull", "restore", "bootstrap", "migrator"})

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    def test_declared_worker_stop_grace_does_not_change_default_services(self, _prefix: object) -> None:
        data = registry_fixture()
        default = ops.stack_command(data, "fixture-web", "stop", identifiers=[CONTAINER_ID])
        self.assertEqual(default, ["docker", "stop", CONTAINER_ID])
        stack = next(row for row in data["stacks"] if row["id"] == "fixture-web")
        stack["stop_timeout_seconds"] = 1815
        self.assertEqual(ops.stack_command(data, "fixture-web", "stop", identifiers=[CONTAINER_ID]),
                         ["docker", "stop", "--timeout", "1815", CONTAINER_ID])
        self.assertEqual(ops.stack_command(data, "fixture-web", "start", identifiers=[CONTAINER_ID]),
                         ["docker", "start", CONTAINER_ID])
        for invalid in (True, 0, 1816, "1815", 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ops.OperationError):
                ops.stop_timeout_seconds({"stop_timeout_seconds": invalid})

    @patch.object(ops.subprocess, "run", return_value=SimpleNamespace(returncode=0))
    def test_command_timeout_can_cover_explicit_drain_without_changing_default(self, run: object) -> None:
        ops.run_command(["synthetic-command"])
        self.assertEqual(run.call_args.kwargs["timeout"], 90)
        ops.run_command(["synthetic-command"], timeout=1830)
        self.assertEqual(run.call_args.kwargs["timeout"], 1830)

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    @patch.object(ops.subprocess, "run")
    def test_restore_container_is_not_selected(self, run: object, _prefix: object) -> None:
        restore_id = "b" * 64
        run.side_effect = [
            SimpleNamespace(stdout=f"{CONTAINER_ID}\n{restore_id}"),
            SimpleNamespace(
                stdout=ownership_record(ops.ROOT / "deploy/core")
                + "\n"
                + ownership_record(ops.ROOT / "deploy/core", service="restore", identifier=restore_id)
            ),
        ]
        self.assertEqual(ops.check_ownership(registry_fixture(), "fixture-web"), [CONTAINER_ID])

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    def test_lifecycle_requires_exact_ids(self, _prefix: object) -> None:
        with self.assertRaises(ops.OperationError):
            ops.stack_command(registry_fixture(), "fixture-web", "start")

    def test_artifact_descriptor_rejects_invalid_json(self) -> None:
        target = self.root / "deploy/local"
        target.mkdir(parents=True)
        (target / "artifacts.json").write_text("not JSON", encoding="utf-8")
        with self.assertRaises(ValueError):
            ops.load_artifacts(self.root)

    def test_artifact_descriptor_rejects_traversal_and_invalid_hash(self) -> None:
        target = self.root / "deploy/local"
        target.mkdir(parents=True)
        for path, digest in (("../outside", "a" * 64), ("asset.json", "not-a-hash")):
            with self.subTest(path=path):
                (target / "artifacts.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "files": [{"id": "proof", "path": path, "sha256": digest}],
                            "directories": [],
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaises(ops.OperationError):
                    ops.load_artifacts(self.root)

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    def test_historical_stack_cannot_start(self, _prefix: object) -> None:
        data = registry_fixture()
        data["stacks"][0]["lifecycle"] = False
        with self.assertRaises(ops.OperationError):
            ops.stack_command(data, "fixture-web", "start")

    @patch.object(ops, "docker_prefix", return_value=["docker"])
    def test_unknown_stack_cannot_execute(self, _prefix: object) -> None:
        with self.assertRaises(ops.OperationError):
            ops.stack_command(registry_fixture(), "unknown", "start")

    def test_structure_check_does_not_claim_product_acceptance(self) -> None:
        result = ops.verification()
        self.assertEqual(result["feature_acceptance"], "not_run")
        self.assertEqual(result["model_calls"], 0)


if __name__ == "__main__":
    unittest.main()
