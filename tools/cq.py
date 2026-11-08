"""Small, dependency-free operations facade; business runners remain separate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path("deploy/local/stacks.json")
CACHES = (".pytest_cache", ".mypy_cache", ".ruff_cache", ".coverage", "htmlcov", "web/dist")
IDENTIFIER = re.compile(r"[a-z][a-z0-9-]*\Z")


class OperationError(Exception):
    """An operation failed without changing the requested scope."""


def within_root(root: Path, relative: str) -> Path:
    """Reject traversal and symlink/junction escapes before any filesystem action."""
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise OperationError(f"Expected a workspace-relative path: {relative!r}")
    root = root.resolve()
    path = root / relative
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or resolved == root:
        raise OperationError(f"Path escapes workspace: {relative!r}")
    # Do not operate through a link, even when its destination is in the workspace.
    cursor = path
    while cursor != root:
        is_junction = getattr(cursor, "is_junction", lambda: False)
        if cursor.is_symlink() or is_junction():
            raise OperationError(f"Symlink/junction is not allowed: {relative!r}")
        cursor = cursor.parent
    return path


def compose_paths(stack: dict[str, Any], root: Path = ROOT) -> list[Path]:
    """Keep the full ordered Compose identity; never select an implicit override."""
    overrides = stack.get("compose_overrides", [])
    if not isinstance(overrides, list) or any(not isinstance(path, str) for path in overrides):
        raise OperationError("Invalid declared Compose overrides")
    names = [stack["compose"], *overrides]
    if any("," in name for name in names):
        raise OperationError("Ambiguous comma in declared Compose path")
    paths = [within_root(root, name).resolve() for name in names]
    if len(set(paths)) != len(paths):
        raise OperationError("Duplicate declared Compose path")
    return paths


def stop_timeout_seconds(stack: dict[str, Any]) -> int | None:
    value = stack.get("stop_timeout_seconds")
    if value is not None and (type(value) is not int or not 1 <= value <= 1815):
        raise OperationError("Declared stop timeout must be an integer from 1 to 1815 seconds")
    return value


def release_descriptor(stack: dict[str, Any], root: Path = ROOT) -> dict[str, Any] | None:
    release = stack.get("active_release")
    if release is None:
        return None
    if not isinstance(release, dict) or set(release) != {
        "image_id", "config_manifest", "config_manifest_sha256", "compose_sha256"
    }:
        raise OperationError("Invalid active release descriptor")
    if not isinstance(release["image_id"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", release["image_id"]):
        raise OperationError("Active release requires an exact image ID")
    if not isinstance(release["config_manifest"], str):
        raise OperationError("Invalid active release configuration reference")
    within_root(root, release["config_manifest"])
    hashes = release["compose_sha256"]
    names = [stack["compose"], *stack.get("compose_overrides", [])]
    if not isinstance(hashes, dict) or set(hashes) != set(names):
        raise OperationError("Active release must bind every declared Compose file")
    if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
           for value in [release["config_manifest_sha256"], *hashes.values()]):
        raise OperationError("Invalid active release SHA256")
    return release


def verify_release_files(stack: dict[str, Any], root: Path = ROOT) -> dict[str, Any] | None:
    """Check only declared non-secret release files before any lifecycle action."""
    release = release_descriptor(stack, root)
    if release is None:
        return None

    def check(path: Path, expected: str) -> None:
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise OperationError("Active release file is missing or changed: " + path.relative_to(root).as_posix())

    for relative, expected in release["compose_sha256"].items():
        check(within_root(root, relative), expected)
    manifest_path = within_root(root, release["config_manifest"])
    check(manifest_path, release["config_manifest_sha256"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("image", {}).get("image_id") != release["image_id"]:
        raise OperationError("Active release image/configuration binding changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise OperationError("Active release configuration file bindings are missing")
    parent = manifest_path.parent.relative_to(root)
    for relative, expected in files.items():
        if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
                or ".." in Path(relative).parts or not isinstance(expected, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected)):
            raise OperationError("Invalid active release configuration file binding")
        check(within_root(root, (parent / relative).as_posix()), expected)
    return release


def load_registry(root: Path = ROOT) -> dict[str, Any]:
    data = json.loads(within_root(root, REGISTRY.as_posix()).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1:
        raise OperationError("Unsupported registry schema")
    seen: set[str] = set()
    projects: set[str] = set()
    for stack in data["stacks"]:
        for key in ("id", "project"):
            if not IDENTIFIER.fullmatch(stack[key]):
                raise OperationError(f"Invalid stack {key}")
        if stack["id"] in seen or stack["project"] in projects:
            raise OperationError("Duplicate stack id/project")
        seen.add(stack["id"])
        projects.add(stack["project"])
        if not isinstance(stack["lifecycle"], bool) or not isinstance(stack["services"], list):
            raise OperationError("Invalid lifecycle/services")
        if any(not IDENTIFIER.fullmatch(service) for service in stack["services"]):
            raise OperationError("Invalid service name")
        if stack["lifecycle"] and not stack["services"]:
            raise OperationError("Managed stack requires explicit services")
        if not within_root(root, stack["compose"]).is_file():
            raise OperationError(f"Missing compose: {stack['compose']}")
        # Local release artifacts may be absent on a clean checkout. Structure
        # checks validate references; lifecycle checks require their exact bytes.
        compose_paths(stack, root)
        release_descriptor(stack, root)
        stop_timeout_seconds(stack)
    if data.get("docker_config") is not None:
        within_root(root, data["docker_config"])
    for runtime in data["runtimes"]:
        for key in ("project", "lock", "requirements", "dockerfile"):
            if key in runtime and not within_root(root, runtime[key]).is_file():
                raise OperationError(f"Missing runtime {key}: {runtime[key]}")
    return data


def verification(root: Path = ROOT) -> dict[str, Any]:
    """Check engineering structure only; never label feature acceptance passed."""
    data = load_registry(root)
    mandatory = (
        "AGENTS.md",
        "README.md",
        "pyproject.toml",
        "deploy/core/Dockerfile",
        "deploy/core/compose.yaml",
        "assets/export/sjtu-proposal.docx",
        "assets/export/harvard.csl",
        "assets/export/toolchain-identity.json",
        "assets/export/capacity-certification.json",
        "runtime/s13/release-profile.json",
        "src/citequest/web_v3/main.py",
        "src/citequest/agents_v3/worker.py",
        "src/citequest/exports_v3/cli.py",
        "scripts/s09/manage_index.py",
        "scripts/s14/serve_decisionrag.py",
        "web/package-lock.json",
        "alembic-app.ini",
        ".github/workflows/engineering.yml",
    )
    missing = [path for path in mandatory if not within_root(root, path).is_file()]
    if missing:
        raise OperationError("Missing engineering files: " + ", ".join(missing))
    if within_root(root, "deploy/local/artifacts.json").is_file():
        load_artifacts(root)
    return {
        "status": "passed",
        "scope": "engineering_structure_only",
        "stacks": len(data["stacks"]),
        "runtimes": len(data["runtimes"]),
        "feature_acceptance": "not_run",
        "docker_mutations": 0,
        "model_calls": 0,
    }


def load_artifacts(root: Path = ROOT) -> dict[str, Any]:
    """Validate descriptors without requiring local data to exist on a clean checkout."""
    manifest = json.loads(within_root(root, "deploy/local/artifacts.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise OperationError("Unsupported artifact manifest")
    seen = set()
    for category in ("files", "directories"):
        if not isinstance(manifest.get(category), list):
            raise OperationError("Invalid artifact category")
        for row in manifest[category]:
            if (
                not isinstance(row, dict)
                or not isinstance(row.get("id"), str)
                or not IDENTIFIER.fullmatch(row["id"])
                or row["id"] in seen
                or not isinstance(row.get("path"), str)
            ):
                raise OperationError("Invalid/duplicate artifact id")
            seen.add(row["id"])
            within_root(root, row["path"])
            if category == "files" and (
                not isinstance(row.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
            ):
                raise OperationError("Invalid artifact SHA256")
    return manifest


def local_artifacts(root: Path = ROOT) -> dict[str, Any]:
    """Probe declared non-secret artifacts; a directory is not a data integrity check."""
    if not within_root(root, "deploy/local/artifacts.json").is_file():
        return {"status": "not_configured", "checks": []}
    manifest = load_artifacts(root)
    checks = []
    for row in manifest["files"]:
        path = within_root(root, row["path"])
        status = "missing"
        if path.is_file():
            with path.open("rb") as stream:
                status = (
                    "verified"
                    if hashlib.file_digest(stream, "sha256").hexdigest() == row["sha256"]
                    else "hash_mismatch"
                )
        checks.append({"id": row["id"], "status": status})
    for row in manifest["directories"]:
        exists = within_root(root, row["path"]).is_dir()
        checks.append({"id": row["id"], "status": "exists_only" if exists else "missing"})
    return {
        "status": "present_not_runtime_ready"
        if all(item["status"] in {"verified", "exists_only"} for item in checks)
        else "needs_setup",
        "checks": checks,
    }


def docker_prefix(data: dict[str, Any], root: Path = ROOT) -> list[str]:
    binary = shutil.which("docker")
    if not binary:
        raise OperationError("Docker CLI not found; no service was started")
    if data.get("docker_config") is None:
        return [binary]
    config = within_root(root, data["docker_config"])
    if not config.is_dir():
        raise OperationError("Declared Docker client config directory is missing")
    return [binary, "--config", str(config)]


def stack_command(
    data: dict[str, Any], name: str, action: str, root: Path = ROOT, *, identifiers: list[str] | None = None
) -> list[str]:
    matches = [item for item in data["stacks"] if item["id"] == name]
    if not matches:
        raise OperationError(f"Unknown stack: {name}")
    stack = matches[0]
    prefix = docker_prefix(data, root)
    if action == "status":
        return [
            *prefix,
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={stack['project']}",
            "--filter",
            f"label=com.docker.compose.project.working_dir={within_root(root, stack['compose']).parent}",
            "--format",
            "table {{.Names}}\t{{.Status}}\t{{.Image}}\t{{.Ports}}",
        ]
    if not stack["lifecycle"]:
        raise OperationError("Historical/test bootstrap stack is read-only through this facade")
    if action not in {"start", "stop", "logs"}:
        raise OperationError("Unsupported lifecycle action")
    if not identifiers or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in identifiers):
        raise OperationError("Exact verified container IDs are required")
    if action == "logs" and len(identifiers) != 1:
        raise OperationError("Read logs one verified container at a time")
    options = ["--tail", "80"] if action == "logs" else []
    grace = stop_timeout_seconds(stack)
    if action == "stop" and grace is not None:
        options = ["--timeout", str(grace)]
    # Bind IDs after verification; Compose cannot reselect a different checkout's containers.
    return [*prefix, action, *options, *identifiers]


def check_ownership(data: dict[str, Any], name: str, root: Path = ROOT) -> list[str]:
    """A project name alone is insufficient: another checkout can use the same name."""
    stack = next(item for item in data["stacks"] if item["id"] == name)
    expected_compose = compose_paths(stack, root)
    if any(not path.is_file() for path in expected_compose):
        raise OperationError("Declared Compose file missing; no fallback was attempted")
    release = verify_release_files(stack, root)
    prefix = docker_prefix(data, root)
    fields = (
        '[{{json .Id}},{{json (index .Config.Labels "com.docker.compose.project.working_dir")}},'
        '{{json (index .Config.Labels "com.docker.compose.project.config_files")}},'
        '{{json (index .Config.Labels "com.docker.compose.service")}},'
        '{{json (index .Config.Labels "com.docker.compose.oneoff")}}'
    )
    result = subprocess.run(
        [*prefix, "ps", "-aq", "--no-trunc", "--filter", f"label=com.docker.compose.project={stack['project']}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    identifiers = result.stdout.split()
    if not identifiers or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in identifiers):
        raise OperationError("No existing containers for this stack; explicit first-time setup is required")
    result = subprocess.run(
        [
            *prefix,
            "inspect",
            "--format",
            fields + (',{{json .Image}}]' if release else ']'),
            *identifiers,
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=20,
    )
    compose = expected_compose[0]
    records = [json.loads(line) for line in result.stdout.splitlines()]
    if len(records) != len(identifiers):
        raise OperationError("Container ownership response incomplete")
    selected: dict[str, list[str]] = {service: [] for service in stack["services"]}
    for record in records:
        if not isinstance(record, list) or len(record) != (6 if release else 5) or not all(isinstance(value, str) for value in record):
            raise OperationError("Malformed container ownership response")
        container, directory, config_files, service, oneoff = record[:5]
        if (
            container not in identifiers
            or Path(directory).resolve() != compose.parent
            or [Path(path.strip()).resolve() for path in config_files.split(",")] != expected_compose
        ):
            raise OperationError("Compose project belongs to another directory/config; refusing operation")
        if service in selected and str(oneoff).lower() == "false":
            if release and record[5] != release["image_id"]:
                raise OperationError("Container image differs from the active release; refusing operation")
            selected[service].append(container)
    if {row[0] for row in records} != set(identifiers) or any(not ids for ids in selected.values()):
        raise OperationError("Missing/duplicate ownership response or registered normal service container")
    return [container for service in stack["services"] for container in sorted(selected[service])]


def run_command(command: list[str], *, timeout: int = 90) -> None:
    result = subprocess.run(command, cwd=ROOT, check=False, timeout=timeout)
    if result.returncode:
        raise OperationError(f"Command failed (exit {result.returncode}); no fallback was attempted")


def cache_plan(root: Path = ROOT) -> list[str]:
    return [relative for relative in CACHES if within_root(root, relative).exists()]


def clean_cache(root: Path = ROOT, *, apply: bool = False) -> dict[str, Any]:
    paths = cache_plan(root)
    removed = []
    errors = []
    if apply:
        for relative in paths:
            path = within_root(root, relative)
            try:
                if path.is_dir():
                    # Check every descendant before deletion, not only the directory root.
                    for child in path.rglob("*"):
                        within_root(root, child.relative_to(root).as_posix())
                    shutil.rmtree(path)
                else:
                    path.unlink()
                removed.append(relative)
            except OSError as exc:
                errors.append({"path": relative, "error": type(exc).__name__})
    return {
        "status": "partial" if errors else "completed" if apply else "preview",
        "applied": apply,
        "paths": paths,
        "removed": removed,
        "errors": errors,
        "data_touched": False,
        "docker_touched": False,
    }


def growth_gib(value: str) -> float:
    """Reject negative/nonfinite/overflowing input rather than weakening the gate."""
    try:
        number = float(value)
        if number < 0 or not math.isfinite((10 + 2 * number) * 1024**3):
            raise ValueError
    except (ValueError, OverflowError):
        raise argparse.ArgumentTypeError("Expected finite non-negative GiB") from None
    return number


def docker_space(*, data_vhd: Path | None = None, expected_growth_gib: float = 0,
                 check: bool = False, root: Path = ROOT) -> dict[str, Any]:
    """Read selected metadata only; no inspect Env/Args, jobs or Docker mutation.

    Docker Desktop's host volume cannot be inferred from /var/lib/docker.
    An explicit trusted VHD path is required; its binding is operator-supplied.
    Docker's own system-df totals are retained without summing virtual sizes.
    """
    expected_growth_gib = growth_gib(str(expected_growth_gib))
    required = math.ceil((10 + 2 * expected_growth_gib) * 1024**3)
    result: dict[str, Any] = {
        "status": "unknown", "check_requested": check, "check_passed": False,
        "required_free_bytes": required, "minimum_free_gib": 10,
        "expected_growth_gib": expected_growth_gib, "temporary_growth_multiplier": 2,
        "storage": {"status": "unknown", "reason": "explicit_trusted_data_vhd_required"},
        "docker_mutations": 0, "model_calls": 0, "errors": [],
    }
    try:
        prefix = docker_prefix(load_registry(root), root)
    except (OperationError, OSError, ValueError, KeyError) as error:
        result["errors"].append({"check": "docker_configuration", "error": type(error).__name__})
        return result

    def read(name: str, args: list[str]) -> list[dict[str, Any]] | None:
        try:
            response = subprocess.run([*prefix, *args], cwd=root, check=True, capture_output=True,
                text=True, encoding="utf-8", timeout=30)
            rows = [json.loads(line) for line in response.stdout.splitlines() if line.strip()]
            if any(not isinstance(row, dict) for row in rows):
                raise ValueError
            return rows
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            # Raw Docker stderr can contain connection details; keep it private.
            result["errors"].append({"check": name, "error": type(error).__name__})
            return None

    info = read("daemon", ["info", "--format", '{"id":{{json .ID}},"os":{{json .OSType}},'
        '"system":{{json .OperatingSystem}},"root":{{json .DockerRootDir}}}'])
    desktop = bool(info and len(info) == 1 and info[0].get("id") and info[0].get("os") == "linux"
        and info[0].get("system") == "Docker Desktop" and info[0].get("root") == "/var/lib/docker")
    result["daemon"] = info[0] if info and len(info) == 1 else None
    if data_vhd is not None:
        try:
            if sys.platform != "win32" or not desktop:
                raise OperationError("Explicit VHD check requires local Windows Docker Desktop Linux backend")
            path = data_vhd.expanduser()
            if not path.is_absolute() or path.suffix.lower() != ".vhdx" or not path.is_file():
                raise OperationError("Existing absolute VHDX file required")
            for ancestor in (path, *path.parents):
                if ancestor.is_symlink() or getattr(ancestor, "is_junction", lambda: False)():
                    raise OperationError("VHD path cannot pass through a link/junction")
            usage = shutil.disk_usage(path.parent)
            result["storage"] = {"status": "known_explicit", "source": "operator_supplied_VHD_binding_not_autodetected",
                "vhd_path": str(path), "host_volume": path.anchor,
                "host_volume_total_bytes": usage.total, "host_volume_free_bytes": usage.free,
                "vhd_logical_file_bytes_not_allocated_bytes": path.stat().st_size}
        except (OSError, OperationError) as error:
            result["errors"].append({"check": "data_vhd", "error": type(error).__name__})

    result["docker_system_df"] = read("system_df", ["system", "df", "--format", "{{json .}}"])
    result["size_note"] = "Official daemon totals; virtual image sizes are not summed or treated as freed host bytes"
    containers = read("containers", ["ps", "-a", "--no-trunc", "--format",
        '{"id":{{json .ID}},"state":{{json .State}},"oneoff":{{json (.Label "com.docker.compose.oneoff")}},'
        '"lifecycle":{{json (.Label "cq.lifecycle")}},"task":{{json (.Label "cq.task")}}}'])
    if containers is not None:
        oneoffs = [row for row in containers if str(row.get("oneoff", "")).lower() == "true"
            or row.get("lifecycle") == "oneoff"]
        result["containers"] = {"total": len(containers), "running": sum(row.get("state") == "running" for row in containers),
            "exited": sum(row.get("state") == "exited" for row in containers),
            "explicit_oneoffs_not_exited": [{"id":row["id"], "state":row.get("state")} for row in oneoffs if row.get("state") != "exited"],
            "task_labelled_not_exited": sum(bool(row.get("task")) and row.get("state") != "exited" for row in containers),
            "classification_note": "Labels are inventory hints, not ownership or a deletion allowlist; unlabelled tasks remain unclassified"}
    images = read("images", ["image", "ls", "--no-trunc", "--format",
        '{"id":{{json .ID}},"repository":{{json .Repository}},"tag":{{json .Tag}}}'])
    if images is not None:
        project = [row for row in images if str(row.get("repository", "")).startswith(("citequest", "cq-"))]
        result["images"] = {"unique_image_ids": len({row["id"] for row in images}),
            "project_named_unique_image_ids": len({row["id"] for row in project}),
            "project_tags": len(project), "classification_note": "Repository-name inventory only, not ownership proof"}
    known = result["storage"]["status"] == "known_explicit" and not result["errors"]
    enough = known and result["storage"]["host_volume_free_bytes"] >= required
    result["check_passed"] = bool(enough)
    result["status"] = ("passed" if enough else "insufficient") if known else "unknown"
    return result


def emit(value: Any, *, as_json: bool = False) -> None:
    if as_json or not isinstance(value, str):
        sys.stdout.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(value + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CiteQuest workspace operations (no automatic jobs)")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("doctor", "verify", "stacks"):
        item = sub.add_parser(name)
        item.add_argument("--json", action="store_true")
    item = sub.add_parser("stack")
    item.add_argument("name")
    item.add_argument("action", choices=("status", "start", "stop", "logs"))
    item = sub.add_parser("clean-cache", help="Preview root tool caches; --apply removes only the allowlist")
    item.add_argument("--apply", action="store_true")
    item = sub.add_parser("docker-space", help="Read-only Docker inventory and explicit host VHD volume space gate")
    item.add_argument("--data-vhd", type=Path, help="Trusted absolute path to the actual Docker Desktop data VHDX")
    item.add_argument("--expected-growth-gib", type=growth_gib, default=0)
    item.add_argument("--check", action="store_true", help="Return nonzero for unknown or insufficient space")
    item.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "docker-space":
            result = docker_space(data_vhd=args.data_vhd, expected_growth_gib=args.expected_growth_gib, check=args.check)
            emit(result, as_json=args.json)
            if args.check and not result["check_passed"]:
                return 1
        elif args.command == "clean-cache":
            result = clean_cache(apply=args.apply)
            emit(result)
            if result["errors"]:
                return 1
        elif args.command == "verify":
            emit(verification(), as_json=args.json)
        else:
            data = load_registry()
            if args.command == "doctor":
                result = verification()
                result["tools"] = {name: bool(shutil.which(name)) for name in ("docker", "git", "uv", "node", "npm")}
                result["python"] = sys.version.split()[0]
                result["local_artifacts"] = local_artifacts()
                result["runtime_note"] = "Uses stdlib only; production runtime/DB/model/daemon health not tested"
                emit(result, as_json=args.json)
            elif args.command == "stacks":
                emit(
                    data["stacks"]
                    if args.json
                    else "\n".join(
                        f"{row['id']:22} {row['tier']:12} {row['project']}  {row['purpose']}" for row in data["stacks"]
                    ),
                    as_json=args.json,
                )
            else:
                if args.action == "status":
                    run_command(stack_command(data, args.name, args.action))
                else:
                    stack = next((row for row in data["stacks"] if row["id"] == args.name), None)
                    if stack is None or not stack["lifecycle"]:
                        raise OperationError("Unknown or read-only historical stack")
                    identifiers = check_ownership(data, args.name)
                    if args.action == "stop":
                        identifiers.reverse()
                    groups = [[value] for value in identifiers] if args.action == "logs" else [identifiers]
                    for group in groups:
                        command = stack_command(data, args.name, args.action, identifiers=group)
                        grace = stop_timeout_seconds(stack)
                        if args.action == "stop" and grace is not None:
                            run_command(command, timeout=grace + 15)
                        else:
                            run_command(command)
    except (OperationError, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
