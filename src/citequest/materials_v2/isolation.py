"""Native Windows Job Object resource boundary for one parser at a time."""
from __future__ import annotations

import ctypes
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from .cpu_affinity import select_affinity, system_cpu_sets
from .text import MaterialError


class WindowsJob:
    def __init__(self, memory_bytes: int, logical_cpus: int, allowed_affinity: list[int]) -> None:
        if sys.platform != "win32":
            raise MaterialError("WINDOWS_JOB_PROFILE_UNAVAILABLE")
        import win32job

        self.module = win32job
        self.handle = win32job.CreateJobObject(None, "")
        try:
            info = win32job.QueryInformationJobObject(self.handle, win32job.JobObjectExtendedLimitInformation)
            chosen, topology = select_affinity(allowed_affinity, logical_cpus, system_cpu_sets())
            flags = (win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY | win32job.JOB_OBJECT_LIMIT_JOB_MEMORY
                     | win32job.JOB_OBJECT_LIMIT_ACTIVE_PROCESS | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                     | win32job.JOB_OBJECT_LIMIT_AFFINITY)
            info["BasicLimitInformation"].update(LimitFlags=flags, ActiveProcessLimit=1, Affinity=sum(1 << cpu for cpu in chosen))
            info["ProcessMemoryLimit"] = info["JobMemoryLimit"] = memory_bytes
            win32job.SetInformationJobObject(self.handle, win32job.JobObjectExtendedLimitInformation, info)

            class CpuRate(ctypes.Structure):
                _fields_ = [("ControlFlags", ctypes.c_ulong), ("CpuRate", ctypes.c_ulong)]

            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.SetInformationJobObject.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong)
            kernel.SetInformationJobObject.restype = ctypes.c_int
            kernel.QueryInformationJobObject.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong, ctypes.c_void_p)
            kernel.QueryInformationJobObject.restype = ctypes.c_int
            rate = max(1, min(10000, int(logical_cpus / (os.cpu_count() or logical_cpus) * 10000)))
            cpu = CpuRate(0x1 | 0x4, rate)  # Official ENABLE | HARD_CAP, 1/10000 units.
            if not kernel.SetInformationJobObject(int(self.handle), 15, ctypes.byref(cpu), ctypes.sizeof(cpu)):
                raise ctypes.WinError(ctypes.get_last_error())
            verified = CpuRate()
            if not kernel.QueryInformationJobObject(int(self.handle), 15, ctypes.byref(verified), ctypes.sizeof(verified), None):
                raise ctypes.WinError(ctypes.get_last_error())
            self.cpu_control = {"flags": verified.ControlFlags, "rate_per_10000": verified.CpuRate,
                                "host_logical_cpus": os.cpu_count(), "chosen_affinity": chosen, "topology_selection": topology}
        except BaseException:
            self.handle.Close()
            raise

    def assign(self, process: subprocess.Popen) -> None:
        self.module.AssignProcessToJobObject(self.handle, int(process._handle))

    def inspect(self) -> dict:
        return {"extended": self.module.QueryInformationJobObject(self.handle, self.module.JobObjectExtendedLimitInformation),
                "accounting": self.module.QueryInformationJobObject(self.handle, self.module.JobObjectBasicAccountingInformation),
                "cpu": self.cpu_control}

    def terminate(self) -> None:
        self.module.TerminateJobObject(self.handle, 0xE004)

    def close(self) -> None:
        self.handle.Close()


@contextmanager
def _single_worker(lock_path: Path) -> Iterator[None]:
    import msvcrt

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as file:
        file.seek(0)
        if not file.read(1):
            file.write(b"0")
            file.flush()
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise MaterialError("PARSER_BUSY") from exc
        try:
            yield
        finally:
            file.seek(0)
            msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)


def _bounded_worker_result(output: Path, limit: int) -> tuple[dict, int, list[str]]:
    """Final post-exit check happens before loading any worker-controlled JSON.

    The observed file cap is a poll-and-acceptance boundary, not an NTFS quota.
    Failed oversized artifacts are preserved and never loaded into the parent.
    """
    files = [path for path in output.rglob("*") if path.is_file()]
    total = sum(path.stat().st_size for path in files)
    path = output / "result.json"
    size = path.stat().st_size if path.exists() else None
    if total > limit or (size is not None and size > limit):
        return ({"state": "failed", "code": "WORKER_ARTIFACT_LIMIT_EXCEEDED", "can_qa": False,
                 "unread_result_path": str(path), "unread_result_bytes": size}, total, ["WORKER_ARTIFACT_LIMIT_EXCEEDED"])
    if path.is_symlink() or (path.exists() and path.resolve().parent != output.resolve()):
        return {"state": "failed", "code": "WORKER_RESULT_PATH_INVALID", "can_qa": False}, total, ["WORKER_RESULT_PATH_INVALID"]
    if not path.exists():
        return {"state": "failed", "code": "WORKER_NO_RESULT", "can_qa": False}, total, []
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return {"state": "failed", "code": "WORKER_INVALID_RESULT_JSON", "can_qa": False}, total, []
    if (type(result) is not dict or type(result.get("state")) is not str or result.get("state") not in {"ready", "partial", "failed", "rejected"}
            or type(result.get("code")) is not str or not result["code"]):
        return {"state": "failed", "code": "WORKER_INVALID_RESULT_SHAPE", "can_qa": False}, total, []
    return result, total, []


def _target_environment(python: Path, root: Path, versions: dict) -> Path:
    environment = python.resolve().parent.parent
    if not python.is_file() or not environment.is_relative_to(root):
        raise MaterialError("INVALID_PINNED_WORKER_ENVIRONMENT")
    cfg = dict(line.split(" = ", 1) for line in (environment / "pyvenv.cfg").read_text(encoding="utf-8").splitlines() if " = " in line)
    if (Path(cfg.get("home", "")).resolve() != Path(sys._base_executable).resolve().parent
            or cfg.get("include-system-site-packages") != "false"
            or cfg.get("version", cfg.get("version_info")) != ".".join(str(v) for v in sys.version_info[:3])):
        raise MaterialError("WORKER_BASE_INTERPRETER_OR_SITE_POLICY_MISMATCH")
    site = environment / "Lib/site-packages"
    installed = {str(distribution.metadata["Name"]).lower().replace("_", "-"): distribution.version
                 for distribution in importlib.metadata.Distribution.discover(path=[str(site)])}
    if any(installed.get(name.lower().replace("_", "-")) != expected for name, expected in versions.items()):
        raise MaterialError("PARSER_DEPENDENCY_VERSION_MISMATCH")
    return site


def run_isolated(request: dict, *, root: Path, profile_path: Path, output: Path, python: Path,
                 on_validated: Callable[[], None] | None = None, cancel_requested: Callable[[], bool] | None = None,
                 deadline_seconds: float | None = None) -> dict:
    """Resource limits are assigned before the worker receives its START line.

    Inputs are internal CAS/developer paths, never an HTTP-supplied filesystem
    path. Product callers must check current owner/ACL and source SHA first.
    """
    import psutil

    root, output = root.resolve(), output.resolve()
    if sys.platform != "win32" or not output.is_relative_to(root) or output.exists():
        raise MaterialError("INVALID_OR_UNAVAILABLE_WORKER_OUTPUT")
    profile_raw = profile_path.read_bytes()
    profile_sha = hashlib.sha256(profile_raw).hexdigest()
    if request.get("expected_profile_sha256", profile_sha) != profile_sha:
        raise MaterialError("PARSER_PROFILE_CHANGED_BEFORE_START")
    profile = json.loads(profile_raw)
    site = _target_environment(python, root, profile["parser_versions"])
    operation = request.get("operation", "parse")
    if operation not in {"parse", "chunk", "probe_memory", "probe_timeout", "probe_cpu", "probe_child"}:
        raise MaterialError("UNKNOWN_WORKER_OPERATION")
    if psutil.virtual_memory().available < profile["minimum_host_available_bytes"]:
        raise MaterialError("INSUFFICIENT_HOST_MEMORY_FOR_PARSER")
    test_probe = operation.startswith("probe_")
    memory = request.get("probe_memory_limit", profile["memory_limit_bytes"]) if test_probe else profile["memory_limit_bytes"]
    deadline = request.get("probe_deadline", profile["deadline_seconds"]) if test_probe else profile["deadline_seconds"]
    if deadline_seconds is not None:
        if type(deadline_seconds) not in {float, int} or not 0 < deadline_seconds <= profile["deadline_seconds"]:
            raise MaterialError("REMAINING_DEADLINE_CANNOT_EXPAND_PROFILE")
        deadline = min(deadline, deadline_seconds)
    if (type(memory) is not int or not 64 * 1024 * 1024 <= memory <= profile["memory_limit_bytes"]
            or type(deadline) not in {int, float} or not 0 < deadline <= profile["deadline_seconds"]):
        raise MaterialError("RESOURCE_PROBE_CANNOT_INCREASE_PROFILE")
    with _single_worker(root / "data/runtime-s04/parser.lock"):
        output.mkdir(parents=True)
        temp = output / "tmp"
        temp.mkdir()
        home = temp / "home"
        home.mkdir()
        token = uuid.uuid4().hex
        job = WindowsJob(memory, profile["logical_cpu_limit"], psutil.Process().cpu_affinity())
        safe_keys = {"SYSTEMROOT", "WINDIR", "PATH", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS"}
        env = {key: value for key, value in os.environ.items() if key.upper() in safe_keys}
        # Windows venv python.exe is a redirector creating another process.
        # Invoke its exact base interpreter without automatic site processing,
        # then expose only the pinned venv packages via explicit PYTHONPATH.
        env.update(PYTHONPATH=os.pathsep.join((str(root / "src"), str(site))), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", TEMP=str(temp), TMP=str(temp),
                   HF_HOME=str(root / "data/runtime-s04/hf"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1", TORCH_HOME=str(root / "data/runtime-s04/torch"),
                   USERPROFILE=str(home), HOMEDRIVE=home.drive, HOMEPATH=str(home)[len(home.drive):], LOCALAPPDATA=str(home / "AppData/Local"), APPDATA=str(home / "AppData/Roaming"),
                   OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false")
        command = [str(Path(sys._base_executable).resolve()), "-S", str(root / "scripts/s04/worker_entry.py"), token]
        base = Path(command[0]).parent
        allowed_paths = [root / "scripts/s04", root / "src", site,
                         base / f"python{sys.version_info.major}{sys.version_info.minor}.zip", base / "DLLs", base / "Lib", base]
        process, threads, logs, violations = None, [], {"stdout": bytearray(), "stderr": bytearray()}, []
        assigned = False
        phase_observed = False
        start = time.monotonic()
        last_cancel_check = start
        job_before = job.inspect()

        def observe_validation() -> None:
            nonlocal phase_observed
            phase_path = output / "phase.json"
            if on_validated is None or phase_observed or not phase_path.exists():
                return
            if phase_path.is_symlink() or phase_path.stat().st_size > 4096:
                raise MaterialError("WORKER_PHASE_EVENT_INVALID")
            phase = json.loads(phase_path.read_text(encoding="utf-8"))
            expected = {"start_token": token, "phase": "parsing", "profile_sha256": profile_sha, "source_sha256": request.get("source_sha256")}
            if phase != expected:
                raise MaterialError("WORKER_PHASE_EVENT_INVALID")
            on_validated()
            phase_observed = True
        try:
            process = subprocess.Popen(command, cwd=output, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(process)
            assigned = True
            for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
                def reader(label: str = name, source: BinaryIO = pipe) -> None:
                    while part := source.read(8192):
                        if len(logs[label]) + len(part) > profile["max_log_bytes_each"]:
                            violations.append("WORKER_LOG_LIMIT_EXCEEDED")
                            break
                        logs[label].extend(part)
                thread = threading.Thread(target=reader, daemon=True)
                thread.start()
                threads.append(thread)
            payload = {**request, "root": str(root), "output": str(output), "profile": str(profile_path.resolve()), "start_token": token,
                       "allowed_sys_path": [str(p.resolve()) for p in allowed_paths], "profile_json": profile_raw.decode("utf-8"), "profile_sha256": profile_sha}
            process.stdin.write((json.dumps(payload, ensure_ascii=False) + "\n").encode())
            process.stdin.flush()
            process.stdin.close()
            while process.poll() is None:
                try:
                    observe_validation()
                    if cancel_requested is not None and time.monotonic() - last_cancel_check >= 0.5:
                        last_cancel_check = time.monotonic()
                        if cancel_requested() is True:
                            violations.append("PARSER_CANCELLED")
                except Exception:
                    violations.append("WORKER_OBSERVER_FAILED")
                if time.monotonic() - start > deadline:
                    violations.append("PARSER_DEADLINE_EXCEEDED")
                if sum(p.stat().st_size for p in output.rglob("*") if p.is_file()) > profile["max_output_artifact_bytes"]:
                    violations.append("WORKER_ARTIFACT_LIMIT_EXCEEDED")
                if violations:
                    job.terminate()
                    break
                time.sleep(0.05)
            process.wait(timeout=10)
            for thread in threads:
                thread.join(timeout=3)
            job_after = job.inspect()
            try:
                observe_validation()
            except Exception:
                violations.append("WORKER_OBSERVER_FAILED")
            if time.monotonic() - start > deadline and "PARSER_DEADLINE_EXCEEDED" not in violations:
                violations.append("PARSER_DEADLINE_EXCEEDED")
        finally:
            if process is not None and process.poll() is None:
                if assigned:
                    job.terminate()
                else:
                    process.kill()
                process.wait(timeout=10)
            for name, body in logs.items():
                (output / f"{name}.log").write_bytes(body)
            job.close()
        result_path = output / "result.json"
        result, artifact_bytes, final_violations = _bounded_worker_result(output, profile["max_output_artifact_bytes"])
        violations.extend(final_violations)
        profile_after_sha = hashlib.sha256(profile_path.read_bytes()).hexdigest() if profile_path.exists() else None
        if profile_after_sha != profile_sha:
            violations.append("PARSER_PROFILE_CHANGED_DURING_EXECUTION")
        if result_path.exists() and result.get("profile_sha256") != profile_sha and not final_violations:
            violations.append("WORKER_PROFILE_BINDING_MISMATCH")
        if time.monotonic() - start > deadline and "PARSER_DEADLINE_EXCEEDED" not in violations:
            violations.append("PARSER_DEADLINE_EXCEEDED")
        if violations:
            result = {"state": "failed", "code": violations[0], "partial_result_preserved": result_path.exists()}
        if process.returncode and result.get("state") not in {"failed", "rejected"}:
            result = {"state": "failed", "code": "WORKER_NONZERO_EXIT", "original_result": result}
        receipt = {"schema_version": "s04-isolated-worker-receipt/1", "result": result, "exit_code": process.returncode,
                   "elapsed_seconds": time.monotonic() - start, "job_before": job_before, "job_after": job_after,
                   "artifact_bytes_observed_before_receipt": artifact_bytes,
                   "requested_memory_bytes": memory, "deadline_seconds": deadline, "violations": violations,
                   "operation": operation, "profile_path": str(profile_path), "no_credentials_in_worker_environment": True,
                   "profile_sha256": profile_sha, "profile_after_sha256": profile_after_sha,
                   "interpreter": command[0], "environment_packages": str(site), "site_processing_disabled": True, "validation_phase_observed": phase_observed,
                   "interpreter_sha256": hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
                   "lock_path": profile.get("environment_lock", "runtime/v2/uv.lock"),
                   "lock_sha256": hashlib.sha256((root / profile.get("environment_lock", "runtime/v2/uv.lock")).read_bytes()).hexdigest(),
                   "entrypoint_sha256": hashlib.sha256((root / "scripts/s04/worker_entry.py").read_bytes()).hexdigest(),
                   "worker_pid": process.pid, "worker_stopped": process.poll() is not None, "resource_scope": "Windows Job limits, not filesystem or AppContainer sandbox"}
        receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode()
        if artifact_bytes + len(receipt_bytes) > profile["max_output_artifact_bytes"]:
            receipt["result"] = {"state": "failed", "code": "WORKER_ARTIFACT_LIMIT_EXCEEDED", "can_qa": False,
                                 "partial_result_preserved": result_path.exists(), "parent_receipt_would_exceed_artifact_limit": True}
            if "WORKER_ARTIFACT_LIMIT_EXCEEDED" not in violations:
                violations.append("WORKER_ARTIFACT_LIMIT_EXCEEDED")
            receipt_bytes = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode()
        (output / "receipt.json").write_bytes(receipt_bytes)
        return receipt
