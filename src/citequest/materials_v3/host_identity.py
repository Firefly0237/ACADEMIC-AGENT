"""Local supervisor identity using OS process birth time, never PID alone.

These observations do not prove anything about another host, unavailable OS,
Docker daemon or cloud workload. Failures remain unknown, never "stopped".
"""
from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4


def boot_identity() -> str:
    if sys.platform == "win32":
        reply = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
            "(Get-CimInstance Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime.ToUniversalTime().ToFileTimeUtc()"],
            capture_output=True, text=True, timeout=10, check=True)
        value = reply.stdout.strip()
        if not value.isdecimal():
            raise RuntimeError("host_boot_identity_unavailable")
        return "windows-boot-filetime:" + value
    return "linux-boot-id:" + Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def process_identity(pid: int) -> dict:
    if type(pid) is not int or pid <= 0:
        raise ValueError("positive_process_id_required")
    if sys.platform == "win32":
        from ctypes import wintypes

        library = ctypes.WinDLL("kernel32", use_last_error=True)
        library.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        library.OpenProcess.restype = wintypes.HANDLE
        library.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
        library.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        library.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = library.OpenProcess(0x1000, False, pid)
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # ERROR_INVALID_PARAMETER: no such process.
                return {"pid": pid, "exists": False}
            raise OSError(error, "process_identity_unavailable")
        try:
            values = [wintypes.FILETIME() for _ in range(4)]
            code = wintypes.DWORD()
            if not library.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
                raise OSError(ctypes.get_last_error(), "process_birth_time_unavailable")
            if not library.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise OSError(ctypes.get_last_error(), "process_exit_state_unavailable")
            birth = (values[0].dwHighDateTime << 32) | values[0].dwLowDateTime
            return {"pid": pid, "exists": True, "birth": str(birth), "running": code.value == 259}
        finally:
            library.CloseHandle(handle)
    path = Path("/proc") / str(pid) / "stat"
    try:
        fields = path.read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return {"pid": pid, "exists": False}
    return {"pid": pid, "exists": True, "birth": fields[19], "running": fields[0] not in {"Z", "X"}}


def host_identity(root: Path) -> dict:
    path = root / ".local/host-control/material-supervisor-instance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as stream:
            json.dump({"version": "material-supervisor-instance/1", "instance_id": str(uuid4())}, stream)
    except FileExistsError:
        pass
    value = json.loads(path.read_bytes())
    if value.get("version") != "material-supervisor-instance/1" or not value.get("instance_id"):
        raise ValueError("trusted_host_instance_invalid")
    return {**value, "boot_identity": boot_identity()}


def stopped_original_process(saved: dict) -> bool:
    current = process_identity(saved["pid"])
    return not current["exists"] or current.get("birth") != saved["birth"] or not current.get("running", True)


def current_process_identity() -> dict:
    return process_identity(os.getpid())
