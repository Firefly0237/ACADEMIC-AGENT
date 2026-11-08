"""Minimal Linux bootstrap: establish parent-death protection before SDK import."""

from __future__ import annotations

import ctypes
import os
import signal
import sys


def tether_parent(expected_parent_pid: int) -> None:
    if sys.platform != "linux" or expected_parent_pid <= 0:
        raise RuntimeError("linux_parent_death_supervision_required")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.restype = ctypes.c_int
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:  # Linux PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "parent_death_signal_setup_failed")
    actual = ctypes.c_int()
    if libc.prctl(2, ctypes.addressof(actual), 0, 0, 0) != 0 or actual.value != signal.SIGKILL:
        raise RuntimeError("parent_death_signal_not_confirmed")
    # Covers the fork/exec -> prctl race: never enter SDK IO after reparenting.
    if os.getppid() != expected_parent_pid:
        raise RuntimeError("parent_changed_before_physical_io")


def main() -> int:
    if len(sys.argv) != 5 or sys.argv[1] != "--parent-pid" or sys.argv[3] != "--job":
        raise RuntimeError("fixed_blob_io_worker_arguments_required")
    parent = int(sys.argv[2])
    tether_parent(parent)
    os.umask(0o077)
    import resource

    for kind, requested in ((resource.RLIMIT_CORE, 0), (resource.RLIMIT_AS, 512 * 1024**2),
                            (resource.RLIMIT_CPU, 120), (resource.RLIMIT_NOFILE, 64),
                            (resource.RLIMIT_FSIZE, 64 * 1024**2 + 65536)):
        _, hard = resource.getrlimit(kind)
        value = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        resource.setrlimit(kind, (value, value))
    # No PG/model client or credential discovery occurs before the tether.
    observations = {"go": False, "network_before_go": 0, "network_after_go": 0,
                    "client_or_auth_before_go": 0, "client_or_auth_after_go": 0,
                    "sensitive_environment_present": any(name.startswith(("DEEPSEEK", "OPENAI", "PGPASSWORD", "DATABASE_URL"))
                        or name == "GOOGLE_APPLICATION_CREDENTIALS" for name in os.environ)}

    def audit(name: str, _args: tuple) -> None:
        if name in {"socket.__new__", "socket.connect", "socket.getaddrinfo"}:
            key = "network_after_go" if observations["go"] else "network_before_go"
            observations[key] += 1
            if not observations["go"]:
                raise RuntimeError("network_before_parent_GO_forbidden")

    def profile(frame: object, event: str, _arg: object) -> None:
        if event != "call":
            return
        module = frame.f_globals.get("__name__", "")
        name = frame.f_code.co_name
        guarded = ((module == "google.cloud.storage.client" and name == "__init__")
                   or (module.startswith("google.auth") and name in {"default", "refresh", "ping", "is_on_gce"})
                   or (module == "google.cloud._http" and name == "api_request"))
        if guarded:
            key = "client_or_auth_after_go" if observations["go"] else "client_or_auth_before_go"
            observations[key] += 1
            if not observations["go"]:
                raise RuntimeError("client_or_auth_before_parent_GO_forbidden")

    sys.addaudithook(audit)
    sys.setprofile(profile)
    from .io_runtime import run

    return run(sys.argv[4], parent, observations)


if __name__ == "__main__":
    raise SystemExit(main())
