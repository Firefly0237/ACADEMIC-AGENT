"""Linux process limits, separate from deployment cgroup memory/CPU policy.

RLIMIT_AS bounds address space (not RSS). A service manager/container should
also set its physical memory/CPU limits; the accompanying systemd unit does.
The parent retains an absolute per-call deadline and kills only its own child.
"""
from __future__ import annotations

import ctypes
import os
import signal
import sys
from pathlib import Path

from citequest.agents_v3.contracts import AgentError

ADDRESS_SPACE_BYTES = 8 * 1024**3


def apply_child_limits(parent_pid: int) -> None:
    if sys.platform != "linux":
        raise AgentError("decisionrag_Linux_process_profile_required")
    import resource
    os.setsid()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        raise AgentError("decisionrag_parent_death_guard_failed")
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    files = 256 if hard == resource.RLIM_INFINITY else min(256, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (files, files))
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise AgentError("decisionrag_CPU_affinity_unavailable")
    os.sched_setaffinity(0, set(available[:2]))


class LinuxProcessHandle:
    def __init__(self, pid: int):
        self.pid = pid

    def observation(self) -> dict:
        import resource
        limits = resource.prlimit(self.pid, resource.RLIMIT_AS)
        if limits != (ADDRESS_SPACE_BYTES, ADDRESS_SPACE_BYTES):
            raise AgentError("decisionrag_Linux_address_space_limit_changed")
        status = Path(f"/proc/{self.pid}/status").read_text()
        values = {}
        for line in status.splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmPeak", "VmHWM"}:
                values[key] = int(value.split()[0]) * 1024
        return {"memory_limit_kind": "Linux_RLIMIT_AS_address_space_not_RSS",
            "memory_limit_bytes": ADDRESS_SPACE_BYTES, "observed_process_bytes": values,
            "CPU_affinity_count": len(os.sched_getaffinity(self.pid)),
            "physical_memory_limit_owner": "deployment_cgroup_or_service_manager"}

    def close(self) -> None:
        pass
