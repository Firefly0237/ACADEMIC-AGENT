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


def deployment_limits() -> dict:
    """Observe external controls; an address-space limit is never called RSS."""
    base = Path("/sys/fs/cgroup")
    roots = []
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                candidate = (base / line[3:].lstrip("/")).resolve()
                if candidate.is_relative_to(base):
                    roots.append(candidate)
    except OSError:
        pass
    roots.append(base)
    for root in roots:
        if not (root / "memory.max").is_file():
            continue
        memory = (root / "memory.max").read_text().strip()
        cpu = (root / "cpu.max").read_text().strip().split() if (root / "cpu.max").is_file() else []
        return {"external_memory_limit_bytes": None if memory == "max" else int(memory),
            "external_CPU_quota": int(cpu[0]) / int(cpu[1]) if len(cpu) == 2 and cpu[0] != "max" else None,
            "external_memory_current_bytes": int((root / "memory.current").read_text()) if (root / "memory.current").is_file() else None,
            "external_memory_peak_bytes": int((root / "memory.peak").read_text()) if (root / "memory.peak").is_file() else None,
            "physical_memory_limit_owner": "deployment_cgroup_or_service_manager"}
    return {"external_memory_limit_bytes": None, "external_CPU_quota": None,
        "physical_memory_limit_owner": "not_observed"}


def apply_child_limits(parent_pid: int, *, cpu_threads: int = 2,
        address_space_bytes: int = ADDRESS_SPACE_BYTES, required_memory_budget_bytes: int | None = None) -> None:
    if sys.platform != "linux":
        raise AgentError("decisionrag_Linux_process_profile_required")
    import resource
    os.setsid()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent_pid:
        raise AgentError("decisionrag_parent_death_guard_failed")
    if (type(cpu_threads) is not int or not 1 <= cpu_threads <= 64 or type(address_space_bytes) is not int
            or not 4 * 1024**3 <= address_space_bytes <= 1024 * 1024**3):
        raise AgentError("decisionrag_Linux_resource_profile_invalid")
    resource.setrlimit(resource.RLIMIT_AS, (address_space_bytes, address_space_bytes))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    files = 256 if hard == resource.RLIM_INFINITY else min(256, hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (files, files))
    available = sorted(os.sched_getaffinity(0))
    if not available:
        raise AgentError("decisionrag_CPU_affinity_unavailable")
    if required_memory_budget_bytes is not None:
        observed = deployment_limits()
        if observed["external_memory_limit_bytes"] is None or observed["external_memory_limit_bytes"] < required_memory_budget_bytes:
            raise AgentError("decisionrag_deployment_memory_below_profile_or_unbounded")
        if (len(available) < cpu_threads or observed["external_CPU_quota"] is not None
                and observed["external_CPU_quota"] < cpu_threads):
            raise AgentError("decisionrag_deployment_CPU_below_profile")
    os.sched_setaffinity(0, set(available[:cpu_threads]))


class LinuxProcessHandle:
    def __init__(self, pid: int, *, address_space_bytes: int = ADDRESS_SPACE_BYTES, cpu_threads: int | None = None):
        self.pid, self.address_space_bytes, self.cpu_threads = pid, address_space_bytes, cpu_threads

    def observation(self) -> dict:
        import resource
        limits = resource.prlimit(self.pid, resource.RLIMIT_AS)
        if limits != (self.address_space_bytes, self.address_space_bytes):
            raise AgentError("decisionrag_Linux_address_space_limit_changed")
        affinity = len(os.sched_getaffinity(self.pid))
        if self.cpu_threads is not None and affinity != self.cpu_threads:
            raise AgentError("decisionrag_Linux_CPU_affinity_changed")
        status = Path(f"/proc/{self.pid}/status").read_text()
        values = {}
        for line in status.splitlines():
            key, _, value = line.partition(":")
            if key in {"VmRSS", "VmPeak", "VmHWM"}:
                values[key] = int(value.split()[0]) * 1024
        return {"memory_limit_kind": "Linux_RLIMIT_AS_address_space_not_RSS",
            "memory_limit_bytes": self.address_space_bytes, "observed_process_bytes": values,
            "CPU_affinity_count": affinity, **deployment_limits()}

    def close(self) -> None:
        pass
