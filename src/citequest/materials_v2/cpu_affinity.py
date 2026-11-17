"""Topology-aware selection inside the existing Windows logical-CPU ceiling.

Uses Microsoft's CPU Set enumeration, not assumptions about even/odd CPUs.
No CPU is enabled, unparked or added to the process's allowed affinity.
"""
from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import asdict, dataclass

from .text import MaterialError


@dataclass(frozen=True)
class CpuSet:
    group: int
    logical: int
    core: int
    efficiency_class: int
    flags: int = 0


def system_cpu_sets() -> list[CpuSet]:
    if sys.platform != "win32":
        raise MaterialError("WINDOWS_CPU_TOPOLOGY_UNAVAILABLE")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    function = kernel.GetSystemCpuSetInformation
    function.argtypes = (ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG), wintypes.HANDLE, wintypes.ULONG)
    function.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    process = kernel.GetCurrentProcess()
    length = wintypes.ULONG()
    function(None, 0, ctypes.byref(length), process, 0)
    for _ in range(3):
        if not 20 <= length.value <= 1024 * 1024:
            raise MaterialError("WINDOWS_CPU_TOPOLOGY_SIZE_INVALID")
        buffer = ctypes.create_string_buffer(length.value)
        capacity = length.value
        if function(buffer, capacity, ctypes.byref(length), process, 0):
            if length.value > capacity:
                raise MaterialError("WINDOWS_CPU_TOPOLOGY_SIZE_INVALID")
            raw = buffer.raw[:length.value]
            break
        if ctypes.get_last_error() != 122:
            raise MaterialError("WINDOWS_CPU_TOPOLOGY_UNAVAILABLE")
    else:
        raise MaterialError("WINDOWS_CPU_TOPOLOGY_UNSTABLE")
    result, offset = [], 0
    while offset < len(raw):
        size = int.from_bytes(raw[offset:offset + 4], "little")
        kind = int.from_bytes(raw[offset + 4:offset + 8], "little")
        if size < 8 or offset + size > len(raw):
            raise MaterialError("WINDOWS_CPU_TOPOLOGY_RECORD_INVALID")
        if kind == 0:
            if size < 20:
                raise MaterialError("WINDOWS_CPU_TOPOLOGY_RECORD_INVALID")
            item = raw[offset:offset + size]
            result.append(CpuSet(int.from_bytes(item[12:14], "little"), item[14], item[15], item[18], item[19]))
        offset += size
    if not result or any(v.group != 0 for v in result):
        raise MaterialError("WINDOWS_SINGLE_PROCESSOR_GROUP_PROFILE_REQUIRED")
    return result


def select_affinity(allowed: list[int], count: int, topology: list[CpuSet]) -> tuple[list[int], dict]:
    if type(count) is not int or count < 1 or len(set(allowed)) != len(allowed) or any(type(v) is not int or not 0 <= v < 64 for v in allowed):
        raise MaterialError("INVALID_LOGICAL_CPU_AFFINITY")
    if len({(v.group, v.logical) for v in topology}) != len(topology) or any(v.group != 0 for v in topology):
        raise MaterialError("WINDOWS_CPU_TOPOLOGY_RECORD_INVALID")
    # Parked CPUs and CPUs reserved exclusively for another process are not
    # silently re-enabled. Querying the current process sets target bit 2.
    candidates = [v for v in topology if v.logical in allowed and not v.flags & 1 and not (v.flags & 2 and not v.flags & 4)]
    candidates.sort(key=lambda v: (-v.efficiency_class, v.core, v.logical))
    chosen, cores = [], set()
    for value in candidates:
        if value.core not in cores and len(chosen) < count:
            chosen.append(value.logical)
            cores.add(value.core)
    for value in candidates:
        if value.logical not in chosen and len(chosen) < count:
            chosen.append(value.logical)
    if len(chosen) != count:
        raise MaterialError("INSUFFICIENT_LOGICAL_CPU_AFFINITY")
    return chosen, {"policy": "windows-cpu-set-distinct-core-first/1", "allowed_affinity": allowed,
                    "selected_topology": [asdict(v) for v in candidates if v.logical in chosen],
                    "distinct_physical_cores": len({v.core for v in candidates if v.logical in chosen}),
                    "logical_cpu_limit_unchanged": count}
