"""Windows Job Object limits for the isolated local CPU probability process.

Uses documented jobapi2 and winnt contracts. No other process is assigned or
modified. Memory figures are committed-byte peaks, not resident set size.
"""
from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


class BasicLimits(ctypes.Structure):
    _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
        ("flags", wintypes.DWORD), ("minimum_working_set", ctypes.c_size_t),
        ("maximum_working_set", ctypes.c_size_t), ("active_process_limit", wintypes.DWORD),
        ("affinity", ctypes.c_size_t), ("priority_class", wintypes.DWORD),
        ("scheduling_class", wintypes.DWORD)]


class IOCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_ulonglong) for name in
        ("read_operations", "write_operations", "other_operations", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(ctypes.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IOCounters), ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t), ("peak_process_memory", ctypes.c_size_t),
        ("peak_job_memory", ctypes.c_size_t)]


class WindowsProcessJob:
    def __init__(self, pid: int, memory_bytes: int = 3758096384):
        if os.name != "nt" or memory_bytes != 3758096384:
            raise RuntimeError("fixed_Windows_local_CPU_job_required")
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel.CloseHandle.restype = wintypes.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = wintypes.BOOL
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.SetInformationJobObject.restype = wintypes.BOOL
        self.kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
            ctypes.c_void_p, wintypes.DWORD, ctypes.c_void_p]
        self.kernel.QueryInformationJobObject.restype = wintypes.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise RuntimeError("decisionrag_process_job_create_failed")
        limits = ExtendedLimits()
        # Parent owns the job, so even abrupt parent exit kills this exact
        # isolated child. No unrelated process is assigned to the job.
        limits.basic.flags = 0x100 | 0x2000
        limits.process_memory_limit = memory_bytes
        child = self.kernel.OpenProcess(0x100 | 0x1, False, pid)
        assigned = False
        try:
            assigned = bool(child and self.kernel.SetInformationJobObject(self.handle, 9,
                ctypes.byref(limits), ctypes.sizeof(limits)) and self.kernel.AssignProcessToJobObject(self.handle, child))
        finally:
            if child:
                self.kernel.CloseHandle(child)
        if not assigned:
            self.close()
            raise RuntimeError("decisionrag_process_memory_limit_not_applied")

    def observation(self) -> dict:
        limits = ExtendedLimits()
        if not self.kernel.QueryInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits), None):
            raise RuntimeError("decisionrag_process_memory_observation_failed")
        return {"memory_limit_kind": "Windows_Job_Object_process_committed_bytes",
            "memory_limit_bytes": limits.process_memory_limit,
            "peak_committed_bytes": limits.peak_process_memory}

    def close(self) -> None:
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None
