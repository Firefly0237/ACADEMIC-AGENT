"""Additional no-subprocess policy after trusted Python starts, using libseccomp."""

from __future__ import annotations

import ctypes
import ctypes.util
import errno


def no_subprocesses() -> dict:
    # Docker's default seccomp stays active. This filter only further restricts it.
    name = ctypes.util.find_library("seccomp")
    if not name:
        raise RuntimeError("libseccomp_required_no_unfiltered_fallback")
    library = ctypes.CDLL(name, use_errno=True)

    class Comparison(ctypes.Structure):
        _fields_ = [
            ("arg", ctypes.c_uint),
            ("op", ctypes.c_uint),
            ("datum_a", ctypes.c_uint64),
            ("datum_b", ctypes.c_uint64),
        ]

    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add_array.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(Comparison),
    ]
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    context = library.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW; stacked on Docker's default policy.
    if not context:
        raise RuntimeError("seccomp_initialization_failed")
    try:
        for syscall in ["fork", "vfork", "execve", "execveat", "clone3", "clone"]:
            number = library.seccomp_syscall_resolve_name(syscall.encode())
            if number < 0:
                raise RuntimeError("required_syscall_not_resolved:" + syscall)
            code = errno.ENOSYS if syscall == "clone3" else errno.EPERM
            comparison = Comparison(0, 7, 0x10000, 0)  # MASKED_EQ: clone without CLONE_THREAD.
            pointer = ctypes.pointer(comparison) if syscall == "clone" else None
            count = 1 if syscall == "clone" else 0
            if library.seccomp_rule_add_array(context, 0x00050000 | code, number, count, pointer) != 0:
                raise RuntimeError("seccomp_rule_failed:" + syscall)
        if library.seccomp_load(context) != 0:
            raise RuntimeError("seccomp_load_failed")
    finally:
        library.seccomp_release(context)
    return {
        "backend": "libseccomp stacked on Docker default",
        "process_creation": "denied",
        "clone_thread": "allowed subject to pids cgroup",
        "clone3": "ENOSYS for pthread clone fallback",
        "exec_after_start": "denied",
        "unfiltered_fallback": False,
    }
