"""Immutable image entry; no database or model API credentials are accepted."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
from pathlib import Path

sys.path[:] = [
    "/app/src",
    "/usr/local/lib/python312.zip",
    "/usr/local/lib/python3.12",
    "/usr/local/lib/python3.12/lib-dynload",
    "/usr/local/lib/python3.12/site-packages",
]


def main() -> int:
    from citequest.materials_v2.worker import execute
    from citequest.materials_v3.fixed_profiles import IMAGE_PROFILE_RELATIVES
    from citequest.materials_v3.seccomp import no_subprocesses

    raw = Path("/app/input/request.json").read_bytes()
    if len(raw) > 262144:
        raise ValueError("bounded_trusted_control_required")
    request = json.loads(raw)
    profile_path = Path(request["profile"])
    if profile_path not in {Path("/app") / relative for relative in IMAGE_PROFILE_RELATIVES}:
        raise ValueError("fixed_image_profile_required")
    profile_raw = profile_path.read_bytes()
    if hashlib.sha256(profile_raw).hexdigest() != request["profile_sha256"]:
        raise ValueError("exact_profile_binding_required")
    forbidden = [
        key for key in os.environ if any(term in key.upper() for term in ["SECRET", "PASSWORD", "API_KEY", "DSN"])
    ]
    if forbidden:
        raise ValueError("worker_credential_environment_rejected")
    request.update(
        root="/app", output="/app/output", source_path="/app/input/material", profile_json=profile_raw.decode()
    )
    sandbox = no_subprocesses()

    def terminate_orphan(_number: int, _frame: object) -> None:
        os._exit(124)

    seconds = request["orphan_deadline_seconds"]
    profile = json.loads(profile_raw)
    if type(seconds) not in {int, float} or not 0 < seconds <= profile["deadline_seconds"]:
        raise ValueError("bounded_orphan_deadline_required")
    signal.signal(signal.SIGALRM, terminate_orphan)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    Path("/app/output/linux-sandbox.json").write_text(json.dumps(sandbox))

    def cgroup_sample() -> dict:
        return {
            name: (Path("/sys/fs/cgroup") / name).read_text().strip()
            for name in ["memory.current", "memory.peak", "memory.events", "cpu.stat", "pids.current"]
        }

    Path("/app/output/linux-cgroup-before.json").write_text(json.dumps(cgroup_sample()))
    try:
        return execute(request)
    finally:
        Path("/app/output/linux-cgroup-after.json").write_text(json.dumps(cgroup_sample()))


if __name__ == "__main__":
    raise SystemExit(main())
