"""Fixed trusted PG resource owner. No model/GCS/parser code or durable DSN."""
from __future__ import annotations

import json
import math
import os
import select
import sys
import time

from .io_worker import tether_parent


def main() -> int:
    if len(sys.argv) != 3 or sys.argv[1] != "--parent-pid":
        return 2
    parent = int(sys.argv[2])
    tether_parent(parent)
    os.umask(0o077)
    import resource

    for kind, value in ((resource.RLIMIT_AS, 256 * 1024**2), (resource.RLIMIT_NOFILE, 32),
                        (resource.RLIMIT_CORE, 0), (resource.RLIMIT_CPU, 120)):
        _, hard = resource.getrlimit(kind)
        value = value if hard == resource.RLIM_INFINITY else min(hard, value)
        resource.setrlimit(kind, (value, value))
    raw = bytearray()
    while b"\n" not in raw:
        part = os.read(0, min(4096, 40001 - len(raw)))
        if not part or len(raw) + len(part) > 40000:
            return 2
        raw.extend(part)
    job = json.loads(raw)
    if (set(job) != {"version", "parent", "expires", "keys", "dsn"}
            or job["version"] != "citequest-pg-resource-job/1" or job["parent"] != parent
            or type(job["expires"]) is not float or not math.isfinite(job["expires"])
            or not isinstance(job["dsn"], str) or not 0 < len(job["dsn"].encode()) <= 32768
            or not isinstance(job["keys"], list) or len(job["keys"]) != 2
            or any(type(key) is not int or not -(2**63) <= key < 2**63 for key in job["keys"])
            or len(set(job["keys"])) != 2):
        return 2
    remaining = job["expires"] - time.monotonic()
    if remaining <= 0 or remaining > 120 or os.getppid() != parent:
        return 2
    import psycopg

    # libpq timeout is an additional guard; parent kill+wait bounds even DNS,
    # TCP handshake and cancellation paths beyond the original deadline.
    with psycopg.connect(job["dsn"], autocommit=True, connect_timeout=max(2, math.ceil(remaining))) as connection:
        remaining = job["expires"] - time.monotonic()
        if remaining <= 0:
            return 3
        connection.execute("SELECT set_config('statement_timeout',%s,false)", (str(max(1, int(remaining * 1000))),))
        key = None
        for candidate in job["keys"]:
            if connection.execute("SELECT pg_try_advisory_lock(%s)", (candidate,)).fetchone()[0] is True:
                key = candidate
                break
        if time.monotonic() >= job["expires"]:
            return 3
        result = {"version": "citequest-pg-resource-ready/1", "parent": parent, "pid": os.getpid(),
                  "status": "acquired" if key is not None else "busy", "key": key,
                  "backend_pid": connection.info.backend_pid}
        print(json.dumps(result, separators=(",", ":")), flush=True)  # noqa: T201
        if key is None:
            return 0
        # Keep the session until the parent confirms SDK termination. No lease
        # expiry unlocks while a possibly-live SDK process still owns the job.
        while True:
            readable, _, _ = select.select([0, connection.fileno()], [], [], 0.1)
            if 0 in readable:
                return 0  # EOF: parent has confirmed SDK exit before close.
            if connection.fileno() in readable:
                connection.pgconn.consume_input()
                if connection.pgconn.status != psycopg.pq.ConnStatus.OK:
                    return 4


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        # Do not print exception text: libpq errors can contain DSN/host fields.
        raise SystemExit(2) from None
