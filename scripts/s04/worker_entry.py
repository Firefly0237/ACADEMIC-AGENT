"""Minimal gate: no parser/file access before parent assigns Windows Job limits."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    # These are trusted parent control bytes, not document text.
    line = sys.stdin.buffer.readline(262144)
    request = json.loads(line)
    if len(sys.argv) != 2 or request.get("start_token") != sys.argv[1]:
        raise ValueError("worker_start_gate_mismatch")
    allowed = {Path(value).resolve() for value in request["allowed_sys_path"]}
    if not sys.flags.no_site or any(Path(value).resolve() not in allowed for value in sys.path):
        raise ValueError("worker_python_search_path_mismatch")
    from citequest.materials_v2.worker import execute

    return execute(request)


if __name__ == "__main__":
    raise SystemExit(main())
