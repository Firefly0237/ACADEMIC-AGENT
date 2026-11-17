"""Separate non-model export queue consumer; never migrates or generates prose."""

# ruff: noqa: T201 -- process entrypoint emits bounded operational state only
from __future__ import annotations

import argparse
import json
import signal
import threading
from pathlib import Path
from uuid import uuid4

from .runtime import configured, create_process


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    process = None
    stopped = threading.Event()

    def stop(_signum: int, _frame: object) -> None:
        stopped.set()

    for name in (signal.SIGINT, signal.SIGTERM):
        signal.signal(name, stop)
    try:
        process = create_process(configured(args.config), worker_id="export-" + str(uuid4()))
        while not stopped.is_set():
            result = process.worker.run_once()
            if result is not None or args.once:
                print(json.dumps(result or {"status": "idle", "actual_model_requests": 0}), flush=True)
            if args.once:
                return 0 if result is None or result.get("state") == "succeeded" else 1
            stopped.wait(1 if result is None else 0.05)
    except Exception as error:
        # Configuration/driver errors may carry DSNs, source text or paths.
        # A retained per-job failure is already stored by the Worker. This
        # operational failure creates no synthetic success and no hidden retry.
        print(json.dumps({"status": "unavailable", "error_type": type(error).__name__, "actual_model_requests": 0}))
        return 2
    finally:
        if process is not None:
            process.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
