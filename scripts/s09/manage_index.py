"""Trusted operator/worker CLI; these identity/config options are never model tools."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from uuid import UUID
from citequest.material_evidence_v3.native import index_from_config
from citequest.v2.domain import Actor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-collection", "enqueue", "run-once", "recover-expired", "reconcile", "preflight", "request-delete", "delete-once", "discover-ready", "discover-withdrawn", "worker-window"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--owner", type=UUID)
    parser.add_argument("--file", action="append", type=UUID, default=[])
    parser.add_argument("--generation", type=UUID)
    parser.add_argument("--worker", default="material-index-worker")
    parser.add_argument("--window-seconds", type=int, default=900)
    parser.add_argument("--max-dispatches", type=int, default=1000)
    args = parser.parse_args()
    index = index_from_config(args.config.resolve(strict=True))
    try:
        if args.command == "prepare-collection":
            result = index.prepare_collection()
        elif args.command == "run-once":
            result = index.run_one(args.worker)
        elif args.command == "delete-once":
            result = index.run_delete_one(args.worker)
        elif args.command == "recover-expired":
            result = index.recover_expired()
        elif args.command == "discover-ready":
            result = index.discover_ready()
        elif args.command == "discover-withdrawn":
            result = index.discover_withdrawn()
        elif args.command == "worker-window":
            if not 1 <= args.window_seconds <= 1800 or not 1 <= args.max_dispatches <= 10000:
                parser.error("worker window/dispatch limits exceeded")
            deadline, dispatched = time.monotonic()+args.window_seconds, 0
            while time.monotonic() < deadline and dispatched < args.max_dispatches:
                index.recover_expired()
                index.discover_withdrawn()
                index.discover_ready()
                deletion = index.run_delete_one(args.worker)
                delivery = index.run_one(args.worker)
                if deletion["state"] != "idle" or delivery["state"] != "idle":
                    dispatched += 1
                    print(json.dumps({"deletion": deletion, "delivery": delivery}), flush=True)
                else:
                    time.sleep(min(1, max(0, deadline-time.monotonic())))
            result = {"worker_window_ended": True, "dispatches": dispatched, "unfinished_jobs_retained": True}
        elif args.command == "reconcile":
            if args.generation is None:
                parser.error("reconcile requires --generation")
            result = index.reconcile(args.generation, args.worker)
        else:
            if args.owner is None or not args.file:
                parser.error("owner-scoped command requires --owner and --file")
            actor = Actor(args.owner)
            if args.command == "enqueue":
                result = [index.enqueue(actor, file_id) for file_id in args.file]
            elif args.command == "request-delete":
                result = [index.request_delete(actor, file_id) for file_id in args.file]
            else:
                index.preflight(actor, args.file)
                result = {"ready": True, "model_calls": 0, "embedding_calls": 0}
        print(json.dumps(result))
    finally:
        index.native.close()


if __name__ == "__main__":
    main()
