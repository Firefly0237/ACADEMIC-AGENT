"""Bounded read-only discovery after a parent/host restart. Never retries/GCs."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from .broker_protocol import rooted, strict_object
from .private_spool_client import bound_stop_received


def _uuid_name(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def discover(root: Path, *, limit: int = 100, target: tuple[UUID, UUID] | None = None,
             include_released: bool = False) -> dict:
    """Trusted operator metadata, not a user-facing source-list endpoint.

    A previous parent PID is not proof of liveness or death (PID reuse/host
    identity). Nothing is deleted, dispatched or republished by this function.
    """
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("bounded_private_input_inventory_limit_required")
    if (type(include_released) is not bool or (target is not None
            and (len(target) != 2 or any(not isinstance(value, UUID) for value in target)))):
        raise ValueError("typed_execution_inventory_target_required")
    root = root.resolve(strict=True)
    artifacts = rooted(root, "data/runtime-s04-linux/artifacts")
    items, partial, inspected = [], False, 0
    if not artifacts.exists():
        return {"items": [], "partial_scan": False, "read_only": True}
    ingestions = [artifacts / str(target[0])] if target is not None else artifacts.iterdir()
    for ingestion in ingestions:
        if ingestion.is_symlink() or not ingestion.is_dir() or not _uuid_name(ingestion.name):
            continue
        executions = [ingestion / str(target[1])] if target is not None else ingestion.iterdir()
        for execution in executions:
            if execution.is_symlink() or not execution.is_dir() or not _uuid_name(execution.name):
                continue
            inspected += 1
            if inspected > limit:
                partial = True
                break
            source = rooted(root, (execution / "input/source.bin").relative_to(root).as_posix())
            ownership_path = execution / "input-ownership.json"
            if not source.exists() and not ownership_path.exists():
                continue
            item = {"ingestion_id": ingestion.name, "execution_id": execution.name,
                    "input_exists": source.exists(), "stop_evidence_observed": False,
                    "parent_liveness": "not_verified", "action": "retain_no_dispatch_no_publication"}
            try:
                if ownership_path.is_symlink() or ownership_path.stat().st_size > 8192:
                    raise ValueError("bounded_owned_input_record_required")
                ownership = json.loads(ownership_path.read_bytes(), object_pairs_hook=strict_object)
                if (ownership.get("version") != "citequest-private-input/1"
                        or ownership.get("ingestion_id") != ingestion.name
                        or ownership.get("execution_id") != execution.name
                        or ownership.get("source_relative") != source.relative_to(root).as_posix()):
                    raise ValueError("input_identity_record_mismatch")
                item["ownership_state"] = ownership.get("state")
                item["source_sha256"] = ownership.get("source_sha256")
                item["recorded_parent_pid"] = ownership.get("parent_pid")
                seen_requests, all_stopped = 0, True
                for sequence in (1, 2):
                    state_path = execution / f"parser-request-state-{sequence}.json"
                    if not state_path.exists():
                        continue
                    if state_path.is_symlink() or state_path.stat().st_size > 8192:
                        raise ValueError("bounded_request_state_required")
                    state = json.loads(state_path.read_bytes(), object_pairs_hook=strict_object)
                    seen_requests += 1
                    if (state.get("version") != "citequest-private-parser-request/1"
                            or state.get("nonce") != ownership.get("nonce") or state.get("sequence") != sequence
                            or state.get("ingestion_id") != ingestion.name or state.get("execution_id") != execution.name):
                        raise ValueError("request_identity_record_mismatch")
                    exchange = rooted(root, state["exchange_relative"])
                    if not exchange.is_relative_to(root / "data/runtime-s04-linux/exchanges"):
                        raise ValueError("owned_exchange_required")
                    raw_path = exchange / f"request-{sequence}.json"
                    if raw_path.is_symlink() or raw_path.stat().st_size > 262144:
                        raise ValueError("bounded_original_request_required")
                    raw = raw_path.read_bytes()
                    message = json.loads(raw, object_pairs_hook=strict_object)
                    request_sha = hashlib.sha256(raw).hexdigest()
                    output = rooted(root, state["output_relative"])
                    expected_output = execution / ("parse" if sequence == 1 else "chunk")
                    if (message.get("nonce") != state["nonce"] or message.get("sequence") != sequence
                            or output != expected_output or message.get("output_relative") != state["output_relative"]
                            or state.get("request_sha256") not in {None, request_sha}
                            or message.get("profile_sha256") != state["parser_profile_sha256"]):
                        raise ValueError("published_request_record_mismatch")
                    if sequence == 1 and (message.get("source_relative") != ownership["source_relative"]
                            or message.get("request", {}).get("source_sha256") != ownership["source_sha256"]
                            or state.get("source_sha256") != ownership["source_sha256"]):
                        raise ValueError("published_original_File_hash_mismatch")
                    stopped = bound_stop_received(exchange / f"response-{sequence}.json", output / "receipt.json",
                        nonce=state["nonce"], request_sha256=request_sha,
                        parser_profile_sha256=state["parser_profile_sha256"], parser_image_id=state["parser_image_id"])
                    all_stopped = all_stopped and stopped
                    item["request_state"] = state.get("state")
                item["stop_evidence_observed"] = seen_requests > 0 and all_stopped
                if (not include_released and not source.exists() and ownership.get("state") in {
                        "input_released_after_stop_or_zero_dispatch", "input_released_after_late_stop"}
                        and (seen_requests == 0 or all_stopped)):
                    continue
                if item["stop_evidence_observed"]:
                    item["action"] = "bound_stop_observed_no_cleanup_or_publication_by_inventory"
            except (OSError, ValueError, TypeError, KeyError) as exc:
                item["control_status"] = "incomplete_or_malformed_retain"
                item["error_type"] = type(exc).__name__
            items.append(item)
        if partial:
            break
    return {"version": "citequest-private-input-inventory/1", "observed_at": datetime.now(UTC).isoformat(),
            "items": items, "inspected_executions": min(inspected, limit), "partial_scan": partial,
            "read_only": True, "dispatches": 0, "deletions": 0, "publications": 0,
            "parent_liveness": "not_verified", "fleet_physical_concurrency": "not_verified"}
