"""Explicit bounded physical-readiness checks after a service restart.

Only missing IDs may be observed again during this declared startup window.
No embedding, flush, upsert or relaxed scalar/vector check is performed. A
physical persistence proof grants neither source adoption nor search visibility.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime

from .native import IncompleteReadbackError, NativeMilvus
from .rows import canonical_sha


def verify_startup_readiness(
    collections: list[dict],
    native_factory: Callable[[], NativeMilvus],
    *,
    maximum_seconds: float,
    interval_seconds: float = 2,
    maximum_attempts: int = 90,
    checkpoint: Callable[[dict], None],
) -> dict:
    if (
        type(collections) is not list
        or not 1 <= len(collections) <= 100
        or len({item["collection"] for item in collections}) != len(collections)
    ):
        raise ValueError("nonempty_bounded_unique_expected_collections_required")
    if (
        type(maximum_seconds) not in {int, float}
        or not 20 <= maximum_seconds <= 300
        or type(interval_seconds) not in {int, float}
        or not 1 <= interval_seconds <= 5
        or type(maximum_attempts) is not int
        or not 1 <= maximum_attempts <= 90
    ):
        raise ValueError("explicit_bounded_startup_readiness_profile_required")
    started = time.monotonic()
    deadline = started + maximum_seconds
    attempts = 0
    results = []
    for entry in collections:
        if not entry["rows"]:
            raise ValueError("nonempty_expected_physical_collection_required")
        while True:
            remaining = deadline - time.monotonic()
            if remaining < 20 or attempts >= maximum_attempts:
                raise TimeoutError("startup_physical_data_not_ready_within_declared_window")
            attempts += 1
            attempt_started = time.monotonic()
            record = {
                "attempt": attempts,
                "collection": entry["collection"],
                "at_start": datetime.now(UTC).isoformat(),
                "elapsed_start_seconds": attempt_started - started,
                "remaining_start_seconds": remaining,
            }
            native = None
            missing = False
            try:
                native = native_factory()
                native.absolute_deadline = deadline
                native.call(
                    "load_collection",
                    collection_name=entry["collection"],
                    seconds=min(60, deadline - time.monotonic() - 2),
                    maximum_calls=12,
                )
                config = native.collection_proof(entry["collection"])
                if config["configuration_sha256"] != entry["configuration_sha256"]:
                    raise ValueError("startup_native_configuration_changed")
                entire = native.scan_all(
                    entry["collection"], entry["rows"], seconds=min(60, deadline - time.monotonic() - 2)
                )
                fixed = []
                for offset in range(0, len(entry["rows"]), 100):
                    rows = entry["rows"][offset : offset + 100]
                    _, proof = native.read_ids(
                        entry["collection"], rows, seconds=min(30, deadline - time.monotonic() - 2)
                    )
                    if not proof["complete"]:
                        # Do not accept a different scalar get view as ready.
                        raise ValueError("startup_fixed_ID_get_incomplete_after_complete_iterator")
                    fixed.append(
                        {
                            "ids": [row["chunk_id"] for row in rows],
                            "vector_values_sha256": proof["vector_values_sha256"],
                            "rows_sha256": proof["rows_sha256"],
                        }
                    )
                vectors = canonical_sha({"fixed_ID_batches": fixed})
                if (
                    entry.get("expected_fixed_ID_vectors_and_scalars_sha256") is not None
                    and vectors != entry["expected_fixed_ID_vectors_and_scalars_sha256"]
                ):
                    raise ValueError("startup_persisted_vector_values_changed")
                if time.monotonic() >= deadline:
                    raise TimeoutError("late_startup_readback_not_accepted")
                result = {
                    "collection": entry["collection"],
                    "generation_id": entry["generation_id"],
                    "coverage": entry["coverage"],
                    "whole_physical_collection": entire,
                    "fixed_ID_vectors_and_scalars_sha256": vectors,
                    "configuration_sha256": config["configuration_sha256"],
                }
                record.update(
                    outcome="full_expected_physical_collection_verified",
                    found_count=entire["matched"],
                    expected_count=len(entry["rows"]),
                )
                results.append(result)
            except IncompleteReadbackError as exc:
                missing = True
                record.update(outcome="missing_IDs_not_ready", observation=exc.observation)
            except Exception as exc:
                record.update(outcome="hard_failure_not_retried", error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                if native is not None:
                    native.close()
                record.update(
                    at_end=datetime.now(UTC).isoformat(),
                    elapsed_end_seconds=time.monotonic() - started,
                    remaining_end_seconds=max(0, deadline - time.monotonic()),
                )
                checkpoint(record)
            if not missing:
                break
            if deadline - time.monotonic() < 20 + interval_seconds:
                raise TimeoutError("startup_missing_IDs_remain_at_declared_window_boundary")
            time.sleep(interval_seconds)
    return {
        "status": "physical_data_ready",
        "maximum_seconds": maximum_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "attempts": attempts,
        "collections": results,
        "only_missing_IDs_reobserved": True,
        "embedding_calls": 0,
        "upsert_calls": 0,
        "flush_calls": 0,
        "search_publication_permission_granted": False,
    }
