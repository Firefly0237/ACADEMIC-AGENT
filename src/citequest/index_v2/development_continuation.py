"""Bounded development-only continuation of one already sealed pilot.

This module never creates a generation/collection, recompiles a source, publishes
an alias, or replays an unknown write. It does not implement a cloud scheduler.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import select, text

from citequest.data_v2 import models as data

from . import models as m
from .rows import canonical_sha
from .schema import IndexProfile

GENERATION = UUID("2077df19-91b4-46c6-aaf3-0d48115db693")
SNAPSHOT = UUID("6f6e49a4-9e17-5328-9eee-ea55f4dfde82")
FIXED = {
    "generation_id": str(GENERATION),
    "snapshot_id": str(SNAPSHOT),
    "collection_name": "cq_v2_clean_2077df1991b446c6aaf30d48115db693",
    "state": "building",
    "build_scope": "pilot",
    "requested_unit_limit": 5000,
    "unit_count": 5000,
    "empty_unit_count": 1,
    "chunk_count": 5873,
    "membership_sha256": "0e8c44c53526f952a2383a7cd1e159d92f6b630172afea2fe0ff936015301620",
    "expected_manifest_sha256": "49277e2d402e0c238d65c05879f0e4f5ec277192ca3c0a117e1850e3f8d02b47",
    "index_profile_sha256": "5776ccddf754ae975f80db548b322dce6314ebc610e15138fb4df89ec19eece6",
    "tokenizer_profile_sha256": "b517586e082f3bc71d1660d52e6deb734619fc98c193852c72229b66f693c5b7",
}
SOURCE_SEAL = "e7afe3123e3b9b6497a6bb145c4906bf74b6d4fb3c04d9d08d29a75288d22e5d"


def read_approved_profile(path: Path, expected_sha256: str) -> tuple[bytes, dict]:
    """Hash and parse the same captured bytes; never reload the approved path."""
    raw = path.read_bytes()
    if not re.fullmatch(r"[a-f0-9]{64}", expected_sha256) or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("exact_approved_profile_bytes_required")

    def strict_object(pairs: list[tuple[str, object]]) -> dict:
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate_profile_JSON_key_forbidden")
            value[key] = item
        return value

    def reject_constant(value: str) -> None:
        raise ValueError("nonfinite_profile_JSON_constant_forbidden_" + value)

    profile = json.loads(raw, object_pairs_hook=strict_object, parse_constant=reject_constant)
    if type(profile) is not dict:
        raise ValueError("approved_profile_JSON_object_required")
    return raw, profile


def claim_profile_execution(root: Path, profile_sha256: str, output: Path, started_at: str) -> Path:
    """Exclusive local dev evidence, not a distributed production lease."""
    if not re.fullmatch(r"[a-f0-9]{64}", profile_sha256) or not output.resolve().is_relative_to(root.resolve()):
        raise ValueError("owned_exact_profile_execution_claim_required")
    path = root / ("execution-claim-" + profile_sha256 + ".json")
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "profile_sha256": profile_sha256,
                "output": str(output.resolve()),
                "started_at": started_at,
                "consumed_once_even_if_failed": True,
            },
            stream,
            ensure_ascii=False,
            indent=2,
        )
    return path


def read_checkpoint(factory: Any) -> tuple[dict, list[dict], list[dict]]:
    """One consistent READ ONLY SQL snapshot; no leases or state are changed."""
    with factory() as db:
        db.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        db.execute(text("SET LOCAL statement_timeout='15000ms'"))
        if db.execute(text("SELECT current_database(),current_user")).one() != ("citequest_s03", "cq_s03_app"):
            raise ValueError("owned_database_required")
        head = db.execute(text("SELECT version_num FROM cq_app.alembic_version")).scalar_one()
        generation = db.get(m.IndexGeneration, GENERATION)
        source = db.get(data.SourceSnapshot, SNAPSHOT)
        if generation is None or source is None:
            raise ValueError("existing_pilot_and_source_required")
        identity = {
            key: str(getattr(generation, "id" if key == "generation_id" else key))
            if key in {"generation_id", "snapshot_id"}
            else getattr(generation, key)
            for key in FIXED
        }
        if identity != FIXED or IndexProfile().fingerprint() != FIXED["index_profile_sha256"]:
            raise ValueError("unchanged_building_pilot_identity_required")
        if (
            source.security_domain != "public"
            or source.source_namespace != "litsearch_clean"
            or source.purpose != "standard"
            or source.representation_profile != "litsearch-clean-exact/1"
            or source.status != "staged"
            or source.profile_adopted is not True
            or source.seal_sha256 != SOURCE_SEAL
            or canonical_sha(source.seal) != SOURCE_SEAL
            or source.policy_sha256 != generation.policy_sha256
            or source.evaluation_sha256 != generation.evaluation_sha256
            or generation.security_domain != "public"
            or generation.source_root_kind != "metadata_snapshot"
            or generation.server_proof is None
        ):
            raise ValueError("unchanged_adopted_standard_source_and_server_proof_required")
        if (
            db.scalar(
                select(m.IndexCatalog.name)
                .where(
                    (m.IndexCatalog.active_generation_id == GENERATION)
                    | (m.IndexCatalog.pending_intent_id.is_not(None))
                )
                .limit(1)
            )
            is not None
        ):
            raise ValueError("no_active_pilot_or_pending_publication_required")
        chunks = list(db.scalars(select(m.IndexChunk).where(m.IndexChunk.generation_id == GENERATION)))
        row_map = {chunk.chunk_id: chunk.payload for chunk in chunks}
        if (
            len(row_map) != FIXED["chunk_count"]
            or any(canonical_sha(chunk.payload) != chunk.row_sha256 for chunk in chunks)
            or canonical_sha({chunk.chunk_id: chunk.row_sha256 for chunk in chunks})
            != FIXED["expected_manifest_sha256"]
        ):
            raise ValueError("persisted_whole_expectation_corrupt")
        units = list(db.scalars(select(m.IndexUnit).where(m.IndexUnit.generation_id == GENERATION)))
        membership = {
            str(unit.projection_id): {
                "native_id": unit.native_id,
                "projection_sha256": unit.projection_sha256,
                "source_pointer_sha256": unit.source_pointer_sha256,
            }
            for unit in units
        }
        if (
            len(units) != 5000
            or sum(not unit.searchable for unit in units) != 1
            or sum(unit.chunk_count for unit in units) != len(chunks)
            or canonical_sha(membership) != FIXED["membership_sha256"]
            or any(
                unit.source_reference_kind != "metadata_projection" or unit.metadata_projection_id != unit.projection_id
                for unit in units
            )
        ):
            raise ValueError("all_original_source_units_and_empty_record_membership_required")
        batches = list(
            db.scalars(
                select(m.IndexBatch).where(m.IndexBatch.generation_id == GENERATION).order_by(m.IndexBatch.sequence)
            )
        )
        records, assigned, settled_ids, ready_ids = [], set(), set(), set()
        for batch in batches:
            ids = batch.chunk_ids
            if (
                batch.state not in {"ready", "verified"}
                or batch.lease_id is not None
                or batch.lease_expires_at is not None
            ):
                raise ValueError("unknown_or_inflight_requires_separate_reconciliation_no_dispatch")
            if (
                not ids
                or len(ids) > 4
                or len(set(ids)) != len(ids)
                or set(ids) & assigned
                or set(ids) - row_map.keys()
                or batch.batch_manifest["rows"] != {key: canonical_sha(row_map[key]) for key in ids}
                or batch.batch_manifest["sha256"] != canonical_sha(batch.batch_manifest["rows"])
                or batch.batch_manifest["coverage_scope"] != "batch_only"
            ):
                raise ValueError("batch_partition_or_manifest_corrupt")
            if batch.state == "verified" and (
                batch.logical_attempts < 1
                or batch.last_receipt is None
                or batch.last_receipt.get("state") != "verified"
                or batch.last_receipt.get("readback", {}).get("complete") is not True
                or batch.last_receipt["readback"]["rows_sha256"] != batch.batch_manifest["sha256"]
            ):
                raise ValueError("settled_batch_receipt_corrupt")
            if batch.state == "ready" and (batch.logical_attempts != 0 or batch.last_receipt is not None):
                raise ValueError("previously_attempted_ready_batch_requires_reconciliation")
            assigned.update(ids)
            (settled_ids if batch.state == "verified" else ready_ids).update(ids)
            records.append(
                {
                    "id": str(batch.id),
                    "sequence": batch.sequence,
                    "state": batch.state,
                    "chunk_ids": ids,
                    "batch_manifest_sha256": batch.batch_manifest["sha256"],
                    "logical_attempts": batch.logical_attempts,
                    "last_receipt_sha256": canonical_sha(batch.last_receipt) if batch.last_receipt else None,
                }
            )
        if assigned != row_map.keys():
            raise ValueError("all_expected_rows_must_belong_to_exactly_one_batch")
        counts = dict(Counter(batch["state"] for batch in records))
        if counts != {"verified": 956, "ready": 513} or len(settled_ids) != 3824 or len(ready_ids) != 2049:
            raise ValueError("original_checkpoint_counts_changed_prepare_new_reviewed_profile")
        checkpoint = {
            "identity": identity,
            "source_seal_sha256": source.seal_sha256,
            "source_manifest_sha256": source.source_manifest_sha256,
            "policy_sha256": source.policy_sha256,
            "evaluation_sha256": source.evaluation_sha256,
            "server_configuration_sha256": generation.server_proof["configuration_sha256"],
            "creator_id": str(generation.creator_id),
            "batch_counts": counts,
            "settled_rows": len(settled_ids),
            "ready_rows": len(ready_ids),
            "ready_tokens": sum(row_map[key]["token_count"] for key in ready_ids),
            "batches": records,
            "settled_rows_sha256": canonical_sha({key: canonical_sha(row_map[key]) for key in settled_ids}),
            "ready_rows_sha256": canonical_sha({key: canonical_sha(row_map[key]) for key in ready_ids}),
        }
        checkpoint["checkpoint_sha256"] = canonical_sha(checkpoint)
        # Migration head is diagnostic, deliberately not the checkpoint identity:
        # an additive accepted descendant must not invalidate immutable metadata.
        checkpoint["sql_head_observed"] = head
        db.rollback()
    return checkpoint, [row_map[key] for key in sorted(settled_ids)], [row_map[key] for key in sorted(row_map)]


def validate_profile(profile: dict, checkpoint: dict) -> None:
    if checkpoint["checkpoint_sha256"] != canonical_sha(
        {key: value for key, value in checkpoint.items() if key not in {"checkpoint_sha256", "sql_head_observed"}}
    ):
        raise ValueError("checkpoint_content_hash_mismatch")
    required = {
        "schema_version",
        "run_id",
        "kind",
        "segment_window_seconds",
        "dispatch_reserve_seconds",
        "final_readback_maximum_seconds",
        "checkpoint_sha256",
        "identity",
        "source_seal_sha256",
        "source_bindings",
        "historical_runs",
        "prior_failure_retained",
        "automatic_uncertain_replay",
        "verified_rows_reencoded",
        "generation_or_collection_created",
        "publication_performed",
        "cloud_or_full_corpus_or_original_performance_gate_passed",
        "runtime_executable",
    }
    if type(profile) is not dict or set(profile) != required:
        raise ValueError("exact_development_continuation_profile_required")
    if (
        profile["schema_version"] != "cq.development-index-continuation/1"
        or profile["run_id"] != "S05-development-5000-continuation-r1"
        or profile["kind"] != "new_bounded_segment_on_unchanged_existing_building_checkpoint"
        or type(profile["segment_window_seconds"]) is not int
        or not 300 <= profile["segment_window_seconds"] <= 3600
        or profile["dispatch_reserve_seconds"] != 180
        or type(profile["dispatch_reserve_seconds"]) is not int
        or profile["final_readback_maximum_seconds"] != 120
        or type(profile["final_readback_maximum_seconds"]) is not int
        or profile["checkpoint_sha256"] != checkpoint["checkpoint_sha256"]
        or profile["identity"] != FIXED
        or type(profile["identity"]) is not dict
        or any(type(profile["identity"][key]) is not type(value) for key, value in FIXED.items())
        or profile["identity"] != checkpoint["identity"]
        or profile["source_seal_sha256"] != SOURCE_SEAL
        or profile["prior_failure_retained"] is not True
        or any(
            profile[key] is not False
            for key in (
                "automatic_uncertain_replay",
                "verified_rows_reencoded",
                "generation_or_collection_created",
                "publication_performed",
                "cloud_or_full_corpus_or_original_performance_gate_passed",
            )
        )
        or profile["runtime_executable"] != "data/runtime-s05-index-next/env/Scripts/python.exe"
        or type(profile["source_bindings"]) is not dict
        or not profile["source_bindings"]
        or type(profile["historical_runs"]) is not list
        or len(profile["historical_runs"]) != 1
    ):
        raise ValueError("unchanged_profile_limits_identity_and_honest_scope_required")
    history = profile["historical_runs"][0]
    if (
        set(history) != {"path", "sha256", "elapsed_seconds", "status", "counted_as_build_execution"}
        or history["path"] != "reports/stages/v2/S05/index-execution/pilot5000-r1/verification.json"
        or history["status"] != "fail"
        or history["counted_as_build_execution"] is not True
        or type(history["elapsed_seconds"]) not in {int, float}
        or not math.isfinite(history["elapsed_seconds"])
        or history["elapsed_seconds"] != 3593.3279999999795
    ):
        raise ValueError("original_failed_run_and_actual_elapsed_must_remain_explicit")


def read_outcome(factory: Any, baseline: dict) -> dict:
    """Keep original settled receipts/counters and all remaining tail visible."""
    with factory() as db:
        db.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"))
        db.execute(text("SET LOCAL statement_timeout='15000ms'"))
        generation = db.get(m.IndexGeneration, GENERATION)
        if generation is None or generation.state not in {"building", "verified"}:
            raise ValueError("pilot_outcome_must_remain_building_or_verified_not_published")
        rows = list(db.scalars(select(m.IndexChunk).where(m.IndexChunk.generation_id == GENERATION)))
        if (
            len(rows) != FIXED["chunk_count"]
            or any(canonical_sha(row.payload) != row.row_sha256 for row in rows)
            or canonical_sha({row.chunk_id: row.row_sha256 for row in rows}) != FIXED["expected_manifest_sha256"]
        ):
            raise ValueError("whole_persisted_source_rows_changed")
        records = {record["id"]: record for record in baseline["batches"]}
        batches = list(db.scalars(select(m.IndexBatch).where(m.IndexBatch.generation_id == GENERATION)))
        if {str(batch.id) for batch in batches} != set(records):
            raise ValueError("original_batch_set_changed")
        after, new_attempts = [], 0
        for batch in batches:
            before = records[str(batch.id)]
            if (
                batch.chunk_ids != before["chunk_ids"]
                or batch.sequence != before["sequence"]
                or batch.batch_manifest["sha256"] != before["batch_manifest_sha256"]
                or batch.logical_attempts < before["logical_attempts"]
            ):
                raise ValueError("batch_identity_or_historical_counter_changed")
            if before["state"] == "verified" and (
                batch.state != "verified"
                or batch.logical_attempts != before["logical_attempts"]
                or canonical_sha(batch.last_receipt) != before["last_receipt_sha256"]
                or batch.lease_id is not None
                or batch.lease_expires_at is not None
            ):
                raise ValueError("original_verified_batch_receipt_or_attempt_changed")
            new_attempts += batch.logical_attempts - before["logical_attempts"]
            after.append(
                {
                    "id": str(batch.id),
                    "state": batch.state,
                    "logical_attempts": batch.logical_attempts,
                    "row_count": len(batch.chunk_ids),
                    "lease_id": str(batch.lease_id) if batch.lease_id else None,
                    "lease_expires_at": batch.lease_expires_at,
                    "last_receipt_sha256": canonical_sha(batch.last_receipt) if batch.last_receipt else None,
                }
            )
        active = db.scalar(
            select(m.IndexCatalog.name).where(m.IndexCatalog.active_generation_id == GENERATION).limit(1)
        )
        if active is not None:
            raise ValueError("pilot_was_published")
        result = {
            "generation_state": generation.state,
            "whole_expected_rows": len(rows),
            "batch_counts": dict(Counter(item["state"] for item in after)),
            "row_counts": {
                state: sum(item["row_count"] for item in after if item["state"] == state)
                for state in sorted({item["state"] for item in after})
            },
            "original_956_verified_batch_receipts_and_attempts_unchanged": True,
            "new_logical_attempts": new_attempts,
            "batches": after,
            "search_published": False,
            "unknown_or_inflight_batches": sum(item["state"] in {"sending", "uncertain"} for item in after),
            "readback_proof": generation.readback_proof,
        }
        db.rollback()
        return result


def continue_delivery(
    *,
    profile: dict,
    baseline: dict,
    rows: list[dict],
    native: Any,
    tx: Callable,
    deadline: float,
    checkpoint: Callable[[dict], None],
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """Only existing ready batches; total segment deadline includes preflight."""
    validate_profile(profile, baseline)
    native.absolute_deadline = deadline
    tail = {record["id"]: record for record in baseline["batches"] if record["state"] == "ready"}
    delivered = []
    while clock() < deadline - profile["dispatch_reserve_seconds"]:
        claim = tx(lambda repository: repository.batch_claim(GENERATION))
        if claim is None:
            break
        if claim["status"] == "busy":
            return {"status": "busy_stopped_no_resend", "batch_id": claim["batch_id"]}
        record = tail.get(claim["batch_id"])
        if (
            record is None
            or claim["batch_id"] in delivered
            or claim["collection_name"] != FIXED["collection_name"]
            or claim["generation_id"] != str(GENERATION)
            or claim["batch_manifest"]["sha256"] != record["batch_manifest_sha256"]
            or [row["chunk_id"] for row in claim["rows"]] != record["chunk_ids"]
            or claim["logical_attempt"] != record["logical_attempts"] + 1
            or claim["read_before_write"] is not False
        ):
            # A concurrent/unknown transition is not permission to resend. The
            # persisted claim and checkpoint remain for separate reconciliation.
            checkpoint({"event": "unexpected_or_unknown_claim_no_dispatch", "batch_id": claim["batch_id"]})
            return {"status": "reconciliation_required", "batch_id": claim["batch_id"], "dispatched": False}
        receipt = native.upsert_and_read(
            claim["collection_name"], claim["rows"], seconds=min(120, deadline - clock() - 15)
        )
        tx(
            lambda repository, c=claim, r=receipt: repository.batch_result(
                GENERATION, UUID(c["batch_id"]), UUID(c["lease_id"]), r
            )
        )
        checkpoint(
            {
                "event": "continuation_batch_result",
                "batch_id": claim["batch_id"],
                "logical_attempt": claim["logical_attempt"],
                "receipt": receipt,
                "within_segment_window": clock() < deadline,
            }
        )
        if receipt["state"] != "verified":
            return {"status": "reconciliation_required", "batch_id": claim["batch_id"]}
        delivered.append(claim["batch_id"])
    outstanding = tx(lambda repository: repository.remaining_batches(GENERATION))
    if outstanding or clock() >= deadline - 15:
        return {"status": "window_stopped", "remaining_batches": outstanding, "new_verified_batches": len(delivered)}
    final_configuration = native.collection_proof(FIXED["collection_name"])
    if final_configuration["configuration_sha256"] != baseline["server_configuration_sha256"]:
        raise ValueError("native_configuration_changed_before_final_acceptance")
    checkpoint({"event": "final_native_configuration_verified", "proof": final_configuration})
    if clock() >= deadline - 15:
        return {"status": "window_stopped", "remaining_batches": 0, "new_verified_batches": len(delivered)}
    proof = native.scan_all(
        FIXED["collection_name"],
        rows,
        seconds=min(profile["final_readback_maximum_seconds"], deadline - clock() - 10),
    )
    if clock() >= deadline - 5:
        checkpoint({"event": "late_complete_readback_not_accepted", "proof": proof})
        return {"status": "window_stopped", "remaining_batches": 0, "new_verified_batches": len(delivered)}
    summary = tx(lambda repository: repository.accept_readback(GENERATION, proof))
    if clock() >= deadline:
        return {"status": "late_segment_not_a_time_gate_pass", "generation": summary, "complete_readback": proof}
    return {
        "status": "verified_not_published",
        "generation": summary,
        "complete_readback": proof,
        "final_native_configuration": final_configuration,
        "new_verified_batches": len(delivered),
        "search_published": False,
        "original_60_minute_performance_gate_passed": False,
    }
