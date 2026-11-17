"""Reviewed continuation in the same cumulative window; SQL-bounded batch claim."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import resource
import sys
import time
import traceback
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url

from citequest.index_v2 import models as m
from citequest.index_v2.native import RPCAudit
from citequest.index_v3.backend import read_profile
from citequest.index_v3.models import BackendBinding
from citequest.index_v3.native import GPUCandidateNative
from citequest.index_v3.repository_bounded import BoundedFullBuildRepository as FullBuildRepository
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor
from citequest.v2.services import Catalog
from citequest.v2.storage_config import app_storage

ROOT = Path(__file__).resolve().parents[2]


def strict_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate_reviewed_JSON_key")
        value[key] = item
    return value


def read_reviewed_profile(path: Path, expected: str) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("exact_reviewed_execution_profile_required")
    profile = json.loads(raw, object_pairs_hook=strict_object)
    if (
        profile["schema_version"] != "cq.full-GPU-candidate-execution/1"
        or profile["total_active_budget_seconds"] != 14400
        or profile["publish"] is not False
        or profile["automatic_unknown_replay"] is not False
        or profile["environment"] != "docker-linux"
    ):
        raise ValueError("explicit_nonpublishing_full_candidate_contract_required")
    return raw, profile


def main() -> None:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--approved-profile-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw, profile = read_reviewed_profile(args.profile, args.approved_profile_sha256)
    output = args.output.resolve()
    if not output.is_relative_to(Path("/evidence")) or output.exists():
        raise ValueError("new_owned_evidence_output_required")
    output.mkdir(parents=True)
    (output / "approved-profile.json").write_bytes(raw)
    claim = Path("/evidence") / ("execution-claim-" + args.approved_profile_sha256 + ".json")
    with claim.open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "profile_sha256": args.approved_profile_sha256,
                "output": str(output),
                "generation_id": profile["generation_id"],
                "consumed_once_even_if_failed": True,
            },
            stream,
        )
    code_manifest = ROOT / "build-source-manifest.json"
    if hashlib.sha256(code_manifest.read_bytes()).hexdigest() != profile["code_manifest_sha256"]:
        raise ValueError("exact_controller_source_capsule_required")
    for relative, expected in json.loads(code_manifest.read_text()).items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("controller_image_source_changed")
    prior_active = 0.0
    for item in profile["prior_execution_evidence"]:
        prior_path = (ROOT / item["path"]).resolve()
        if not prior_path.is_relative_to(ROOT):
            raise ValueError("owned_prior_evidence_required")
        proof_raw = prior_path.read_bytes()
        if hashlib.sha256(proof_raw).hexdigest() != item["sha256"]:
            raise ValueError("prior_execution_proof_changed")
        prior = json.loads(proof_raw)
        measured = prior.get("elapsed_seconds", prior.get("controller_elapsed_seconds"))
        if prior["generation_id"] != profile["generation_id"] or measured != item["elapsed_seconds"]:
            raise ValueError("prior_execution_identity_or_elapsed_changed")
        prior_active += measured
    remaining = profile["total_active_budget_seconds"] - prior_active
    if remaining <= 420:
        raise ValueError("insufficient_total_generation_budget")
    deadline = started + remaining
    generation = UUID(profile["generation_id"])
    backend_path = ROOT / "runtime/s05-full/gpu-candle-f32-candidate-r1.json"
    _, backend_sha, _ = read_profile(backend_path)
    if backend_sha != profile["backend_profile_sha256"]:
        raise ValueError("approved_backend_profile_changed")
    dsn = Path("/run/secrets/index_app_dsn").read_text().strip()
    parsed = make_url(dsn)
    if (
        parsed.username != "cq_s03_app"
        or parsed.database != "citequest_s03"
        or parsed.host != "citequest-v2-s03-postgres-1"
        or parsed.port != 5432
    ):
        raise ValueError("trusted_main_app_secretref_required")
    database, native, guard = engine(dsn), None, None
    factory, store = sessions(database), app_storage().cas()
    with factory() as db:
        db.execute(text("SET TRANSACTION READ ONLY"))
        value = db.get(m.IndexGeneration, generation)
        binding = db.get(BackendBinding, generation)
        if value is None or binding is None or binding.backend_profile_sha256 != backend_sha:
            raise ValueError("migrator_backend_binding_required_before_any_GPU_write")
        if (
            value.state != "building"
            or value.build_scope != "full"
            or value.expected_manifest_sha256 != profile["expected_manifest_sha256"]
            or value.unit_count != profile["source_units"]
            or value.chunk_count != profile["chunks"]
        ):
            raise ValueError("sealed_full_generation_identity_required")
        actor = Actor(value.creator_id, publisher=True)
        policy_sha, evaluation_sha = value.policy_sha256, value.evaluation_sha256
        db.rollback()
    audit = RPCAudit(output / "rpc.jsonl")
    proof = {
        "status": "failed",
        "generation_id": str(generation),
        "profile_sha256": args.approved_profile_sha256,
        "backend_profile_sha256": backend_sha,
        "prior_active_seconds": prior_active,
        "total_active_budget_seconds": 14400,
        "remaining_controller_window_seconds": remaining,
        "search_published": False,
        "CPU_equivalence_claimed": False,
        "new_generative_API_calls": 0,
        "runtime_packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions())),
    }

    def checkpoint(event: dict) -> None:
        row = {
            "elapsed_seconds": time.monotonic() - started,
            "peak_controller_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            **event,
        }
        with (output / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, default=str) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def tx(action: Callable[[FullBuildRepository], Any]) -> Any:
        ms = int((deadline - time.monotonic()) * 1000)
        if ms <= 0:
            raise TimeoutError("cumulative_four_hour_budget_exhausted")
        with factory.begin() as db:
            db.execute(text("SELECT set_config('statement_timeout',:ms,true)"), {"ms": str(min(300000, ms))})
            db.execute(text("SET LOCAL lock_timeout='15000ms'"))
            return action(
                FullBuildRepository(
                    Catalog(db, actor), store=store, policy_sha256=policy_sha, evaluation_sha256=evaluation_sha
                )
            )

    try:
        guard = database.connect().execution_options(isolation_level="AUTOCOMMIT")
        key = int.from_bytes(hashlib.sha256(generation.bytes).digest()[:8], signed=True)
        if guard.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": key}).scalar_one() is not True:
            raise ValueError("one_owned_controller_per_generation_required")
        proof["advisory_lock_key"] = key
        native = GPUCandidateNative(audit=audit, profile_path=backend_path, environment="docker-linux")
        native.absolute_deadline = deadline
        summary = tx(lambda repo: repo.describe(generation))
        configuration = native.create(summary["collection_name"])
        tx(lambda repo: repo.bind_server_proof(generation, configuration))
        checkpoint({"event": "candidate_native_configuration_bound", "configuration": configuration})
        while time.monotonic() < deadline - 420:
            if (output / "STOP").exists():
                checkpoint({"event": "explicit_stop_before_new_dispatch"})
                break
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > 1800 * 1024**2:
                checkpoint({"event": "controller_RSS_admission_stop"})
                break
            claim = tx(lambda repo: repo.batch_claim(generation))
            if claim is None:
                break
            if claim["status"] == "busy":
                proof["status"] = "busy_not_retried"
                break
            if claim["read_before_write"]:
                _, before = native.read_ids(claim["collection_name"], claim["rows"], seconds=45)
                receipt = {
                    "state": "verified" if before["complete"] else "uncertain",
                    "readback": before,
                    "recovery": "actual_read_before_any_resend",
                    "application_automatic_retries": 0,
                }
            else:
                receipt = native.upsert_and_read(claim["collection_name"], claim["rows"], seconds=120)
            tx(
                lambda repo, c=claim, r=receipt: repo.batch_result(
                    generation, UUID(c["batch_id"]), UUID(c["lease_id"]), r
                )
            )
            checkpoint(
                {
                    "event": "batch_result",
                    "batch_id": claim["batch_id"],
                    "logical_attempt": claim["logical_attempt"],
                    "receipt": receipt,
                }
            )
            if receipt["state"] != "verified":
                proof["status"] = "unknown_stop_no_automatic_resend"
                break
        outstanding = tx(lambda repo: repo.remaining_batches(generation))
        proof["remaining_batches"] = outstanding
        if not outstanding and time.monotonic() < deadline - 330:
            expected = tx(lambda repo: repo.expected_rows(generation))
            actual_config = native.collection_proof(summary["collection_name"])
            if actual_config["configuration_sha256"] != configuration["configuration_sha256"]:
                raise ValueError("backend_configuration_changed")
            whole = native.scan_all(summary["collection_name"], expected, seconds=300)
            if time.monotonic() >= deadline - 10:
                raise TimeoutError("late_full_readback_not_accepted")
            proof["generation"] = tx(lambda repo: repo.accept_readback(generation, whole))
            proof["complete_readback"] = whole
            proof["status"] = "verified_candidate_NOT_published_NOT_quality_adopted"
        elif proof["status"] == "failed":
            proof["status"] = "window_or_resource_stopped_checkpoint_retained"
    except Exception as exc:
        proof.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    finally:
        try:
            with factory() as db:
                db.execute(text("SET TRANSACTION READ ONLY"))
                states = list(
                    db.execute(
                        select(m.IndexBatch.state, func.count())
                        .where(m.IndexBatch.generation_id == generation)
                        .group_by(m.IndexBatch.state)
                    )
                )
                proof["final_batch_counts"] = dict(states)
                proof["remaining_unknown_or_inflight"] = sum(
                    count for state, count in states if state in {"sending", "uncertain"}
                )
                db.rollback()
        except Exception as exc:
            proof["final_SQL_observation_error"] = type(exc).__name__
            proof["status"] = "final_state_unverified"
        for name, action in [
            ("native_close", lambda: native.close() if native else None),
            ("guard_close", lambda: guard.close() if guard else None),
        ]:
            try:
                action()
            except Exception as exc:
                proof.setdefault("cleanup_errors", []).append({"phase": name, "type": type(exc).__name__})
                proof["status"] = "cleanup_unverified"
        database.dispose()
        proof["controller_elapsed_seconds"] = time.monotonic() - started
        proof["cumulative_active_seconds"] = prior_active + proof["controller_elapsed_seconds"]
        proof["within_four_hour_budget"] = proof["cumulative_active_seconds"] <= 14400
        if not proof["within_four_hour_budget"]:
            proof["status"] = "late_not_budget_pass"
        proof["physical_RPC_counts"] = dict(
            Counter(row["method"].rsplit("/", 1)[-1] for row in audit.events if row["event"] == "rpc_start")
        )
        proof["peak_controller_RSS_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        (output / "verification.json").write_text(
            json.dumps(proof, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )
    print(  # noqa: T201
        json.dumps(
            {
                key: proof.get(key)
                for key in ["status", "generation_id", "cumulative_active_seconds", "final_batch_counts"]
            }
        )
    )
    if proof["status"] != "verified_candidate_NOT_published_NOT_quality_adopted":
        raise SystemExit(1)


if __name__ == "__main__":
    boot_started = time.monotonic()
    try:
        main()
    except Exception as startup_error:
        # Initialization failures (before the main execution try/finally) also
        # receive an honest durable receipt; never silently reopen their window.
        if "--output" in sys.argv:
            failed_output = Path(sys.argv[sys.argv.index("--output") + 1]).resolve()
            if failed_output.is_relative_to(Path("/evidence")):
                failed_output.mkdir(parents=True, exist_ok=True)
                receipt_path = failed_output / "verification.json"
                if not receipt_path.exists():
                    captured = failed_output / "approved-profile.json"
                    captured_profile = json.loads(captured.read_bytes()) if captured.exists() else {}
                    receipt_path.write_text(
                        json.dumps(
                            {
                                "status": "initialization_failed_NOT_budget_or_quality_pass",
                                "generation_id": captured_profile.get("generation_id"),
                                "controller_elapsed_seconds": time.monotonic() - boot_started,
                                "error_type": type(startup_error).__name__,
                                "error": str(startup_error),
                                "traceback": traceback.format_exc(),
                                "encoding_dispatch_state": "requires_RPC_and_SQL_review_if_main_receipt_missing",
                                "search_published": False,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
        raise
