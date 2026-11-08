"""Pure continuation fault controls, never native encoding or SQL admission."""

from __future__ import annotations

import copy
import hashlib
import json
from uuid import uuid4

import pytest

from citequest.index_v2.development_continuation import (
    FIXED,
    GENERATION,
    SOURCE_SEAL,
    claim_profile_execution,
    continue_delivery,
    read_approved_profile,
    validate_profile,
)
from citequest.index_v2.rows import canonical_sha


def fixture():
    batch_id, old_id = str(uuid4()), str(uuid4())
    row = {"chunk_id": "a" * 64, "text": "actual source text"}
    manifest = {"rows": {row["chunk_id"]: canonical_sha(row)}}
    manifest["sha256"] = canonical_sha(manifest["rows"])
    baseline = {
        "identity": copy.deepcopy(FIXED),
        "server_configuration_sha256": "f" * 64,
        "batches": [
            {
                "id": batch_id,
                "state": "ready",
                "chunk_ids": [row["chunk_id"]],
                "logical_attempts": 0,
                "batch_manifest_sha256": manifest["sha256"],
            },
            {
                "id": old_id,
                "state": "verified",
                "chunk_ids": ["b" * 64],
                "logical_attempts": 1,
                "batch_manifest_sha256": "c" * 64,
            },
        ],
    }
    baseline["checkpoint_sha256"] = canonical_sha(baseline)
    baseline["sql_head_observed"] = "s05_index_0002"
    profile = {
        "schema_version": "cq.development-index-continuation/1",
        "run_id": "S05-development-5000-continuation-r1",
        "kind": "new_bounded_segment_on_unchanged_existing_building_checkpoint",
        "segment_window_seconds": 3600,
        "dispatch_reserve_seconds": 180,
        "final_readback_maximum_seconds": 120,
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "identity": copy.deepcopy(FIXED),
        "source_seal_sha256": SOURCE_SEAL,
        "source_bindings": {"synthetic_fixture_only": "d" * 64},
        "historical_runs": [
            {
                "path": "reports/stages/v2/S05/index-execution/pilot5000-r1/verification.json",
                "sha256": "e" * 64,
                "elapsed_seconds": 3593.3279999999795,
                "status": "fail",
                "counted_as_build_execution": True,
            }
        ],
        "prior_failure_retained": True,
        "automatic_uncertain_replay": False,
        "verified_rows_reencoded": False,
        "generation_or_collection_created": False,
        "publication_performed": False,
        "cloud_or_full_corpus_or_original_performance_gate_passed": False,
        "runtime_executable": "data/runtime-s05-index-next/env/Scripts/python.exe",
    }
    claim = {
        "status": "claimed",
        "batch_id": batch_id,
        "lease_id": str(uuid4()),
        "generation_id": str(GENERATION),
        "collection_name": FIXED["collection_name"],
        "rows": [row],
        "batch_manifest": manifest,
        "logical_attempt": 1,
        "read_before_write": False,
    }
    return profile, baseline, claim


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class Native:
    def __init__(self, clock, *, state="verified", elapsed=2.0, scan_elapsed=1.0):
        self.clock, self.state, self.elapsed, self.scan_elapsed = clock, state, elapsed, scan_elapsed
        self.upserts, self.scans = [], []
        self.configuration_sha256 = "f" * 64

    def collection_proof(self, collection):
        return {"configuration_sha256": self.configuration_sha256}

    def upsert_and_read(self, collection, rows, *, seconds):
        assert seconds <= 120 and {"dense", "sparse"}.isdisjoint(rows[0])
        self.upserts.append(rows)
        self.clock.now += self.elapsed
        return {"state": self.state, "readback": {"complete": self.state == "verified"}}

    def scan_all(self, collection, rows, *, seconds):
        assert 0 < seconds <= 120
        self.scans.append(rows)
        self.clock.now += self.scan_elapsed
        return {"complete": True, "matched": len(rows)}


class Repo:
    def __init__(self, claim, *, outstanding=0):
        self.claim, self.outstanding = claim, outstanding
        self.claim_calls, self.results, self.accepted = 0, [], []

    def batch_claim(self, generation):
        assert generation == GENERATION
        self.claim_calls += 1
        return self.claim if self.claim_calls == 1 else None

    def batch_result(self, generation, batch_id, lease_id, receipt):
        self.results.append(receipt)

    def remaining_batches(self, generation):
        return self.outstanding

    def accept_readback(self, generation, proof):
        self.accepted.append(proof)
        return {"state": "verified"}


def run(profile, baseline, claim, *, outstanding=0, elapsed=2, scan_elapsed=1, deadline=3600):
    clock = Clock()
    native = Native(clock, elapsed=elapsed, scan_elapsed=scan_elapsed)
    repository = Repo(claim, outstanding=outstanding)
    logs = []
    result = continue_delivery(
        profile=profile,
        baseline=baseline,
        rows=claim["rows"],
        native=native,
        tx=lambda action: action(repository),
        deadline=deadline,
        checkpoint=logs.append,
        clock=clock,
    )
    return result, native, repository, logs


@pytest.mark.parametrize("window", [True, 0, 299, 3601, float("inf"), float("nan"), 3600.0, "3600"])
def test_invalid_segment_window_is_rejected_before_any_dispatch(window):
    profile, baseline, _ = fixture()
    profile["segment_window_seconds"] = window
    with pytest.raises(ValueError):
        validate_profile(profile, baseline)


@pytest.mark.parametrize(
    "field",
    [
        "automatic_uncertain_replay",
        "verified_rows_reencoded",
        "generation_or_collection_created",
        "publication_performed",
        "cloud_or_full_corpus_or_original_performance_gate_passed",
    ],
)
def test_historical_reset_replay_or_scope_upgrade_is_not_a_legal_profile(field):
    profile, baseline, _ = fixture()
    profile[field] = True
    with pytest.raises(ValueError):
        validate_profile(profile, baseline)


def test_original_failed_run_cannot_be_presented_as_pass_or_zero_elapsed():
    profile, baseline, _ = fixture()
    profile["historical_runs"][0].update(status="pass", elapsed_seconds=0)
    with pytest.raises(ValueError):
        validate_profile(profile, baseline)


def test_boolean_is_not_a_valid_integer_in_the_fixed_pilot_identity():
    profile, baseline, _ = fixture()
    profile["identity"]["empty_unit_count"] = True
    with pytest.raises(ValueError):
        validate_profile(profile, baseline)


def test_same_profile_cannot_open_another_window_after_a_failed_or_successful_attempt(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    path = claim_profile_execution(tmp_path, "a" * 64, first, "2026-11-05T00:00:00Z")
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        claim_profile_execution(tmp_path, "a" * 64, second, "2026-11-05T01:00:00Z")
    assert path.read_bytes() == original


def test_execution_uses_captured_approved_bytes_even_when_path_changes_later(tmp_path):
    profile, _, _ = fixture()
    path = tmp_path / "profile.json"
    approved = (json.dumps(profile) + "\n").encode()
    path.write_bytes(approved)
    raw, captured = read_approved_profile(path, hashlib.sha256(approved).hexdigest())
    different = copy.deepcopy(profile)
    different["segment_window_seconds"] = 300
    path.write_text(json.dumps(different) + "\n\n", encoding="utf-8")
    assert raw == approved and captured["segment_window_seconds"] == 3600
    assert json.loads(path.read_bytes())["segment_window_seconds"] == 300
    with pytest.raises(ValueError, match="approved_profile_bytes"):
        read_approved_profile(path, hashlib.sha256(approved).hexdigest())


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b"[]"])
def test_approved_hash_does_not_legalize_duplicate_nonfinite_or_nonobject_JSON(tmp_path, raw):
    path = tmp_path / "profile.json"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        read_approved_profile(path, hashlib.sha256(raw).hexdigest())


def test_checkpoint_content_tamper_and_unknown_profile_field_are_rejected():
    profile, baseline, _ = fixture()
    baseline["batches"][0]["logical_attempts"] = 1
    with pytest.raises(ValueError, match="checkpoint_content"):
        validate_profile(profile, baseline)
    profile, baseline, _ = fixture()
    profile["allow_replay"] = True
    with pytest.raises(ValueError, match="exact_development"):
        validate_profile(profile, baseline)


def test_additive_migration_head_is_diagnostic_and_does_not_change_metadata_identity():
    profile, baseline, _ = fixture()
    baseline["sql_head_observed"] = "s05_data_0003"
    validate_profile(profile, baseline)


def test_only_ready_tail_is_encoded_and_success_still_is_not_original_or_publication_pass():
    profile, baseline, claim = fixture()
    result, native, repository, _ = run(profile, baseline, claim)
    assert result["status"] == "verified_not_published"
    assert len(native.upserts) == len(native.scans) == len(repository.accepted) == 1
    assert not result["search_published"] and not result["original_60_minute_performance_gate_passed"]
    assert native.absolute_deadline == 3600 and result["new_verified_batches"] == 1


@pytest.mark.parametrize("fault", ["old_verified", "unknown", "manifest", "attempt", "collection", "row_ID"])
def test_unexpected_or_uncertain_claim_is_preserved_without_any_encoding(fault):
    profile, baseline, claim = fixture()
    if fault == "old_verified":
        claim["batch_id"] = baseline["batches"][1]["id"]
    if fault == "unknown":
        claim["read_before_write"] = True
    if fault == "manifest":
        claim["batch_manifest"]["sha256"] = "f" * 64
    if fault == "attempt":
        claim["logical_attempt"] = 2
    if fault == "collection":
        claim["collection_name"] = "another_collection"
    if fault == "row_ID":
        claim["rows"][0]["chunk_id"] = "f" * 64
    result, native, repository, logs = run(profile, baseline, claim)
    assert result["status"] == "reconciliation_required" and not result["dispatched"]
    assert not native.upserts and not native.scans and not repository.accepted
    assert logs[0]["event"] == "unexpected_or_unknown_claim_no_dispatch"


def test_uncertain_delivery_stops_after_saved_receipt_without_second_call_or_final_scan():
    profile, baseline, claim = fixture()
    clock = Clock()
    native, repository = Native(clock, state="uncertain"), Repo(claim)
    result = continue_delivery(
        profile=profile,
        baseline=baseline,
        rows=claim["rows"],
        native=native,
        tx=lambda action: action(repository),
        deadline=3600,
        checkpoint=lambda event: None,
        clock=clock,
    )
    assert result["status"] == "reconciliation_required" and len(native.upserts) == 1
    assert repository.results == [{"state": "uncertain", "readback": {"complete": False}}]
    assert repository.claim_calls == 1 and not native.scans and not repository.accepted


@pytest.mark.parametrize("remaining", [1, 513])
def test_remaining_tail_never_triggers_whole_generation_scan(remaining):
    profile, baseline, claim = fixture()
    result, native, repository, _ = run(profile, baseline, claim, outstanding=remaining)
    assert result["status"] == "window_stopped" and result["remaining_batches"] == remaining
    assert not native.scans and not repository.accepted


def test_dispatch_reserve_exhaustion_does_not_claim_a_new_batch():
    profile, baseline, claim = fixture()
    result, native, repository, _ = run(profile, baseline, claim, outstanding=513, deadline=179)
    assert result["status"] == "window_stopped"
    assert not native.upserts and repository.claim_calls == 0 and not native.scans


def test_late_whole_readback_never_marks_generation_accepted():
    profile, baseline, claim = fixture()
    result, native, repository, logs = run(profile, baseline, claim, scan_elapsed=3600)
    assert result["status"] == "window_stopped" and len(native.scans) == 1
    assert not repository.accepted and logs[-1]["event"] == "late_complete_readback_not_accepted"


def test_busy_claim_is_not_retried_or_treated_as_success():
    profile, baseline, claim = fixture()
    claim.update(status="busy")
    result, native, repository, _ = run(profile, baseline, claim)
    assert result["status"] == "busy_stopped_no_resend" and repository.claim_calls == 1
    assert not native.upserts and not native.scans


def test_native_configuration_drift_is_not_accepted_after_tail_delivery():
    profile, baseline, claim = fixture()
    clock = Clock()
    native, repository = Native(clock), Repo(claim)
    native.configuration_sha256 = "c" * 64
    with pytest.raises(ValueError, match="configuration_changed"):
        continue_delivery(
            profile=profile,
            baseline=baseline,
            rows=claim["rows"],
            native=native,
            tx=lambda action: action(repository),
            deadline=3600,
            checkpoint=lambda event: None,
            clock=clock,
        )
    assert len(native.upserts) == 1 and not native.scans and not repository.accepted
