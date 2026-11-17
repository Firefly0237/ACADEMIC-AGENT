"""Meaningful fault controls; real cold-cycle evidence is separate."""

from __future__ import annotations

from uuid import uuid4

import pytest

from citequest.index_v2.controller import IndexController
from citequest.index_v2.native import IncompleteReadbackError
from citequest.index_v2.publication import AliasTransport, publication_alias
from citequest.index_v2.startup import verify_startup_readiness


class Clock:
    value = 0.0

    def monotonic(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class Native:
    def __init__(self, action):
        self.action = action
        self.closed = False

    def call(self, *args, **kwargs):
        assert args == ("load_collection",)

    def collection_proof(self, name):
        return {"configuration_sha256": "b" * 64}

    def scan_all(self, name, rows, *, seconds):
        if self.action == "missing":
            raise IncompleteReadbackError(name, {row["chunk_id"] for row in rows}, set(), [])
        if self.action == "extra":
            raise ValueError("unknown_or_duplicate_full_readback_ID")
        if self.action == "hash":
            raise ValueError("actual_scalar_text_hash_or_config_mismatch")
        if self.action == "dense":
            raise ValueError("actual_dense_1024_finite_nonzero_required")
        return {"matched": len(rows), "complete": True, "rows_sha256": "c" * 64}

    def read_ids(self, name, rows, *, seconds):
        return rows, {"complete": True, "rows_sha256": "c" * 64, "vector_values_sha256": "d" * 64}

    def close(self):
        self.closed = True


def entry():
    return {
        "collection": "cq_v2_startup_control",
        "rows": [{"chunk_id": "a" * 64}],
        "configuration_sha256": "b" * 64,
        "generation_id": str(uuid4()),
        "coverage": "settled_subset_only",
    }


def setup(monkeypatch, actions):
    import citequest.index_v2.startup as module

    clock = Clock()
    monkeypatch.setattr(module.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(module.time, "sleep", clock.sleep)
    natives = []

    def factory():
        native = Native(actions.pop(0))
        natives.append(native)
        return native

    return clock, natives, factory


def test_startup_missing_is_transparently_observed_again_but_grants_no_search_permission(monkeypatch):
    clock, natives, factory = setup(monkeypatch, ["missing", "pass"])
    logs = []
    result = verify_startup_readiness([entry()], factory, maximum_seconds=180, checkpoint=logs.append)
    assert result["status"] == "physical_data_ready" and result["attempts"] == 2 and clock.value == 2
    assert not result["search_publication_permission_granted"]
    assert logs[0]["observation"]["expected_count"] == 1 and logs[0]["observation"]["found_count"] == 0
    assert logs[0]["observation"]["missing_ids"] == ["a" * 64]
    assert (
        all(n.closed for n in natives)
        and result["upsert_calls"] == result["flush_calls"] == result["embedding_calls"] == 0
    )


@pytest.mark.parametrize("action", ["extra", "hash", "dense"])
def test_extra_or_scalar_or_dense_failure_is_never_retried(monkeypatch, action):
    _, natives, factory = setup(monkeypatch, [action, "pass"])
    logs = []
    with pytest.raises(ValueError):
        verify_startup_readiness([entry()], factory, maximum_seconds=180, checkpoint=logs.append)
    assert len(natives) == 1 and len(logs) == 1 and logs[0]["outcome"] == "hard_failure_not_retried"


def test_changed_persisted_dense_fingerprint_is_never_retried(monkeypatch):
    _, natives, factory = setup(monkeypatch, ["pass", "pass"])
    expected = entry()
    expected["expected_fixed_ID_vectors_and_scalars_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="persisted_vector_values_changed"):
        verify_startup_readiness([expected], factory, maximum_seconds=180, checkpoint=lambda event: None)
    assert len(natives) == 1


def test_attempt_budget_does_not_become_hidden_retry_until_pass(monkeypatch):
    _, natives, factory = setup(monkeypatch, ["missing", "pass"])
    logs = []
    with pytest.raises(TimeoutError):
        verify_startup_readiness([entry()], factory, maximum_seconds=180, maximum_attempts=1, checkpoint=logs.append)
    assert len(natives) == 1 and len(logs) == 1


def test_connection_failure_is_logged_and_not_treated_as_missing(monkeypatch):
    setup(monkeypatch, [])
    logs = []

    def factory():
        raise ConnectionError("controlled transport unavailable")

    with pytest.raises(ConnectionError):
        verify_startup_readiness([entry()], factory, maximum_seconds=180, checkpoint=logs.append)
    assert len(logs) == 1 and logs[0]["outcome"] == "hard_failure_not_retried"


def test_alias_names_are_canonical_and_isolated_across_real_domain_identifiers():
    owner = "owner:" + str(uuid4())
    other = "owner:" + str(uuid4())
    values = [publication_alias("standard_metadata", domain) for domain in ("public", owner, other)]
    assert len(set(values)) == 3 and all(len(name) <= 128 for name in values)
    assert publication_alias("standard_metadata", owner) == values[1]
    with pytest.raises(ValueError):
        publication_alias("standard_metadata", "owner:not-a-UUID")


def test_pinned_sdk_dict_shape_is_required_instead_of_inaccurate_annotation():
    class BadShape:
        def call(self, *args, **kwargs):
            return []

    with pytest.raises(ValueError, match="scope_invalid"):
        AliasTransport(BadShape()).target("cq_v2_alias_control")


@pytest.mark.parametrize("remaining", [1, 2])
def test_real_metadata_controller_never_scans_partial_delivery_at_window_exit(remaining):
    class Repo:
        def begin(self, *args, **kwargs):
            return {"state": "building", "collection_name": "cq_v2_partial_control"}

        def describe(self, *args):
            return self.begin()

        def bind_server_proof(self, *args):
            pass

        def batch_claim(self, *args):
            return None

        def remaining_batches(self, *args):
            return remaining

        def expected_rows(self, *args):
            raise AssertionError("partial manifest must not be scanned")

    class Wire:
        def create(self, *args):
            return {}

        def scan_all(self, *args, **kwargs):
            raise AssertionError("partial native collection must not be accepted")

    controller = IndexController.__new__(IndexController)
    controller.native = Wire()
    controller.chunker = type("Chunker", (), {"profile_sha256": "b" * 64})()
    controller.tx = lambda operation: operation(Repo())
    result = controller.run(uuid4(), uuid4(), unit_limit=5000, seconds=900, checkpoint=lambda event: None)
    assert result["status"] == "window_stopped" and result["remaining_batches"] == remaining


@pytest.mark.parametrize("typed", [False, True])
def test_capture_wrappers_validate_the_joined_object_without_fresh_generation_reload(typed):
    from types import SimpleNamespace

    from citequest.index_v2.repository import IndexRepository
    from citequest.index_v2.repository_v2 import VariantIndexRepository
    from citequest.v2.domain import Actor

    repository = (VariantIndexRepository if typed else IndexRepository).__new__(
        VariantIndexRepository if typed else IndexRepository
    )
    generation = SimpleNamespace(
        id=uuid4(),
        collection_name="cq_v2_atomic_capture_control",
        expected_manifest_sha256="a" * 64,
        source_seal_sha256="b" * 64,
    )
    repository.actor = Actor(uuid4())
    repository._capture_pointer_snapshot = lambda catalog, domain: (generation, 17)

    def no_reload(*args, **kwargs):
        raise AssertionError("captured generation must not be reloaded after JOIN")

    repository.generation = no_reload
    checked = []
    if typed:
        repository._validate_generation_value = lambda value: checked.append(value)
        repository._read_ready = lambda value: checked.append(value)
    else:
        repository._validate_generation_source = lambda value: checked.append(value)
        repository._publication_ready = lambda value: checked.append(value)

    class DB:
        lease = None

        def add(self, value):
            self.lease = value

        def flush(self):
            self.lease.id = uuid4()

    repository.db = DB()
    result = repository.capture_reader()
    assert checked == [generation, generation]
    assert result["generation_id"] == str(generation.id) and result["epoch"] == 17
    assert repository.db.lease.generation_id == generation.id and repository.db.lease.epoch == 17
