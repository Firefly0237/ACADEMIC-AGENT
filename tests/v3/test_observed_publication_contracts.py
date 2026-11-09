"""Pure cursor, membership and profile contracts; no broker/index resources."""
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from citequest.updates_v2.locks import LINUX_PROFILE, WINDOWS_PROFILE, policy, policy_sha
from citequest.updates_v2.oai_cursor import next_request, state_for
from citequest.updates_v2.oai_enumeration import OAIEnumeration
from citequest.updates_v2.publication_contracts import merge_members
from citequest.updates_v2.schemas import WindowInput


def configuration():
    return SimpleNamespace(lower_bound=datetime(2026, 1, 1, tzinfo=UTC), upper_bound=datetime(2026, 1, 3, tzinfo=UTC), maximum_pages=5)


def test_durable_opaque_cursor_restarts_without_resetting_cost_or_old_state():
    config, now = configuration(), datetime(2026, 1, 1, 12, tzinfo=UTC)
    first, reserved = next_request(config, {}, now)
    assert "until=2026-01-02" in first["url"] and first["oai_request"] == 1
    state = state_for(config, {"oai": reserved})
    state.accept({"records": [{"native_id": "2601.00001", "record_sha256": "a" * 64}],
        "token": "opaque&next=1", "token_expiration": "2026-01-02T00:00:00Z"})
    durable = state.cursor()
    second, reserved = next_request(config, {"oai": durable}, now)
    assert second["oai_request"] == 2 and "resumptionToken=opaque%26next%3D1" in second["url"]
    state = OAIEnumeration.restore(reserved)
    state.accept({"error": "badResumptionToken", "records": []})
    assert state.requests == 2 and state.restarts == 1 and state.records == {}
    assert durable["records"] and durable["requests"] == 1
    third, reserved = next_request(config, {"oai": state.cursor()}, now)
    assert third["oai_request"] == 3 and third["oai_epoch"] == 1 and "metadataPrefix=arXiv" in third["url"]
    state = OAIEnumeration.restore(reserved)
    state.accept({"records": [], "error": "noRecordsMatch"})
    assert state.finished and not state.awaiting_response
    assert next_request(config, {"oai": state.cursor()}, now)[0] is None


def test_oai_rejects_conflicts_and_preserves_request_caps():
    config, now = configuration(), datetime(2026, 1, 1, 12, tzinfo=UTC)
    _, reserved = next_request(config, {}, now)
    state = OAIEnumeration.restore(reserved)
    state.accept({"records": [{"native_id": "x", "record_sha256": "a" * 64}], "token": "next"})
    _, reserved = next_request(config, {"oai": state.cursor()}, now)
    state = OAIEnumeration.restore(reserved)
    with pytest.raises(ValueError, match="content_conflict"):
        state.accept({"records": [{"native_id": "x", "record_sha256": "b" * 64}], "token": None})
    state = OAIEnumeration.restore(reserved)
    with pytest.raises(ValueError, match="after_continuation"):
        state.accept({"records": [], "error": "noRecordsMatch"})
    bad = {**reserved, "requests": True}
    with pytest.raises(ValueError, match="accounting_invalid"):
        OAIEnumeration.restore(bad)


def test_complete_merge_preserves_base_and_source_scoped_quarantine():
    base = [{"source_unit_id": "A", "source_namespace": "one", "native_id": "same", "source_revision": "v1"},
        {"source_unit_id": "B", "source_namespace": "two", "native_id": "same", "source_revision": None}]
    added = [{"source_unit_id": "C", "source_namespace": "one", "native_id": "same", "source_revision": "v2"},
        {"source_unit_id": "D", "source_namespace": "one", "native_id": "new", "source_revision": None}]
    assert [item["source_unit_id"] for item in merge_members(base, added, {("one", "new")})] == ["B", "C"]
    assert base[0]["source_unit_id"] == "A" and added[0]["source_unit_id"] == "C"
    with pytest.raises(ValueError, match="regression"):
        merge_members([{**base[0], "source_revision": "v3"}], added, set())
    with pytest.raises(ValueError, match="unique"):
        merge_members([base[0], base[0]], [], set())
    with pytest.raises(ValueError, match="nonempty"):
        merge_members(base, [], {("one", "same"), ("two", "same")})


def test_explicit_linux_policy_and_legacy_window_wire():
    assert policy_sha(WINDOWS_PROFILE) == "6bb143507ddc8bc8bf4b7d31ce3f38d4e377de3bc595bc097ccd1f725e270509"
    assert policy_sha(LINUX_PROFILE) != policy_sha(WINDOWS_PROFILE)
    assert "UnixFileLock" in policy(LINUX_PROFILE)["backend"]
    values = {"source_namespace": "arxiv", "mode": "protocol_canary", "lower_bound": configuration().lower_bound,
        "upper_bound": configuration().upper_bound, "managed_ids": [], "maximum_network_calls": 10,
        "maximum_active_seconds": 60, "client_operation_id": uuid4()}
    old = WindowInput.model_validate(values, strict=True)
    assert "protocol_profile" not in old.model_dump(mode="json")
    assert WindowInput.model_validate({**values, "protocol_profile": "oai-fixed-window/1"}, strict=True).protocol_profile
    with pytest.raises(ValidationError, match="UTC_day"):
        WindowInput.model_validate({**values, "protocol_profile": "oai-fixed-window/1", "lower_bound": datetime(2026, 1, 1, 12, tzinfo=UTC)}, strict=True)
