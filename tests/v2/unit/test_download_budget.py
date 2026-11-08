"""Operational source-byte accounting controls; no model money or product DB."""

import json

import pytest

from citequest.data_v2.download_budget import DownloadBudget


def test_unknown_reservation_survives_restart_and_never_resets_cap(tmp_path):
    budget = DownloadBudget(tmp_path, 100)
    first, amount = budget.reserve(60)
    second, _ = budget.reserve(30)
    assert amount == 60
    budget.settle(second, 10)
    assert budget.summary()["remaining"] == 30
    budget.close()
    resumed = DownloadBudget(tmp_path, 100)
    assert resumed.summary() == {
        "limit": 100,
        "settled_bytes": 10,
        "unresolved_reserved_bytes": 60,
        "remaining": 30,
        "settled_requests": 1,
        "unresolved_requests": 1,
    }
    resumed.settle(first, 25)
    assert resumed.remaining == 65
    resumed.close()
    with pytest.raises(ValueError, match="limit_or_history"):
        DownloadBudget(tmp_path, 200)


def test_overspend_double_settle_and_concurrent_writer_are_rejected(tmp_path):
    budget = DownloadBudget(tmp_path, 100)
    try:
        with pytest.raises(FileExistsError):
            DownloadBudget(tmp_path, 100)
        key, _ = budget.reserve(100)
        for invalid in (101, -1, True, 0.5):
            with pytest.raises(ValueError, match="outside_reservation"):
                budget.settle(key, invalid)
        with pytest.raises(ValueError, match="exhausted"):
            budget.reserve(1)
        assert budget.remaining == 0
        budget.settle(key, 100)
        with pytest.raises(ValueError):
            budget.settle(key, 0)
    finally:
        budget.close()


def test_invalid_or_partial_journal_fails_closed(tmp_path):
    for contents in (
        '{"kind":"limit","bytes":100}\n{"kind":',
        json.dumps({"kind": "limit", "bytes": 100}) + '\n{"kind":"settle","id":"unreserved","bytes":0}\n',
    ):
        (tmp_path / "requests.jsonl").write_text(contents, encoding="utf-8")
        with pytest.raises((ValueError, json.JSONDecodeError)):
            DownloadBudget(tmp_path, 100)
        assert not (tmp_path / "writer.lock").exists()


def test_nonpositive_or_noninteger_admission_has_no_reservation(tmp_path):
    budget = DownloadBudget(tmp_path, 100)
    try:
        for invalid in (0, -1, True, 0.5):
            with pytest.raises(ValueError):
                budget.reserve(invalid)
        assert budget.summary()["remaining"] == 100
    finally:
        budget.close()
