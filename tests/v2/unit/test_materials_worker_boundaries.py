"""Post-exit artifact bounds: no native parsing, database or model calls."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from citequest.materials_v2.isolation import _bounded_worker_result


def test_fast_exit_oversized_result_is_not_deserialized(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "result.json").write_bytes(b"X" * 257)
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: pytest.fail("oversized JSON was read"))
    result, size, violations = _bounded_worker_result(tmp_path, 256)
    assert result["code"] == "WORKER_ARTIFACT_LIMIT_EXCEEDED" and result["can_qa"] is False
    assert size == 257 and violations == ["WORKER_ARTIFACT_LIMIT_EXCEEDED"]


def test_final_logs_and_nested_temporary_files_count_toward_same_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "result.json").write_text('{"state":"ready","code":"OK"}', encoding="utf-8")
    (tmp_path / "stdout.log").write_bytes(b"x" * 100)
    (tmp_path / "tmp").mkdir()
    (tmp_path / "tmp/render.bin").write_bytes(b"x" * 100)
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: pytest.fail("aggregate excess JSON was read"))
    result, size, violations = _bounded_worker_result(tmp_path, 200)
    assert size > 200 and result["code"] == "WORKER_ARTIFACT_LIMIT_EXCEEDED" and violations


@pytest.mark.parametrize("raw, expected", [(b"{", "WORKER_INVALID_RESULT_JSON"), (b"\xff", "WORKER_INVALID_RESULT_JSON"),
                                         (b"[]", "WORKER_INVALID_RESULT_SHAPE"), (b'{"state":"ready"}', "WORKER_INVALID_RESULT_SHAPE"),
                                         (b'{"state":[]}', "WORKER_INVALID_RESULT_SHAPE"), (b'{"state":{}}', "WORKER_INVALID_RESULT_SHAPE")])
def test_malformed_worker_result_has_a_recordable_failure(tmp_path: Path, raw: bytes, expected: str) -> None:
    (tmp_path / "result.json").write_bytes(raw)
    result, _, _ = _bounded_worker_result(tmp_path, 4096)
    assert result["code"] == expected and result["can_qa"] is False


def test_valid_bounded_positive_and_missing_result(tmp_path: Path) -> None:
    assert _bounded_worker_result(tmp_path, 4096)[0]["code"] == "WORKER_NO_RESULT"
    data = {"state": "ready", "code": "TEXT_PARSED", "can_qa": True}
    (tmp_path / "result.json").write_text(json.dumps(data), encoding="utf-8")
    result, _, violations = _bounded_worker_result(tmp_path, 4096)
    assert result == data and not violations
