"""Immutable profile and observation deadline controls, with no live worker."""
from __future__ import annotations

import hashlib
import io
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

from citequest.materials_v2 import isolation, worker
from citequest.materials_v2.adapter import JobParser
from citequest.materials_v2.interfaces import ParserInput
from citequest.materials_v2.text import MaterialError, load_limits

ROOT = Path(__file__).resolve().parents[3]


def test_raw_profile_bytes_and_current_path_must_match(tmp_path: Path) -> None:
    path = tmp_path / "profile.json"
    raw = b'{"parser_versions":{"filetype":"1.2.0"},"variant":"A"}'
    path.write_bytes(raw)
    request = {"profile": str(path), "profile_json": raw.decode(), "profile_sha256": hashlib.sha256(raw).hexdigest()}
    assert worker._bound_profile(request)["variant"] == "A"
    with pytest.raises(MaterialError, match="PARSER_PROFILE_BINDING_MISMATCH"):
        worker._bound_profile({**request, "profile_json": raw.decode().replace('"A"', '"B"')})
    path.write_bytes(raw.replace(b'"A"', b'"B"'))
    with pytest.raises(MaterialError, match="PARSER_PROFILE_BINDING_MISMATCH"):
        worker._bound_profile(request)


def test_profile_mutated_during_parse_cannot_be_returned_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = tmp_path / "profile.json"
    raw = b'{"parser_versions":{"filetype":"1.2.0"},"variant":"A"}'
    profile.write_bytes(raw)
    source = tmp_path / "input.txt"
    source.write_bytes(b"actual fixed input")

    def parse(request: dict, captured: dict) -> dict:
        assert captured["variant"] == "A"
        profile.write_bytes(raw.replace(b'"A"', b'"B"'))
        assert captured["variant"] == "A"
        return {"state": "ready", "code": "TEXT_PARSED", "can_qa": True}

    monkeypatch.setattr(worker, "_parse", parse)
    worker.execute({"profile": str(profile), "profile_json": raw.decode(), "profile_sha256": hashlib.sha256(raw).hexdigest(),
                    "output": str(tmp_path), "source_path": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["state"] == "rejected" and result["can_qa"] is False
    assert result["code"] == "PARSER_PROFILE_CHANGED_DURING_EXECUTION"


@pytest.mark.parametrize("which", ["result", "receipt"])
def test_adapter_rejects_profile_label_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str) -> None:
    import citequest.materials_v2.adapter as adapter

    profile = ROOT / "runtime/s04/worker-profile.json"
    sha = hashlib.sha256(profile.read_bytes()).hexdigest()

    def fake_run(request: dict, **kwargs: dict) -> dict:
        assert request["expected_profile_sha256"] == sha
        return {"profile_sha256": "f" * 64 if which == "receipt" else sha,
                "result": {"state": "ready", "code": "TEXT_PARSED", "profile_sha256": "f" * 64 if which == "result" else sha, "source_sha256": "a" * 64, "blocks": []}}

    monkeypatch.setattr(adapter, "run_isolated", fake_run)
    request = ParserInput(tmp_path / "source.txt", "a" * 64, "text/plain", load_limits(ROOT / "contracts/v2/policy.json"), sha)
    with pytest.raises(MaterialError, match="PARSER_RESULT_PROFILE_MISMATCH"):
        JobParser(ROOT, profile, tmp_path).parse(request)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific worker profile; no native process is started")
@pytest.mark.parametrize("elapsed, expected", [(0.999, "PROBE_COMPLETED"), (1.001, "PARSER_DEADLINE_EXCEEDED")])
def test_fast_exit_observed_after_deadline_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, elapsed: float, expected: str) -> None:
    profile = ROOT / "runtime/s04/worker-profile.json"
    sha = hashlib.sha256(profile.read_bytes()).hexdigest()

    class Job:
        def __init__(self, *args: object) -> None:
            pass

        def inspect(self) -> dict:
            return {}

        def assign(self, process: object) -> None:
            pass

        def close(self) -> None:
            pass

    class ExitedProcess:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self.stdin, self.stdout, self.stderr = io.BytesIO(), io.BytesIO(), io.BytesIO()
            self.returncode, self.pid = 0, 12345
            (Path(kwargs["cwd"]) / "result.json").write_text(json.dumps({"state": "ready", "code": "PROBE_COMPLETED", "profile_sha256": sha}), encoding="utf-8")

        def poll(self) -> int:
            return 0

        def wait(self, **kwargs: object) -> int:
            return 0

    clock = iter([0] + [elapsed] * 100)
    monkeypatch.setattr(isolation, "WindowsJob", Job)
    monkeypatch.setattr(isolation, "_single_worker", lambda path: nullcontext())
    monkeypatch.setattr(isolation.subprocess, "Popen", ExitedProcess)
    monkeypatch.setattr(isolation.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=8 * 1024**3))
    receipt = isolation.run_isolated({"operation": "probe_cpu", "probe_deadline": 1}, root=ROOT, profile_path=profile,
                                    output=tmp_path / "worker-output", python=Path(sys.executable))
    assert receipt["result"]["code"] == expected
    assert receipt["worker_stopped"] is True
