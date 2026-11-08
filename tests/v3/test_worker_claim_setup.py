"""Post-claim setup failures settle a synthetic execution, with no IO/model."""
import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from citequest.agents_v3 import worker


@pytest.mark.parametrize("failure", ["memory", "provider", "temporary_directory"])
def test_owned_setup_failure_finishes_without_leaving_a_running_command(monkeypatch, failure):
    finished, closed, observed = [], [], []
    campaign = SimpleNamespace(require_worker=lambda *_: None)
    database = SimpleNamespace(development_dispatch_policy=None, development_campaign_policy=campaign)
    execution = SimpleNamespace(task_id=uuid4(), run_id=uuid4(),
        finish=lambda state, **kwargs: finished.append((state, kwargs)))

    @contextmanager
    def claimed(*_args, **_kwargs):
        yield execution

    def flush():
        if failure == "memory":
            raise worker.AgentError("synthetic_memory_setup_failed")

    class Provider:
        def __init__(self, *_args, **_kwargs):
            if failure == "provider":
                raise worker.AgentError("live_provider_metadata_expired_or_changed")

        async def close(self):
            closed.append(True)

    def mkdir(**_kwargs):
        raise OSError("synthetic_unwritable_temporary_directory")

    config = SimpleNamespace(mode="live", database=lambda: database, limits=object(),
        api_key_file=Path("synthetic-key-reference"), metadata_receipt_file=Path("synthetic-receipt-reference"),
        temporary_directory=SimpleNamespace(mkdir=mkdir))
    monkeypatch.setattr(worker, "claim", claimed)
    monkeypatch.setattr(worker, "MemoryService", lambda _db: SimpleNamespace(flush=flush))
    monkeypatch.setattr(worker, "ResponsesProvider", Provider)
    monkeypatch.setattr(worker, "private_runtime_observation", lambda *args, **kwargs: observed.append(args))
    result = asyncio.run(worker.run_once(config))
    expected = {"memory": "synthetic_memory_setup_failed", "provider": "live_provider_metadata_expired_or_changed",
        "temporary_directory": "OSError"}[failure]
    assert result == {"status": "failed", "task_id": str(execution.task_id), "error_code": expected}
    assert finished == [("failed", {"error": expected})]
    assert closed == ([True] if failure == "temporary_directory" else [])
    assert observed == [("agent_worker_failure", "run_once", "failed")]
