"""Service drain controls use a synthetic consumer and never touch a queue/model."""
import asyncio
import json
import signal

import pytest

from citequest.agents_v3 import worker


@pytest.mark.parametrize("status", ["processed", "failed"])
def test_signal_drains_active_command_without_cancelling_or_claiming_again(monkeypatch, capsys, status):
    callbacks, restored, calls = {}, [], []

    async def scenario():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_signal_handler", lambda signum, callback: callbacks.setdefault(signum, callback))
        monkeypatch.setattr(loop, "remove_signal_handler", lambda signum: callbacks.pop(signum))
        monkeypatch.setattr(signal, "getsignal", lambda signum: signal.SIG_DFL)
        monkeypatch.setattr(signal, "signal", lambda signum, previous: restored.append((signum, previous)))

        async def consume(config):
            calls.append(config)
            loop.call_soon(callbacks[signal.SIGTERM])
            await asyncio.sleep(0)
            # This line must still run after the drain request.
            return {"status": status, "task_id": "synthetic-task", "error_code": "synthetic" if status == "failed" else None}

        monkeypatch.setattr(worker, "run_once", consume)
        await worker.main_async("synthetic-config", False, drain_on_signal=True)

    asyncio.run(scenario())
    assert calls == ["synthetic-config"]
    receipts = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["status"] for row in receipts] == ["draining", status]
    assert not callbacks and restored == [(signal.SIGTERM, signal.SIG_DFL), (signal.SIGINT, signal.SIG_DFL)]


def test_once_default_never_installs_a_signal_handler(monkeypatch, capsys):
    async def scenario():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_signal_handler", lambda *_: pytest.fail("default changed signal handling"))

        async def consume(_config):
            return {"status": "idle"}

        monkeypatch.setattr(worker, "run_once", consume)
        await worker.main_async("synthetic-config", True)

    asyncio.run(scenario())
    assert json.loads(capsys.readouterr().out) == {"status": "idle"}


def test_admission_exception_exits_without_restart_and_restores_handlers(monkeypatch):
    callbacks, restored = {}, []

    async def scenario():
        loop = asyncio.get_running_loop()
        monkeypatch.setattr(loop, "add_signal_handler", lambda signum, callback: callbacks.setdefault(signum, callback))
        monkeypatch.setattr(loop, "remove_signal_handler", lambda signum: callbacks.pop(signum))
        monkeypatch.setattr(signal, "getsignal", lambda signum: signal.SIG_DFL)
        monkeypatch.setattr(signal, "signal", lambda signum, previous: restored.append(signum))

        async def consume(_config):
            raise worker.AgentError("synthetic_unknown_admission_hold")

        monkeypatch.setattr(worker, "run_once", consume)
        with pytest.raises(worker.AgentError, match="synthetic_unknown_admission_hold"):
            await worker.main_async("synthetic-config", False, drain_on_signal=True)

    asyncio.run(scenario())
    assert not callbacks and restored == [signal.SIGTERM, signal.SIGINT]
