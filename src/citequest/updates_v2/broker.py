"""Dedicated Linux source-broker worker; startup never creates schema/resources."""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

from pydantic import Field
from sqlalchemy import select

from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor, Input
from . import models as m
from .gate import PgSourceGate, network_engine
from .locks import LINUX_PROFILE, native_lock, policy_sha
from .managed_scope import candidate_managed_scope
from .scheduler import create_dispatcher
from .transport import SourceFetcher
from .worker import UpdateWorker


class BrokerConfig(Input):
    schema_version: str = Field(pattern=r"^citequest-source-broker/1$")
    profile_id: str = Field(pattern=r"^linux-source-broker/1$")
    workspace: Path
    state_dir: Path
    business_dsn_file: Path
    owner_id: UUID
    dispatch_interval_seconds: int = Field(default=10, ge=1, le=300)


def configuration(path: Path) -> BrokerConfig:
    path = path.resolve(strict=True)
    value = json.loads(path.read_text("utf8"))
    for key in ("workspace", "state_dir", "business_dsn_file"):
        selected = Path(value[key])
        value[key] = str(selected if selected.is_absolute() else path.parent / selected)
    result = BrokerConfig.model_validate_json(json.dumps(value), strict=True)
    if result.workspace.resolve(strict=True) != Path(__file__).resolve().parents[3]:
        raise ValueError("source_broker_workspace_must_match_application_storage_root")
    return result


def adopt_policy(gate_engine, lock_root: Path) -> None:
    """Explicit maintenance action after the former broker has drained/stopped."""
    lock_root.mkdir(parents=True, exist_ok=True)
    held = [native_lock(lock_root / (provider + ".lock"), LINUX_PROFILE) for provider in ("arxiv", "crossref")]
    try:
        for lock in held:
            lock.acquire()
        with gate_engine.begin() as db:
            states = db.execute(select(m.NetworkState).order_by(m.NetworkState.provider).with_for_update()).mappings().all()
            live = db.scalar(select(m.NetworkAttempt.id).where(m.NetworkAttempt.state.in_(["in_flight", "unknown"]),
                m.NetworkAttempt.deadline_at > datetime.now(UTC)).limit(1))
            if live is not None or {row["provider"] for row in states} != {"arxiv", "crossref"}:
                raise ValueError("source_broker_drain_and_expired_unknown_window_required")
            from sqlalchemy import update
            db.execute(update(m.NetworkState).values(policy_sha256=policy_sha(LINUX_PROFILE)))
    finally:
        for lock in reversed(held):
            if lock.is_locked:
                lock.release()


async def execute(config: BrokerConfig, *, once: bool, adopt: bool) -> None:
    raw_dsn = config.business_dsn_file.read_text("utf8").strip()
    dsn = raw_dsn.replace("postgresql://", "postgresql+psycopg://", 1)
    database, gate_database = engine(dsn), network_engine(dsn)
    runtime = config.state_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    singleton = native_lock(runtime / "broker.lock", LINUX_PROFILE)
    singleton.acquire()
    try:
        if adopt:
            adopt_policy(gate_database, runtime / "source-locks")
        gate = PgSourceGate(gate_database, runtime / "source-locks", wait_for_slot=True, profile_id=config.profile_id)
        worker = UpdateWorker(sessions(database), Actor(config.owner_id, publisher=True), SourceFetcher(gate), runtime,
            managed_scope_factory=candidate_managed_scope, source_gate_profile=config.profile_id)
        if once:
            with worker.factory() as db:
                identifier = db.scalar(select(m.UpdateWindow.id).where(m.UpdateWindow.owner_id == config.owner_id,
                    m.UpdateWindow.state.in_(["queued", "running"])).order_by(m.UpdateWindow.created_at, m.UpdateWindow.id).limit(1))
            if identifier is not None:
                await worker.run(identifier)
            return
        scheduler = create_dispatcher(worker, interval_seconds=config.dispatch_interval_seconds)
        stopped = asyncio.Event()
        worker.stop_requested = stopped.is_set
        loop = asyncio.get_running_loop()
        for selected in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(selected, stopped.set)
        scheduler.start()
        try:
            await stopped.wait()
        finally:
            scheduler.pause()
            drain_deadline = loop.time() + 45
            while worker.running and loop.time() < drain_deadline:
                await asyncio.sleep(0.1)
            scheduler.shutdown(wait=True)
    finally:
        singleton.release()
        gate_database.dispose()
        database.dispose()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--adopt-network-policy", action="store_true")
    args = parser.parse_args()
    asyncio.run(execute(configuration(args.config), once=args.once, adopt=args.adopt_network_policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
