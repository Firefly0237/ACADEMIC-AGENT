"""One-host kernel exclusion plus persistent PostgreSQL request accounting."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from importlib.metadata import version
from pathlib import Path
from uuid import UUID, uuid4

import anyio
from filelock import Timeout
from sqlalchemy import create_engine, insert, select, text, update
from sqlalchemy.engine import Connection, Engine

from citequest.materials_v2.reference_fetch import FetchContext, FetchError
from citequest.updates_v2 import models as m
from citequest.updates_v2.context import WindowFetchContext
from .locks import WINDOWS_PROFILE, native_lock, policy, policy_sha

POLICY = {"id": "cq.shared-source-network/1", "backend": "WindowsFileLock4.2.9+PG-session-advisory",
          "per_hop_cooldown_seconds": {"arxiv": 3.1, "crossref": 1.0}, "unknown_ack": "retain_unknown_fail_closed"}
POLICY_SHA = hashlib.sha256(json.dumps(POLICY, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def network_engine(url: str) -> Engine:
    if not url.startswith("postgresql+psycopg://"):
        raise ValueError("source_gate_requires_product_PostgreSQL")
    return create_engine(url, pool_size=1, max_overflow=0, pool_timeout=1, pool_pre_ping=True,
                         hide_parameters=True, connect_args={"connect_timeout": 3, "application_name": "citequest-source-gate",
                         "options": "-c statement_timeout=5000 -c lock_timeout=1000 -c idle_in_transaction_session_timeout=5000"})


class PgSourceGate:
    """A fresh native lock instance per attempt prevents recursive sharing.

    The OS lock covers the socket lifetime even when the DB session dies. The
    deliberately conservative cooldown occurs after acquiring that lock, so a
    restart/unknown previous acknowledgement cannot produce an immediate burst.
    No network runs in a SQL transaction; the small gate pool is independent.
    """
    production = True

    def __init__(self, engine: Engine, lock_root: Path, *, wait_for_slot: bool = False,
                 profile_id: str = WINDOWS_PROFILE) -> None:
        self.profile_id, self.policy = profile_id, policy(profile_id)
        self.policy_sha256 = policy_sha(profile_id)
        native_lock(lock_root / "profile-check.lock", profile_id)
        if version("filelock") != "4.2.9":
            raise RuntimeError("source_gate_filelock_version_mismatch")
        self.engine, self.lock_root = engine, lock_root.resolve()
        self.wait_for_slot = wait_for_slot
        self.lock_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _slot(provider: str) -> int:
        return int.from_bytes(hashlib.sha256(("citequest-source:" + provider).encode()).digest()[:8], "big", signed=True)

    def _claim(self, provider: str, wait_seconds: float = 0.0, *, reconcile: bool = True) -> Connection:
        conn = self.engine.connect()
        try:
            with conn.begin():
                if wait_seconds:
                    milliseconds = max(1, min(30000, int(wait_seconds * 1000)))
                    conn.exec_driver_sql(f'SET LOCAL statement_timeout={milliseconds}')
                    conn.exec_driver_sql(f'SET LOCAL lock_timeout={milliseconds}')
                    conn.execute(text('SELECT pg_advisory_lock(:slot)'), {'slot': self._slot(provider)})
                elif not conn.execute(text("SELECT pg_try_advisory_lock(:slot)"), {"slot": self._slot(provider)}).scalar():
                    raise FetchError("SOURCE_PROVIDER_DATABASE_BUSY")
                row = conn.execute(select(m.NetworkState).where(m.NetworkState.provider == provider)).mappings().first()
                if row is None or row["policy_sha256"] != self.policy_sha256:
                    raise FetchError("SOURCE_GATE_POLICY_OR_TABLE_UNAVAILABLE")
                if reconcile and row["current_attempt_id"]:
                    previous = conn.execute(select(m.NetworkAttempt.receipt).where(m.NetworkAttempt.id == row['current_attempt_id'])).scalar_one()
                    conn.execute(update(m.NetworkAttempt).where(m.NetworkAttempt.id == row["current_attempt_id"], m.NetworkAttempt.state == "in_flight")
                                 .values(state="unknown", finished_at=datetime.now(UTC), receipt={**(previous or {}), "recovery": "exclusive_kernel_lock_acquired; old execution no longer owns local socket gate; original request outcome remains unknown"}))
            conn.execution_options(isolation_level="AUTOCOMMIT")
            return conn
        except BaseException:
            conn.invalidate()  # Never return an accidentally held session lock to the pool.
            conn.close()
            raise

    def _reconcile_after_kernel(self, conn: Connection, provider: str) -> None:
        conn.execution_options(isolation_level='READ COMMITTED')
        with conn.begin():
            row = conn.execute(select(m.NetworkState).where(m.NetworkState.provider == provider)).mappings().one()
            if row['current_attempt_id']:
                previous = conn.execute(select(m.NetworkAttempt.receipt).where(m.NetworkAttempt.id == row['current_attempt_id'])).scalar_one()
                conn.execute(update(m.NetworkAttempt).where(m.NetworkAttempt.id == row['current_attempt_id'], m.NetworkAttempt.state == 'in_flight')
                             .values(state='unknown', finished_at=datetime.now(UTC), receipt={**(previous or {}), 'recovery': 'exclusive_kernel_lock_acquired; prior local socket owner ended; outcome remains unknown'}))
        conn.execution_options(isolation_level='AUTOCOMMIT')

    def _start(self, conn: Connection, provider: str, url: str, remaining: float, caller: dict) -> dict:
        conn.execution_options(isolation_level="READ COMMITTED")
        attempt_id = uuid4()
        with conn.begin():
            row = conn.execute(select(m.NetworkState).where(m.NetworkState.provider == provider).with_for_update()).mappings().one()
            fence = row["fence"] + 1
            now = conn.execute(text("SELECT clock_timestamp()")).scalar_one()
            request_sha = hashlib.sha256(json.dumps({"method": "GET", "url": url}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            conn.execute(insert(m.NetworkAttempt).values(id=attempt_id, provider=provider, fence=fence, url=url,
                         request_sha256=request_sha, owner_pid=os.getpid(), started_at=now,
                         deadline_at=now + timedelta(seconds=remaining), state="in_flight", receipt=caller or None))
            conn.execute(update(m.NetworkState).where(m.NetworkState.provider == provider).values(fence=fence, current_attempt_id=attempt_id, last_started_at=now))
            backend_pid = conn.execute(text("SELECT pg_backend_pid()")).scalar_one()
        conn.execution_options(isolation_level="AUTOCOMMIT")
        return {**caller, "attempt_id": str(attempt_id), "provider": provider, "fence": fence, "started_at": now.isoformat(),
                "backend_pid": backend_pid, "gate_mode": "shared_kernel_lock_and_PG", "source_gate_profile": self.profile_id,
                "source_gate_policy_sha256": self.policy_sha256, "request_intent_sha256": request_sha,
                "network_transaction_open": False}

    def _finish(self, conn: Connection, receipt: dict, error: BaseException | None) -> None:
        attempt = UUID(receipt["attempt_id"])
        payload = {key: value for key, value in receipt.items() if key != "backend_pid"}
        payload["error_type"] = type(error).__name__ if error else None
        try:
            conn.commit()
            conn.execution_options(isolation_level="READ COMMITTED")
            with conn.begin():
                updated = conn.execute(update(m.NetworkAttempt).where(m.NetworkAttempt.id == attempt, m.NetworkAttempt.state == "in_flight")
                                       .values(state="settled_error" if error else "settled", finished_at=datetime.now(UTC), receipt=payload))
                if updated.rowcount != 1:
                    raise FetchError("SOURCE_GATE_ACK_FENCE_MISMATCH")
                conn.execute(update(m.NetworkState).where(m.NetworkState.provider == receipt["provider"], m.NetworkState.fence == receipt["fence"],
                                                         m.NetworkState.current_attempt_id == attempt).values(current_attempt_id=None))
        except Exception as exc:
            # A lost SQL session never implies the HTTP request did not happen.
            conn.invalidate()
            conn.close()
            try:
                with self.engine.begin() as recovery:
                    recovery.execute(update(m.NetworkAttempt).where(m.NetworkAttempt.id == attempt).values(
                        state="unknown", finished_at=datetime.now(UTC), receipt={**payload, "gate_ack_lost": True, "ack_error_type": type(exc).__name__}))
                    recovery.execute(update(m.NetworkState).where(m.NetworkState.provider == receipt["provider"], m.NetworkState.current_attempt_id == attempt)
                                     .values(current_attempt_id=None))
            except Exception:
                # The original durable in_flight row remains for a later reconciler.
                pass
            raise FetchError("SOURCE_GATE_ACK_UNKNOWN") from exc

    def _close(self, conn: Connection, provider: str) -> None:
        try:
            if not conn.closed:
                conn.rollback()
                conn.execution_options(isolation_level="AUTOCOMMIT")
                conn.execute(text("SELECT pg_advisory_unlock(:slot)"), {"slot": self._slot(provider)})
                conn.commit()
        except Exception:
            conn.invalidate()
            raise
        finally:
            conn.close()

    def record_reference_transport(self, transport: dict) -> None:
        """Keep the full S05 audit while the public reference response uses v1."""
        gate = transport.get('gate')
        if not gate:
            return  # Rejection before a durable attempt is not a sent request.
        trace = {key: value for key, value in transport.items() if key != 'gate'}
        trace_sha = hashlib.sha256(json.dumps(trace, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        with self.engine.begin() as db:
            attempt = db.execute(select(m.NetworkAttempt).where(m.NetworkAttempt.id == UUID(gate['attempt_id'])).with_for_update()).mappings().one()
            if attempt['fence'] != gate['fence'] or attempt['url'] != transport['url']:
                raise FetchError('SOURCE_REFERENCE_TRACE_FENCE_MISMATCH')
            payload = dict(attempt['receipt'] or {})
            if payload.get('reference_transport_sha256') not in {None, trace_sha}:
                raise FetchError('SOURCE_REFERENCE_TRACE_REPLAY_CONFLICT')
            payload.update(reference_transport_trace=trace, reference_transport_sha256=trace_sha)
            db.execute(update(m.NetworkAttempt).where(m.NetworkAttempt.id == attempt['id']).values(receipt=payload))

    @asynccontextmanager
    async def permit(self, provider: str, url: str, context: FetchContext) -> AsyncIterator[dict]:
        if provider not in self.policy["per_hop_cooldown_seconds"]:
            raise FetchError("SOURCE_PROVIDER_REJECTED")
        lock = native_lock(self.lock_root / (provider + ".lock"), self.profile_id)
        conn = None
        receipt = None
        error = None
        try:
            if self.wait_for_slot:
                # PG's session-lock waiter queue prevents the metadata owner
                # from barging ahead with PDF while another process is queued.
                # It still cannot open a socket before the old OS owner ends.
                try:
                    conn = await anyio.to_thread.run_sync(lambda: self._claim(provider, context.remaining(), reconcile=False))
                except FetchError:
                    raise
                except Exception as exc:
                    raise FetchError('SOURCE_GATE_DATABASE_UNAVAILABLE_OR_QUEUE_TIMEOUT') from exc
            try:
                timeout = max(0.0, context.remaining() - 0.1) if self.wait_for_slot else 0.0
                await anyio.to_thread.run_sync(lambda: lock.acquire(timeout=timeout, poll_interval=0.05))
            except Timeout as exc:
                raise FetchError("SOURCE_PROVIDER_KERNEL_BUSY") from exc
            try:
                if conn is None:
                    conn = await anyio.to_thread.run_sync(self._claim, provider)
                else:
                    await anyio.to_thread.run_sync(self._reconcile_after_kernel, conn, provider)
            except FetchError:
                raise
            except Exception as exc:
                raise FetchError("SOURCE_GATE_DATABASE_UNAVAILABLE") from exc
            with anyio.fail_after(context.remaining()):
                await anyio.sleep(self.policy["per_hop_cooldown_seconds"][provider])
            caller = {}
            if isinstance(context, WindowFetchContext):
                caller = {'source_window_id': str(UUID(context.source_window_id)), 'source_sequence': context.source_sequence}
                if type(context.source_sequence) is not int or context.source_sequence < 1:
                    raise FetchError('SOURCE_WINDOW_NETWORK_CALLER_INVALID')
            receipt = await anyio.to_thread.run_sync(self._start, conn, provider, url, context.remaining(), caller)
            try:
                yield receipt
            except BaseException as exc:
                error = exc
                raise
            finally:
                with anyio.CancelScope(shield=True):
                    await anyio.to_thread.run_sync(self._finish, conn, receipt, error)
        finally:
            with anyio.CancelScope(shield=True):
                try:
                    if conn is not None:
                        await anyio.to_thread.run_sync(self._close, conn, provider)
                finally:
                    if lock.is_locked:
                        await anyio.to_thread.run_sync(lock.release)
