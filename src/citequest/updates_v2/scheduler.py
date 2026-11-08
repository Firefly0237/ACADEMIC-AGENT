"""APScheduler only dispatches fixed PG jobs; no serialized arbitrary callables."""

import logging
from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import select

from citequest.updates_v2.models import UpdateWindow
from citequest.v2.domain import DomainError

logger = logging.getLogger(__name__)


def create_dispatcher(worker, *, interval_seconds: int = 10) -> AsyncIOScheduler:
    if type(interval_seconds) is not int or not 1 <= interval_seconds <= 300:
        raise ValueError('bounded_dispatch_interval_required')

    async def dispatch_one():
        with worker.factory.begin() as db:
            identifier = db.scalar(select(UpdateWindow.id).where(UpdateWindow.owner_id == worker.actor.owner_id,
                                   UpdateWindow.state.in_(['queued', 'running'])).order_by(UpdateWindow.created_at, UpdateWindow.id).limit(1))
        if identifier is not None:
            try:
                await worker.run(identifier)
            except DomainError as exc:
                if exc.code != 'source_window_worker_busy':
                    logger.error('source_dispatch_failed code=%s', exc.code)

    # Business rows, counters and result receipts persist in PG/CAS, not pickled
    # APScheduler jobs. Restart always recreates this one trusted callable.
    scheduler = AsyncIOScheduler(timezone=UTC, job_defaults={'coalesce': True, 'max_instances': 1, 'misfire_grace_time': 30})
    scheduler.add_job(dispatch_one, 'interval', seconds=interval_seconds, id='source_pg_dispatch', replace_existing=True,
                      next_run_time=datetime.now(UTC))
    return scheduler
