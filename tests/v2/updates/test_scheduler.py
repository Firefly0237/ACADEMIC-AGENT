"""Real APScheduler clock/executor, controlled no-network worker dependency."""

from contextlib import contextmanager
from uuid import uuid4

import anyio

from citequest.updates_v2.scheduler import create_dispatcher
from citequest.v2.domain import Actor


def test_framework_never_overlaps_long_dispatch_and_no_job_pickle():
    identifier = uuid4()
    class Factory:
        @contextmanager
        def begin(self):
            yield self
        def scalar(self, query):
            return identifier

    class Worker:
        actor = Actor(uuid4())
        factory = Factory()
        def __init__(self):
            self.calls, self.active, self.maximum = 0, 0, 0
        async def run(self, value):
            assert value == identifier
            self.active += 1
            self.calls += 1
            self.maximum = max(self.maximum, self.active)
            try:
                await anyio.sleep(2.1)
            finally:
                self.active -= 1

    async def run():
        worker = Worker()
        scheduler = create_dispatcher(worker, interval_seconds=1)
        scheduler.start()
        try:
            await anyio.sleep(3.3)
            assert worker.calls == 2 and worker.maximum == 1
            job = scheduler.get_job('source_pg_dispatch')
            assert job.coalesce and job.max_instances == 1
            assert type(scheduler._jobstores['default']).__name__ == 'MemoryJobStore'
        finally:
            scheduler.shutdown(wait=False)
            await anyio.sleep(0.1)
    anyio.run(run)
