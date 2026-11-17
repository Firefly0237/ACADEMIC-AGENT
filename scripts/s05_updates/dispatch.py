"""Bounded trusted-operator dispatcher; existing frozen-input jobs only."""

import argparse
import sys
from pathlib import Path
from uuid import UUID

import anyio

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/s03'))
from bootstrap import dsn  # noqa: E402

from citequest.updates_v2.gate import PgSourceGate, network_engine  # noqa: E402
from citequest.updates_v2.managed_scope import candidate_managed_scope  # noqa: E402
from citequest.updates_v2.scheduler import create_dispatcher  # noqa: E402
from citequest.updates_v2.transport import SourceFetcher  # noqa: E402
from citequest.updates_v2.worker import UpdateWorker  # noqa: E402
from citequest.v2 import models as core  # noqa: E402
from citequest.v2.db import engine, sessions  # noqa: E402
from citequest.v2.domain import Actor  # noqa: E402


async def run(args):
    database, gate_db = engine(dsn('cq_s03_app')), network_engine(dsn('cq_s03_app'))
    factory = sessions(database)
    with factory.begin() as db:
        owner = db.get(core.Owner, args.owner)
        if owner is None or not owner.can_publish:
            raise ValueError('trusted_existing_publisher_owner_required')
    worker = UpdateWorker(factory, Actor(args.owner, publisher=True),
                          SourceFetcher(PgSourceGate(gate_db, ROOT / 'data/runtime-s05-updates/source-locks', wait_for_slot=True)),
                          ROOT / 'data/runtime-s05-updates', managed_scope_factory=candidate_managed_scope)
    scheduler = create_dispatcher(worker)
    try:
        scheduler.start()
        await anyio.sleep(args.run_for_seconds)
    finally:
        scheduler.shutdown(wait=False)
        gate_db.dispose()
        database.dispose()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner', type=UUID, required=True)
    parser.add_argument('--run-for-seconds', type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.run_for_seconds <= 86400:
        raise ValueError('explicit_dispatcher_runtime_1_to_86400_seconds_required')
    expected = ROOT / 'data/runtime-s05-updates/env/Scripts/python.exe'
    if Path(sys.executable).resolve() != expected.resolve():
        raise RuntimeError('locked_s05_updates_runtime_required')
    anyio.run(run, args)


if __name__ == '__main__':
    main()
