"""Run official Alembic against the new S03 database; URL stays out of argv."""

import argparse
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from bootstrap import dsn, permissions

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["upgrade", "downgrade", "check"])
    parser.add_argument("revision", nargs="?", default="head")
    args = parser.parse_args()
    os.environ["CITEQUEST_S03_MIGRATION_URL"] = dsn("cq_s03_migrator")
    cfg = Config(str(ROOT / "alembic-app.ini"))
    if args.action == "check":
        command.check(cfg)
    else:
        getattr(command, args.action)(cfg, args.revision)
    if args.action == "upgrade":
        permissions()
