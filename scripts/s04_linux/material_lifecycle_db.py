"""Exact fixture business actions/observations; no model or object-store call."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url

from citequest.materials_v2 import models as m
from citequest.materials_v2.services import MaterialService
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile",type=Path,required=True)
    parser.add_argument("--dsn",type=Path,required=True)
    parser.add_argument("--action",choices=("cancel","observe"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    profile=json.loads(args.profile.read_bytes())
    dsn=args.dsn.read_text().strip();url=make_url(dsn)
    assert url.database==profile["database_name"]=="cq_s15_blob_20261106_r2" and url.username=="cq_s08_business"
    database=engine(dsn,connect_timeout=3,pool_timeout=3);factory=sessions(database)
    try:
        with factory.begin() as db:
            if args.action=="observe":db.execute(text("SET TRANSACTION READ ONLY"))
            db.execute(text("SET LOCAL statement_timeout='5s'"))
            actor=Actor(UUID(profile["assigned_owner_id"]));file_id=UUID(profile["assigned_file_id"])
            row=db.scalar(select(m.MaterialIngestion).where(m.MaterialIngestion.file_id==file_id))
            assert row and row.owner_id==actor.owner_id and row.execution_id==UUID(profile["assigned_execution_id"])
            if args.action=="cancel":
                assert row.state in {"validating","parsing"}
                MaterialService(db,actor,None).cancel(file_id)
                db.flush()
            value={"state":row.state,"execution_outcome":row.execution_outcome,"error_code":row.error_code,
                "execution_id":str(row.execution_id),"lease_id":str(row.lease_id) if row.lease_id else None,
                "representation_id":row.representation_id,"revision":row.revision,"budget_seconds":row.execution_budget_seconds,
                "representations":db.scalar(select(func.count()).select_from(m.ParsedRepresentation).where(m.ParsedRepresentation.file_id==file_id)),
                "fragments":db.scalar(select(func.count()).select_from(m.MaterialFragment).where(m.MaterialFragment.file_id==file_id))}
        args.out.write_text(json.dumps({"status":"observed","action":args.action,"ingestion":value,
            "model_calls":0,"main_database":False},indent=2)+'\n')
    finally:database.dispose()


if __name__=="__main__":main()
