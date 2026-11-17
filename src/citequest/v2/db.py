"""Product PostgreSQL engine; no runtime DDL, implicit SQLite or probe database."""

from __future__ import annotations

import math

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker


def engine(url: str, *, pool_size: int = 3, max_overflow: int = 0,
           pool_timeout: float = 10, connect_timeout: int = 10) -> Engine:
    if (type(pool_size) is not int or not 1 <= pool_size <= 32
        or type(max_overflow) is not int or not 0 <= max_overflow <= 16
        or type(connect_timeout) is not int or not 1 <= connect_timeout <= 30
        or type(pool_timeout) not in {int, float} or not 0 < pool_timeout <= 60
        or not math.isfinite(pool_timeout)):
        raise ValueError("invalid_explicit_database_pool_profile")
    parsed = make_url(url)
    if parsed.drivername != "postgresql+psycopg":
        raise ValueError("Product database requires explicit postgresql+psycopg")
    return create_engine(
        parsed,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        hide_parameters=True,
        connect_args={"connect_timeout": connect_timeout},
        pool_timeout=pool_timeout,
    )


def sessions(db: Engine) -> sessionmaker[Session]:
    return sessionmaker(db, expire_on_commit=False)
