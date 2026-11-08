"""Cooperating-publisher session fence around the existing SQL/alias protocol.

The PostgreSQL advisory lock serializes trusted publication controllers for one
catalog. It is not an atomic Milvus fencing token: late server execution after
an unknown RPC still requires actual alias reconciliation. Queries retain the
SQL collection/reader-lease authority and never trust an alias as permission.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from psycopg import Connection

from citequest.index_v2.publication import publication_alias


class PublicationSessionFence:
    def __init__(self, connection: Connection, catalog: str, security_domain: str) -> None:
        self.connection = connection
        self.alias = publication_alias(catalog, security_domain)
        self.key = int.from_bytes(hashlib.sha256(("cq-publication:" + self.alias).encode()).digest()[:8], "big", signed=True)
        self.backend_pid: int | None = None

    @contextmanager
    def held(self) -> Iterator[PublicationSessionFence]:
        if not self.connection.autocommit or self.backend_pid is not None:
            raise ValueError("dedicated_autocommit_publication_session_required")
        result = self.connection.execute("SELECT pg_try_advisory_lock(%s) AS acquired,pg_backend_pid() AS pid", (self.key,)).fetchone()
        if not result["acquired"]:
            raise RuntimeError("catalog_publication_controller_busy")
        self.backend_pid = result["pid"]
        try:
            yield self
        finally:
            if not self.connection.closed:
                self.connection.execute("SELECT pg_advisory_unlock(%s)", (self.key,))
            self.backend_pid = None

    def check(self) -> None:
        if self.backend_pid is None or self.connection.closed:
            raise RuntimeError("publication_session_fence_not_held")
        unsigned = self.key & ((1 << 64) - 1)
        row = self.connection.execute("""SELECT pg_backend_pid() AS pid,EXISTS(
            SELECT 1 FROM pg_locks WHERE locktype='advisory' AND pid=pg_backend_pid() AND granted
             AND classid::bigint=%s AND objid::bigint=%s AND objsubid=1) AS held""",
                                      (unsigned >> 32, unsigned & 0xFFFFFFFF)).fetchone()
        if row["pid"] != self.backend_pid or row["held"] is not True:
            raise RuntimeError("publication_session_fence_lost")


class FencedAliasCalls:
    """Only declared alias operations; recheck SQL intent immediately before mutation."""

    METHODS = frozenset({"list_aliases", "describe_alias", "create_alias", "alter_alias", "drop_alias"})
    MUTATIONS = frozenset({"create_alias", "alter_alias", "drop_alias"})

    def __init__(self, native: Any, fence: PublicationSessionFence, authority: Callable[[], str | None]) -> None:
        self.native, self.fence, self.authority = native, fence, authority

    @property
    def database(self) -> str:
        # Preserve the immutable namespace through the SQL-session fence. The
        # wrapper must not silently reclassify an isolated transport as default.
        return getattr(self.native, "database", "default")

    def call(self, method: str, **kwargs: Any) -> Any:
        if method not in self.METHODS:
            raise ValueError("alias_only_native_transport")
        if method == "list_aliases":
            if kwargs != {"collection_name": ""}:
                raise ValueError("global_alias_scope_required")
        elif kwargs.get("alias") != self.fence.alias:
            raise ValueError("one_fenced_catalog_alias_required")
        self.fence.check()
        if method in self.MUTATIONS:
            wanted = self.authority()
            if (method == "drop_alias" and wanted is not None) or (
                method != "drop_alias" and kwargs.get("collection_name") != wanted
            ):
                raise RuntimeError("current_SQL_intent_forbids_alias_mutation")
            self.fence.check()
        return self.native.call(method, **kwargs)
