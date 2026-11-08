"""Shared SQL-intent/alias protocol, with real readback after unknown RPC outcomes.

The repository owns source admission and SQL CAS. The alias is a repairable
convenience; every query must capture an explicit SQL collection + reader lease.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Protocol
from uuid import UUID

from .native import NativeMilvus, checked_database
from .schema import checked_collection_name


def publication_alias(catalog_name: str, security_domain: str) -> str:
    if (
        catalog_name not in {"standard_metadata", "application_papers", "managed_metadata"}
        or type(security_domain) is not str
    ):
        raise ValueError("declared_catalog_and_domain_required")
    if security_domain != "public" and (
        not security_domain.startswith("owner:") or security_domain != "owner:" + str(UUID(security_domain[6:]))
    ):
        raise ValueError("canonical_owner_or_public_domain_required")
    return checked_collection_name("cq_v2_" + catalog_name + "_" + hashlib.sha256(security_domain.encode()).hexdigest())


class Repository(Protocol):
    def publish_intent(self, identifier: UUID, intent_id: UUID, *, expected_epoch: int) -> dict: ...
    def record_alias(self, intent_id: UUID, receipt: dict) -> None: ...
    def commit_publish(self, intent_id: UUID) -> dict: ...
    def recovery_target(self, intent_id: UUID) -> dict: ...
    def acknowledge_rollback(self, intent_id: UUID, actual_collection: str | None) -> dict: ...


class AliasTransport:
    def __init__(self, native: NativeMilvus) -> None:
        self.native = native

    def target(self, alias: str) -> str | None:
        checked_collection_name(alias)
        database = checked_database(getattr(self.native, "database", "default"))
        response = self.native.call("list_aliases", collection_name="")
        # Pymilvus2.6.17's annotated List[str] is inaccurate: its real pinned
        # GrpcHandler returns {aliases, optional collection_name, db_name}.
        # The global empty-collection request must retain that exact scope.
        if (
            type(response) is not dict
            or set(response) - {"aliases", "collection_name", "db_name"}
            or response.get("collection_name", "") != ""
            or response.get("db_name", "default") != database
        ):
            raise ValueError("actual_global_alias_response_scope_invalid")
        aliases = response.get("aliases")
        if type(aliases) is not list or any(type(value) is not str for value in aliases):
            raise ValueError("actual_alias_list_shape_invalid")
        if alias not in aliases:
            return None
        actual = self.native.call("describe_alias", alias=alias)
        if (type(actual) is not dict or actual.get("alias", alias) != alias
                or actual.get("db_name", "default") != database):
            raise ValueError("actual_alias_description_database_scope_invalid")
        return checked_collection_name(actual["collection_name"])

    def point(self, alias: str, collection: str | None) -> dict:
        checked_collection_name(alias)
        if collection is not None:
            checked_collection_name(collection)
        before = self.target(alias)
        error = None
        if before != collection:
            try:
                if collection is None:
                    self.native.call("drop_alias", alias=alias)
                elif before is None:
                    self.native.call("create_alias", collection_name=collection, alias=alias)
                else:
                    self.native.call("alter_alias", collection_name=collection, alias=alias)
            except Exception as exc:
                # A deadline is an unknown mutation result. Always re-read;
                # never infer that the alias remained unchanged from the error.
                error = {
                    "type": type(exc).__name__,
                    "message": str(exc)[:600],
                    "outcome": "unknown_until_actual_alias_readback",
                }
        after = self.target(alias)
        if after != collection:
            raise RuntimeError("actual_alias_target_not_confirmed")
        receipt = {
            "alias_name": alias,
            "before_collection": before,
            "observed_collection": after,
            "mutation_error": error,
            "verified_by_real_alias_readback": True,
        }
        if getattr(self.native, "database", "default") != "default":
            receipt["milvus_database"] = checked_database(self.native.database)
        return receipt


class PublicationCoordinator:
    def __init__(self, transaction: Callable[[Callable[[Repository], Any]], Any], aliases: AliasTransport) -> None:
        self.tx, self.aliases = transaction, aliases

    def publish(
        self,
        generation_id: UUID,
        intent_id: UUID,
        *,
        expected_epoch: int,
        failpoint: Callable[[str], None] | None = None,
    ) -> dict:
        def boundary(name: str) -> None:
            if failpoint:
                failpoint(name)

        plan = self.tx(
            lambda repository: repository.publish_intent(generation_id, intent_id, expected_epoch=expected_epoch)
        )
        if plan["state"] == "committed":
            return {"status": "already_committed_sql_authoritative", "recovery": self.recover(intent_id)}
        if plan["state"] not in {"prepared", "alias_confirmed", "reconcile_required"}:
            raise ValueError("closed_publication_intent_cannot_be_reopened")
        boundary("after_durable_intent_before_alias")
        receipt = self.aliases.point(plan["alias_name"], plan["target_collection"])
        boundary("after_alias_before_sql_ack")
        self.tx(lambda repository: repository.record_alias(intent_id, receipt))
        boundary("after_alias_ack_before_sql_commit")
        result = self.tx(lambda repository: repository.commit_publish(intent_id))
        boundary("after_sql_commit")
        return {"status": "committed_sql_authoritative", "intent": result, "alias_receipt": receipt}

    def recover(self, intent_id: UUID) -> dict:
        plan = self.tx(lambda repository: repository.recovery_target(intent_id))
        receipt = self.aliases.point(plan["alias_name"], plan["authoritative_collection"])
        if plan["action"] == "rollback_uncommitted_alias":
            result = self.tx(
                lambda repository: repository.acknowledge_rollback(intent_id, receipt["observed_collection"])
            )
            return {"status": "uncommitted_alias_rolled_back", "intent": result, "alias_receipt": receipt}
        if plan["action"] != "repair_committed_alias":
            raise ValueError("unknown_recovery_action")
        return {"status": "committed_alias_reconciled_to_sql", "intent": plan, "alias_receipt": receipt}
