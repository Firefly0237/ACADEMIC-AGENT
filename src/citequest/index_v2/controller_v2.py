"""Bounded typed-source staging; native/update admission never implies QA."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from citequest.materials_v2.chunking import SourceChunker
from citequest.v2.domain import Actor
from citequest.v2.services import Catalog
from citequest.v2.storage import CAS

from .native import NativeMilvus
from .repository_v2 import VariantIndexRepository
from .variant_compiler import compile_unit


class VariantIndexController:
    def before_native_configuration(self, generation_id: UUID) -> None:
        """Trusted subclasses can require deployment binding before native writes."""
    @staticmethod
    def admit_window(kind: str, unit_limit: int | None, seconds: float) -> None:
        if (type(unit_limit) is not int or not 1 <= unit_limit <= 100
            or type(seconds) not in {float, int} or not 0 < seconds <= 900):
            raise ValueError("typed_pilot_maximum_100_units_900_seconds")
    def __init__(
        self,
        factory: sessionmaker[Session],
        actor: Actor,
        *,
        store: CAS,
        chunker: SourceChunker,
        native: NativeMilvus,
        policy_sha256: str,
        evaluation_sha256: str,
    ) -> None:
        self.factory, self.actor, self.store = factory, actor, store
        self.chunker, self.native = chunker, native
        self.policy_sha256, self.evaluation_sha256 = policy_sha256, evaluation_sha256

    def tx(self, operation: Callable[[VariantIndexRepository], Any]) -> Any:
        with self.factory.begin() as db:
            return operation(
                VariantIndexRepository(
                    Catalog(db, self.actor),
                    store=self.store,
                    policy_sha256=self.policy_sha256,
                    evaluation_sha256=self.evaluation_sha256,
                )
            )

    def run(
        self,
        kind: str,
        root_id: UUID,
        generation_id: UUID,
        *,
        unit_limit: int | None,
        seconds: float,
        checkpoint: Callable[[dict], None],
        allow_uncertain_replay: bool = False,
    ) -> dict:
        # This controller is intentionally only a bounded technical pilot. A
        # whole-corpus profile must be declared after measured pilot resources.
        self.admit_window(kind, unit_limit, seconds)
        started = time.monotonic()
        deadline = started + seconds
        self.native.absolute_deadline = deadline
        summary = self.tx(
            lambda r: r.begin_source(
                kind,
                root_id,
                generation_id,
                tokenizer_profile_sha256=self.chunker.profile_sha256,
                unit_limit=unit_limit,
            )
        )
        checkpoint({"event": "typed_generation_opened", "generation": summary})
        while summary["state"] == "preparing" and time.monotonic() < deadline - 10:
            units = self.tx(lambda r: r.pending_units(generation_id, limit=10))
            if not units:
                summary = self.tx(lambda r: r.seal_preparation(generation_id))
                checkpoint({"event": "typed_source_preparation_sealed", "generation": summary})
                break
            with self.chunker.verified_batch():
                compiled = [(unit, compile_unit(unit, self.chunker)) for unit in units]
            if time.monotonic() >= deadline - 10:
                return {"status": "window_stopped", "phase": "compilation_not_accepted", "generation": summary}
            for unit, result in compiled:
                self.tx(lambda r, u=unit, c=result: r.stage_source_unit(generation_id, u, c))
            checkpoint({"event": "typed_source_units_staged_not_searchable", "units": len(compiled)})
        summary = self.tx(lambda r: r.describe(generation_id))
        if summary["state"] == "verified":
            return {"status": "already_verified_not_published", "generation": summary}
        if summary["state"] == "empty_unindexed":
            return {
                "status": "explicit_empty_or_denied_no_native_collection",
                "generation": summary,
                "search_published": False,
            }
        if summary["state"] != "building" or time.monotonic() >= deadline - 20:
            return {"status": "window_stopped", "generation": summary}
        self.before_native_configuration(generation_id)
        configuration = self.native.create(summary["collection_name"])
        self.tx(lambda r: r.bind_server_proof(generation_id, configuration))
        checkpoint({"event": "native_configuration_verified", "proof": configuration})
        while time.monotonic() < deadline - 15:
            claim = self.tx(lambda r: r.batch_claim(generation_id))
            if claim is None:
                break
            if claim["status"] == "busy":
                return {"status": "busy", "batch": claim, "generation": summary}
            receipt = None
            if claim["read_before_write"]:
                try:
                    _, prior = self.native.read_ids(
                        claim["collection_name"], claim["rows"], seconds=min(45, deadline - time.monotonic() - 10)
                    )
                    receipt = {
                        "state": "verified" if prior["complete"] else "uncertain",
                        "readback": prior,
                        "recovery": "actual_read_before_any_replay",
                        "application_automatic_retries": 0,
                    }
                except Exception as exc:
                    receipt = {
                        "state": "uncertain",
                        "readback_error": {"type": type(exc).__name__, "message": str(exc)[:600]},
                        "recovery": "read_failed_no_resend",
                    }
                if receipt["state"] != "verified" and not allow_uncertain_replay:
                    self.tx(
                        lambda r, c=claim, v=receipt: r.batch_result(
                            generation_id, UUID(c["batch_id"]), UUID(c["lease_id"]), v
                        )
                    )
                    checkpoint({"event": "uncertain_not_resent", "batch": claim["batch_id"], "receipt": receipt})
                    return {"status": "reconciliation_required", "generation": summary}
            if receipt is None or receipt["state"] != "verified":
                prior = receipt
                receipt = self.native.upsert_and_read(
                    claim["collection_name"], claim["rows"], seconds=min(120, deadline - time.monotonic() - 10)
                )
                if prior is not None:
                    receipt["prior_readback"] = prior
                    receipt["explicit_replay"] = (
                        "bounded at-least-once upsert after actual readback; not inferred never-written"
                    )
            self.tx(
                lambda r, c=claim, v=receipt: r.batch_result(generation_id, UUID(c["batch_id"]), UUID(c["lease_id"]), v)
            )
            checkpoint({"event": "batch_result", "batch_id": claim["batch_id"], "receipt": receipt})
            if receipt["state"] != "verified":
                return {"status": "reconciliation_required", "generation": summary}
        # Never scan an incomplete whole generation as a substitute for its
        # remaining delivery: the durable outbox/batches are the checkpoint.
        outstanding = self.tx(lambda r: r.remaining_batches(generation_id))
        if outstanding or time.monotonic() >= deadline - 10:
            return {
                "status": "window_stopped",
                "remaining_batches": outstanding,
                "generation": summary,
                "elapsed_seconds": time.monotonic() - started,
            }
        rows = self.tx(lambda r: r.expected_rows(generation_id))
        proof = self.native.scan_all(
            summary["collection_name"], rows, seconds=min(300, deadline - time.monotonic() - 5)
        )
        if time.monotonic() >= deadline:
            checkpoint({"event": "late_complete_readback_not_accepted", "proof": proof})
            return {"status": "window_stopped", "generation": summary}
        summary = self.tx(lambda r: r.accept_readback(generation_id, proof))
        return {
            "status": "verified_not_published",
            "generation": summary,
            "complete_readback": proof,
            "elapsed_seconds": time.monotonic() - started,
            "search_published": False,
            "index_staging_is_not_QA_permission": True,
        }
