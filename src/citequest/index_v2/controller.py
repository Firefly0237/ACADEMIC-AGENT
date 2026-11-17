"""Sequential delivery around short SQL units of work, never network in SQL."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session, sessionmaker

from citequest.materials_v2.chunking import SourceChunker
from citequest.v2.domain import Actor
from citequest.v2.services import Catalog

from .native import NativeMilvus
from .projection import compile_projection
from .repository import IndexRepository


class IndexController:
    def __init__(
        self,
        factory: sessionmaker[Session],
        actor: Actor,
        *,
        chunker: SourceChunker,
        native: NativeMilvus,
        policy_sha256: str,
        evaluation_sha256: str,
    ) -> None:
        self.factory, self.actor, self.chunker, self.native = factory, actor, chunker, native
        self.policy_sha256, self.evaluation_sha256 = policy_sha256, evaluation_sha256

    def tx(self, action: Callable[[IndexRepository], Any]) -> Any:
        with self.factory.begin() as db:
            return action(
                IndexRepository(
                    Catalog(db, self.actor), policy_sha256=self.policy_sha256, evaluation_sha256=self.evaluation_sha256
                )
            )

    def run(
        self,
        snapshot_id: UUID,
        generation_id: UUID,
        *,
        unit_limit: int | None,
        seconds: float,
        checkpoint: Callable[[dict], None],
        allow_uncertain_replay: bool = False,
    ) -> dict:
        if type(seconds) not in {int, float} or not 0 < seconds <= 3600:
            raise ValueError("bounded_controller_window_required")
        if unit_limit == 100 and seconds > 900:
            raise ValueError("pilot100_maximum_900_second_window")
        started = time.monotonic()
        deadline = started + seconds
        self.native.absolute_deadline = deadline
        summary = self.tx(
            lambda repository: repository.begin(
                snapshot_id, generation_id, tokenizer_profile_sha256=self.chunker.profile_sha256, unit_limit=unit_limit
            )
        )
        checkpoint({"event": "generation_opened", "generation": summary})
        while summary["state"] == "preparing" and time.monotonic() < deadline - 10:
            projections = self.tx(lambda repository: repository.pending_projections(generation_id, limit=100))
            if not projections:
                summary = self.tx(lambda repository: repository.seal_preparation(generation_id))
                checkpoint({"event": "preparation_sealed_not_published", "generation": summary})
                break
            # No acceptance or SQL write until the entire verified operation's
            # exit check succeeds, including scope lifetime/asset verification.
            with self.chunker.verified_batch():
                compiled = [(projection, compile_projection(projection, self.chunker)) for projection in projections]
            if time.monotonic() >= deadline - 10:
                return {"status": "window_stopped", "phase": "compilation_not_accepted", "generation": summary}
            for projection, unit in compiled:
                self.tx(lambda repository, p=projection, u=unit: repository.stage_unit(generation_id, p, u))
            checkpoint({"event": "units_staged_not_published", "source_units": len(compiled)})
        summary = self.tx(lambda repository: repository.describe(generation_id))
        if summary["state"] == "verified":
            return {
                "status": "already_verified_not_published",
                "generation": summary,
                "elapsed_seconds": time.monotonic() - started,
            }
        if summary["state"] != "building" or time.monotonic() >= deadline - 20:
            return {"status": "window_stopped", "generation": summary, "elapsed_seconds": time.monotonic() - started}
        proof = self.native.create(summary["collection_name"])
        self.tx(lambda repository: repository.bind_server_proof(generation_id, proof))
        checkpoint({"event": "native_configuration_verified", "proof": proof})
        while time.monotonic() < deadline - 15:
            claim = self.tx(lambda repository: repository.batch_claim(generation_id))
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
                        "recovery": "read_before_any_replay",
                        "application_automatic_retries": 0,
                    }
                except Exception as exc:
                    receipt = {
                        "state": "uncertain",
                        "readback_error": {"type": type(exc).__name__, "message": str(exc)[:600]},
                        "recovery": "read_before_any_replay_failed_no_resend",
                    }
                if receipt["state"] != "verified" and not allow_uncertain_replay:
                    self.tx(
                        lambda repository, c=claim, r=receipt: repository.batch_result(
                            generation_id, UUID(c["batch_id"]), UUID(c["lease_id"]), r
                        )
                    )
                    checkpoint(
                        {"event": "uncertain_batch_not_resent", "batch_id": claim["batch_id"], "receipt": receipt}
                    )
                    return {"status": "reconciliation_required", "batch_id": claim["batch_id"], "generation": summary}
            if receipt is None or receipt["state"] != "verified":
                prior_receipt = receipt
                receipt = self.native.upsert_and_read(
                    claim["collection_name"], claim["rows"], seconds=min(120, deadline - time.monotonic() - 10)
                )
                if prior_receipt:
                    receipt["prior_reconciliation"] = prior_receipt
                    receipt["explicit_replay"] = (
                        "bounded at-least-once upsert after actual readback; not inferred never-written"
                    )
            self.tx(
                lambda repository, c=claim, r=receipt: repository.batch_result(
                    generation_id, UUID(c["batch_id"]), UUID(c["lease_id"]), r
                )
            )
            checkpoint(
                {
                    "event": "batch_result",
                    "batch_id": claim["batch_id"],
                    "logical_attempt": claim["logical_attempt"],
                    "receipt": receipt,
                }
            )
            if receipt["state"] != "verified":
                return {"status": "reconciliation_required", "batch_id": claim["batch_id"], "generation": summary}
        outstanding = self.tx(lambda repository: repository.remaining_batches(generation_id))
        if outstanding or time.monotonic() >= deadline - 10:
            return {
                "status": "window_stopped",
                "remaining_batches": outstanding,
                "generation": summary,
                "elapsed_seconds": time.monotonic() - started,
            }
        rows = self.tx(lambda repository: repository.expected_rows(generation_id))
        full = self.native.scan_all(summary["collection_name"], rows, seconds=min(300, deadline - time.monotonic() - 5))
        if time.monotonic() >= deadline:
            checkpoint({"event": "late_readback_not_accepted", "proof": full})
            return {"status": "window_stopped", "generation": summary, "elapsed_seconds": time.monotonic() - started}
        summary = self.tx(lambda repository: repository.accept_readback(generation_id, full))
        return {
            "status": "verified_not_published",
            "generation": summary,
            "complete_readback": full,
            "elapsed_seconds": time.monotonic() - started,
            "within_controller_window": time.monotonic() < deadline,
            "search_published": False,
        }
