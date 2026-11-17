"""Private native BM25+TEI/RRF with PG generations, durable jobs and current ACL.

One collection per trusted backend/schema, with tenant partition-key pruning.
The PG job is a transactional outbox; unknown writes are reconciled by exact
ID/content readback, never silently retried. Paper/CanonicalWork is not involved.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, ConfigDict
from sqlalchemy import exists, func, or_, select
from citequest.index_v2.native import schema_signature, verify_readback
from citequest.index_v2.rows_v2 import canonical_sha
from citequest.index_v2.schema import IndexProfile
from citequest.index_v3.models import BackendAdoption, BackendBinding
from citequest.materials_v2.models import MaterialFragment, MaterialIngestion, ParsedRepresentation
from citequest.materials_v2.services import MaterialService
from citequest.materials_v2.chunking import SourceChunker
from citequest.search_v3.native import SearchNative
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor, DomainError
from citequest.v2.models import Blob, File
from citequest.v2.storage import CAS, StoredBlob
from . import models as m

INDEX_SPECS = {
    "dense": {"index_type": "HNSW", "metric_type": "COSINE", "M": "16", "efConstruction": "200"},
    "sparse": {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "BM25", "inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": "1.2", "bm25_b": "0.75"},
    "owner_id": {"index_type": "INVERTED"}, "file_id": {"index_type": "INVERTED"}, "generation_id": {"index_type": "INVERTED"},
}


def text_sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def material_schema(endpoint: str):
    from pymilvus import DataType, Function, FunctionType, MilvusClient
    if not re.fullmatch(r"https?://[a-zA-Z0-9_.:-]+(?::[0-9]+)?", endpoint):
        raise ValueError("trusted_material_embedding_endpoint_required")
    profile = IndexProfile()
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    for name, length in [("chunk_id", 64), ("row_key", 64), ("owner_id", 36), ("file_id", 36),
                         ("generation_id", 36), ("representation_id", 64), ("material_version_id", 36),
                         ("fragment_id", 64), ("source_file_sha256", 64), ("source_text_sha256", 64),
                         ("backend_profile_sha256", 64), ("tokenizer_profile_sha256", 64)]:
        schema.add_field(name, DataType.VARCHAR, max_length=length, is_primary=name == "chunk_id", is_partition_key=name == "owner_id")
    for name in ("ordinal", "span_start", "span_end", "token_count"):
        schema.add_field(name, DataType.INT64)
    schema.add_field("text", DataType.VARCHAR, max_length=65535, enable_analyzer=True, analyzer_params={"type": "english"})
    schema.add_field("dense", DataType.FLOAT_VECTOR, dim=1024)
    schema.add_field("sparse", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(Function(name="text_bm25", function_type=FunctionType.BM25, input_field_names=["text"], output_field_names=["sparse"]))
    schema.add_function(Function(name="text_dense", function_type=FunctionType.TEXTEMBEDDING, input_field_names=["text"], output_field_names=["dense"],
        params={"provider": "tei", "endpoint": endpoint, "truncate": "false", "truncation_direction": "Right", "max_client_batch_size": 2}))
    indexes = MilvusClient.prepare_index_params()
    indexes.add_index(field_name="dense", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
    indexes.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": .75})
    for field in ("owner_id", "file_id", "generation_id"):
        indexes.add_index(field_name=field, index_type="INVERTED")
    return schema, indexes


@dataclass(frozen=True)
class MaterialNativeHit:
    file_id: UUID
    representation_id: str
    fragment_id: str
    generation_id: UUID
    row_id: str
    row_sha256: str
    source_text_sha256: str
    score: float


class MaterialNativeIndex:
    def __init__(self, factory, store: CAS, native: SearchNative, chunker: SourceChunker, *, collection_name: str,
                 embedding_endpoint: str, backend_profile_sha256: str, backend_proof: dict):
        if re.fullmatch(r"cq_v2_material_[a-z0-9_]{1,80}", collection_name) is None or re.fullmatch(r"[a-f0-9]{64}", backend_profile_sha256) is None:
            raise ValueError("trusted_material_collection_backend_required")
        if backend_proof.get("backend_profile_sha256") != backend_profile_sha256 or backend_proof.get("profile_sha256") != IndexProfile().fingerprint():
            raise ValueError("explicit_same_semantic_profile_and_backend_proof_required")
        self.factory, self.store, self.native, self.chunker = factory, store, native, chunker
        self.collection, self.endpoint, self.backend = collection_name, embedding_endpoint, backend_profile_sha256
        self.schema, self.indexes = material_schema(embedding_endpoint)
        signature = schema_signature(self.schema.to_dict())
        self.schema_sha256 = canonical_sha({"schema": signature, "indexes": INDEX_SPECS, "partitions": 64,
            "semantic": IndexProfile().fingerprint(), "backend": self.backend})
        self.proof = {**backend_proof, "schema_signature": signature}

    def _registered(self, db, *, adopted: bool):
        binding = db.scalar(select(BackendBinding).where(BackendBinding.backend_profile_sha256 == self.backend).limit(1))
        if binding is None or hashlib.sha256(binding.profile_canonical.encode()).hexdigest() != self.backend:
            raise DomainError("material_embedding_backend_not_registered", 503)
        if adopted:
            approval = db.scalar(select(BackendAdoption).where(BackendAdoption.backend_profile_sha256 == self.backend).limit(1))
            if approval is None or approval.metrics.get("status") != "pass" or approval.metrics.get("attempt_errors") != 0:
                raise DomainError("material_embedding_backend_quality_not_adopted", 503)

    def _generation_binding(self, generation):
        if (generation.collection_name != self.collection or generation.backend_profile_sha256 != self.backend
                or generation.schema_profile_sha256 != self.schema_sha256
                or generation.tokenizer_profile_sha256 != self.chunker.profile_sha256):
            raise DomainError("material_runtime_index_profile_mismatch", 503)

    def _source(self, db, actor, file_id, *, verify_blob: bool = True):
        material = MaterialService(db, actor, self.store)
        file = material.catalog.file(file_id)
        if file.outbound_policy != "selected_cloud_processing_allowed" or file.rights_status == "restricted":
            raise DomainError("material_cloud_processing_not_permitted", 403)
        if file.status not in {"ready", "partial"}:
            raise DomainError("material_not_readable", 409)
        parsed = material.get_representation(file_id)
        rep = db.get(ParsedRepresentation, parsed["representation_id"])
        if rep is None or rep.material_version_id is None or rep.source_sha256 != file.sha256:
            raise DomainError("material_source_version_changed", 409)
        blob = db.get(Blob, (file.security_domain, file.sha256))
        if blob is None:
            raise DomainError("material_blob_missing", 503)
        if verify_blob:
            self.store.verify(StoredBlob(file.security_domain, file.sha256, blob.size_bytes))
        return file, rep

    def prepare_collection(self):
        from pymilvus import MilvusClient
        with self.factory() as db:
            self._registered(db, adopted=False)
        if not self.native.call("has_collection", collection_name=self.collection):
            self.native.call("create_collection", collection_name=self.collection, schema=self.schema,
                num_partitions=64, consistency_level="Strong")
        # One official index operation per bounded RPC window. A multi-index
        # SDK convenience call polls/allocates timestamps internally and can
        # exceed a query-sized RPC budget; partial setup remains inspectable.
        existing = set(self.native.call("list_indexes", collection_name=self.collection))
        for field in INDEX_SPECS:
            if field not in existing:
                one = MilvusClient.prepare_index_params()
                if field == "dense":
                    one.add_index(field_name=field, index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
                elif field == "sparse":
                    one.add_index(field_name=field, index_type="SPARSE_INVERTED_INDEX", metric_type="BM25",
                        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": .75})
                else:
                    one.add_index(field_name=field, index_type="INVERTED")
                self.native.call("create_index", collection_name=self.collection, index_params=one, seconds=30, maximum_calls=12)
        self.native.call("load_collection", collection_name=self.collection, seconds=30, maximum_calls=12)
        self._verify_collection()
        actual = self.native.call("describe_collection", collection_name=self.collection)
        if actual.get("num_partitions") != 64:
            raise DomainError("material_partition_profile_changed", 503)
        return {"schema_profile_sha256": self.schema_sha256, "collection": self.collection, "product_activated": False}

    def _verify_collection(self):
        self.native.verify(self.collection, self.proof, dense=True)
        names = self.native.call("list_indexes", collection_name=self.collection)
        observed = {}
        for name in names:
            description = self.native.call("describe_index", collection_name=self.collection, index_name=name)
            observed[description["field_name"]] = {**description, **description.get("params", {})}
        for field, expected in INDEX_SPECS.items():
            if any(str(observed.get(field, {}).get(key)) != value for key, value in expected.items()):
                raise DomainError("material_native_index_parameters_changed", 503)

    def enqueue(self, actor: Actor, file_id: UUID) -> dict:
        self.chunker.verify_assets()
        with self.factory.begin() as db:
            self._registered(db, adopted=False)
            file, rep = self._source(db, actor, file_id)
            # Serialize enqueue and pointer publication by the same owner lock.
            MaterialService(db, actor, self.store).catalog.lock_owner()
            identity = {"owner": str(actor.owner_id), "file": str(file_id), "representation": rep.id,
                        "file_revision": file.revision, "backend": self.backend, "schema": self.schema_sha256,
                        "tokenizer": self.chunker.profile_sha256, "collection": self.collection}
            identifier = uuid5(NAMESPACE_URL, canonical_sha(identity))
            existing = db.get(m.MaterialIndexGeneration, identifier)
            if existing is not None:
                self._generation_binding(existing)
                return {"generation_id": str(existing.id), "state": existing.state, "expected_rows": existing.expected_rows}
            fragments = list(db.scalars(select(MaterialFragment).where(MaterialFragment.owner_id == actor.owner_id,
                MaterialFragment.file_id == file_id, MaterialFragment.representation_id == rep.id).order_by(MaterialFragment.ordinal).limit(5001)))
            if not 1 <= len(fragments) <= 5000:
                raise DomainError("material_native_fragment_count_unsupported", 413)
            rows = []
            for fragment in fragments:
                text = fragment.source_text
                tokens = self.chunker.count(text)
                if (not text or len(text.encode()) > 20000 or tokens != fragment.token_count or not 1 <= tokens <= 2048
                        or fragment.payload.get("source_text") != text or fragment.payload.get("source_text_sha256") != text_sha(text)):
                    raise DomainError("material_fragment_integrity_or_token_profile_mismatch", 409)
                payload = {"owner_id": str(actor.owner_id), "file_id": str(file_id), "representation_id": rep.id,
                    "generation_id": str(identifier), "material_version_id": str(rep.material_version_id), "fragment_id": fragment.id,
                    "row_key": fragment.id, "source_file_sha256": file.sha256, "source_text_sha256": text_sha(text),
                    "backend_profile_sha256": self.backend, "tokenizer_profile_sha256": self.chunker.profile_sha256,
                    "ordinal": fragment.ordinal, "span_start": 0, "span_end": len(text), "token_count": tokens, "text": text}
                payload["chunk_id"] = canonical_sha(payload)
                rows.append((fragment, payload))
            manifest = canonical_sha(dict(sorted((payload["chunk_id"], canonical_sha(payload)) for _, payload in rows)))
            generation = m.MaterialIndexGeneration(id=identifier, owner_id=actor.owner_id, file_id=file_id,
                representation_id=rep.id, material_version_id=rep.material_version_id, file_revision=file.revision,
                source_file_sha256=file.sha256, backend_profile_sha256=self.backend, schema_profile_sha256=self.schema_sha256,
                tokenizer_profile_sha256=self.chunker.profile_sha256, collection_name=self.collection,
                manifest_sha256=manifest, expected_rows=len(rows), state="ready")
            db.add(generation)
            db.flush()
            db.add_all([m.MaterialIndexRow(generation_id=identifier, row_id=payload["chunk_id"], owner_id=actor.owner_id,
                file_id=file_id, representation_id=rep.id, fragment_id=fragment.id, ordinal=fragment.ordinal,
                row_sha256=canonical_sha(payload), payload=payload, state="ready") for fragment, payload in rows])
            db.add(m.MaterialIndexJob(generation_id=identifier, operation="upsert", state="ready", epoch=0, batch_ids=[], attempts=[]))
            return {"generation_id": str(identifier), "state": "ready", "expected_rows": len(rows), "product_activated": False}

    def discover_ready(self, limit: int = 20) -> dict:
        """Durable anti-join catch-up after parser commits, including worker restarts."""
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("bounded_material_discovery_required")
        with self.factory() as db:
            self._registered(db, adopted=False)
            known = exists(select(m.MaterialIndexGeneration.id).where(m.MaterialIndexGeneration.owner_id == File.owner_id,
                m.MaterialIndexGeneration.file_id == File.id, m.MaterialIndexGeneration.representation_id == ParsedRepresentation.id,
                m.MaterialIndexGeneration.file_revision == File.revision, m.MaterialIndexGeneration.backend_profile_sha256 == self.backend,
                m.MaterialIndexGeneration.schema_profile_sha256 == self.schema_sha256,
                m.MaterialIndexGeneration.tokenizer_profile_sha256 == self.chunker.profile_sha256,
                m.MaterialIndexGeneration.collection_name == self.collection))
            candidates = db.execute(select(File.owner_id, File.id).join(MaterialIngestion,
                (MaterialIngestion.owner_id == File.owner_id) & (MaterialIngestion.file_id == File.id)).join(ParsedRepresentation,
                (ParsedRepresentation.id == MaterialIngestion.representation_id) & (ParsedRepresentation.owner_id == File.owner_id) &
                (ParsedRepresentation.file_id == File.id)).where(File.deleted_at.is_(None), File.status.in_(["ready", "partial"]),
                File.outbound_policy == "selected_cloud_processing_allowed", File.rights_status != "restricted",
                ParsedRepresentation.source_sha256 == File.sha256, ~known).order_by(File.created_at, File.id).limit(limit)).all()
        queued = [self.enqueue(Actor(owner), file_id) for owner, file_id in candidates]
        return {"new_generations": queued, "discovered": len(candidates)}

    def discover_withdrawn(self, limit: int = 100) -> dict:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("bounded_material_withdrawal_discovery_required")
        with self.factory() as db:
            candidates = db.execute(select(m.MaterialIndexGeneration.owner_id, m.MaterialIndexGeneration.file_id).join(File,
                (File.owner_id == m.MaterialIndexGeneration.owner_id) & (File.id == m.MaterialIndexGeneration.file_id)).where(
                m.MaterialIndexGeneration.backend_profile_sha256 == self.backend, m.MaterialIndexGeneration.collection_name == self.collection,
                or_(File.deleted_at.is_not(None), File.outbound_policy != "selected_cloud_processing_allowed"),
                ~exists(select(m.MaterialIndexJob.id).where(m.MaterialIndexJob.generation_id == m.MaterialIndexGeneration.id,
                    m.MaterialIndexJob.operation == "delete"))).distinct().limit(limit)).all()
        queued = [self.request_delete(Actor(owner), file_id) for owner, file_id in candidates]
        return {"sources_queued": len(queued)}

    def _expected(self, db, generation):
        rows = list(db.scalars(select(m.MaterialIndexRow).where(m.MaterialIndexRow.generation_id == generation.id).order_by(m.MaterialIndexRow.ordinal)))
        if (len(rows) != generation.expected_rows or canonical_sha(dict(sorted((row.row_id, row.row_sha256) for row in rows))) != generation.manifest_sha256
                or any(canonical_sha(row.payload) != row.row_sha256 for row in rows)):
            raise DomainError("material_full_expected_manifest_changed", 409)
        return rows

    def _readback(self, payloads):
        actual = self.native.call("get", collection_name=self.collection, ids=[row["chunk_id"] for row in payloads],
            output_fields=[*payloads[0], "dense"], consistency_level="Strong")
        return verify_readback(payloads, actual, complete=False)

    def _settle_receipt(self, attempt_id, job_id, epoch, request_sha256, outcome, receipt):
        # A late wire response is an accounting/audit fact, not permission to
        # advance a reclaimed job. Settle its immutable attempt independently,
        # then enforce the current job fence in the following transaction.
        with self.factory.begin() as db:
            audit = db.get(m.MaterialIndexAttempt, attempt_id, with_for_update=True, populate_existing=True)
            if audit is None or audit.job_id != job_id or audit.epoch != epoch or audit.request_sha256 != request_sha256:
                raise DomainError("material_attempt_receipt_binding_mismatch", 409)
            if audit.write_outcome == "sending":
                audit.write_outcome, audit.receipt = outcome, receipt
            elif audit.write_outcome != outcome:
                raise DomainError("material_settled_attempt_cannot_be_reclassified", 409)

    def run_one(self, worker_id: str) -> dict:
        if type(worker_id) is not str or re.fullmatch(r"[a-zA-Z0-9_.:-]{1,100}", worker_id) is None:
            raise ValueError("bounded_worker_identity_required")
        # Preflight is outside a reserved write. A preflight failure has not sent
        # an upsert and cannot strand a write in unknown state.
        self._verify_collection()
        with self.factory.begin() as db:
            self._registered(db, adopted=False)
            job = db.scalar(select(m.MaterialIndexJob).join(m.MaterialIndexGeneration, m.MaterialIndexGeneration.id == m.MaterialIndexJob.generation_id)
                .where(m.MaterialIndexJob.state == "ready", m.MaterialIndexJob.operation == "upsert",
                    m.MaterialIndexGeneration.backend_profile_sha256 == self.backend, m.MaterialIndexGeneration.collection_name == self.collection,
                    m.MaterialIndexGeneration.schema_profile_sha256 == self.schema_sha256,
                    m.MaterialIndexGeneration.tokenizer_profile_sha256 == self.chunker.profile_sha256)
                .order_by(m.MaterialIndexJob.created_at, m.MaterialIndexJob.id).with_for_update(skip_locked=True, of=m.MaterialIndexJob).limit(1))
            if job is None:
                return {"state": "idle"}
            generation = db.get(m.MaterialIndexGeneration, job.generation_id)
            try:
                # Raw bytes are verified when preparing the immutable complete
                # source manifest and on evidence reads. Each two-row delivery
                # checks current SQL ACL/source identity and exact row hashes,
                # rather than re-reading a 50MiB PDF for every pair of chunks.
                current_file, current_rep = self._source(db, Actor(generation.owner_id), generation.file_id, verify_blob=False)
                if current_file.sha256 != generation.source_file_sha256 or current_rep.id != generation.representation_id or current_file.revision != generation.file_revision:
                    raise DomainError("material_job_source_changed", 409)
            except DomainError as error:
                job.state, generation.state, job.error_code = "cancelled", "cancelled", error.code
                return {"generation_id": str(generation.id), "state": "cancelled", "reason": error.code}
            self._generation_binding(generation)
            if db.scalar(select(func.count()).select_from(m.MaterialIndexRow).where(m.MaterialIndexRow.generation_id == generation.id)) != generation.expected_rows:
                raise DomainError("material_full_expected_count_changed", 409)
            pending = list(db.scalars(select(m.MaterialIndexRow).where(m.MaterialIndexRow.generation_id == generation.id,
                m.MaterialIndexRow.state == "ready").order_by(m.MaterialIndexRow.ordinal).limit(2)))
            if any(canonical_sha(row.payload) != row.row_sha256 for row in pending):
                raise DomainError("material_delivery_row_hash_changed", 409)
            if not pending:
                raise DomainError("material_job_requires_reconciliation", 409)
            job.state, job.epoch, job.worker_id, job.lease_id = "sending", job.epoch+1, worker_id, uuid4()
            job.lease_until = db.scalar(select(func.clock_timestamp()))+timedelta(seconds=120)
            job.batch_ids = [row.row_id for row in pending]
            attempt = {"id": str(uuid4()), "epoch": job.epoch, "batch_ids": job.batch_ids,
                       "at": datetime.now(UTC).isoformat(), "write_outcome": "sending"}
            job.attempts = [*job.attempts, attempt][-20:]
            db.add(m.MaterialIndexAttempt(id=UUID(attempt["id"]), job_id=job.id, epoch=job.epoch, operation="upsert",
                request_sha256=canonical_sha({"rows": [row.payload for row in pending]}), batch_ids=job.batch_ids,
                write_outcome="sending", receipt=None))
            for row in pending:
                row.state = "sending"
            job_id, lease_id, epoch, payloads = job.id, job.lease_id, job.epoch, [row.payload for row in pending]
        write_error = None
        try:
            self.native.call("upsert", collection_name=self.collection, data=payloads)
        except Exception as error:
            write_error = type(error).__name__
        proof, read_error = None, None
        try:
            proof = self._readback(payloads)
        except Exception as error:
            read_error = type(error).__name__
        outcome = "unknown" if write_error else "acknowledged"
        receipt = {**attempt, "write_outcome": outcome, "write_error": write_error, "read_error": read_error,
            "readback": proof, "finished_at": datetime.now(UTC).isoformat()}
        self._settle_receipt(UUID(attempt["id"]), job_id, epoch, canonical_sha({"rows": payloads}), outcome, receipt)
        with self.factory.begin() as db:
            job = db.get(m.MaterialIndexJob, job_id, with_for_update=True, populate_existing=True)
            if (job.state != "sending" or job.lease_id != lease_id or job.epoch != epoch
                    or job.lease_until is None or job.lease_until <= db.scalar(select(func.clock_timestamp()))):
                raise DomainError("material_late_job_result_fenced", 409)
            entries = [dict(value) for value in job.attempts]
            entries[-1].update(write_outcome="unknown" if write_error else "acknowledged", write_error=write_error,
                read_error=read_error, readback=proof, finished_at=datetime.now(UTC).isoformat())
            job.attempts = entries
            complete = bool(proof and proof["complete"])
            for row in db.scalars(select(m.MaterialIndexRow).where(m.MaterialIndexRow.generation_id == job.generation_id,
                m.MaterialIndexRow.row_id.in_(job.batch_ids))):
                row.state = "verified" if complete else "unknown"
            job.state, job.error_code = ("ready", None) if complete else ("unknown", "material_write_requires_exact_readback")
            job.lease_id, job.lease_until = None, None
            db.flush()
            remaining = db.scalar(select(func.count()).select_from(m.MaterialIndexRow).where(
                m.MaterialIndexRow.generation_id == job.generation_id, m.MaterialIndexRow.state != "verified"))
            if complete and remaining == 0:
                job.state = "verified"
            current_file = db.get(File, db.get(m.MaterialIndexGeneration, job.generation_id).file_id, populate_existing=True)
            if current_file.deleted_at is not None or current_file.outbound_policy != "selected_cloud_processing_allowed":
                job.state = "cancelled"
                db.get(m.MaterialIndexGeneration, job.generation_id).state = "cancelled"
            identifier, state = job.generation_id, job.state
        if state == "verified":
            return self.reconcile(identifier, worker_id)
        return {"generation_id": str(identifier), "state": state, "write_outcome": "unknown" if write_error else "acknowledged"}

    def reconcile(self, generation_id: UUID, worker_id: str) -> dict:
        """Read-only native reconciliation. Missing rows never trigger a retry."""
        self._verify_collection()
        with self.factory.begin() as db:
            generation = db.get(m.MaterialIndexGeneration, generation_id)
            if generation is None or generation.backend_profile_sha256 != self.backend or generation.collection_name != self.collection:
                raise DomainError("not_found", 404)
            self._generation_binding(generation)
            self._source(db, Actor(generation.owner_id), generation.file_id)
            expected = [row.payload for row in self._expected(db, generation)]
        proofs, all_found = [], True
        for start in range(0, len(expected), 50):
            proof = self._readback(expected[start:start+50])
            proofs.append(proof)
            all_found &= proof["complete"]
        with self.factory.begin() as db:
            job = db.scalar(select(m.MaterialIndexJob).where(m.MaterialIndexJob.generation_id == generation_id,
                m.MaterialIndexJob.operation == "upsert").with_for_update())
            generation = db.get(m.MaterialIndexGeneration, generation_id, with_for_update=True, populate_existing=True)
            if job.state == "sending" and job.lease_until is not None and job.lease_until > db.scalar(select(func.clock_timestamp())):
                raise DomainError("material_writer_still_has_live_lease", 409)
            receipt = {"at": datetime.now(UTC).isoformat(), "operation": "reconcile_read_only", "worker_id": worker_id,
                "complete": all_found, "proofs": proofs}
            job.attempts = [*job.attempts, receipt][-20:]
            db.add(m.MaterialIndexAttempt(job_id=job.id, epoch=job.epoch, operation="reconcile",
                request_sha256=canonical_sha({"expected_manifest": generation.manifest_sha256}), batch_ids=[],
                write_outcome="read_only", receipt=receipt))
            if not all_found:
                job.state, job.error_code = "unknown", "material_missing_rows_no_automatic_resend"
                return {"generation_id": str(generation_id), "state": "unknown", "automatic_resend": False}
            generation.readback_proof = {"complete": True, "expected": len(expected), "matched": len(expected),
                "manifest_sha256": generation.manifest_sha256, "batch_proofs": proofs}
            generation.state, job.state, job.lease_id, job.lease_until = "verified", "verified", None, None
            for row in db.scalars(select(m.MaterialIndexRow).where(m.MaterialIndexRow.generation_id == generation_id)):
                row.state = "verified"
        # Quality adoption is an independent release gate, never an embedding
        # equivalence claim. A verified candidate may wait here indefinitely.
        try:
            return self.adopt(Actor(generation.owner_id), generation_id)
        except DomainError as error:
            if error.code == "material_embedding_backend_quality_not_adopted":
                return {"generation_id": str(generation_id), "state": "verified", "product_activated": False, "reason": error.code}
            raise

    def adopt(self, actor: Actor, generation_id: UUID) -> dict:
        with self.factory.begin() as db:
            self._registered(db, adopted=True)
            MaterialService(db, actor, self.store).catalog.lock_owner()
            generation = db.get(m.MaterialIndexGeneration, generation_id, with_for_update=True, populate_existing=True)
            if generation is None or generation.owner_id != actor.owner_id or generation.backend_profile_sha256 != self.backend:
                raise DomainError("not_found", 404)
            self._generation_binding(generation)
            file, rep = self._source(db, actor, generation.file_id)
            if (generation.state not in {"verified", "adopted"} or not generation.readback_proof or not generation.readback_proof.get("complete")
                    or generation.readback_proof.get("matched") != generation.expected_rows
                    or generation.readback_proof.get("manifest_sha256") != generation.manifest_sha256
                    or rep.id != generation.representation_id or file.revision != generation.file_revision
                    or file.sha256 != generation.source_file_sha256):
                raise DomainError("material_complete_current_generation_required", 409)
            self._expected(db, generation)
            pointer = db.get(m.MaterialIndexPointer, (actor.owner_id, file.id), with_for_update=True, populate_existing=True)
            if pointer is None:
                pointer = m.MaterialIndexPointer(owner_id=actor.owner_id, file_id=file.id, generation_id=None, epoch=0)
                db.add(pointer)
            if pointer.generation_id != generation.id:
                if pointer.generation_id is not None:
                    db.get(m.MaterialIndexGeneration, pointer.generation_id).state = "superseded"
                pointer.generation_id, pointer.epoch = generation.id, pointer.epoch+1
            generation.state = "adopted"
            return {"generation_id": str(generation.id), "state": "adopted", "epoch": pointer.epoch, "product_activated": True}

    def search(self, actor: Actor, file_ids: list[UUID], query: str, limit: int = 3) -> list[MaterialNativeHit]:
        from pymilvus import AnnSearchRequest, RRFRanker
        if (not 1 <= len(file_ids) <= 20 or len(set(file_ids)) != len(file_ids) or type(limit) is not int or not 1 <= limit <= 3
                or type(query) is not str or not query.strip() or len(query) > 2000 or "\x00" in query
                or not 1 <= self.chunker.count(query) <= 2048):
            raise DomainError("invalid_material_evidence_scope", 422)
        with self.factory.begin() as db:
            self._registered(db, adopted=True)
            captured, representations = {}, {}
            for file_id in file_ids:
                file, rep = self._source(db, actor, file_id)
                pointer = db.get(m.MaterialIndexPointer, (actor.owner_id, file_id), populate_existing=True)
                generation = db.get(m.MaterialIndexGeneration, pointer.generation_id) if pointer and pointer.generation_id else None
                if generation is not None:
                    self._generation_binding(generation)
                if (generation is None or generation.state != "adopted" or generation.representation_id != rep.id or generation.file_revision != file.revision
                        or generation.source_file_sha256 != file.sha256 or generation.backend_profile_sha256 != self.backend
                        or generation.schema_profile_sha256 != self.schema_sha256 or generation.tokenizer_profile_sha256 != self.chunker.profile_sha256):
                    raise DomainError("material_native_index_not_ready", 503)
                captured[file_id] = generation.id
                representations[file_id] = rep.id
        values = {"owner": str(actor.owner_id)}
        clauses = []
        for ordinal, (file_id, generation_id) in enumerate(captured.items()):
            clauses.append(f"(file_id == {{file_{ordinal}}} and generation_id == {{gen_{ordinal}}} and representation_id == {{rep_{ordinal}}})")
            values[f"file_{ordinal}"], values[f"gen_{ordinal}"] = str(file_id), str(generation_id)
            values[f"rep_{ordinal}"] = representations[file_id]
        expression = "owner_id == {owner} and (" + " or ".join(clauses) + ")"
        self._verify_collection()
        requests = [AnnSearchRequest(data=[query], anns_field=field, param=params, limit=50, expr=expression, expr_params=values)
            for field, params in [("dense", {"metric_type": "COSINE", "params": {"ef": 100}}),
                                  ("sparse", {"metric_type": "BM25", "params": {"drop_ratio_search": 0.0}})]]
        raw = self.native.call("hybrid_search", collection_name=self.collection, reqs=requests, ranker=RRFRanker(k=60),
            limit=limit, output_fields=["chunk_id"], consistency_level="Strong")
        if not isinstance(raw, list) or len(raw) != 1 or len(raw[0]) > limit:
            raise DomainError("material_native_result_shape_invalid", 503)
        result = []
        with self.factory.begin() as db:
            for file_id, generation_id in captured.items():
                file, rep = self._source(db, actor, file_id)
                pointer = db.get(m.MaterialIndexPointer, (actor.owner_id, file_id), populate_existing=True)
                generation = db.get(m.MaterialIndexGeneration, generation_id, populate_existing=True)
                self._generation_binding(generation)
                if (pointer is None or pointer.generation_id != generation_id or generation.state != "adopted"
                        or rep.id != generation.representation_id or file.revision != generation.file_revision
                        or file.sha256 != generation.source_file_sha256 or rep.material_version_id != generation.material_version_id):
                    raise DomainError("material_source_changed_during_retrieval", 409)
            seen = set()
            for value in raw[0]:
                row_id = value.get("chunk_id", value.get("id"))
                matches = list(db.scalars(select(m.MaterialIndexRow).where(m.MaterialIndexRow.owner_id == actor.owner_id,
                    m.MaterialIndexRow.generation_id.in_(captured.values()), m.MaterialIndexRow.row_id == row_id)))
                if len(matches) != 1 or row_id in seen:
                    raise DomainError("material_native_result_outside_captured_scope", 503)
                row = matches[0]
                fragment = db.get(MaterialFragment, row.fragment_id)
                if (fragment is None or row.state != "verified" or canonical_sha(row.payload) != row.row_sha256 or row.payload["source_text_sha256"] != text_sha(fragment.source_text)
                        or row.payload["text"] != fragment.source_text or fragment.owner_id != actor.owner_id
                        or fragment.file_id != row.file_id or fragment.representation_id != row.representation_id):
                    raise DomainError("material_native_result_source_integrity_failure", 503)
                import math
                score = value.get("distance")
                if type(score) not in {float, int} or not math.isfinite(score):
                    raise DomainError("material_native_score_invalid", 503)
                seen.add(row_id)
                result.append(MaterialNativeHit(row.file_id, row.representation_id, row.fragment_id, row.generation_id,
                    row.row_id, row.row_sha256, row.payload["source_text_sha256"], float(score)))
        return result

    def preflight(self, actor: Actor, file_ids: list[UUID]) -> None:
        """No model, TEI or native RPC. Admit only current adopted private scope."""
        if type(file_ids) is not list or not 1 <= len(file_ids) <= 20 or len(set(file_ids)) != len(file_ids):
            raise DomainError("invalid_material_evidence_scope", 422)
        with self.factory.begin() as db:
            self._registered(db, adopted=True)
            if getattr(self.native, "backend_profile_sha256", None) != self.backend:
                raise DomainError("native_runtime_backend_binding_mismatch", 503)
            if getattr(self.native, "function_endpoint", None) != self.endpoint:
                raise DomainError("native_runtime_function_endpoint_binding_mismatch", 503)
            if hasattr(self.native, "check_binding"):
                self.native.check_binding(self.proof)
            for file_id in file_ids:
                file, rep = self._source(db, actor, file_id)
                pointer = db.get(m.MaterialIndexPointer, (actor.owner_id, file_id), populate_existing=True)
                generation = db.get(m.MaterialIndexGeneration, pointer.generation_id) if pointer and pointer.generation_id else None
                if generation is not None:
                    self._generation_binding(generation)
                if (generation is None or generation.state != "adopted" or generation.representation_id != rep.id
                        or generation.file_revision != file.revision or generation.source_file_sha256 != file.sha256
                        or generation.material_version_id != rep.material_version_id or generation.backend_profile_sha256 != self.backend
                        or generation.schema_profile_sha256 != self.schema_sha256 or generation.tokenizer_profile_sha256 != self.chunker.profile_sha256
                        or not generation.readback_proof or generation.readback_proof.get("complete") is not True
                        or generation.readback_proof.get("matched") != generation.expected_rows
                        or generation.readback_proof.get("manifest_sha256") != generation.manifest_sha256):
                    raise DomainError("material_native_index_not_ready", 503)

    def request_delete(self, actor: Actor, file_id: UUID) -> dict:
        """Immediate SQL denial, transactional native-tombstone jobs for all old generations."""
        with self.factory.begin() as db:
            MaterialService(db, actor, self.store).catalog.lock_owner()
            file = db.get(File, file_id, populate_existing=True)
            if file is None or file.owner_id != actor.owner_id:
                raise DomainError("not_found", 404)
            if file.deleted_at is None and file.outbound_policy == "selected_cloud_processing_allowed":
                raise DomainError("material_delete_requires_deleted_or_withdrawn_source", 409)
            pointer = db.get(m.MaterialIndexPointer, (actor.owner_id, file_id), with_for_update=True)
            if pointer and pointer.generation_id is not None:
                pointer.generation_id, pointer.epoch = None, pointer.epoch+1
            generations = list(db.scalars(select(m.MaterialIndexGeneration).where(m.MaterialIndexGeneration.owner_id == actor.owner_id,
                m.MaterialIndexGeneration.file_id == file_id, m.MaterialIndexGeneration.backend_profile_sha256 == self.backend)))
            for generation in generations:
                upsert = db.scalar(select(m.MaterialIndexJob).where(m.MaterialIndexJob.generation_id == generation.id,
                    m.MaterialIndexJob.operation == "upsert").with_for_update())
                if upsert and upsert.state == "ready":
                    upsert.state = "cancelled"
                deletion = db.scalar(select(m.MaterialIndexJob).where(m.MaterialIndexJob.generation_id == generation.id,
                    m.MaterialIndexJob.operation == "delete"))
                if deletion is None:
                    db.add(m.MaterialIndexJob(generation_id=generation.id, operation="delete", state="ready", epoch=0, batch_ids=[], attempts=[]))
            return {"logical_access_denied": True, "generations_queued": len(generations), "native_erasure_complete": False}

    def recover_expired(self) -> dict:
        """Crash observation only; a lease fence cannot cancel an old RPC socket."""
        with self.factory.begin() as db:
            jobs = list(db.scalars(select(m.MaterialIndexJob).where(m.MaterialIndexJob.state == "sending",
                m.MaterialIndexJob.lease_until <= func.clock_timestamp()).with_for_update(skip_locked=True).limit(100)))
            for job in jobs:
                job.state, job.error_code = "unknown", "material_expired_writer_outcome_unknown"
                job.attempts = [*job.attempts, {"operation": "expired_lease_observed", "at": datetime.now(UTC).isoformat(),
                    "epoch": job.epoch, "automatic_resend": False}][-20:]
                db.add(m.MaterialIndexAttempt(job_id=job.id, epoch=job.epoch, operation="reconcile",
                    request_sha256=canonical_sha({"expired_lease": str(job.lease_id), "epoch": job.epoch}),
                    batch_ids=list(job.batch_ids), write_outcome="read_only", receipt=job.attempts[-1]))
                if job.operation == "upsert":
                    for row in db.scalars(select(m.MaterialIndexRow).where(m.MaterialIndexRow.generation_id == job.generation_id,
                        m.MaterialIndexRow.row_id.in_(job.batch_ids), m.MaterialIndexRow.state == "sending")):
                        row.state = "unknown"
            return {"observed_expired_jobs": len(jobs), "automatic_resend": False}

    def run_delete_one(self, worker_id: str) -> dict:
        """Observe native tombstone/absence, never claim SSD or CAS byte erasure.

        An old uncertain writer may still arrive. Such a generation is retained
        as unknown; read-absence cannot prove that socket has finished and does
        not authorize automatic clearing/resending of its original attempt.
        """
        if type(worker_id) is not str or re.fullmatch(r"[a-zA-Z0-9_.:-]{1,100}", worker_id) is None:
            raise ValueError("bounded_worker_identity_required")
        with self.factory.begin() as db:
            job = db.scalar(select(m.MaterialIndexJob).join(m.MaterialIndexGeneration, m.MaterialIndexGeneration.id == m.MaterialIndexJob.generation_id)
                .where(m.MaterialIndexJob.operation == "delete", m.MaterialIndexJob.state == "ready",
                    m.MaterialIndexGeneration.backend_profile_sha256 == self.backend, m.MaterialIndexGeneration.collection_name == self.collection)
                .order_by(m.MaterialIndexJob.created_at).with_for_update(skip_locked=True, of=m.MaterialIndexJob).limit(1))
            if job is None:
                return {"state": "idle"}
            generation = db.get(m.MaterialIndexGeneration, job.generation_id)
            file = db.get(File, generation.file_id, populate_existing=True)
            if file.deleted_at is None and file.outbound_policy == "selected_cloud_processing_allowed":
                raise DomainError("material_source_is_not_deleted_or_withdrawn", 409)
            upsert = db.scalar(select(m.MaterialIndexJob).where(m.MaterialIndexJob.generation_id == generation.id,
                m.MaterialIndexJob.operation == "upsert").with_for_update())
            unresolved = db.scalar(select(m.MaterialIndexAttempt.id).where(m.MaterialIndexAttempt.job_id == upsert.id,
                m.MaterialIndexAttempt.operation == "upsert", m.MaterialIndexAttempt.write_outcome.in_(["sending", "unknown"])).limit(1))
            if upsert.state == "sending" or unresolved is not None:
                job.state, job.error_code = "unknown", "material_prior_writer_settlement_unknown"
                return {"state": "unknown", "reason": job.error_code, "logical_access_denied": True, "automatic_delete": False}
            rows = list(db.scalars(select(m.MaterialIndexRow).where(m.MaterialIndexRow.generation_id == generation.id,
                m.MaterialIndexRow.state != "deleted").order_by(m.MaterialIndexRow.ordinal).limit(50)))
            if not rows:
                job.state, generation.state = "verified", "cancelled"
                return {"state": "verified", "native_tombstones_observed": True, "disk_byte_erasure_claimed": False}
            job.state, job.epoch, job.worker_id, job.lease_id = "sending", job.epoch+1, worker_id, uuid4()
            job.lease_until, job.batch_ids = db.scalar(select(func.clock_timestamp()))+timedelta(seconds=120), [row.row_id for row in rows]
            attempt = {"id": str(uuid4()), "at": datetime.now(UTC).isoformat(), "operation": "delete", "epoch": job.epoch,
                "batch_ids": job.batch_ids, "write_outcome": "sending"}
            job.attempts = [*job.attempts, attempt][-20:]
            db.add(m.MaterialIndexAttempt(id=UUID(attempt["id"]), job_id=job.id, epoch=job.epoch, operation="delete",
                request_sha256=canonical_sha({"delete_ids": job.batch_ids}), batch_ids=job.batch_ids, write_outcome="sending", receipt=None))
            job_id, lease_id, epoch, identifiers = job.id, job.lease_id, job.epoch, list(job.batch_ids)
        write_error, absent = None, False
        try:
            self.native.call("delete", collection_name=self.collection, ids=identifiers)
        except Exception as error:
            write_error = type(error).__name__
        try:
            actual = self.native.call("get", collection_name=self.collection, ids=identifiers, output_fields=["chunk_id"], consistency_level="Strong")
            absent = actual == []
        except Exception:
            pass
        outcome = "unknown" if write_error else "acknowledged"
        receipt = {**attempt, "write_outcome": outcome, "write_error": write_error,
            "native_absence_observed": absent, "finished_at": datetime.now(UTC).isoformat()}
        self._settle_receipt(UUID(attempt["id"]), job_id, epoch, canonical_sha({"delete_ids": identifiers}), outcome, receipt)
        with self.factory.begin() as db:
            job = db.get(m.MaterialIndexJob, job_id, with_for_update=True, populate_existing=True)
            if (job.state != "sending" or job.lease_id != lease_id or job.epoch != epoch
                    or job.lease_until is None or job.lease_until <= db.scalar(select(func.clock_timestamp()))):
                raise DomainError("material_late_delete_result_fenced", 409)
            history = [dict(attempt) for attempt in job.attempts]
            history[-1].update(write_outcome="unknown" if write_error else "acknowledged", write_error=write_error,
                native_absence_observed=absent, finished_at=datetime.now(UTC).isoformat())
            job.attempts, job.lease_id, job.lease_until = history, None, None
            job.state = "ready" if absent else "unknown"
            if absent:
                for row in db.scalars(select(m.MaterialIndexRow).where(m.MaterialIndexRow.generation_id == job.generation_id,
                    m.MaterialIndexRow.row_id.in_(identifiers))):
                    row.state = "deleted"
            return {"generation_id": str(job.generation_id), "state": job.state,
                    "native_absence_observed": absent, "disk_byte_erasure_claimed": False}


class NativeConfig(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    version: str
    search_config_file: str
    backend_proof_file: str
    backend_profile_sha256: str
    collection_name: str
    embedding_endpoint: str


def index_from_config(path: Path) -> MaterialNativeIndex:
    from citequest.search_v3.runtime import service_from_config
    path = Path(path).resolve(strict=True)
    config = NativeConfig.model_validate_json(path.read_bytes())
    if config.version != "citequest-material-native/1":
        raise ValueError("material_native_config_version_required")
    def resolved(value):
        candidate = Path(value)
        return (candidate if candidate.is_absolute() else path.parent/candidate).resolve(strict=True)
    service = service_from_config(resolved(config.search_config_file))
    chunker = service.query_token_counter.__self__
    return MaterialNativeIndex(service.factory, service.store, service.native, chunker,
        collection_name=config.collection_name, embedding_endpoint=config.embedding_endpoint,
        backend_profile_sha256=config.backend_profile_sha256,
        backend_proof=json.loads(resolved(config.backend_proof_file).read_bytes()))
