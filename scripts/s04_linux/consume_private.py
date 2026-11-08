"""Trusted one-File parent: official SDK storage, fixed secretless parser child.

The parent has PG/runtime storage identity but no Docker socket or model key.
No migrations, retries, failed-File reset, or cloud activation happen here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.engine import make_url

from citequest.business_storage_v3.io_pg import PGResourceConfig
from citequest.business_storage_v3.io_process import SupervisedBlobIO
from citequest.business_storage_v3.service import BusinessStorage
from citequest.materials_v2 import models as m
from citequest.materials_v2.processor import MaterialProcessor
from citequest.materials_v3.broker_protocol import rooted, strict_object
from citequest.materials_v3.private_spool_client import PrivateSpoolBrokerParser
from citequest.materials_v3.private_spool_protocol import TRANSPORT
from citequest.platform_v3.blob_config import BlobStoreConfig
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Actor
from citequest.v2.models import Blob
from citequest.v2.services import Catalog

ROOT = Path("/app")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-id", type=UUID, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--dsn-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists() or not args.receipt.resolve().is_relative_to(Path("/evidence")):
        raise ValueError("new_owned_material_receipt_required")
    raw = args.profile.read_bytes()
    if len(raw) > 1024**2 or hashlib.sha256(raw).hexdigest() != args.profile_sha256:
        raise ValueError("exact_bounded_private_parent_profile_required")
    profile = json.loads(raw, object_pairs_hook=strict_object)
    if (profile.get("schema_version") != "cq.linux-material-parent-execution/3"
            or profile.get("source_transport") != TRANSPORT or profile.get("product_api_docker_access") is not False
            or profile.get("parent_docker_socket") is not False or profile.get("parent_model_keys") is not False
            or str(args.file_id) != profile.get("assigned_file_id")):
        raise ValueError("fixed_one_File_private_parent_required")
    for relative, expected in profile["parent_source_bindings"].items():
        if hashlib.sha256(rooted(ROOT, relative).read_bytes()).hexdigest() != expected:
            raise ValueError("private_parent_source_changed")
    storage_raw = args.storage.read_bytes()
    if hashlib.sha256(storage_raw).hexdigest() != profile["storage_config_sha256"]:
        raise ValueError("private_parent_storage_config_changed")
    config = BlobStoreConfig.model_validate_json(storage_raw)
    dsn_raw = args.dsn_file.read_text().strip()
    url = make_url(dsn_raw)
    if (url.username != profile["database_role"] or url.database != profile["database_name"]
            or url.host != profile["database_host"]):
        raise ValueError("private_parent_database_identity_mismatch")
    # One authoritative DSN drives both business SQL and resource sessions.
    # No independently configured PG lock namespace can evade fleet binding.
    database = engine(dsn_raw, connect_timeout=3, pool_timeout=3)
    factory = sessions(database)
    storage = BusinessStorage(factory, config, io_executor=SupervisedBlobIO(config,
        resource=PGResourceConfig(url.set(drivername="postgresql").render_as_string(hide_password=False))))
    try:
        with factory.begin() as db:
            actor = Actor(UUID(profile["assigned_owner_id"]))
            file = Catalog(db, actor).file(args.file_id)
            ingestion = db.scalar(select(m.MaterialIngestion).where(m.MaterialIngestion.file_id == file.id,
                m.MaterialIngestion.owner_id == actor.owner_id))
            blob = db.get(Blob, (file.security_domain, file.sha256))
            if (file.status != "queued" or ingestion is None or ingestion.state != "queued"
                    or str(ingestion.id) != profile["assigned_ingestion_id"]
                    or file.sha256 != profile["assigned_source_sha256"] or file.media_type != profile["assigned_media_type"]
                    or blob is None or blob.size_bytes != profile["assigned_size_bytes"]):
                raise ValueError("assigned_File_is_not_current_unattempted_source")
        exchange = rooted(ROOT, profile["exchange_relative"])
        runner = PrivateSpoolBrokerParser(root=ROOT, exchange=exchange, nonce=profile["nonce"],
                                         parser_image_id=profile["image_id"])
        processor = MaterialProcessor(factory, None, business_storage=storage,
            execution_id=UUID(profile["assigned_execution_id"]), root=ROOT,
            budget_seconds=profile["resource_contract"]["active_seconds"],
            parser_python=Path("/usr/local/bin/python"), artifact_root=ROOT / "data/runtime-s04-linux/artifacts",
            runner=runner, parser_profile=rooted(ROOT, profile["parser_profile_relative"]),
            chunk_profile=ROOT / "runtime/s04-linux/chunk-worker-profile.json",
            extra_configuration_files=("runtime/material-parent/requirements.txt",
                "runtime/material-parent/dependency-provenance.json", "scripts/s04_linux/consume_private.py",
                args.profile.resolve().relative_to(ROOT).as_posix()))
        result = processor.run_next(file_id=args.file_id)
        args.receipt.write_text(json.dumps({"status": "observed", "result": result,
            "profile_sha256": args.profile_sha256, "parser_image_id": profile["image_id"],
            "parent_runtime_image_id_declared": profile["parent_runtime_image_id"],
            "source_transport": TRANSPORT, "storage_binding_sha256": storage.binding.sha256,
            "input_quarantined": runner.quarantined, "parent_resource_owner_retained": not runner.termination_confirmed,
            "model_calls": 0, "cloud_activation": "closed", "scope": "one controlled File, not a cloud deployment/SLA"}, indent=2) + "\n")
        if runner.quarantined and not runner.termination_confirmed:
            # Keep the process/verified FD/local quota until read-only receipt
            # reconciliation confirms physical stop. No new parser window/job.
            while not runner.termination_confirmed:
                time.sleep(0.1)
            reconciliation = processor.observe_failed_stop(args.file_id)
            args.receipt.with_name("late-stop-observation.json").write_text(json.dumps({
                "file_id": str(args.file_id), "profile_sha256": args.profile_sha256,
                "physical_stop_observed": True, "original_failure_preserved": True,
                "publication_permitted": False, "reconciliation": reconciliation}) + "\n")
    finally:
        database.dispose()
        exchange = rooted(ROOT, profile["exchange_relative"])
        (exchange / "parent-done.json").write_text(json.dumps({"file_id": str(args.file_id),
            "profile_sha256": args.profile_sha256, "uncertain_input_may_remain": True}))


if __name__ == "__main__":
    main()
