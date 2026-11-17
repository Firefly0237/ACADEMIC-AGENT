"""Explicit migrator-only binding/adoption using real files and existing SQL gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field
from sqlalchemy import select, text

from citequest.agents_v3.contracts import canonical, digest
from citequest.v2.db import engine, sessions
from citequest.v2.domain import Input


class MaintenanceCommand(Input):
    schema_version: Literal["citequest-publication-backend-maintenance/1"]
    phase: Literal["bind", "adopt"]
    business_dsn_file: Path
    generation_id: UUID
    backend_profile_file: Path
    adoption_file: Path | None = None
    quality_report_file: Path | None = None
    approval_file: Path | None = None


def execute(config: MaintenanceCommand) -> dict:
    from citequest.index_v2.models import IndexGeneration
    from citequest.index_v3.backend import read_profile
    from citequest.index_v3.models import BackendAdoption, BackendBinding
    profile, fingerprint, _ = read_profile(config.backend_profile_file)
    database = engine(config.business_dsn_file.read_text("utf8").strip().replace("postgresql://", "postgresql+psycopg://", 1))
    try:
        with sessions(database).begin() as db:
            if db.scalar(text("SELECT current_user")) != "cq_s03_migrator":
                raise ValueError("dedicated_migrator_identity_required")
            generation = db.get(IndexGeneration, config.generation_id, with_for_update=True)
            if generation is None or generation.source_root_kind != "publication_snapshot" or generation.expected_manifest_sha256 is None:
                raise ValueError("actually_prepared_publication_generation_required")
            if config.phase == "bind":
                existing = db.get(BackendBinding, generation.id)
                if existing is not None:
                    if existing.backend_profile_sha256 != fingerprint or existing.expected_manifest_sha256 != generation.expected_manifest_sha256:
                        raise ValueError("backend_binding_idempotency_conflict")
                else:
                    db.add(BackendBinding(generation_id=generation.id, backend_profile_sha256=fingerprint,
                        expected_manifest_sha256=generation.expected_manifest_sha256, profile=profile,
                        profile_canonical=canonical(profile).decode()))
                return {"state": "prepared", "backend_binding": "bound", "search_published": False}
            if any(value is None for value in (config.adoption_file, config.quality_report_file, config.approval_file)):
                raise ValueError("actual_adoption_quality_and_generation_approval_files_required")
            request = json.loads(config.adoption_file.read_text("utf8"))
            required = {column.key for column in BackendAdoption.__table__.columns} - {"adopted_at"}
            if set(request) != required:
                raise ValueError("exact_existing_backend_adoption_contract_required")
            approval_raw = config.approval_file.read_bytes()
            approval = json.loads(approval_raw)
            if (request["generation_id"] != str(generation.id) or request["backend_profile_sha256"] != fingerprint
                or request["expected_manifest_sha256"] != generation.expected_manifest_sha256
                or request["evaluation_sha256"] != generation.evaluation_sha256
                or digest(config.quality_report_file.read_bytes()) != request["quality_report_sha256"]
                or digest(approval_raw) != request["approval_sha256"]
                or approval != {"scope": "this_generation_using_existing_adopted_standard_backend_quality",
                    "generation_id": str(generation.id), "expected_manifest_sha256": generation.expected_manifest_sha256,
                    "backend_profile_sha256": fingerprint, "quality_report_sha256": request["quality_report_sha256"]}):
                raise ValueError("real_quality_report_and_explicit_generation_approval_pins_required")
            standard = db.get(BackendAdoption, UUID(request["standard_quality_generation_id"]))
            if (standard is None or standard.backend_profile_sha256 != fingerprint
                or standard.quality_report_sha256 != request["quality_report_sha256"] or standard.metrics != request["metrics"]):
                raise ValueError("actual_existing_standard_quality_required_no_invented_metrics")
            request["generation_id"], request["standard_quality_generation_id"] = generation.id, UUID(request["standard_quality_generation_id"])
            existing = db.get(BackendAdoption, generation.id)
            if existing is None:
                db.add(BackendAdoption(**request))
            elif any(getattr(existing, key) != value for key, value in request.items()):
                raise ValueError("backend_adoption_idempotency_conflict")
            return {"state": "prepared", "backend_adoption": "adopted_existing_standard_quality",
                "new_corpus_quality_measurement_claimed": False, "search_published": False}
    finally:
        database.dispose()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        path = args.config.resolve(strict=True)
        value = json.loads(path.read_text("utf8"))
        for key in ("business_dsn_file", "backend_profile_file", "adoption_file", "quality_report_file", "approval_file"):
            if value.get(key) is not None:
                selected = Path(value[key])
                value[key] = str(selected if selected.is_absolute() else path.parent / selected)
        print(json.dumps(execute(MaintenanceCommand.model_validate_json(json.dumps(value), strict=True))))
        return 0
    except Exception as error:
        print(json.dumps({"state": "prepared", "status": "blocked", "error": type(error).__name__, "search_published": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
