"""The adopted application stop rule supplements, never rewrites, policy v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal
from uuid import UUID

from citequest.v2.domain import SHA, DomainError, Input

SNAPSHOT = UUID("e7cabf2f-bf73-5cb9-8846-c1afb3b3ed35")
POLICY = "79d74a66007e3967a080f36703d470204ec5c5c875478bb9ba0f60221d8af17a"
EVALUATION = "26c005f6cc56cda7553f8b42e34aaf3b693944aba17f41274fc21ffeb1676e1b"
ADOPTION = "20f0b743e94e538dad69152e7812c3a7f61270736b3f29d08522c90512b7db24"
PROFILE_PATH = "contracts/v3/application-freeze-profile.json"
PROFILE_SHA = "e7abb3442a8ab9c000b3c8000e53e589e4743bfcd75694e6519355029d61bd05"
CHAIN_PATH = "reports/stages/v2/S05/rights-preparation/pdf-rights-20261104T222257393069Z/source-chain-reconciliation/outcomes.jsonl"
RIGHTS_PATH = "reports/stages/v2/S05/rights-preparation/pdf-rights-20261104T222257393069Z/rights-outcomes.jsonl"


class ApplicationFreezeInput(Input):
    """Only a digest precondition is supplied. Counts and membership are derived."""

    client_operation_id: UUID
    expected_profile_sha256: SHA
    expected_qualification_sha256: SHA


class FreezeProfile(Input):
    schema_version: Literal["cq.application-freeze-profile/1"]
    profile_id: Literal["litsearch-native-application-adr0015/1"]
    source_snapshot_id: UUID
    original_policy_sha256: SHA
    evaluation_sha256: SHA
    stop_rule_adoption_sha256: SHA
    minimum_qualified: Literal[500]
    maximum_qualified: Literal[2000]
    primary_positions: Literal[3000]
    pinned_component_candidates: Literal[692]
    stop_rule: Literal["audit_all_fixed_primary_then_freeze_all_qualified_if_minimum_met"]
    fallback: Literal["only_if_primary_qualified_below_minimum_separate_admission_required"]
    source_representation: Literal["entire_native_literal_text"]
    coordinate_system: Literal["unicode_codepoint"]
    minimum_characters: Literal[2000]
    maximum_characters: Literal[200000]
    minimum_paragraphs: Literal[5]
    minimum_section_headers: Literal[2]
    original_pdf_complete_extraction_claimed: Literal[False]
    third_party_figure_rights_claimed: Literal[False]
    freeze_grants_qa_or_stage_closure: Literal[False]


def read_pinned(root: Path, relative: str, expected: str, maximum: int) -> bytes:
    base = root.resolve(strict=True)
    path = base
    parts = Path(relative).parts
    if Path(relative).is_absolute() or ".." in parts:
        raise DomainError("application_evidence_path_escape", 422)
    for part in parts:
        path /= part
        if path.is_symlink():
            raise DomainError("application_evidence_symlink", 422)
    if not path.resolve(strict=True).is_relative_to(base):
        raise DomainError("application_evidence_path_escape", 422)
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != expected:
        raise DomainError("application_pinned_evidence_changed", 422)
    return raw


def profile(root: Path) -> tuple[FreezeProfile, str]:
    raw = read_pinned(root, PROFILE_PATH, PROFILE_SHA, 65536)
    value = FreezeProfile.model_validate_json(raw)
    if (
        value.source_snapshot_id != SNAPSHOT
        or value.original_policy_sha256 != POLICY
        or value.evaluation_sha256 != EVALUATION
        or value.stop_rule_adoption_sha256 != ADOPTION
    ):
        raise DomainError("application_profile_binding_conflict", 409)
    read_pinned(root, "contracts/v2/policy.json", POLICY, 1024**2)
    read_pinned(root, "contracts/v2/evaluation.json", EVALUATION, 1024**2)
    read_pinned(root, "docs/decisions/ADR-0015-application-corpus-stop-rule-adoption.md", ADOPTION, 1024**2)
    return value, hashlib.sha256(raw).hexdigest()


def original_inventory(raw: bytes) -> list[dict]:
    values = [json.loads(line) for line in raw.splitlines()]
    if (
        len(values) != 3000
        or [v.get("rank") for v in values] != list(range(3000))
        or any(type(v.get("native_id")) is not int for v in values)
        or len({v["native_id"] for v in values}) != 3000
    ):
        raise DomainError("application_original_candidate_order_changed", 422)
    return values
