"""Typed artifact provenance from the Task's existing settled model ledger."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from citequest.agents_v3.contracts import AgentError, Strict, digest

SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]
Key = Annotated[str, Field(min_length=1, max_length=600)]


class HandoffUsage(Strict):
    input: int = Field(ge=0)
    cached: int = Field(ge=0)
    output: int = Field(ge=0)

    @model_validator(mode="after")
    def cached_within_input(self) -> HandoffUsage:
        if self.cached > self.input:
            raise ValueError("cached tokens exceed input")
        return self


class ModelReceipt(Strict):
    attempt_id: str = Field(min_length=1, max_length=64)
    logical_key: Key
    request_sha256: SHA
    response_sha256: SHA
    schema_name: str = Field(min_length=1, max_length=160)
    usage: HandoffUsage


class TargetRevision(Strict):
    target_id: str = Field(pattern=r"^(?:(?:RQ|E)[1-9][0-9]?|[a-z][a-z0-9_]{0,79})$")
    before_sha256: SHA
    after_sha256: SHA
    unaffected_sha256: SHA
    resolved_conflict_refs: list[Key] = Field(max_length=8)


class HandoffBinding(Strict):
    sources: dict[Key, Annotated[str, Field(min_length=1, max_length=160)]] = Field(max_length=4)
    parents: dict[Annotated[str, Field(min_length=1, max_length=100)], SHA] = Field(max_length=32)
    revision: TargetRevision | None = None


class HandoffAudit(Strict):
    task_id: str = Field(min_length=1, max_length=64)
    command_id: str = Field(min_length=1, max_length=64)
    brief_sha256: SHA
    profile_sha256: SHA
    role: str = Field(min_length=1, max_length=64)
    artifact_sha256: SHA
    parent_artifact_sha256s: dict[str, SHA] = Field(max_length=32)
    sources: list[ModelReceipt] = Field(max_length=4)
    authority: Literal["versioned_user_brief", "settled_model_artifact"]
    revision: TargetRevision | None = None


def bind_handoff(entry: dict, sources: dict[str, str], parents: dict[str, str], *,
                 revision: dict | None = None) -> dict:
    """Bind trusted call keys and observed parent identities, never model declarations."""
    binding = HandoffBinding(sources=sources, parents=parents,
        revision=TargetRevision.model_validate(revision, strict=True) if revision is not None else None)
    return {**entry, "audit_binding": binding.model_dump(mode="json")}


def source_keys(entries: list[dict]) -> list[str]:
    return list(dict.fromkeys(key for entry in entries for key in entry.get("audit_binding", {}).get("sources", {})))


def collect_handoff_audit(entries: list[dict], receipts: dict[str, dict], *, task_id: str,
                          command_id: str, brief_sha256: str, profile_sha256: str) -> dict:
    records = []
    for entry in entries:
        if digest(entry["artifact"]) != entry["sha256"]:
            raise AgentError("research_handoff_artifact_hash_changed")
        user = entry["role"] == "user"
        if not user and "audit_binding" not in entry:
            raise AgentError("research_handoff_model_binding_missing")
        binding = HandoffBinding.model_validate(entry.get("audit_binding", {"sources": {}, "parents": {}}), strict=True)
        if user and binding.sources or not user and not binding.sources:
            raise AgentError("research_handoff_source_authority_invalid")
        if binding.revision is not None:
            revised = entry["artifact"].get("replacement", entry["artifact"])
            target = revised.get("id", revised.get("section_id"))
            before = binding.parents.get("current_target", binding.parents.get("previous_section"))
            if (target != binding.revision.target_id or digest(revised) != binding.revision.after_sha256
                    or before != binding.revision.before_sha256
                    or entry["role"] == "reviser" and entry["artifact"].get("resolved_conflict_ids")
                        != binding.revision.resolved_conflict_refs):
                raise AgentError("research_handoff_revision_scope_invalid")
        sources = []
        for key, schema in binding.sources.items():
            if key not in receipts:
                raise AgentError("research_handoff_settled_receipt_missing")
            source = ModelReceipt.model_validate(receipts[key], strict=True)
            if source.logical_key != key or source.schema_name != schema or not key.startswith(command_id + ":"):
                raise AgentError("research_handoff_receipt_identity_invalid")
            sources.append(source)
        records.append(HandoffAudit(task_id=task_id, command_id=command_id,
            brief_sha256=brief_sha256, profile_sha256=profile_sha256, role=entry["role"],
            artifact_sha256=entry["sha256"], parent_artifact_sha256s=binding.parents,
            sources=sources, authority="versioned_user_brief" if user else "settled_model_artifact",
            revision=binding.revision).model_dump(mode="json"))
    return {"protocol": "citequest-research-handoff-audit/1", "scope": "artifact_and_receipt_integrity",
        "handoffs": records}
