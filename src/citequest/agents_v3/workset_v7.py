"""Versioned host-identity-bound goals on the existing full-source Workset core."""
import copy

from citequest.research_v3.current_requirements import RequirementReview
from citequest.research_v3.requirements_identity import bound_requirement_review_model

from .contracts import AgentError, digest
from .dialogue_reference import add_reference
from .requirements_lifecycle import POLICY as LIFECYCLE_POLICY
from .workset_v6 import Workset6, Workset6Sources, WorksetRequirementsGate

VERSION = "citequest-typed-workset/7"
STAGE_COVERAGE_POLICY = """Stage assignment for the existing Workset workflow:
The server coverage_assignment defines this job's coverage obligation. In
document_contribution, extract and review this document's contribution to the
requested dimensions; do not require another document's findings or a completed
bilateral comparison. Preserve genuinely unavailable local details as bounded
gaps. In atomic_comparison, cover the exact immutable atomic question and its
assigned documents and requirement IDs, not every other comparison question.
Requirement IDs link this contribution to global goals; they do not assign the
whole parent goal to every local worker.
All claim/Q relation, numeric and condition judgments remain mandatory.
context_manifest.source_scope describes only the evidence supplied to this model
stage. not_in_current_stage does not mean unreadable, missing, unauthorized or
absent from the whole workset. The server workset_documents ledger records actual
available-text descriptors and completed window reads, not scientific findings
or proof of complete PDF extraction. Never turn local context omission into a
claim or limitation that another selected document has no readable source.
Without a coverage_assignment, comparison planning and final global_coverage
must still account for every selected document and every current requirement.
Local coverage cannot establish or override complete global comparison coverage.
"""


class Workset7RequirementsGate(WorksetRequirementsGate):
    profile_id = "workset-v7"
    review_output_cap = 4096
    strict_requirement_identity = True
    use_dialogue_references = True

    def goal_review_model(self, state: dict) -> type[RequirementReview]:
        return bound_requirement_review_model(digest(state["qa5"]["candidate"]), digest(state["qa5"]["turn"]["text"]))

    def prompt(self, state: dict, *, reviewer: bool = False, planner: bool = False) -> tuple[dict, dict]:
        value, manifest = super().prompt(state, reviewer=reviewer, planner=planner)
        if reviewer:
            value["expected_goal_bindings"] = {"candidate_sha256": digest(state["qa5"]["candidate"]),
                "user_turn_sha256": digest(state["qa5"]["turn"]["text"]),
                "identity_kind": "RequirementsCandidate; never another answer, review grid or parent state"}
        return value, manifest

    def finalize(self, state: dict) -> dict:
        value = super().finalize(state)
        value["workset_requirements"]["dialogue_reference"] = state["qa5"].get("dialogue_reference")
        return value


class Workset7Sources(Workset6Sources):
    profile_id = "workset-v7"
    review_output_cap = 4096
    protocol_version = VERSION
    lifecycle_policy = LIFECYCLE_POLICY + "\n" + STAGE_COVERAGE_POLICY

    def make_job(self, data, *, job_id, role, evidence, contexts, candidate=None,
                 review_required=False, task_data=None, atomic=None):
        job = super().make_job(data, job_id=job_id, role=role, evidence=evidence,
            contexts=contexts, candidate=candidate, review_required=review_required,
            task_data=task_data, atomic=atomic)
        manifest = self.manifest(data)
        if atomic is not None:
            targets = list(atomic["document_ids"])
            assignment = {"stage": "atomic_comparison", "claim_id": atomic["claim_id"],
                "question": atomic["question"], "requirement_ids": list(atomic["requirement_ids"])}
        else:
            targets = [(task_data or {}).get("document_id")]
            assignment = {"stage": "document_contribution"}
        if (not targets or any(target not in {doc.document_id for doc in manifest.documents} for target in targets)):
            raise AgentError("workset7_stage_document_assignment_invalid")
        assignment.update(protocol="citequest-workset-stage-coverage/1", document_ids=targets,
            workset_documents=[{"document_id": doc.document_id,
                "selection": doc.selection.model_dump(), "source_version_sha256": doc.source_version_sha256,
                "coverage_kind": doc.coverage_kind, "available_codepoints": doc.available_codepoints,
                "planned_windows": len(doc.windows),
                "read_windows": sum(window.window_id in data["windows"] for window in doc.windows),
                "in_current_stage_evidence": any(item.get("document_id") == doc.document_id for item in evidence),
                "complete_pdf_claimed": False} for doc in manifest.documents])
        job["coverage_assignment"] = assignment
        job["task_data"] = {**copy.deepcopy(job["task_data"]), "coverage_assignment": copy.deepcopy(assignment)}
        return job

    def project(self, anchor: dict, *, reviewer: bool = False, evidence: list[dict] | None = None) -> tuple[dict, dict]:
        value, _ = super().project(anchor, reviewer=reviewer, evidence=evidence)
        value = add_reference(value, anchor["typed_requirements"].get("dialogue_reference"))
        manifest = value["context_manifest"]
        manifest["source_scope_basis"] = "evidence_supplied_to_this_model_stage_not_global_workset_availability"
        for item in manifest.get("source_scope", []):
            if item.get("state") == "uncovered":
                item["state"] = "not_in_current_stage"
        return value, value["context_manifest"]

    def finish_document(self, state):
        value = super().finish_document(state)
        value["data"]["document_request_coverage"][-1]["scope"] = "document_contribution"
        return value

    def publish(self, state: dict) -> dict:
        value = super().publish(state)
        metadata = value["data"].get("last_wire_context_manifest", {}).get("dialogue_reference")
        if metadata is not None:
            value["result"]["context_manifest"]["dialogue_reference"] = metadata
        return value


class Workset7(Workset6):
    gate_class = Workset7RequirementsGate
    sources_class = Workset7Sources
    protocol_version = VERSION
