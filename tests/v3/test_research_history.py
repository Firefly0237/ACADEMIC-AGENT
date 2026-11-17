"""Offline checks for immutable intake identity and source inheritance."""
from copy import deepcopy
from uuid import uuid4

import pytest

from citequest.agents_v3.contracts import AgentError, VersionedBriefRef, digest
from citequest.agents_v3.research_history import source_ref_ids
from citequest.agents_v3.research_runtime import load_document_revision
from citequest.research_v3.brief import BriefDocument


class RevisionDB:
    def __init__(self, row):
        self.row = row

    def execute(self, _query, _parameters):
        return self

    def fetchone(self):
        return self.row


def test_persisted_intake_uuid_json_requires_exact_archived_identity():
    owner, project, brief_id = uuid4(), uuid4(), uuid4()
    document = BriefDocument(id=brief_id, project_id=project, revision=2)
    payload = document.model_dump(mode="json")
    reference = VersionedBriefRef(id=brief_id, revision=2, sha256=digest(payload))
    row = {"id": brief_id, "project_id": project, "revision": 2,
           "document": payload, "document_sha256": digest(payload)}
    actual, sha256 = load_document_revision(RevisionDB(row), owner, project, reference)
    assert actual == document and sha256 == reference.sha256
    for changed in ({**row, "revision": 3}, {**row, "document_sha256": "f" * 64},
                    {**row, "document": {**payload, "project_id": str(uuid4())}}):
        with pytest.raises(AgentError, match="research_brief_revision_integrity_failure"):
            load_document_revision(RevisionDB(changed), owner, project, reference)
    with pytest.raises(AgentError, match="research_brief_revision_unavailable"):
        load_document_revision(RevisionDB(None), owner, project, reference)


def test_only_exact_draft_ledger_refs_can_be_inherited():
    entry = {"ref_id": "pinned:1", "source_sha256": "a" * 64,
             "unit_text_sha256": "b" * 64,
             "source_ref": {"kind": "native_representation", "id": "c" * 64}}
    result = {"context_manifest": {"evidence_refs": [entry]}, "evidence_ledger": [deepcopy(entry)],
              "school_document": {"source_aliases": {"S1": ["pinned:1"]}},
              "citations": [{"ref_id": "pinned:1", "quote": "the actual source quote"}]}
    assert source_ref_ids(result) == ["pinned:1"]
    changed = deepcopy(result)
    changed["evidence_ledger"][0]["source_sha256"] = "d" * 64
    outside = deepcopy(result)
    outside["school_document"]["source_aliases"]["S1"].append("unissued:2")
    duplicate = deepcopy(result)
    duplicate["context_manifest"]["evidence_refs"].append(deepcopy(entry))
    for candidate in (changed, outside, duplicate):
        with pytest.raises(AgentError, match="research_source_evidence_binding_invalid"):
            source_ref_ids(candidate)
