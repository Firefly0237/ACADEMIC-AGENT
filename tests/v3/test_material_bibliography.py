"""Strict persisted bibliography JSON and unchanged source/version admission."""
from copy import deepcopy
from types import SimpleNamespace
from uuid import UUID

import pytest
from pydantic import ValidationError

from citequest.agents_v3.contracts import digest
from citequest.research_v3.school_contracts import ReferenceMetadataSource
from citequest.research_v3.school_export import current_references, reference_metadata
from citequest.search_v3.schemas import PaperDetail, PaperLocator, SourceCitation


def material_fixture(kind="material_fragment"):
    identifier = str(UUID(int=11))
    bibliography = {"file_id": identifier, "file_sha256": "a" * 64,
        "representation_id": "b" * 64, "title": "Synthetic research paper",
        "authors": ["Synthetic Author"], "year": 2025, "confirmed": True,
        "provenance": "user_confirmed", "confirmation_message_id": str(UUID(int=12))}
    source_kind = "material_file" if kind == "source_window" else kind
    source = {"kind": source_kind, "id": identifier if source_kind == "material_file" else "c" * 64}
    evidence = {"ref_id": "material:fixture", "source_ref": source, "file_id": identifier,
        "source_file_sha256": "a" * 64, "representation_id": "b" * 64,
        "source_sha256": "d" * 64, "unit_text_sha256": "e" * 64}
    confirmed = {"bibliography": bibliography, "bibliography_sha256": digest(bibliography),
        "provenance": "user_confirmed", "source": {"file_id": identifier,
            "file_sha256": "a" * 64, "representation_id": "b" * 64}}
    if kind == "source_window":
        evidence.update(evidence_type="source_window", selection={"kind": "material_file", "id": identifier},
            source_version_pin={"source_file_sha256": evidence.pop("source_file_sha256"),
                "representation_id": evidence.pop("representation_id")})
        evidence.pop("file_id")
    return evidence, confirmed


def resolve(evidence, confirmed, origins=None):
    def no_paper(_locator):
        pytest.fail("uploaded metadata must use its authorized material reader")
    return reference_metadata({"S1": [evidence["ref_id"]]}, [deepcopy(evidence)],
        lambda _ref: evidence, no_paper, read_material_bibliography=lambda _file: confirmed,
        metadata_sources=origins)


@pytest.mark.parametrize("kind", ["material_fragment", "material_file", "source_window"])
def test_persisted_uuid_strings_use_strict_json_and_keep_origin(kind):
    evidence, confirmed = material_fixture(kind)
    origins = {}
    references = resolve(evidence, confirmed, origins)
    assert references == [{"id": "S1", "type": "article", "title": "Synthetic research paper",
        "author": [{"literal": "Synthetic Author"}], "issued": {"date-parts": [[2025]]}}]
    assert origins["S1"]["provenance"] == "user_confirmed"
    assert origins["S1"]["bibliography_sha256"] == confirmed["bibliography_sha256"]
    assert origins["S1"]["representation_id"] == confirmed["source"]["representation_id"]


@pytest.mark.parametrize("patch", [{"year": "2025"}, {"confirmed": False},
    {"unexpected": "not a schema field"}, {"file_id": "invalid-uuid"}])
def test_json_boundary_does_not_weaken_types_or_schema(patch):
    evidence, confirmed = material_fixture()
    confirmed["bibliography"].update(patch)
    confirmed["bibliography_sha256"] = digest(confirmed["bibliography"])
    with pytest.raises(ValidationError):
        resolve(evidence, confirmed)


def test_current_source_pin_change_is_rejected():
    evidence, confirmed = material_fixture()
    confirmed["source"]["representation_id"] = "f" * 64
    with pytest.raises(ValueError, match="no longer matches"):
        resolve(evidence, confirmed)


@pytest.mark.parametrize("old_metadata_sources", [None, {}])
def test_existing_public_paper_mapping_needs_no_material_callback(old_metadata_sources):
    locator = PaperLocator(kind="native_representation", id="a" * 64)
    source = SourceCitation(source_namespace="synthetic_fixture", native_id="paper-1",
        source_revision=None, projection_sha256="b" * 64, source_pointer_sha256="c" * 64)
    detail = PaperDetail(paper=locator, title="Public fixture", abstract=None,
        authors=["Fixture Author"], year=2024, work_id=None, work_version_id=None,
        coverage="native_literal", source=source, source_snapshot_id=UUID(int=1),
        literal_text_available=True, coordinate_system="unicode_codepoints_in_native_literal",
        links_status="not_in_source", citation_status="complete")
    evidence = {"ref_id": "public:fixture", "source_ref": locator.model_dump(),
        "source_sha256": "d" * 64, "unit_text_sha256": "e" * 64}
    references = reference_metadata({"S1": [evidence["ref_id"]]}, [evidence],
        lambda _ref: evidence, lambda _locator: detail)
    assert references[0]["title"] == "Public fixture"
    assert references[0]["issued"] == {"date-parts": [[2024]]}
    document = SimpleNamespace(sections=[SimpleNamespace(body=lambda: "[@S1]")],
        source_aliases={"S1": [evidence["ref_id"]]}, context_manifest={"evidence_refs": [evidence]},
        reference_admission=SimpleNamespace(metadata_sources=old_metadata_sources))
    assert current_references(document, lambda _ref: evidence, lambda _locator: detail) == references


def test_changed_confirmation_cannot_reuse_prior_export_admission():
    evidence, confirmed = material_fixture()
    origins = {}
    resolve(evidence, confirmed, origins)
    document = SimpleNamespace(sections=[SimpleNamespace(body=lambda: "[@S1]")],
        source_aliases={"S1": [evidence["ref_id"]]}, context_manifest={"evidence_refs": [evidence]},
        reference_admission=SimpleNamespace(metadata_sources={"S1": ReferenceMetadataSource(**origins["S1"])}))
    confirmed["bibliography"]["confirmation_message_id"] = str(UUID(int=13))
    confirmed["bibliography_sha256"] = digest(confirmed["bibliography"])
    with pytest.raises(ValueError, match="provenance or source version changed"):
        current_references(document, lambda _ref: evidence, lambda _locator: None,
            read_material_bibliography=lambda _file: confirmed)
