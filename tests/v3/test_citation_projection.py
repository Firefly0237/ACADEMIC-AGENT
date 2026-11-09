"""Source-only CSL projection; no formatter, database or provider invocation."""

from uuid import uuid4

from citequest.search_v3.citation import project_csl
from citequest.search_v3.schemas import PaperDetail


def project(**changes):
    values = {"identifier": "metadata_projection:source", "title": "Actual title", "authors": ["Full Source Name"],
        "year": 2025, "url": "https://example.org/paper", "metadata": {}}
    return project_csl(**{**values, **changes})


def test_unknown_publication_type_has_honest_generic_reference():
    item, missing = project()
    assert item.model_dump(mode="json") == {"id": "metadata_projection:source", "type": "document",
        "title": "Actual title", "author": [{"literal": "Full Source Name"}],
        "issued": {"date-parts": [[2025]]}, "URL": "https://example.org/paper"}
    assert missing == ["publication_type"]
    paper, gaps = project(verified_paper=True, metadata={"doi": "10.18653/v1/source", "publisher": "Actual Publisher"})
    native = paper.model_dump(mode="json")
    assert native["type"] == "article" and native["DOI"] == "10.18653/v1/source"
    assert native["publisher"] == "Actual Publisher" and gaps == ["publication_type"]
    assert not {"container-title", "volume", "issue", "page"}.intersection(native)


def test_explicit_source_classification_and_fields_are_retained():
    item, missing = project(metadata={"type": "journal-article", "journal": "Actual Journal",
        "volume": "12", "issue": "2", "pages": "10-18", "publisher": "Actual Publisher"})
    value = item.model_dump(mode="json")
    assert not missing
    assert {key: value[key] for key in ("type", "container-title", "volume", "issue", "page", "publisher")} == {
        "type": "article-journal", "container-title": "Actual Journal", "volume": "12", "issue": "2",
        "page": "10-18", "publisher": "Actual Publisher"}
    conference, gaps = project(metadata={"entry_type": "inproceedings", "booktitle": "Actual Proceedings"})
    assert conference.type == "paper-conference" and not gaps
    report, gaps = project(metadata={"type": "report", "number": "TR-5"})
    assert "issue" not in report.model_dump() and gaps == ["publisher"]


def test_invalid_or_missing_values_are_omitted_without_completion():
    item, missing = project(title=None, authors=["Real name", "bad\x00name"], year=True,
        url="https://user:secret@example.org/paper", metadata={"doi": "not a DOI", "venue": "Actual Container"})
    value = item.model_dump(mode="json")
    assert value == {"id": "metadata_projection:source", "type": "document", "container-title": "Actual Container"}
    assert missing == ["publication_type", "title", "author", "issued"]
    for url in ("javascript:alert(1)", "https://example.org/\r\nheader", "https://[broken"):
        projected, _ = project(url=url)
        assert "URL" not in projected.model_dump()


def test_detail_optional_fields_preserve_old_wire_and_aliases():
    values = {"paper": {"kind": "metadata_projection", "id": str(uuid4())}, "title": "Actual title",
        "abstract": None, "authors": None, "year": None, "work_id": None, "work_version_id": None,
        "coverage": "metadata", "source": {"source_namespace": "fixture", "native_id": "paper",
            "source_revision": None, "projection_sha256": "a" * 64, "source_pointer_sha256": "b" * 64},
        "source_snapshot_id": uuid4(), "literal_text_available": False,
        "coordinate_system": "unicode_codepoints_in_metadata_recipe", "links_status": "not_in_source",
        "citation_status": "incomplete_metadata"}
    old = PaperDetail.model_validate(values, strict=True).model_dump(mode="json")
    assert "csl" not in old and "citation_missing_fields" not in old
    item, missing = project(metadata={"type": "paper-conference", "venue": "Actual Proceedings"})
    detail = PaperDetail.model_validate({**values, "csl": item, "citation_missing_fields": missing or None}, strict=True)
    nested = detail.model_dump(mode="json")["csl"]
    assert nested["issued"] == {"date-parts": [[2025]]}
    assert nested["container-title"] == "Actual Proceedings"
    assert "date_parts" not in nested["issued"] and "container_title" not in nested
