"""Local viewer destinations must never admit automatic executable actions."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import (
    ArrayObject,
    BooleanObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NullObject,
    NumberObject,
    TextStringObject,
)

from citequest.materials_v2.pdf_navigation import safe_local_open_destination
from citequest.materials_v2.text import MaterialError, load_limits
from citequest.materials_v2.worker import _pdf_preflight


def reader() -> PdfReader:
    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    stream = io.BytesIO()
    writer.write(stream)
    stream.seek(0)
    return PdfReader(stream, strict=True)


@pytest.mark.parametrize(
    "view,parameters",
    [
        ("/Fit", []),
        ("/FitB", []),
        ("/XYZ", [NullObject(), NullObject(), NumberObject(0)]),
        ("/XYZ", [NumberObject(-4), FloatObject(796.5), FloatObject(1.25)]),
        ("/XYZ", [NullObject(), NullObject(), NullObject()]),
    ],
)
def test_current_document_inert_destinations(view: str, parameters: list) -> None:
    pdf = reader()
    page = pdf.pages[0].indirect_reference
    refs = {(page.idnum, page.generation)}
    value = ArrayObject([page, NameObject(view), *parameters])
    assert safe_local_open_destination(value, pdf, refs)
    assert safe_local_open_destination(
        DictionaryObject({NameObject("/S"): NameObject("/GoTo"), NameObject("/D"): value}), pdf, refs
    )


@pytest.mark.parametrize(
    "parameters",
    [
        [],
        [NullObject()],
        [NullObject(), NullObject()],
        [NullObject(), NullObject(), NumberObject(-1)],
        [BooleanObject(True), NullObject(), NumberObject(0)],
        [TextStringObject("1"), NullObject(), NumberObject(0)],
        [NameObject("/Null"), NullObject(), NumberObject(0)],
        [FloatObject(float("inf")), NullObject(), NumberObject(0)],
        [FloatObject(float("nan")), NullObject(), NumberObject(0)],
        [NullObject(), NullObject(), NumberObject(0), NullObject()],
    ],
)
def test_xyz_requires_exact_finite_typed_parameters(parameters: list) -> None:
    pdf = reader()
    page = pdf.pages[0].indirect_reference
    assert not safe_local_open_destination(
        ArrayObject([page, NameObject("/XYZ"), *parameters]), pdf, {(page.idnum, page.generation)}
    )


def test_page_reference_requires_same_reader_and_actual_page() -> None:
    pdf, foreign = reader(), reader()
    page = pdf.pages[0].indirect_reference
    refs = {(page.idnum, page.generation)}
    # Same object/generation numbers in a different PDF are insufficient.
    other = foreign.pages[0].indirect_reference
    assert (other.idnum, other.generation) in refs
    for target in (other, pdf.trailer.raw_get("/Root"), NumberObject(0), NullObject()):
        assert not safe_local_open_destination(
            ArrayObject([target, NameObject("/XYZ"), NullObject(), NullObject(), NumberObject(0)]), pdf, refs
        )
    assert not safe_local_open_destination(ArrayObject([page, TextStringObject("/Fit")]), pdf, refs)


@pytest.mark.parametrize("action", ("/URI", "/GoToR", "/JavaScript", "/Launch", "/SubmitForm"))
def test_remote_or_executable_open_action_is_rejected(action: str) -> None:
    pdf = reader()
    page = pdf.pages[0].indirect_reference
    value = DictionaryObject(
        {NameObject("/S"): NameObject(action), NameObject("/D"): ArrayObject([page, NameObject("/Fit")])}
    )
    assert not safe_local_open_destination(value, pdf, {(page.idnum, page.generation)})


@pytest.mark.parametrize("extra", ("/Next", "/F", "/URI", "/JS", "/AA"))
def test_goto_cannot_smuggle_extra_action_fields(extra: str) -> None:
    pdf = reader()
    page = pdf.pages[0].indirect_reference
    value = DictionaryObject(
        {
            NameObject("/S"): NameObject("/GoTo"),
            NameObject("/D"): ArrayObject([page, NameObject("/Fit")]),
            NameObject(extra): TextStringObject("untrusted"),
        }
    )
    assert not safe_local_open_destination(value, pdf, {(page.idnum, page.generation)})


@pytest.mark.parametrize("active", ("/JS", "/JavaScript", "/AA", "/Launch", "/EmbeddedFiles", "/RichMedia", "/XFA"))
def test_existing_recursive_active_content_gate_remains(tmp_path: Path, active: str) -> None:
    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    writer._root_object[NameObject(active)] = TextStringObject("explicit adversarial control")
    target = tmp_path / "active.pdf"
    writer.write(target)
    limits = load_limits(Path(__file__).resolve().parents[3] / "contracts/v2/policy.json")
    with pytest.raises(MaterialError, match="UNSUPPORTED_PDF_ACTIVE_CONTENT"):
        _pdf_preflight(target, limits)


def test_real_office_xyz_pdf_unchanged_bytes_is_admitted() -> None:
    root = Path(__file__).resolve().parents[3]
    target = root / "reports/stages/v3/S12-material-inputs/render-r3/research06-draft.pdf"
    if not target.exists():
        pytest.skip("explicit real Office product input fixture unavailable")
    before = target.read_bytes()
    pages, _ = _pdf_preflight(target, load_limits(root / "contracts/v2/policy.json"))
    assert len(pages) == 1 and pages[0]["status"] == "text"
    assert "八周" in pages[0]["text"] and "十二周" in pages[0]["text"]
    assert target.read_bytes() == before
