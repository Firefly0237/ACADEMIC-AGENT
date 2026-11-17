"""Small synthetic OOXML admission controls; not Office rendering conformance."""
from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from citequest.materials_v2.text import load_limits
from citequest.materials_v2.worker import _docx_preflight

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("extra, reason", [
    ('<w:r><a:t>unmapped DrawingML text</a:t></w:r>', "DOCX_MAIN_DRAWING_TEXT_NOT_YET_MAPPED"),
    ('<w:altChunk r:id="alt1"/>', "DOCX_ALTERNATIVE_CONTENT_NOT_YET_MAPPED"),
])
def test_unmapped_main_part_content_is_explicitly_partial_candidate(tmp_path: Path, extra: str, reason: str) -> None:
    path = tmp_path / "synthetic-main-part.docx"
    content = ('<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
               'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
               'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
               '<w:body><w:p><w:r><w:t>verified main paragraph</w:t></w:r>' + extra + '</w:p></w:body></w:document>')
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        archive.writestr("word/document.xml", content)
    paragraphs, _, missing = _docx_preflight(path, load_limits(ROOT / "contracts/v2/policy.json"))
    assert paragraphs[0]["text"] == "verified main paragraph"
    assert [row["reason"] for row in missing] == [reason]
