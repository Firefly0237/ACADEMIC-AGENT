"""Fail-closed literal export coverage; not scientific or mathematical review."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from zipfile import ZipFile

from lxml import etree as ET
from pypdf import PageObject, PdfReader

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def normalized(text: str, *, math: bool = False) -> str:
    value = unicodedata.normalize("NFKC", text)
    if math:
        value = "".join(char for char in value if char != "^" and unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", "", value)


def math_text(node: ET._Element) -> str:
    local = ET.QName(node).localname
    if local == "t":
        return node.text or ""
    if local.endswith("Pr"):
        return ""
    if local == "nary":
        char = node.find("m:naryPr/m:chr", NS)
        operator = char.get("{" + NS["m"] + "}val") if char is not None else "∫"
        return operator + "".join(math_text(child) for child in node)
    if local == "d":
        props = node.find("m:dPr", NS)

        def delimiter(name: str, default: str) -> str:
            value = props.find("m:" + name, NS) if props is not None else None
            return value.get("{" + NS["m"] + "}val", default) if value is not None else default

        return (
            delimiter("begChr", "(")
            + delimiter("sepChr", "|").join(math_text(child) for child in node if ET.QName(child).localname != "dPr")
            + delimiter("endChr", ")")
        )
    return "".join(math_text(child) for child in node)


def _page_text(page: PageObject) -> tuple[str, str]:
    body, footer = [], []

    def visitor(text: str, matrix: list[float], position: list[float], font: dict | None, size: float) -> None:
        (footer if matrix[5] + position[5] < 70 else body).append(text)

    page.extract_text(visitor_text=visitor)
    return "".join(body), normalized("".join(footer))


def verify_pdf(pdf: Path, subdocuments: list[Path]) -> dict:
    reader = PdfReader(pdf)
    if (
        not reader.pages
        or len(reader.pages) > 100
        or any(
            abs(float(page.mediabox.width) - 595.3) > 2 or abs(float(page.mediabox.height) - 841.9) > 2
            for page in reader.pages
        )
    ):
        raise ValueError("export_pdf_A4_page_profile_failed")
    texts, footers = [], []
    for page in reader.pages:
        body, footer = _page_text(page)
        texts.append(body)
        footers.append(footer)
    start = next((index for index, text in enumerate(texts) if "Proposed Title" in text), None)
    if (
        start != 2
        or "Venue" not in texts[0]
        or "Instruction" not in texts[1]
        or "fordoctoralguidance." not in normalized(texts[1])
    ):
        raise ValueError("export_pinned_frontmatter_layout_overflow_or_content_missing")
    if start is None or any(footers[index] != str(index - start + 1) for index in range(start, len(reader.pages))):
        raise ValueError("export_body_page_number_restart_or_sequence_failed")
    combined = normalized("".join(texts))
    math_combined = normalized("".join(texts), math=True)
    literal_units, equations = [], []
    for path in subdocuments:
        with ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
        for paragraph in root.xpath("//w:body//w:p", namespaces=NS):
            chunks = []
            for node in paragraph.iter():
                if node.tag == "{" + NS["m"] + "}oMath":
                    if chunks:
                        literal_units.append(normalized("".join(chunks)))
                        chunks = []
                    equations.append(normalized(math_text(node), math=True))
                elif node.tag == "{" + NS["w"] + "}t":
                    chunks.append(node.text or "")
            if chunks:
                literal_units.append(normalized("".join(chunks)))
    missing = [index for index, value in enumerate(literal_units) if value and value not in combined]
    invisible = [index for index, value in enumerate(equations) if value and value not in math_combined]
    if missing or invisible:
        raise ValueError("export_pdf_literal_or_formula_visibility_coverage_failed")
    return {
        "status": "passed",
        "scope": "normalized literal paragraph/table-cell coverage and formula visibility, not mathematical semantics or per-document human visual review",
        "pages": len(reader.pages),
        "first_body_physical_page": start + 1,
        "literal_units": len(literal_units),
        "equations": len(equations),
        "missing_literal_units": missing,
        "unmatched_formula_units": invisible,
        "body_page_numbers_restart_at_one": True,
    }
