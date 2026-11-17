"""Explicit Linux rendering derivative; the public source DOCX stays immutable.

LibreOffice's DOCX importer needs a paragraph anchor before a table-first
section to retain the section's page-number restart. This adapter also names
the actual open fonts selected by a versioned Linux profile, instead of
depending on undocumented missing-font fallback. It changes no content.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from lxml import etree as ET

PROFILE_ID = "citequest-linux-arphic-ukai-uming-liberation/2"
FONT_MAPPING = {
    "楷体": "AR PL UKai CN",
    "楷体_GB2312": "AR PL UKai CN",
    "华文楷体": "AR PL UKai CN",
    "华文仿宋": "AR PL UKai CN",
    "仿宋_GB2312": "AR PL UKai CN",
    "宋体": "AR PL UMing CN",
    "SimSun": "AR PL UMing CN",
    "黑体": "Noto Sans CJK SC",
    "MS Gothic": "Noto Sans CJK SC",
    "Times New Roman": "Liberation Serif",
    "Cambria Math": "DejaVu Sans",
}
NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}


def _q(name: str) -> str:
    prefix, local = name.split(":", 1)
    return "{" + NS[prefix] + "}" + local


def _content(root: ET._Element) -> dict:
    return {
        "text": root.xpath("//w:t/text()|//m:t/text()", namespaces=NS),
        "controls": root.xpath("//w:sdtPr/w:id/@w:val", namespaces=NS),
        "permissions": {
            name: dict(Counter(root.xpath("//w:" + name + "/@w:id", namespaces=NS)))
            for name in ("permStart", "permEnd")
        },
        "equations": [ET.tostring(node).decode("utf8") for node in root.xpath("//m:oMath", namespaces=NS)],
        "sections": len(root.xpath("//w:sectPr", namespaces=NS)),
        "page_restarts": root.xpath("//w:sectPr/w:pgNumType/@w:start", namespaces=NS),
    }


def prepare_linux_render_input(source: Path, output: Path, *, expected_sha256: str) -> dict:
    """Prepare a private derivative of an already trusted generated source DOCX.

    Never call this with an arbitrary uploaded Office package. The host supplies
    the exact exporter receipt SHA. The original DOCX remains the Word download;
    this private derivative exists solely as the input of the Linux PDF job.
    """
    if source.is_symlink() or not source.is_file() or source.stat().st_size > 16 * 1024**2:
        raise ValueError("trusted_generated_docx_required")
    original = source.read_bytes()
    actual_sha = hashlib.sha256(original).hexdigest()
    if actual_sha != expected_sha256:
        raise ValueError("trusted_generated_docx_hash_mismatch")
    if output.exists() or output.resolve() == source.resolve():
        raise FileExistsError("preserve_prior_source_and_derivative")
    with ZipFile(source) as archive:
        if sum(item.file_size for item in archive.infolist()) > 64 * 1024**2:
            raise ValueError("generated_docx_unpacked_size_exceeded")
        parts = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(parts["word/document.xml"])
    before = _content(root)
    if before["sections"] != 2 or before["page_restarts"] != ["1"]:
        raise ValueError("expected_school_section_contract_required")
    body = root.find("w:body", NS)
    if body is None:
        raise ValueError("school_document_body_required")
    anchors = [
        node
        for node in body
        if node.tag == _q("w:p")
        and node.find("w:pPr/w:sectPr", NS) is not None
        and node.getnext() is not None
        and node.getnext().tag == _q("w:tbl")
    ]
    if len(anchors) != 1:
        raise ValueError("exactly_one_table_first_school_section_required")
    anchor = ET.Element(_q("w:p"))
    props = ET.SubElement(anchor, _q("w:pPr"))
    ET.SubElement(
        props, _q("w:spacing"), {_q("w:before"): "0", _q("w:after"): "0", _q("w:line"): "20", _q("w:lineRule"): "exact"}
    )
    run_props = ET.SubElement(props, _q("w:rPr"))
    ET.SubElement(run_props, _q("w:sz"), {_q("w:val"): "2"})
    anchors[0].addnext(anchor)
    grid_paragraphs = 0
    for container in body:
        if container is anchors[0]:
            break
        for paragraph in container.iter(_q("w:p")):
            properties = paragraph.find(_q("w:pPr"))
            if properties is None:
                properties = ET.Element(_q("w:pPr"))
                paragraph.insert(0, properties)
            snap = properties.find(_q("w:snapToGrid"))
            if snap is None:
                snap = ET.SubElement(properties, _q("w:snapToGrid"))
            snap.set(_q("w:val"), "0")
            grid_paragraphs += 1
    replacements: Counter[str] = Counter()
    for name in ("word/document.xml", "word/styles.xml", "word/settings.xml"):
        tree = root if name == "word/document.xml" else ET.fromstring(parts[name])
        for fonts in tree.xpath("//w:rFonts", namespaces=NS):
            for attribute in ("ascii", "eastAsia", "hAnsi", "cs"):
                value = fonts.get(_q("w:" + attribute))
                if value in FONT_MAPPING:
                    fonts.set(_q("w:" + attribute), FONT_MAPPING[value])
                    replacements[value] += 1
        for math_font in tree.xpath("//m:mathFont", namespaces=NS):
            value = math_font.get(_q("m:val"))
            if value in FONT_MAPPING:
                math_font.set(_q("m:val"), FONT_MAPPING[value])
                replacements[value] += 1
        parts[name] = ET.tostring(tree, xml_declaration=True, encoding="UTF-8", standalone=True)
    if _content(root) != before:
        raise ValueError("linux_render_derivative_changed_text_or_source_controls")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x", ZIP_DEFLATED) as archive:
        for name, value in parts.items():
            archive.writestr(name, value)
    return {
        "schema_version": "citequest-linux-render-derivative/1",
        "status": "prepared_render_audit_and_visual_pending",
        "font_profile": PROFILE_ID,
        "original_docx_sha256": actual_sha,
        "render_input_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "public_source_docx_changed": source.read_bytes() != original,
        "text_controls_equations_sections_and_page_restart_preserved": True,
        "table_first_section_anchor": "one empty 1pt paragraph after original section break",
        "frontmatter_paragraphs_using_explicit_line_spacing_not_Windows_grid": grid_paragraphs,
        "frontmatter_sizes_and_source_line_spacing_changed": False,
        "font_mapping": FONT_MAPPING,
        "font_attribute_replacements": dict(replacements),
        "font_equivalence": "explicit open-family PDF derivative; not Microsoft font-family or pixel identity",
        "public_word_download": "original_source_DOCX_with_source_family_requests",
        "school_submission_approval": False,
    }


def write_derivative_receipt(receipt: dict, path: Path) -> None:
    if path.exists():
        raise FileExistsError("preserve_prior_derivative_receipt")
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
