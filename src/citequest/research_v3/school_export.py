"""Pinned CSL/Pandoc/docxtpl export with current ACL and exact-content receipts.

This is the product exporter, not a second agent runtime. Rendered PDF/PNGs and
their visual QA remain an explicit subsequent gate; a DOCX file is not that gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from citequest.search_v3.schemas import PaperDetail, PaperLocator

from .school_contracts import (
    CSL_SHA,
    TEMPLATE_SHA,
    TOKEN,
    SchoolDocument,
    han,
    review_issues,
    section_wire_bytes,
    section_wire_limit,
)

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
}
SLOTS = {"review": 21, "references": 27, "objectives": 32, "methods": 37, "innovation": 43, "schedule": 47, "prior": 52}
PERMITTED = {
    "word/document.xml",
    "word/styles.xml",
    "word/numbering.xml",
    "word/settings.xml",
    "word/_rels/document.xml.rels",
    "[Content_Types].xml",
    "docProps/core.xml",
    "docProps/app.xml",
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def reference_context(state: dict) -> dict:
    return {
        "brief_sha256": state["intake"]["brief_sha256"],
        "aliases": state["aliases"],
        "evidence_refs": [
            {key: item[key] for key in ("ref_id", "source_ref", "source_sha256", "unit_text_sha256")}
            for item in state["evidence"]
        ],
    }


def reference_preflight(
    state: dict, *, read_evidence: Callable[[str], dict], read_paper: Callable[[PaperLocator], PaperDetail]
) -> dict:
    """Inject as a server-owned callback before any school-writing model call."""
    from citequest.agents_v3.contracts import digest

    context = reference_context(state)
    return {
        "status": "admitted",
        "provenance": "current_authorized_metadata",
        "context_sha256": digest(context),
        "references": reference_metadata(context["aliases"], context["evidence_refs"], read_evidence, read_paper),
    }


def reference_metadata(
    aliases: dict[str, list[str]],
    expected_refs: list[dict],
    read_evidence: Callable[[str], dict],
    read_paper: Callable[[PaperLocator], PaperDetail],
) -> list[dict]:
    """Only current real source metadata supplies bibliography. Never model fields."""
    expected = {item["ref_id"]: item for item in expected_refs}
    references = []
    for alias, identifiers in aliases.items():
        if not re.fullmatch(r"S[1-9][0-9]*", alias) or not identifiers:
            raise ValueError("invalid reference alias")
        locator = None
        for identifier in identifiers:
            actual = read_evidence(identifier)
            if identifier not in expected or any(
                actual.get(key) != expected[identifier].get(key)
                for key in ("source_ref", "source_sha256", "unit_text_sha256")
            ):
                raise ValueError("reference source changed or is outside the document context")
            if actual["source_ref"].get("kind") == "material_file":
                raise ValueError("material_bibliographic_metadata_confirmation_required")
            current_locator = PaperLocator.model_validate(actual["source_ref"], strict=True)
            if locator is not None and locator != current_locator:
                raise ValueError("one bibliography alias cannot merge unrelated papers")
            locator = current_locator
        detail = read_paper(locator)
        if not isinstance(detail, PaperDetail) or detail.paper != locator:
            raise ValueError("current bibliography metadata contract mismatch")
        if not detail.title or not detail.authors or detail.year is None:
            raise ValueError("bibliography_metadata_incomplete: title, authors and year must come from the source")
        item = {
            "id": alias,
            "type": "article",
            "title": detail.title,
            "author": [{"literal": author} for author in detail.authors],
            "issued": {"date-parts": [[detail.year]]},
        }
        if detail.source_url:
            if not detail.source_url.startswith(("https://", "http://")):
                raise ValueError("unsafe bibliography source URL")
            item["URL"] = detail.source_url
        # A DOI, journal or venue absent from the actual PaperDetail is omitted.
        references.append(item)
    return references


def current_references(
    document: SchoolDocument, read_evidence: Callable[[str], dict], read_paper: Callable[[PaperLocator], PaperDetail]
) -> list[dict]:
    used = {alias for section in document.sections for alias in TOKEN.findall(section.body())}
    return reference_metadata(
        {alias: refs for alias, refs in document.source_aliases.items() if alias in used},
        document.context_manifest["evidence_refs"],
        read_evidence,
        read_paper,
    )


def validate_export_document(document: SchoolDocument) -> None:
    from citequest.agents_v3.contracts import digest

    if (
        document.content_hash() != document.cross_review.content_sha256
        or document.content_hash() != document.request_coverage.content_sha256
        or document.content_hash() != document.context_manifest.get("document_content_sha256")
    ):
        raise ValueError("all reviews and export context must bind the exact final content")
    ids = [item.id for item in document.section_specs]
    if (
        len(set(ids)) != 14
        or [item.section_id for item in document.sections] != ids
        or [item.section_id for item in document.section_reviews] != ids
    ):
        raise ValueError("every ordered section needs its current independent review")
    for section, spec, review in zip(document.sections, document.section_specs, document.section_reviews, strict=True):
        if (
            section.brief_sha256 != document.brief_sha256
            or review_issues(section, spec, review)
            or not spec.minimum_han <= han(section.body()) <= spec.maximum_han
            or len(section.blocks) > document.profile.maximum_blocks_per_section
            or section_wire_bytes(section) > section_wire_limit(spec)
        ):
            raise ValueError("section no longer passes its bound content and length review")
    if (
        any(item.verdict != "passed" for item in document.cross_review.sections)
        or document.cross_review.conflicts
        or {item.section_id for item in document.cross_review.sections} != set(ids)
        or any(item.state not in {"covered", "explicit_unknown"} for item in document.request_coverage.requests)
        or not document.request_coverage.school_sections_complete
        or document.request_coverage.performed_experiments_invented
    ):
        raise ValueError("cross-section/request review is incomplete")
    review_han = sum(
        han(section.body())
        for section, spec in zip(document.sections, document.section_specs, strict=True)
        if spec.group == "review"
    )
    total_han = sum(han(section.body()) for section in document.sections)
    if review_han < 5000 or not 8000 <= total_han <= 12000:
        raise ValueError("school literature and project capacity gates must both pass")
    if digest(document.profile.model_dump(mode="json")) != document.context_manifest.get("school_profile_sha256"):
        raise ValueError("school profile changed since content review")


def export_school_document(
    document: SchoolDocument,
    output: Path,
    *,
    template: Path,
    csl: Path,
    authorize_manifest: Callable[[dict], None],
    read_evidence: Callable[[str], dict],
    read_paper: Callable[[PaperLocator], PaperDetail],
) -> dict:
    authorize_manifest(document.context_manifest)
    if document.reference_admission.provenance != "current_authorized_metadata":
        raise ValueError("explicit_reference_fixture_is_not_a_product_export")
    validate_export_document(document)
    references = current_references(document, read_evidence, read_paper)
    groups: dict[str, str] = {key: "" for key in SLOTS if key != "references"}
    for spec, section in zip(document.section_specs, document.sections, strict=True):
        groups[spec.group] += "## " + spec.title + "\n\n" + section.body() + "\n\n"
    receipt = render_template(
        document.title,
        groups,
        references,
        output,
        template=template,
        csl=csl,
        mode="reviewed_school_draft",
        content_sha256=document.content_hash(),
    )
    # Do not release a completed export if authority changed while conversion ran.
    authorize_manifest(document.context_manifest)
    current_references(document, read_evidence, read_paper)
    return receipt


def render_template(
    title: str,
    groups: dict[str, str],
    references: list[dict],
    output: Path,
    *,
    template: Path,
    csl: Path,
    mode: str,
    content_sha256: str,
) -> dict:
    """Shared physical renderer; layout_fixture mode is never school-quality PASS."""
    import pypandoc
    from docx import Document
    from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Pt, RGBColor
    from docxtpl import DocxTemplate
    from lxml import etree as ET

    if mode not in {"reviewed_school_draft", "explicit_layout_fixture"}:
        raise ValueError("explicit export scope required")
    if output.exists():
        raise FileExistsError("preserve prior exports; use a fresh artifact version directory")
    if sha(template.read_bytes()) != TEMPLATE_SHA or sha(csl.read_bytes()) != CSL_SHA:
        raise ValueError("school source template or CSL pin changed")
    if set(groups) != set(SLOTS) - {"references"}:
        raise ValueError("all official content slots must be present")
    reference_ids = {item["id"] for item in references}
    if len(reference_ids) != len(references):
        raise ValueError("duplicate bibliographic aliases")
    for text in [title, *groups.values()]:
        if (
            set(TOKEN.findall(text)) - reference_ids
            or "[@" in TOKEN.sub("", text)
            or re.search(r"!\[|<\s*(script|iframe|img|object|embed)\b|\]\s*\(", text, re.I)
        ):
            raise ValueError("unknown citation or active/remote Markdown content is forbidden")
    output.mkdir(parents=True)
    with ZipFile(template) as archive:
        source_parts = {name: archive.read(name) for name in archive.namelist()}
    original = ET.fromstring(source_parts["word/document.xml"])
    root = ET.fromstring(source_parts["word/document.xml"])
    body = root.find("w:body", NS)

    def plain(node: Any, text: str) -> None:
        for child in list(node):
            if child.tag not in {qn("w:pPr"), qn("w:permStart"), qn("w:permEnd")}:
                node.remove(child)
        for numbering in node.xpath("./w:pPr/w:numPr", namespaces=NS):
            numbering.getparent().remove(numbering)
        run = ET.SubElement(node, qn("w:r"))
        props = ET.SubElement(run, qn("w:rPr"))
        ET.SubElement(props, qn("w:rFonts"), {qn("w:ascii"): "Times New Roman", qn("w:eastAsia"): "楷体"})
        ET.SubElement(props, qn("w:color"), {qn("w:val"): "000000"})
        ET.SubElement(run, qn("w:t")).text = text

    for key, index in SLOTS.items():
        node = body[index]
        if node.xpath(".//w:permStart|.//w:permEnd", namespaces=NS):
            raise ValueError("template slot crosses a protected boundary")
        plain(node, "{{p " + key + " }}")
    plain(body[26], "")
    official_heading_prefixes = (
        "请综述课题国内外研究进展",
        "参考文献 Reference",
        "课题研究目标",
        "拟采取的研究方法",
        "课题的创新点",
        "计划进度",
        "与本课题有关的工作积累",
    )
    heading_chains = []
    for index, paragraph in enumerate(body):
        if paragraph.tag != qn("w:p"):
            continue
        text = "".join(paragraph.xpath(".//w:t/text()", namespaces=NS)).strip()
        prefix = next((value for value in official_heading_prefixes if text.startswith(value)), None)
        if prefix is None:
            continue
        heading_chains.append(prefix)
        # Glue the actual official heading and each reserved empty spacer to
        # the first imported content paragraph. Do not remove source spacing.
        chain = [paragraph]
        for following in body[index + 1 :]:
            if (
                following.tag != qn("w:p")
                or "".join(following.xpath(".//w:t/text()", namespaces=NS)).strip()
                or following.xpath(".//w:drawing|.//w:br", namespaces=NS)
            ):
                break
            chain.append(following)
        for item in chain:
            props = item.find(qn("w:pPr"))
            if props is None:
                props = ET.Element(qn("w:pPr"))
                item.insert(0, props)
            keep = props.find(qn("w:keepNext"))
            if keep is None:
                keep = ET.SubElement(props, qn("w:keepNext"))
            keep.set(qn("w:val"), "true")
    if set(heading_chains) != set(official_heading_prefixes) or len(heading_chains) != 7:
        raise ValueError("all_seven_pinned_official_heading_chains_required")
    for index, paragraph in enumerate(body[: SLOTS["references"]]):
        if paragraph.tag == qn("w:p") and "参考文献 Reference" in "".join(
            paragraph.xpath(".//w:t/text()", namespaces=NS)
        ):
            for following in body[index : SLOTS["references"]]:
                if following.tag == qn("w:p"):
                    props = following.find(qn("w:pPr"))
                    if props is None:
                        props = ET.Element(qn("w:pPr"))
                        following.insert(0, props)
                    keep = props.find(qn("w:keepNext"))
                    if keep is None:
                        keep = ET.SubElement(props, qn("w:keepNext"))
                    keep.set(qn("w:val"), "true")
    label = (
        "明确离线排版夹具 非研究结论 非提交材料"
        if mode == "explicit_layout_fixture"
        else "研究计划草案 拟议未执行 未经学校审核"
    )
    plain(body[3], label)
    for props in body[3].xpath("./w:r/w:rPr", namespaces=NS):
        ET.SubElement(props, qn("w:sz"), {qn("w:val"): "20"})
    tables = root.xpath("/w:document/w:body/w:tbl", namespaces=NS)
    for table_index, row_index in ((0, 5), (1, 0)):
        cells = tables[table_index].xpath("./w:tr", namespaces=NS)[row_index].xpath("./w:tc", namespaces=NS)
        target = cells[1].find("w:p", NS)
        if target is None or target.xpath(".//w:sdt|.//w:permStart|.//w:permEnd|.//w:t[text()]", namespaces=NS):
            raise ValueError("verified empty thesis-title slot changed")
        plain(target, title)
    # Remove the entire dynamic signature-date field; never fill a person's date.
    inside = False
    for node in list(body[58]):
        field = node.find("w:fldChar", NS)
        if field is not None and field.get(qn("w:fldCharType")) == "begin":
            inside = True
        if inside:
            end = field is not None and field.get(qn("w:fldCharType")) == "end"
            body[58].remove(node)
            if end:
                inside = False
    if root.xpath("//w:instrText[contains(., ' DATE ')]", namespaces=NS):
        raise ValueError("unmapped dynamic date field")
    working_parts = {
        **source_parts,
        "word/document.xml": ET.tostring(root, encoding="UTF-8", xml_declaration=True, standalone=True),
    }
    adapted = output / "adapted-template.docx"
    with ZipFile(adapted, "w", ZIP_DEFLATED) as archive:
        for name, value in working_parts.items():
            archive.writestr(name, value)
    bibliography = output / "references.csl.json"
    bibliography.write_text(json.dumps(references, ensure_ascii=False, indent=2), encoding="utf-8")
    conversion = [
        "--citeproc",
        "--bibliography=" + str(bibliography.resolve()),
        "--csl=" + str(csl.resolve()),
        "--metadata=lang:en-GB",
    ]
    markdown_format = "markdown-raw_html-raw_tex-yaml_metadata_block"

    def math_count(value: object) -> int:
        if isinstance(value, dict):
            return int(value.get("t") == "Math") + sum(math_count(item) for item in value.values())
        return sum(math_count(item) for item in value) if isinstance(value, list) else 0

    expected_math = 0
    for markdown in groups.values():
        if len(re.findall(r"(?<!\\)\$", markdown)) % 2:
            raise ValueError("unbalanced_math_delimiters")
        ast = json.loads(pypandoc.convert_text(markdown, "json", format=markdown_format))
        expected_math += math_count(ast)
    documents = {}
    for key, markdown in {
        **groups,
        "references": "\n".join("[@" + item + "]" for item in sorted(reference_ids, key=lambda value: int(value[1:]))),
    }.items():
        path = output / (key + ".docx")
        extra = conversion + (["--metadata=suppress-bibliography:true"] if key != "references" else [])
        # A reference-only bibliography is generated by citeproc; its citation
        # marker paragraph is removed, not any bibliography text.
        pypandoc.convert_text(markdown, "docx", format=markdown_format, outputfile=str(path), extra_args=extra)
        doc = Document(path)
        if key == "references":
            for paragraph in list(doc.paragraphs):
                if paragraph.style.name != "Bibliography" and paragraph.text.strip():
                    paragraph._element.getparent().remove(paragraph._element)
        paragraphs = list(doc.paragraphs)
        for table in doc.tables:
            table_props = table._tbl.tblPr
            borders = table_props.find(qn("w:tblBorders"))
            if borders is None:
                borders = OxmlElement("w:tblBorders")
                table_props.append(borders)
            for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
                line = borders.find(qn("w:" + edge))
                if line is None:
                    line = OxmlElement("w:" + edge)
                    borders.append(line)
                for name, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
                    line.set(qn("w:" + name), value)
            table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
            for number, row in enumerate(table.rows):
                row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
                for cell in row.cells:
                    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    cell_props = cell._tc.get_or_add_tcPr()
                    margins = OxmlElement("w:tcMar")
                    for side in ("top", "left", "bottom", "right"):
                        margin = OxmlElement("w:" + side)
                        margin.set(qn("w:w"), "90")
                        margin.set(qn("w:type"), "dxa")
                        margins.append(margin)
                    cell_props.append(margins)
                    if number == 0:
                        shade = OxmlElement("w:shd")
                        shade.set(qn("w:fill"), "F2F2F2")
                        cell_props.append(shade)
                    paragraphs.extend(cell.paragraphs)
        for paragraph in paragraphs:
            props = paragraph._p.get_or_add_pPr()
            for name, value in (("snapToGrid", "0"), ("overflowPunct", "1")):
                element = props.find(qn("w:" + name))
                if element is None:
                    element = OxmlElement("w:" + name)
                    props.append(element)
                element.set(qn("w:val"), value)
            paragraph.paragraph_format.line_spacing = 1.25
            paragraph.paragraph_format.space_before = Pt(7.8)
            paragraph.paragraph_format.space_after = Pt(0)
            paragraph.paragraph_format.first_line_indent = Pt(24)
            paragraph.paragraph_format.keep_together = False
            paragraph.paragraph_format.widow_control = True
            paragraph.paragraph_format.keep_with_next = paragraph.style.name.startswith("Heading")
            if key == "references":
                paragraph.paragraph_format.keep_together = True
                paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
                paragraph.paragraph_format.left_indent = Pt(18)
                paragraph.paragraph_format.first_line_indent = Pt(-18)
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "楷体")
                run.font.size = Pt(12)
                run.font.color.rgb = RGBColor(0, 0, 0)
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.first_line_indent = Pt(0)
                        paragraph.paragraph_format.space_before = Pt(3)
                        paragraph.paragraph_format.space_after = Pt(3)
                        for run in paragraph.runs:
                            run.font.size = Pt(10)
        doc.save(path)
        documents[key] = path
    tpl = DocxTemplate(adapted)
    tpl.render({key: tpl.new_subdoc(str(path)) for key, path in documents.items()}, autoescape=True)
    final = output / "proposal.docx"
    tpl.save(final)
    with ZipFile(final) as archive:
        final_parts = {name: archive.read(name) for name in archive.namelist()}
    styles = ET.fromstring(final_parts["word/styles.xml"])
    source_styles = ET.fromstring(source_parts["word/styles.xml"])
    defaults = {
        style.get(qn("w:type")): style.get(qn("w:styleId"))
        for style in source_styles.findall("w:style", NS)
        if style.get(qn("w:default")) == "1"
    }
    aliases = {}
    for path in documents.values():
        with ZipFile(path) as archive:
            imported = ET.fromstring(archive.read("word/styles.xml"))
        aliases.update(
            {
                style.get(qn("w:styleId")): defaults[style.get(qn("w:type"))]
                for style in imported.findall("w:style", NS)
                if style.get(qn("w:default")) == "1" and style.get(qn("w:type")) in defaults
            }
        )
    style_ids = set(styles.xpath("//w:style/@w:styleId", namespaces=NS))
    for reference in styles.xpath("//w:basedOn|//w:next|//w:link", namespaces=NS):
        old = reference.get(qn("w:val"))
        if old not in style_ids:
            replacement = aliases.get(old) or defaults.get(reference.getparent().get(qn("w:type")))
            if replacement and replacement != reference.getparent().get(qn("w:styleId")):
                reference.set(qn("w:val"), replacement)
    for style in styles.findall("w:style", NS):
        if style.get(qn("w:default")) == "1" and style.get(qn("w:styleId")) != defaults.get(style.get(qn("w:type"))):
            del style.attrib[qn("w:default")]
    final_parts["word/styles.xml"] = ET.tostring(styles, encoding="UTF-8", xml_declaration=True, standalone=True)
    settings = ET.fromstring(source_parts["word/settings.xml"])
    spacing = settings.find("w:characterSpacingControl", NS)
    if spacing is None:
        spacing = ET.SubElement(settings, qn("w:characterSpacingControl"))
    spacing.set(qn("w:val"), "doNotCompress")
    update = settings.find("w:updateFields", NS)
    if update is None:
        update = ET.SubElement(settings, qn("w:updateFields"))
    update.set(qn("w:val"), "true")
    final_parts["word/settings.xml"] = ET.tostring(settings, encoding="UTF-8", xml_declaration=True, standalone=True)
    for name, content in source_parts.items():
        if name not in PERMITTED:
            final_parts[name] = content
    with ZipFile(final, "w", ZIP_DEFLATED) as archive:
        for name, content in final_parts.items():
            archive.writestr(name, content)
    document_root = ET.fromstring(final_parts["word/document.xml"])
    checks = {
        "source_template_unchanged": sha(template.read_bytes()) == TEMPLATE_SHA,
        "all_seven_official_heading_spacer_chains_keep_with_content": len(heading_chains) == 7,
        "source_two_sections_preserved": len(document_root.xpath("//w:sectPr", namespaces=NS))
        == len(original.xpath("//w:sectPr", namespaces=NS))
        == 2,
        "nine_source_content_controls_preserved": len(document_root.xpath("//w:sdt", namespaces=NS))
        == len(original.xpath("//w:sdt", namespaces=NS))
        == 9,
        "source_permission_boundaries_preserved": all(
            Counter(document_root.xpath("//w:" + tag + "/@w:id", namespaces=NS))
            == Counter(original.xpath("//w:" + tag + "/@w:id", namespaces=NS))
            for tag in ("permStart", "permEnd")
        ),
        "opaque_parts_preserved": all(
            final_parts.get(name) == value for name, value in source_parts.items() if name not in PERMITTED
        ),
        "all_style_references_resolved": all(
            reference.get(qn("w:val")) in style_ids
            for reference in styles.xpath("//w:basedOn|//w:next|//w:link", namespaces=NS)
        ),
        "dynamic_signature_date_removed": not document_root.xpath(
            "//w:instrText[contains(., ' DATE ')]", namespaces=NS
        ),
        "no_jinja_placeholders": "{{p " not in "".join(document_root.xpath("//w:t/text()", namespaces=NS)),
    }
    checks["all_parsed_equations_are_editable_OMML"] = (
        len(document_root.xpath("//m:oMath", namespaces=NS)) == expected_math
    )
    if not all(checks.values()):
        raise ValueError("export source-structure gate failed:" + json.dumps(checks))
    all_markdown = "# " + title + "\n\n" + label + "\n\n" + "\n\n".join(groups.values())
    (output / "proposal.md").write_text(all_markdown, encoding="utf-8")
    html = pypandoc.convert_text(
        all_markdown,
        "html5",
        format=markdown_format,
        extra_args=[*conversion, "--standalone", "--mathml", "--metadata=title:" + title],
    )
    css = "<style>body{max-width:880px;margin:2rem auto;padding:0 1rem;line-height:1.85;color:#161616;font-family:serif;overflow-wrap:anywhere}h1,h2,h3{color:#000}table{border-collapse:collapse;max-width:100%}th,td{border:1px solid #d9d9d9;padding:.5rem}math{max-width:100%;overflow:auto}pre{overflow:auto}@media print{body{max-width:none}h1,h2,h3{break-after:avoid}tr{break-inside:avoid}thead{display:table-header-group}}</style>"
    csp = '<meta http-equiv="Content-Security-Policy" content="default-src &apos;none&apos;; style-src &apos;unsafe-inline&apos;; base-uri &apos;none&apos;; form-action &apos;none&apos;">'
    (output / "proposal.html").write_text(html.replace("</head>", csp + css + "</head>"), encoding="utf-8")
    receipt = {
        "schema_version": "citequest-school-export/1",
        "status": "structure_passed_render_and_visual_pending",
        "scope": mode,
        "content_sha256": content_sha256,
        "template_sha256": TEMPLATE_SHA,
        "csl_sha256": CSL_SHA,
        "docx_sha256": sha(final.read_bytes()),
        "html_sha256": sha((output / "proposal.html").read_bytes()),
        "bibliography_sha256": sha(bibliography.read_bytes()),
        "checks": checks,
        "native_omml_count": len(document_root.xpath("//m:oMath", namespaces=NS)),
        "references_from_model": False,
        "school_submission_approval": False,
        "pdf_render": "not_run",
        "visual_review": "not_run",
        "font_mapping": "Source retained; generated body requests 楷体 12pt and Times New Roman; renderer substitutions must be audited separately.",
        "pandoc_version": str(pypandoc.get_pandoc_version()),
    }
    (output / "export-receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    return receipt


def render_pdf(docx: Path, output: Path, *, renderer: Path, renderer_sha256: str, timeout_seconds: int = 90) -> dict:
    """Actual headless office conversion with an isolated profile and deadline.

    The host supplies a server-owned, pinned binary, never a user-selected path.
    A successful conversion is not visual QA or a scientific approval.
    """
    import os
    import subprocess

    from pypdf import PdfReader

    if not 1 <= timeout_seconds <= 180 or sha(renderer.read_bytes()) != renderer_sha256.lower():
        raise ValueError("invalid renderer pin or bounded conversion deadline")
    if output.exists():
        raise FileExistsError("preserve old render attempts")
    output.mkdir(parents=True)
    profile = output / "office-profile"
    profile.mkdir()
    arguments = [
        str(renderer.resolve()),
        "-env:UserInstallation=" + profile.resolve().as_uri(),
        "--headless",
        "--convert-to",
        "pdf:writer_pdf_Export",
        "--outdir",
        str(output.resolve()),
        str(docx.resolve()),
    ]
    try:
        result = subprocess.run(
            arguments,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError("school_pdf_conversion_deadline_exceeded") from None
    (output / "office.stdout.log").write_bytes(result.stdout)
    (output / "office.stderr.log").write_bytes(result.stderr)
    pdf = output / (docx.stem + ".pdf")
    if result.returncode != 0 or not pdf.exists():
        raise ValueError("office_did_not_produce_actual_pdf")
    pages = PdfReader(pdf).pages
    if not pages or any(
        abs(float(page.mediabox.width) - 595.3) > 2 or abs(float(page.mediabox.height) - 841.9) > 2 for page in pages
    ):
        raise ValueError("school_pdf_page_geometry_invalid")
    receipt = {
        "schema_version": "citequest-school-pdf/1",
        "status": "rendered_visual_and_coverage_pending",
        "docx_sha256": sha(docx.read_bytes()),
        "pdf_sha256": sha(pdf.read_bytes()),
        "renderer_sha256": renderer_sha256.lower(),
        "page_count": len(pages),
        "visual_review": "not_run",
        "full_text_coverage": "not_run",
        "conversion_exit_code": result.returncode,
    }
    (output / "pdf-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt
