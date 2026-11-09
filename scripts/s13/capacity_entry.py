"""CI-only, deterministic physical fixture. No model or database access."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from citequest.exports_v3.pdf_gate import verify_pdf
from citequest.exports_v3.release_build import file_sha256, sha256, write_json
from citequest.exports_v3.renderer_identity import actual_identity
from citequest.research_v3.school_contracts import han
from citequest.research_v3.school_export import SLOTS, render_pdf, render_template
from citequest.research_v3.school_linux import prepare_linux_render_input
from citequest.research_v3.school_profiles import default_profile_id, get_profile


def fixture_groups() -> dict[str, str]:
    groups = {}
    for group in (key for key in SLOTS if key != "references"):
        paragraphs = []
        target = 5100 if group == "review" else 1320
        number = 1
        while han("".join(paragraphs)) < target:
            paragraphs.append(f"夹具{group}{number:03d}：" +
                "这是公开的离线排版容量夹具，检查中文段落、分页和完整内容保留，不表达真实研究结论。" * 3)
            number += 1
        groups[group] = "\n\n".join(paragraphs) + f"\n\n组尾{group}完整保留。[@S1]"
    groups["methods"] += "\n\n$$E = mc^2$$\n\n$$y = \\frac{a}{b}$$\n\n$$x^2 + y^2 = z^2$$"
    groups["schedule"] += "\n\n| 序号 | 排版夹具内容 |\n| --- | --- |\n" + "\n".join(
        f"| {number:02d} | 表行{number:02d}：" + "中文表格跨页内容完整保留，重复表头便于阅读。" * 3 + " |"
        for number in range(1, 37)) + "\n\n表格终点三十六行。"
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, default=Path("/outputs"))
    parser.add_argument("--image-id", required=True)
    args = parser.parse_args()
    output = args.outputs.resolve(strict=True)
    identity = actual_identity()
    groups = fixture_groups()
    profile = get_profile(default_profile_id())
    references = [{"id": "S1", "type": "article", "title": "Explicit offline layout fixture",
        "author": [{"literal": "CiteQuest fixture"}], "issued": {"date-parts": [[2025]]}}]
    rendered = render_template("离线容量排版夹具", groups, references, output / "artifacts",
        template=Path("/app/assets/export/sjtu-proposal.docx"), csl=Path("/app/assets/export/harvard.csl"),
        mode="explicit_layout_fixture", content_sha256=sha256(groups), profile=profile)
    if han("".join(groups.values())) < profile.demonstration_maximum_han or rendered["native_omml_count"] != 3:
        raise ValueError("maximum_length_and_three_editable_equations_fixture_required")
    derivative = output / "private/proposal.docx"
    adapter = prepare_linux_render_input(output / "artifacts/proposal.docx", derivative,
        expected_sha256=rendered["docx_sha256"])
    write_json(output / "linux-adapter.json", adapter)
    render_pdf(derivative, output / "pdf", renderer=Path("/usr/bin/soffice"),
        renderer_sha256=identity["file_sha256"]["/usr/bin/soffice"])
    pdf = output / "pdf/proposal.pdf"
    coverage = verify_pdf(pdf, [output / "artifacts" / (key + ".docx") for key in SLOTS])
    coverage.update(docx_sha256=rendered["docx_sha256"], pdf_sha256=file_sha256(pdf))
    png_dir = output / "pages"
    png_dir.mkdir()
    subprocess.run(["pdftoppm", "-png", "-r", "96", str(pdf), str(png_dir / "page")], check=True, timeout=120)
    pngs = sorted(png_dir.glob("page-*.png"))
    if len(pngs) != coverage["pages"]:
        raise ValueError("all_actual_pdf_pages_must_have_png_artifacts")
    write_json(output / "capacity.json", {
        "schema_version": "citequest-actual-export-capacity/1", "status": "automatic_gates_passed",
        "image_id": args.image_id, "toolchain_sha256": sha256(identity), "font_profile": identity["font_profile"],
        "school_profile_ids": [profile.id], "fixture_sha256": sha256(groups),
        "fixture_han": {key: han(value) for key, value in groups.items()},
        "fixture_scope": "synthetic physical capacity; no research quality or school approval",
        "coverage": coverage, "visual_review": {"status": "not_provided"},
        "page_png_sha256": {path.relative_to(output).as_posix(): file_sha256(path) for path in pngs},
        "actual_model_requests": 0,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
