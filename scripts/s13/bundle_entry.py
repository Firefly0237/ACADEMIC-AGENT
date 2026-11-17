"""Isolated offline render job. No database, model key, URL or arbitrary DOCX."""

# ruff: noqa: T201 -- isolated command emits a small machine-readable receipt
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from citequest.agents_v3.contracts import digest
from citequest.exports_v3.bibliography import assert_safe_export_html, literal_csl
from citequest.exports_v3.pdf_gate import verify_pdf
from citequest.exports_v3.renderer_identity import actual_identity
from citequest.research_v3.school_contracts import SchoolDocument
from citequest.research_v3.school_export import SLOTS, render_pdf, render_template, validate_export_document
from citequest.research_v3.school_linux import PROFILE_ID, prepare_linux_render_input, write_derivative_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--describe-toolchain", action="store_true")
    parser.add_argument("--inputs", type=Path, default=Path("/inputs/job.json"))
    parser.add_argument("--outputs", type=Path, default=Path("/outputs"))
    args = parser.parse_args()
    identity = actual_identity()
    toolchain_sha = digest(identity)
    if args.describe_toolchain:
        print(json.dumps({"identity": identity, "sha256": toolchain_sha}, ensure_ascii=False))
        return 0
    if args.inputs.is_symlink() or args.inputs.stat().st_size > 2 * 1024**2:
        raise ValueError("bounded_host_generated_job_required")
    job = json.loads(args.inputs.read_text(encoding="utf8"))
    if set(job) != {"document", "references", "formats", "document_sha256", "toolchain_sha256"}:
        raise ValueError("host_generated_job_contract_failed")
    if job["toolchain_sha256"] != toolchain_sha or set(job["formats"]) - {"docx", "html", "pdf"} or not job["formats"]:
        raise ValueError("pinned_toolchain_or_formats_mismatch")
    document = SchoolDocument.model_validate_json(json.dumps(job["document"]), strict=True)
    validate_export_document(document)
    if (
        document.reference_admission.provenance != "current_authorized_metadata"
        or digest(document.model_dump(mode="json")) != job["document_sha256"]
    ):
        raise ValueError("trusted_reviewed_document_hash_required")
    groups = {key: "" for key in SLOTS if key != "references"}
    for spec, section in zip(document.section_specs, document.sections, strict=True):
        groups[spec.group] += "## " + spec.title + "\n\n" + section.body() + "\n\n"
    args.outputs.mkdir(parents=True, exist_ok=True)
    artifacts = args.outputs / "artifacts"
    source_receipt = render_template(
        document.title,
        groups,
        literal_csl(job["references"]),
        artifacts,
        template=Path("/assets/sjtu.docx"),
        csl=Path("/assets/harvard.csl"),
        mode="reviewed_school_draft",
        content_sha256=document.content_hash(),
        profile=document.profile,
    )
    files = {"docx": artifacts / "proposal.docx", "html": artifacts / "proposal.html"}
    html_audit = assert_safe_export_html(files["html"])
    pdf_audit = None
    if "pdf" in job["formats"]:
        derivative = args.outputs / "private-render-input/proposal.docx"
        receipt = prepare_linux_render_input(files["docx"], derivative, expected_sha256=source_receipt["docx_sha256"])
        write_derivative_receipt(receipt, args.outputs / "linux-adapter-receipt.json")
        render_pdf(
            derivative,
            args.outputs / "pdf",
            renderer=Path("/usr/bin/soffice"),
            renderer_sha256=identity["file_sha256"]["/usr/bin/soffice"],
        )
        files["pdf"] = args.outputs / "pdf/proposal.pdf"
        pdf_audit = verify_pdf(files["pdf"], [artifacts / (key + ".docx") for key in SLOTS])
    receipt = {
        "status": "automatic_export_gates_passed",
        "document_sha256": job["document_sha256"],
        "toolchain_sha256": toolchain_sha,
        "font_profile": PROFILE_ID,
        "actual_model_requests": 0,
        "raw_source_bibliography_sha256": digest(job["references"]),
        "CSL_rendering": "original metadata, pinned citeproc escaping plus output DOM safety gate; no double-escaping",
        "html_safety": html_audit,
        "template_structure": source_receipt["checks"],
        "pdf_coverage": pdf_audit,
        "per_document_human_visual_review": "not_run; trusted profile requires separately certified all-page capacity fixture",
        "school_submission_approval": False,
        "artifacts": {
            kind: {
                "sha256": hashlib.sha256(files[kind].read_bytes()).hexdigest(),
                "size_bytes": files[kind].stat().st_size,
            }
            for kind in job["formats"]
        },
    }
    (args.outputs / "bundle-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print(json.dumps({"status": receipt["status"], "formats": job["formats"], "actual_model_requests": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
