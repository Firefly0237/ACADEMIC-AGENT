"""Untrusted-file parser implementation. Run only through the gated Job worker."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import stat
import sys
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .provenance import native_pdf_view
from .text import MaterialError, MaterialLimits, load_limits, prepare_text


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _report_validation(request: dict) -> None:
    output = Path(request["output"])
    temporary = output / "phase.pending"
    _write(temporary, {"start_token": request["start_token"], "phase": "parsing", "profile_sha256": request["profile_sha256"], "source_sha256": request["source_sha256"]})
    os.replace(temporary, output / "phase.json")


def _probe(request: dict) -> dict:
    import psutil

    operation = request["operation"]
    if operation == "probe_timeout":
        time.sleep(5)
    elif operation == "probe_memory":
        try:
            _ = bytearray(request["probe_memory_limit"] + 64 * 1024 * 1024)
        except MemoryError:
            return {"state": "rejected", "code": "MEMORY_ALLOCATION_LIMIT_ENFORCED"}
        raise MaterialError("MEMORY_LIMIT_NOT_ENFORCED")
    elif operation == "probe_cpu":
        start = time.monotonic()
        while time.monotonic() - start < 1:
            hashlib.sha256(b"parser-resource-control" * 1000).hexdigest()
    elif operation == "probe_child":
        import subprocess

        try:
            child = subprocess.Popen([sys.executable, "-c", "pass"], creationflags=subprocess.CREATE_NO_WINDOW)
            code = child.wait(timeout=5)
        except OSError as exc:
            return {"state": "rejected", "code": "CHILD_PROCESS_LIMIT_ENFORCED", "os_error": str(exc)}
        if code:
            return {"state": "rejected", "code": "CHILD_PROCESS_LIMIT_ENFORCED", "child_exit": code}
        raise MaterialError("CHILD_PROCESS_LIMIT_NOT_ENFORCED")
    return {"state": "ready", "code": "PROBE_COMPLETED", "affinity": psutil.Process().cpu_affinity(),
            "credential_environment_keys_present": sorted(k for k in os.environ if "API_KEY" in k or ("TOKEN" in k and k not in {"HF_HUB_DISABLE_IMPLICIT_TOKEN", "TOKENIZERS_PARALLELISM"}))}


def _pdf_preflight(path: Path, limits: MaterialLimits) -> tuple[list[dict], list[str]]:
    from pypdf import PdfReader
    from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

    if path.stat().st_size > limits.pdf_bytes:
        raise MaterialError("PDF_BYTES_EXCEEDED")
    with path.open("rb") as source:
        if source.read(5) != b"%PDF-":
            raise MaterialError("INVALID_PDF_SIGNATURE")
    try:
        reader = PdfReader(path, strict=True)
        if reader.is_encrypted:
            raise MaterialError("ENCRYPTED_PDF")
        if len(reader.pages) > limits.pdf_pages:
            raise MaterialError("PDF_PAGES_EXCEEDED")
        page_refs = {(p.indirect_reference.idnum, p.indirect_reference.generation) for p in reader.pages if p.indirect_reference}
        from .pdf_navigation import safe_local_open_destination

        stack, seen, nodes = [reader.trailer["/Root"]], set(), 0
        while stack:
            obj = stack.pop()
            if isinstance(obj, IndirectObject):
                key = (obj.idnum, obj.generation)
                if key in seen:
                    continue
                seen.add(key)
                obj = obj.get_object()
            nodes += 1
            if nodes > 100000:
                raise MaterialError("PDF_OBJECT_COMPLEXITY_EXCEEDED")
            if isinstance(obj, DictionaryObject):
                if any(key in obj for key in ("/JS", "/JavaScript", "/AA", "/Launch", "/EmbeddedFiles", "/RichMedia", "/XFA")):
                    raise MaterialError("UNSUPPORTED_PDF_ACTIVE_CONTENT")
                if "/OpenAction" in obj and not safe_local_open_destination(obj["/OpenAction"], reader, page_refs):
                    raise MaterialError("UNSUPPORTED_PDF_ACTIVE_CONTENT")
                if str(obj.get("/S", "")) in {"/JavaScript", "/Launch", "/SubmitForm", "/ImportData", "/Rendition", "/RichMediaExecute"}:
                    raise MaterialError("UNSUPPORTED_PDF_ACTIVE_CONTENT")
                stack.extend(obj.values())
            elif isinstance(obj, ArrayObject):
                stack.extend(obj)
        pages = []
        for number, page in enumerate(reader.pages, 1):
            text = page.extract_text() or ""
            image_count = len(page.images)
            status = "text" if text.strip() else "ocr_required" if image_count else "empty_or_unextractable"
            pages.append({"page_1based": number, "width": float(page.mediabox.width), "height": float(page.mediabox.height),
                          "text": text, "image_count": image_count, "status": status})
        return pages, ["PDF scan detector is a text-layer/image heuristic, not OCR or proof of all formula semantics.",
                       "S04 admits inert current-document /Fit, /FitB or exact /XYZ OpenAction destinations with finite/null coordinates and nonnegative/null zoom, no action chain; parser never executes actions or fetches URI links."]
    except MaterialError:
        raise
    except Exception as exc:
        raise MaterialError("CORRUPT_OR_UNSUPPORTED_PDF") from exc


def _docx_preflight(path: Path, limits: MaterialLimits) -> tuple[list[dict], list[str], list[dict]]:
    from defusedxml import ElementTree as safe_xml

    if path.stat().st_size > limits.docx_bytes:
        raise MaterialError("DOCX_BYTES_EXCEEDED")
    if not zipfile.is_zipfile(path):
        raise MaterialError("INVALID_DOCX_ZIP")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > limits.docx_zip_entries:
                raise MaterialError("DOCX_ENTRIES_EXCEEDED")
            if sum(e.file_size for e in entries) > limits.docx_uncompressed_bytes:
                raise MaterialError("DOCX_EXPANDED_BYTES_EXCEEDED")
            names, document, additional_parts = set(), None, []
            text_tags = {"{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t", "{http://schemas.openxmlformats.org/officeDocument/2006/math}t", "{http://schemas.openxmlformats.org/drawingml/2006/main}t"}
            for entry in entries:
                name = entry.filename.replace("\\", "/")
                if (name.startswith("/") or ".." in PurePosixPath(name).parts or ":" in name
                        or "\x00" in entry.orig_filename or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise MaterialError("DOCX_PATH_ESCAPE")
                if name in names:
                    raise MaterialError("DOCX_DUPLICATE_ENTRY")
                names.add(name)
                lower = name.lower()
                if entry.flag_bits & 1:
                    raise MaterialError("DOCX_ENCRYPTED_ENTRY")
                if "vbaproject" in lower or lower.endswith(".bin") or lower.startswith(("word/embeddings/", "word/activex/")):
                    raise MaterialError("UNSUPPORTED_DOCX_ACTIVE_OR_EMBEDDED_CONTENT")
                if lower.endswith((".xml", ".rels")):
                    with archive.open(entry) as source:
                        tree = safe_xml.parse(source, forbid_dtd=True, forbid_entities=True, forbid_external=True)
                    if lower.endswith(".rels") and any(n.attrib.get("TargetMode", "").lower() == "external" for n in tree.getroot()):
                        raise MaterialError("UNSUPPORTED_DOCX_EXTERNAL_RELATIONSHIP")
                    if name == "word/document.xml":
                        document = tree.getroot()
                    elif name.startswith("word/") and any(n.tag in text_tags and (n.text or "").strip() for n in tree.getroot().iter()):
                        additional_parts.append({"item_ref": f"part:{name}", "reason": "DOCX_ADDITIONAL_TEXT_PART_NOT_YET_MAPPED"})
            if not {"[Content_Types].xml", "word/document.xml"} <= names or document is None:
                raise MaterialError("MISSING_DOCX_PARTS")
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        math_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/math}"
        if any(n.tag == "{http://schemas.openxmlformats.org/drawingml/2006/main}t" and (n.text or "").strip() for n in document.iter()):
            additional_parts.append({"item_ref": "part:word/document.xml:DrawingML", "reason": "DOCX_MAIN_DRAWING_TEXT_NOT_YET_MAPPED"})
        if any(n.tag == ns + "altChunk" for n in document.iter()):
            additional_parts.append({"item_ref": "part:word/document.xml:altChunk", "reason": "DOCX_ALTERNATIVE_CONTENT_NOT_YET_MAPPED"})
        paragraphs = []
        for number, paragraph in enumerate(document.iter(ns + "p"), 1):
            text = "".join(n.text or "" if n.tag == ns + "t" else "\t" if n.tag == ns + "tab" else "\n" if n.tag in {ns + "br", ns + "cr"} else "" for n in paragraph.iter())
            paragraphs.append({"block_id": f"p:{number:08d}", "part": "word/document.xml", "text": text,
                               "has_math": any(n.tag.startswith(math_ns) for n in paragraph.iter()),
                               "source_xml_sha256": hashlib.sha256(__import__("xml.etree.ElementTree", fromlist=["tostring"]).tostring(paragraph)).hexdigest()})
        return paragraphs, ["DOCX scope is word/document.xml w:t/tab/br/cr paragraph text; no Word page numbers are invented.",
                            "Text in additional parts is explicitly missing/partial; images and formula meaning are not verified."], additional_parts
    except MaterialError:
        raise
    except Exception as exc:
        from defusedxml.common import DefusedXmlException

        if isinstance(exc, DefusedXmlException):
            raise MaterialError("DOCX_XML_ENTITY_OR_DTD") from exc
        raise MaterialError("CORRUPT_OR_UNSUPPORTED_DOCX") from exc


def _docling(path: Path, output: Path, root: Path, profile: dict) -> Any:
    manifest_path = root / profile["model_manifest"]
    if _sha(manifest_path) != profile["model_manifest_sha256"]:
        raise MaterialError("MODEL_MANIFEST_CHANGED")
    models = json.loads(manifest_path.read_text(encoding="utf-8"))
    for model in models["models"]:
        for item in model["files"]:
            if _sha(root / item["path"]) != item["sha256"]:
                raise MaterialError("CACHED_MODEL_BYTES_CHANGED")
    from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
    from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.object_detection_engine_options import TransformersObjectDetectionEngineOptions
    from docling.datamodel.pipeline_options import (
        LayoutObjectDetectionOptions,
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    layout = LayoutObjectDetectionOptions.from_preset("layout_heron_default")
    layout.model_spec.revision = models["models"][0]["resolved_revision"]
    layout.engine_options = TransformersObjectDetectionEngineOptions(torch_dtype="float32", compile_model=False)
    opts = PdfPipelineOptions(artifacts_path=root / profile["model_cache_root"],
                             accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU, num_threads=2),
                             do_ocr=False, do_table_structure=True,
                             table_structure_options=TableStructureOptions(mode=TableFormerMode.FAST), layout_options=layout,
                             do_formula_enrichment=False, do_code_enrichment=False, enable_remote_services=False,
                             allow_external_plugins=False, document_timeout=120, layout_batch_size=1, table_batch_size=1, ocr_batch_size=1)
    converter = DocumentConverter(allowed_formats=[InputFormat.PDF, InputFormat.DOCX],
                                  format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts, backend=PyPdfiumDocumentBackend)})
    _write(output / "docling-options.json", opts.model_dump(mode="json"))
    converted = converter.convert(path, max_num_pages=200, max_file_size=50 * 1024 * 1024, raises_on_error=True)
    converted.document.save_as_json(output / "docling-document.json")
    return converted.document


def _parse(request: dict, profile: dict) -> dict:
    root, output = Path(request["root"]), Path(request["output"])
    source = Path(request["source_path"]).resolve(strict=True)
    if source.is_symlink() or not source.is_relative_to(root):
        raise MaterialError("INTERNAL_SOURCE_PATH_OUTSIDE_WORKSPACE")
    sha = _sha(source)
    if sha != request["source_sha256"]:
        raise MaterialError("ORIGINAL_SOURCE_HASH_MISMATCH")
    limits = load_limits(root / "contracts/v2/policy.json")
    media = request["media_type"]
    if media in {"text/plain", "text/markdown"}:
        if source.stat().st_size > limits.text_bytes:
            raise MaterialError("TEXT_BYTES_EXCEEDED")
        import filetype

        with source.open("rb") as stream:
            if filetype.guess_mime(stream.read(8192)) is not None:
                raise MaterialError("BINARY_SIGNATURE_IN_TEXT_MATERIAL")
        material = prepare_text(source.read_bytes(), media, limits)
        _report_validation(request)
        blocks, start = [], 0
        for line in material.normalized_text.splitlines(keepends=True):
            end = start + len(line)
            if line.strip():
                locator = material.locator(start, end)
                blocks.append({"block_id": f"line:{len(blocks) + 1}", "text": line, "locator": locator,
                               "source_quote": material.read_locator(locator), "anchor_valid": True})
            start = end
        _write(output / "normalized-text.json", {"text": material.normalized_text, "sha256": material.normalized_sha256,
                                                  "original_boundary_map": material.normalized_boundaries_in_original})
        return {"state": "ready", "code": "TEXT_PARSED", "source_sha256": sha, "blocks": blocks,
                "coverage": "all_nonempty_text_lines", "can_qa": bool(blocks), "model_api_calls": 0}
    if media == "application/pdf":
        pages, warnings = _pdf_preflight(source, limits)
        _report_validation(request)
        _write(output / "preflight-pages.json", pages)
        missing = [p["page_1based"] for p in pages if p["status"] != "text"]
        if missing:
            return {"state": "partial", "code": "OCR_OR_PAGE_REVIEW_REQUIRED", "source_sha256": sha,
                    "pages_requiring_ocr_or_review": missing, "blocks": [], "can_qa": False, "ocr_enabled": False, "warnings": warnings}
        cached = request.get("cached_docling_document")
        cache_proof = None
        if cached:
            cached_path = Path(cached).resolve(strict=True)
            if not cached_path.is_relative_to(root / "reports/stages/v2/S04") or _sha(cached_path) != request.get("cached_docling_sha256"):
                raise MaterialError("INVALID_DEVELOPMENT_PARSE_CACHE")
            old_receipt = cached_path.parent / "receipt.json"
            prior = json.loads(old_receipt.read_text(encoding="utf-8"))
            if prior["result"].get("source_sha256") != sha or prior["result"].get("state") not in {"partial", "ready"} or prior.get("worker_stopped") is not True:
                raise MaterialError("DEVELOPMENT_PARSE_CACHE_SOURCE_MISMATCH")
            native_bytes = cached_path.read_bytes()
            (output / "docling-document.json").write_bytes(native_bytes)
            cache_proof = {"scope": "Explicit developer-only reuse of real Docling artifact; no prior anchor verdict reused", "native_path": str(cached_path),
                           "native_sha256": _sha(cached_path), "receipt_sha256": _sha(old_receipt), "inference_reexecuted": False}
        else:
            _docling(source, output, root, profile)
        native_path = output / "docling-document.json"
        native = json.loads(native_path.read_text(encoding="utf-8"))
        view = native_pdf_view(native, source_sha256=sha, native_sha256=_sha(native_path), config_sha256=request["profile_sha256"], page_count=len(pages))
        _write(output / "source-representation.json", view)
        _write(output / "source-pages.json", view["pages"])
        _write(output / "unmapped-blocks.json", view["missing_ranges"])
        missing_ranges = view["missing_ranges"]
        return {"state": "partial" if missing_ranges else "ready", "code": "PDF_PARTIAL_ANCHORS" if missing_ranges else "PDF_PARSED",
                "source_sha256": sha, "representation_id": view["representation_id"], "pages": len(pages), "blocks": view["blocks"], "unmapped_count": len(missing_ranges),
                "missing_ranges": missing_ranges, "coverage": view["coverage"], "usable_scope": [b["block_id"] for b in view["blocks"]],
                "tables": len(native.get("tables", [])), "can_qa": bool(view["blocks"]) and not missing_ranges, "ocr_enabled": False,
                "formula_semantics_verified": False, "formula_warnings": ["Formula recognition/semantics are unverified; formula enrichment is disabled"] + view["formula_warnings"],
                "table_warnings": view["table_warnings"], "qa_scope": view["qa_scope"], "parse_cache_proof": cache_proof, "warnings": warnings}
    if media == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        paragraphs, warnings, missing_parts = _docx_preflight(source, limits)
        _report_validation(request)
        _write(output / "source-paragraphs.json", paragraphs)
        document = _docling(source, output, root, profile)
        blocks = [{"block_id": p["block_id"], "text": p["text"], "source_quote": p["text"],
                   "locator": {"kind": "docx", "file_sha256": sha, "part": p["part"], "block_id": p["block_id"], "char_start": 0, "char_end": len(p["text"])},
                   "source_xml_sha256": p["source_xml_sha256"], "anchor_valid": True,
                   "locator_text_basis": "OOXML paragraph w:t/tab/br/cr text/1"} for p in paragraphs if p["text"].strip()]
        math = sum(p["has_math"] for p in paragraphs)
        missing_ranges = missing_parts + [{"item_ref": f"{p['block_id']}:math", "reason": "DOCX_NATIVE_MATH_REQUIRES_REVIEW"} for p in paragraphs if p["has_math"]]
        return {"state": "partial" if missing_ranges else "ready", "code": "DOCX_PARTIAL_ADDITIONAL_PARTS" if missing_parts else "DOCX_MATH_REVIEW_REQUIRED" if math else "DOCX_PARSED",
                "source_sha256": sha, "blocks": blocks, "tables": len(document.tables), "math_paragraphs": math,
                "missing_ranges": missing_ranges, "usable_scope": [b["block_id"] for b in blocks], "coverage": "partial" if missing_ranges else "text_full",
                "can_qa": bool(blocks) and not missing_ranges, "warnings": warnings, "page_numbers_invented": False}
    raise MaterialError("UNSUPPORTED_MEDIA_TYPE")


def _bound_profile(request: dict) -> dict:
    raw = request["profile_json"].encode("utf-8")
    if hashlib.sha256(raw).hexdigest() != request["profile_sha256"] or _sha(Path(request["profile"])) != request["profile_sha256"]:
        raise MaterialError("PARSER_PROFILE_BINDING_MISMATCH")
    return json.loads(raw)


def _chunk(request: dict, profile: dict) -> dict:
    from .chunking import SourceChunker

    path = Path(request["source_path"]).resolve(strict=True)
    if not path.is_relative_to(Path(request["root"])) or path.stat().st_size > 64 * 1024 * 1024 or _sha(path) != request["source_sha256"]:
        raise MaterialError("CHUNK_INPUT_SOURCE_MISMATCH")
    payload = json.loads(path.read_text(encoding="utf-8"))
    chunker = SourceChunker(Path(request["root"]))
    if (request.get("expected_tokenizer_profile_sha256", chunker.profile_sha256) != chunker.profile_sha256
            or profile["fine_chunk_target_tokens"] != chunker.profile["chunk_target_tokens"]
            or profile["fine_stride_tokens"] != chunker.profile["chunk_overlap_tokens"]
            or profile["semantic_unit_maximum_tokens"] != chunker.profile["maximum_chunk_tokens"]
            or profile["semantic_unit_overlap"] != 0):
        raise MaterialError("CHUNKER_PROFILE_BINDING_MISMATCH")
    chunks = chunker.split(payload["blocks"])
    return {"state": "ready", "code": "CHUNKS_READY", "chunks": chunks, "source_sha256": request["source_sha256"],
            "original_file_sha256": payload["file_sha256"], "tokenizer_profile_sha256": chunker.profile_sha256, "model_api_calls": 0}


def execute(request: dict) -> int:
    output = Path(request["output"])
    profile = None
    try:
        profile = _bound_profile(request)
        operation = request.get("operation", "parse")
        result = _chunk(request, profile) if operation == "chunk" else _probe(request) if operation.startswith("probe_") else _parse(request, profile)
        if operation in {"parse", "chunk"} and _sha(Path(request["source_path"])) != request["source_sha256"]:
            raise MaterialError("ORIGINAL_SOURCE_CHANGED_DURING_PARSE")
        if _sha(Path(request["profile"])) != request["profile_sha256"]:
            raise MaterialError("PARSER_PROFILE_CHANGED_DURING_EXECUTION")
    except MaterialError as exc:
        result = {"state": "rejected", "code": exc.code, "can_qa": False}
    except MemoryError:
        result = {"state": "failed", "code": "PARSER_MEMORY_LIMIT", "can_qa": False}
    except BaseException as exc:
        result = {"state": "failed", "code": "PARSER_FAILED", "exception": type(exc).__name__, "detail": str(exc)[:1000], "can_qa": False}
    result["profile_sha256"] = request.get("profile_sha256")
    result["parser_versions"] = {name: importlib.metadata.version(name) for name in profile["parser_versions"]} if profile else {}
    if profile and result["parser_versions"] != profile["parser_versions"]:
        result.update(state="failed", code="WORKER_DEPENDENCY_VERSION_MISMATCH", can_qa=False)
    result["worker_runtime"] = {"executable": sys.executable, "python_version": sys.version, "no_site": bool(sys.flags.no_site), "sys_path": sys.path}
    _write(output / "result.json", result)
    return 0
