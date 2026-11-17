"""Trusted, offline Docling capabilities and portable profile assembly.

Version 1 keeps native-text parsing. Version 2 explicitly selects text, OCR,
formula enrichment, or both, with every model file bound to a verified manifest.
This module never downloads assets, imports a model, or runs document inference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .text import MaterialError

VERSION = "citequest-material-worker-profile/2"
OPTIONS_VERSION = "citequest-docling-options/1"
MODES = ("text", "ocr", "formula", "ocr_formula")
REPOSITORIES = {
    "layout": "docling-project/docling-layout-heron",
    "table": "docling-project/docling-models",
    "ocr": "RapidAI/RapidOCR",
    "formula": "docling-project/CodeFormulaV2",
}
OCR_FILES = frozenset({"det_model_path", "cls_model_path", "rec_model_path", "rec_keys_path"})


@dataclass(frozen=True)
class ParserFeatures:
    mode: str = "text"
    ocr_language: str = "ch"
    ocr_files: tuple[tuple[str, str], ...] = ()

    @property
    def ocr_enabled(self) -> bool:
        return self.mode in {"ocr", "ocr_formula"}

    @property
    def formula_enabled(self) -> bool:
        return self.mode in {"formula", "ocr_formula"}


def features(profile: dict) -> ParserFeatures:
    version = profile.get("schema_version")
    if version not in {"citequest-material-worker-profile/1", VERSION}:
        raise MaterialError("PARSER_PROFILE_VERSION_UNSUPPORTED")
    if profile.get("remote_services") is not False or profile.get("external_plugins") is not False:
        raise MaterialError("PARSER_REMOTE_FEATURES_FORBIDDEN")
    if version == "citequest-material-worker-profile/1":
        if (profile.get("ocr_enabled") is not False or profile.get("formula_enrichment") is not False
                or "docling_options" in profile):
            raise MaterialError("ENHANCED_PARSER_REQUIRES_VERSIONED_PROFILE")
        return ParserFeatures()
    options = profile.get("docling_options")
    if (not isinstance(options, dict) or set(options) != {"schema_version", "mode", "ocr_language", "ocr_files"}
            or options.get("schema_version") != OPTIONS_VERSION or options.get("mode") not in MODES
            or options.get("ocr_language") not in {"ch", "en"} or not isinstance(options.get("ocr_files"), dict)):
        raise MaterialError("DOCLING_OPTIONS_INVALID")
    selected = ParserFeatures(options["mode"], options["ocr_language"], tuple(sorted(options["ocr_files"].items())))
    if (type(profile.get("ocr_enabled")) is not bool or type(profile.get("formula_enrichment")) is not bool
            or profile["ocr_enabled"] != selected.ocr_enabled
            or profile["formula_enrichment"] != selected.formula_enabled):
        raise MaterialError("PARSER_FEATURE_FLAGS_DISAGREE")
    if set(options["ocr_files"]) != (OCR_FILES if selected.ocr_enabled else set()):
        raise MaterialError("OCR_LOCAL_MODEL_PATHS_REQUIRED")
    if any(not isinstance(value, str) or not value for value in options["ocr_files"].values()):
        raise MaterialError("OCR_LOCAL_MODEL_PATHS_REQUIRED")
    return selected


def _owned(root: Path, relative: str, *, directory: bool = False) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or "\\" in relative:
        raise MaterialError("PARSER_ASSET_RELATIVE_PATH_REQUIRED")
    raw = root / relative
    try:
        path = raw.resolve(strict=True)
    except OSError:
        raise MaterialError("PARSER_MODEL_ASSET_MISSING") from None
    if raw.is_symlink() or not path.is_relative_to(root) or (not path.is_dir() if directory else not path.is_file()):
        raise MaterialError("PARSER_ASSET_PATH_INVALID")
    return path


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def verified_assets(root: Path, profile: dict, selected: ParserFeatures) -> dict:
    """Validate exact manifest bytes and all files before constructing a pipeline."""
    root = root.resolve(strict=True)
    expected = profile.get("model_manifest_sha256")
    if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
        raise MaterialError("MODEL_MANIFEST_SHA256_REQUIRED")
    manifest = _owned(root, profile["model_manifest"])
    if _sha(manifest) != expected:
        raise MaterialError("MODEL_MANIFEST_CHANGED")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MaterialError("MODEL_MANIFEST_STRUCTURE_INVALID")
    models = data.get("models")
    if not isinstance(models, list) or not 1 <= len(models) <= 32:
        raise MaterialError("MODEL_MANIFEST_STRUCTURE_INVALID")
    cache = _owned(root, profile["model_cache_root"], directory=True)
    indexed: dict[Path, str] = {}
    by_repo: dict[str, dict] = {}
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("files"), list) or not model["files"]:
            raise MaterialError("MODEL_MANIFEST_STRUCTURE_INVALID")
        for item in model["files"]:
            path = _owned(root, item["path"])
            if _sha(path) != item["sha256"]:
                raise MaterialError("CACHED_MODEL_BYTES_CHANGED")
            if path in indexed and indexed[path] != item["sha256"]:
                raise MaterialError("MODEL_MANIFEST_FILE_CONFLICT")
            indexed[path] = item["sha256"]
        if profile["schema_version"] == VERSION:
            repository, revision = model.get("repo_id"), model.get("resolved_revision")
            if (not isinstance(repository, str) or repository in by_repo or not isinstance(revision, str)
                    or re.fullmatch(r"[a-f0-9]{40}", revision) is None):
                raise MaterialError("MODEL_MANIFEST_PINNED_REPOSITORY_REQUIRED")
            by_repo[repository] = model
    if profile["schema_version"] != VERSION:
        # Preserve the existing v1 manifest shape and its first-model layout pin.
        return {"manifest": data, "cache": cache, "layout_revision": models[0]["resolved_revision"], "ocr_files": {}}

    required = {"layout", "table"} | ({"ocr"} if selected.ocr_enabled else set()) | (
        {"formula"} if selected.formula_enabled else set())
    if any(REPOSITORIES[role] not in by_repo for role in required):
        raise MaterialError("PARSER_CAPABILITY_MODEL_NOT_IN_MANIFEST")
    # A verified subset cannot authorize loading additional unlisted local files.
    for path in cache.rglob("*"):
        if path.is_symlink() or path.is_file() and path.resolve() not in indexed:
            raise MaterialError("UNMANIFESTED_PARSER_MODEL_ASSET")
    for role in {"layout", "table"} | ({"formula"} if selected.formula_enabled else set()):
        folder = cache / REPOSITORIES[role].replace("/", "--")
        members = [path for path in indexed if path.is_relative_to(folder)]
        declared = {_owned(root, item["path"]) for item in by_repo[REPOSITORIES[role]]["files"]}
        if not members or not set(members) <= declared:
            raise MaterialError("PARSER_MODEL_ARTIFACT_LAYOUT_INVALID")
    ocr_files = {}
    if selected.ocr_enabled:
        declared = {_owned(root, item["path"]) for item in by_repo[REPOSITORIES["ocr"]]["files"]}
        for name, relative in selected.ocr_files:
            path = _owned(root, relative)
            if path not in declared or not path.is_relative_to(cache):
                raise MaterialError("OCR_MODEL_FILE_NOT_IN_MANIFEST")
            ocr_files[name] = str(path)
    if selected.formula_enabled:
        folder = cache / REPOSITORIES["formula"].replace("/", "--")
        names = {path.name for path in indexed if path.is_relative_to(folder)}
        if (not {"config.json", "tokenizer_config.json"} <= names
                or not {"processor_config.json", "preprocessor_config.json"} & names
                or not any(name.endswith((".safetensors", ".bin")) for name in names)):
            raise MaterialError("FORMULA_MODEL_BUNDLE_INCOMPLETE")
    return {"manifest": data, "cache": cache,
        "layout_revision": by_repo[REPOSITORIES["layout"]]["resolved_revision"],
        "formula_revision": by_repo[REPOSITORIES["formula"]]["resolved_revision"] if selected.formula_enabled else None,
        "ocr_files": ocr_files}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--ocr-language", choices=("ch", "en"), default="ch")
    for name in sorted(OCR_FILES):
        parser.add_argument("--" + name.replace("_", "-"), type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = args.root.resolve(strict=True)

        def relative(path: Path) -> str:
            return (path if path.is_absolute() else root / path).resolve(strict=True).relative_to(root).as_posix()

        profile = copy.deepcopy(json.loads((root / relative(args.base)).read_bytes()))
        features(profile)  # An invalid base must not become valid by overwriting its flags.
        profile.update(schema_version=VERSION, profile_id=profile["profile_id"] + "-" + args.mode + "-v2",
            model_manifest=relative(args.manifest), model_cache_root=relative(args.cache_dir),
            ocr_enabled=args.mode in {"ocr", "ocr_formula"}, formula_enrichment=args.mode in {"formula", "ocr_formula"})
        profile["model_manifest_sha256"] = _sha(root / profile["model_manifest"])
        profile["docling_options"] = {"schema_version": OPTIONS_VERSION, "mode": args.mode,
            "ocr_language": args.ocr_language,
            "ocr_files": {name: relative(value) for name in sorted(OCR_FILES)
                          if (value := getattr(args, name)) is not None}}
        selected = features(profile)
        verified_assets(root, profile, selected)
        output = (args.output if args.output.is_absolute() else root / args.output).resolve()
        if not output.is_relative_to(root) or output.exists() or not output.parent.is_dir():
            raise MaterialError("NEW_PARSER_PROFILE_OUTPUT_REQUIRED")
        raw = (json.dumps(profile, ensure_ascii=False, indent=2) + "\n").encode()
        with output.open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"status": "assembled", "profile_id": profile["profile_id"],
            "profile_sha256": hashlib.sha256(raw).hexdigest(), "mode": selected.mode,
            "model_manifest_sha256": profile["model_manifest_sha256"], "model_inference_calls": 0}), flush=True)  # noqa: T201
    except Exception as error:
        print(json.dumps({"status": "failed", "code": error.code if isinstance(error, MaterialError)
            else "PARSER_PROFILE_ASSEMBLY_FAILED", "automatic_retry": False}), flush=True)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
