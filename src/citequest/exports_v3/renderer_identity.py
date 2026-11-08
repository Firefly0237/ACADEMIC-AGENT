"""Fingerprint the actual pinned offline toolchain, including source and fonts."""

from __future__ import annotations

import hashlib
import importlib.metadata
import subprocess
from pathlib import Path

import pypandoc

from citequest.research_v3.school_contracts import CSL_SHA, TEMPLATE_SHA
from citequest.research_v3.school_linux import FONT_MAPPING, PROFILE_ID


def actual_identity() -> dict:
    fonts = {}
    for family in sorted(set(FONT_MAPPING.values())):
        result = subprocess.run(
            ["fc-match", "-f", "%{family}\n%{file}", family], capture_output=True, text=True, check=True, timeout=10
        )
        matched, filename = result.stdout.strip().split("\n", 1)
        if family not in matched.split(","):
            raise ValueError("pinned_open_font_family_unavailable")
        fonts[family] = {"family": matched, "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest()}
    root = Path(__file__).resolve().parents[2]
    files = [
        Path("/usr/bin/soffice"),
        Path("/usr/lib/libreoffice/program/soffice.bin"),
        Path("/usr/lib/libreoffice/program/libsmlo.so"),
        Path("/usr/lib/libreoffice/share/registry/math.xcd"),
        Path(pypandoc.get_pandoc_path()),
        Path(__file__),
        Path(__file__).with_name("pdf_gate.py"),
        Path(__file__).with_name("bibliography.py"),
        root / "citequest/research_v3/school_export.py",
        root / "citequest/research_v3/school_contracts.py",
        root / "citequest/research_v3/school_linux.py",
        root / "citequest/agents_v3/contracts.py",
        Path("/app/scripts/s13/bundle_entry.py"),
    ]
    return {
        "schema_version": "citequest-offline-school-toolchain/1",
        "font_profile": PROFILE_ID,
        "file_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "fonts": fonts,
        "template_sha256": TEMPLATE_SHA,
        "csl_sha256": CSL_SHA,
        "python_packages": {
            name: importlib.metadata.version(name)
            for name in ("python-docx", "docxtpl", "docxcompose", "pypandoc-binary", "pypdf", "lxml")
        },
    }
