"""Offline converter entrypoint. Trusted host authorizes/admits before enqueue.

The job has no model/database credentials. It converts only a preauthorized
reviewed document whose content hash was bound in the job contract, never an
arbitrary user-provided path, URL or executable bundle.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from citequest.research_v3.school_export import render_pdf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/inputs"))
    parser.add_argument("--output-root", type=Path, default=Path("/outputs"))
    parser.add_argument("--renderer", type=Path, default=Path("/usr/bin/soffice"))
    parser.add_argument("--renderer-sha256", required=True)
    args = parser.parse_args()
    source = args.input_root.resolve() / "proposal.docx"
    if not source.is_file() or source.is_symlink():
        raise ValueError("trusted proposal input is required")
    # Output is mounted by the trusted job host, never passed from request text.
    receipt = render_pdf(
        source,
        args.output_root.resolve() / "pdf",
        renderer=args.renderer,
        renderer_sha256=args.renderer_sha256,
        timeout_seconds=90,
    )
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
