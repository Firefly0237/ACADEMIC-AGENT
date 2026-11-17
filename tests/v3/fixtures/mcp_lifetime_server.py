"""Explicit source-free official MCP server for process lifetime controls only."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

parser = argparse.ArgumentParser()
parser.add_argument("--pid-file", type=Path, required=True)
parser.add_argument("--bad-catalog", action="store_true")
args = parser.parse_args()
args.pid_file.write_text(str(os.getpid()))
server = FastMCP("EXPLICIT_SOURCE_FREE_LIFETIME_FIXTURE")


@server.tool()
def search_evidence(query: str, limit: int = 2) -> dict:
    return {"fixture": True, "status": "ok", "evidence": [], "source_reads": False}


if not args.bad_catalog:

    @server.tool()
    def read_evidence(ref_id: str) -> dict:
        return {"fixture": True}

    @server.tool()
    def plan_workset() -> dict:
        return {"fixture": True}

    @server.tool()
    def read_source_window(window_id: str) -> dict:
        return {"fixture": True}

    @server.tool()
    def verify_evidence_batch(ref_ids: list[str]) -> dict:
        return {"fixture": True}


@server.resource("citequest://task/scope")
def scope() -> str:
    return json.dumps(
        {"status": "ready", "brief_sha256": hashlib.sha256(b"{}").hexdigest(), "fixture": True, "source_reads": False}
    )


if __name__ == "__main__":
    server.run(transport="stdio")
