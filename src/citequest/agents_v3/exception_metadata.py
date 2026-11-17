"""Bounded exception structure for private diagnostics, never exception values.

Do not include messages, SQL, arguments, source lines, locals, DSNs or env. MCP
public errors stay unchanged; these observations identify the code boundary.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import traceback
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

COMPONENTS = frozenset({"mcp_client_startup", "mcp_server_boot", "agent_worker_failure"})
PHASES = frozenset({"runtime_imports", "config_load", "build_server", "task_lookup", "source_selection",
    "search_service", "material_service", "tool_registration", "stdio_run", "stdio_open", "session_enter",
    "initialize", "list_tools", "catalog_validation", "run_once"})


def exception_metadata(error: BaseException) -> dict:
    root = Path(__file__).resolve().parents[2]
    leaves = []
    nodes = []
    visited = set()
    truncated = False

    def visit(value: BaseException, depth: int, parent: int | None = None, relation: str = "root") -> None:
        nonlocal truncated
        if id(value) in visited:
            return
        if len(nodes) >= 32 or depth > 8:
            truncated = True
            return
        visited.add(id(value))
        item = {"type":type(value).__name__,"frames":[]}
        if type(value).__module__.startswith("citequest."):
            code = getattr(value,"code",None)
            if isinstance(code,str) and re.fullmatch(r"[a-z][a-z0-9_]{1,120}",code):
                item["code"] = code
        sqlstate = getattr(value,"sqlstate",None)
        if isinstance(sqlstate,str) and re.fullmatch(r"[A-Z0-9]{5}",sqlstate):
            item["SQLSTATE"] = sqlstate
        for frame in traceback.extract_tb(value.__traceback__)[-8:]:
            path = Path(frame.filename)
            name = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
            item["frames"].append({"file":name,"line":frame.lineno,"function":frame.name})
        index = len(nodes)
        nodes.append({**item, "parent": parent, "relation": relation})
        if isinstance(value, BaseExceptionGroup):
            for child in value.exceptions:
                visit(child, depth + 1, index, "group_member")
        elif relation != "cause" and relation != "context":
            leaves.append(item)
        if value.__cause__ is not None:
            visit(value.__cause__, depth + 1, index, "cause")
        elif not value.__suppress_context__ and value.__context__ is not None:
            visit(value.__context__, depth + 1, index, "context")

    visit(error,0)
    return {"type":type(error).__name__,"leaves":leaves,"causal_nodes":nodes,
        "truncated":truncated,"exception_values_included":False}


def private_runtime_observation(component: str, phase: str, state: str, *, task_id: UUID | None = None,
                                elapsed_seconds: float = 0.0, error: BaseException | None = None) -> None:
    """Best-effort fixed-field stderr only; never change a runtime outcome.

    The official stdio client does not expose its process/exit code. Record
    unknown explicitly rather than introspecting or changing that SDK.
    """
    try:
        if component not in COMPONENTS or phase not in PHASES or state not in {"started", "completed", "failed"}:
            return
        if task_id is not None and not isinstance(task_id, UUID):
            return
        if not isinstance(elapsed_seconds, (int, float)) or not math.isfinite(elapsed_seconds):
            return
        value = {"event":component,"phase":phase,"state":state,
            "task_id":str(task_id) if task_id is not None else None,
            "monotonic_elapsed_seconds":round(max(0.0,elapsed_seconds),6),
            "child_exit_code":"unknown"}
        if error is not None:
            value["diagnostic"] = exception_metadata(error)
        print(json.dumps(value,separators=(",",":")),file=sys.stderr,flush=True)  # noqa: T201 - Private data-minimized runtime diagnostics only.
    except Exception:
        # Broken diagnostics must not mask the original application failure.
        pass


@contextmanager
def private_runtime_phase(component: str, phase: str, *, task_id: UUID | None = None) -> Iterator[None]:
    started = time.monotonic()
    private_runtime_observation(component,phase,"started",task_id=task_id)
    try:
        yield
    except BaseException as error:
        private_runtime_observation(component,phase,"failed",task_id=task_id,
            elapsed_seconds=time.monotonic()-started,error=error)
        raise
    else:
        private_runtime_observation(component,phase,"completed",task_id=task_id,
            elapsed_seconds=time.monotonic()-started)
