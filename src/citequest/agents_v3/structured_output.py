"""Compile dynamic output schemas with the exact locked official SDK rules.

The SDK's raw-schema helper is private. Its dependency is intentional, pinned
and covered by compatibility controls. Compilation happens before request
hashing and context admission; the provider never silently rewrites the wire.
"""
from __future__ import annotations

import copy
import importlib.metadata

from openai.lib._pydantic import _ensure_strict_json_schema

from .contracts import AgentError
from .provider import SDK_VERSION


def strict_output_schema(schema: dict) -> dict:
    """Require all advertised fields without weakening enums or local checks.

    In particular, Pydantic defaults do not make a field optional on the wire.
    The SDK retains its declared type and constraints; no nullable alternative
    is invented. Callers keep their original Pydantic and cross-field validators.
    """
    if importlib.metadata.version("openai") != SDK_VERSION:
        raise AgentError("structured_output_sdk_lock_mismatch")
    if not isinstance(schema, dict):
        raise AgentError("structured_output_schema_not_object")
    compiled = copy.deepcopy(schema)
    try:
        return _ensure_strict_json_schema(compiled, path=(), root=compiled)
    except (AssertionError, KeyError, TypeError, ValueError, RecursionError) as error:
        # SDK errors may contain schema values; callers receive only a safe code.
        raise AgentError("structured_output_schema_compile_failed") from error
