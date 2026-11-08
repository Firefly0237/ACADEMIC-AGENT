"""Agent Skills SKILL.md instruction-only subset with safe bounded YAML.

Input is content, never a filesystem path, ZIP, URL or executable package.
Optional scripts/references/assets are not loaded by this implementation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import yaml
from yaml.tokens import AliasToken, AnchorToken, TagToken

from citequest.agents_v3.contracts import AgentError

COMPATIBILITY = "citequest-skill-md-instruction-only/1"
MAX_BYTES = 16384
NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")


class UniqueSafeLoader(yaml.SafeLoader):
    """SafeLoader plus rejection of duplicate/merge keys; no custom constructors."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str) or key in result or key == "<<":
                raise ValueError("duplicate_or_non_string_yaml_key")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True)
class ParsedSkill:
    name: str
    description: str
    metadata: dict
    body: str
    content_sha256: str
    utf8_bytes: int


def parse_skill(markdown: str) -> ParsedSkill:
    if not isinstance(markdown, str):
        raise AgentError("skill_content_required", 422)
    try:
        raw = markdown.encode("utf-8", errors="strict")
    except UnicodeError:
        raise AgentError("skill_utf8_required", 422) from None
    if not 1 <= len(raw) <= MAX_BYTES or "\x00" in markdown:
        raise AgentError("skill_content_byte_limit_or_nul", 413)
    # Permit standard CRLF without rewriting the immutable original bytes/hash.
    text = markdown.replace("\r\n", "\n")
    lines = text.splitlines()
    if not lines or lines[0] != "---" or len(lines) > 500:
        raise AgentError("skill_frontmatter_and_at_most_500_lines_required", 422)
    try:
        closing = lines.index("---", 1)
    except ValueError:
        raise AgentError("skill_frontmatter_terminator_required", 422) from None
    header = "\n".join(lines[1:closing])
    body = "\n".join(lines[closing + 1:]).strip()
    if len(header.encode()) > 8192 or not body:
        raise AgentError("skill_header_limit_or_empty_body", 422)
    try:
        tokens = list(yaml.scan(header))
        if any(isinstance(token, (AliasToken, AnchorToken, TagToken)) for token in tokens):
            raise ValueError("yaml_alias_anchor_and_tags_unsupported")
        value = yaml.load(header, Loader=UniqueSafeLoader)
    except (yaml.YAMLError, ValueError, TypeError, RecursionError):
        raise AgentError("skill_yaml_invalid_or_unsupported", 422) from None
    supported = {"name", "description", "license", "compatibility", "metadata"}
    if not isinstance(value, dict) or not {"name", "description"}.issubset(value) or set(value) - supported:
        # allowed-tools is experimental in the spec. This instruction-only
        # subset rejects it explicitly rather than creating or ignoring powers.
        raise AgentError("skill_frontmatter_fields_unsupported", 422)
    name, description = value["name"], value["description"]
    if not isinstance(name, str) or not 1 <= len(name) <= 64 or NAME.fullmatch(name) is None:
        raise AgentError("skill_name_invalid", 422)
    if not isinstance(description, str) or not 1 <= len(description.strip()) <= 1024:
        raise AgentError("skill_description_invalid", 422)
    for key, maximum in (("license", 1024), ("compatibility", 500)):
        if key in value and (not isinstance(value[key], str) or not 1 <= len(value[key].strip()) <= maximum):
            raise AgentError("skill_optional_frontmatter_invalid", 422)
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict) or len(metadata) > 16 or any(
        not isinstance(key, str) or not 1 <= len(key) <= 128 or not isinstance(item, str) or len(item) > 1024
        for key, item in metadata.items()
    ):
        raise AgentError("skill_metadata_invalid", 422)
    return ParsedSkill(name, description.strip(), value, body, hashlib.sha256(raw).hexdigest(), len(raw))
