"""Source-scoped complete-membership merge; no latest/global freshness claim."""
from __future__ import annotations

import re
from copy import deepcopy

PROFILE = "cq.bounded-observed-metadata-union/1"
MAXIMUM_MEMBERS = 100000


def merge_members(base: list[dict], updates: list[dict], denials: set[tuple[str, str]]) -> list[dict]:
    def key(item):
        return item["source_namespace"], item["native_id"]
    values = {}
    for item in base:
        if key(item) in values:
            raise ValueError("complete_base_source_members_must_be_unique")
        values[key(item)] = deepcopy(item)
    for item in updates:
        prior = values.get(key(item))
        old, new = (prior or {}).get("source_revision"), item.get("source_revision")
        if old and new and re.fullmatch(r"v[1-9][0-9]*", old) and re.fullmatch(r"v[1-9][0-9]*", new) and int(new[1:]) < int(old[1:]):
            raise ValueError("explicit_source_version_regression")
        values[key(item)] = deepcopy(item)
    result = [value for item_key, value in values.items() if item_key not in denials]
    if not result or len(result) > MAXIMUM_MEMBERS or len({item["source_unit_id"] for item in result}) != len(result):
        raise ValueError("bounded_nonempty_unique_complete_membership_required")
    return sorted(result, key=lambda item: item["source_unit_id"])
