"""Explicit, bounded import and immutable activation identities."""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from citequest.agents_v3.contracts import ID, Strict, SkillPin

SHA = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class SkillWrite(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=0)
    # UTF-8 byte and format limits are additionally checked before any DB write.
    markdown: Annotated[str, Field(min_length=1, max_length=16384)]


class SkillLifecycle(Strict):
    operation_id: ID
    expected_revision: int = Field(ge=1)
    deleted: bool
