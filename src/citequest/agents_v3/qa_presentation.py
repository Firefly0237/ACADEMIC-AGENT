"""Typed claim-only paragraph composition and deterministic display rendering."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from citequest.research_v3.contracts import Strict
from citequest.research_v3.qa_v5_contracts import SelectorAnswerV5
from citequest.research_v3.quote_spans import ResearchAnswerV3

ClaimID = Annotated[str, Field(pattern=r"^C[1-9][0-9]?$", max_length=3)]
Paragraph = Annotated[list[ClaimID], Field(min_length=1, max_length=8)]
LINE_SEPARATORS = frozenset("\r\n\v\f\x1c\x1d\x1e\x85\u2028\u2029")


def inline_statement(value: str) -> str:
    if any(character in LINE_SEPARATORS for character in value) or any(marker in value for marker in ("$$", r"\[", r"\]")):
        raise PydanticCustomError("qa_presentation_inline_statement",
            "Paragraph claim statements require one physical line and inline math")
    return value


class Presentation(Strict):
    format: Literal["prose", "blocks"]
    paragraphs: list[Paragraph] = Field(min_length=1, max_length=8)


def validate_partition(presentation: Presentation, identifiers: list[str]) -> None:
    selected = [identifier for paragraph in presentation.paragraphs for identifier in paragraph]
    if len(selected) != len(identifiers) or len(set(selected)) != len(selected) or set(selected) != set(identifiers):
        raise PydanticCustomError("qa_presentation_claim_partition",
            "Every current claim must occur exactly once in the paragraph partition")


class PresentedSelectorAnswer(SelectorAnswerV5):
    presentation: Presentation

    @model_validator(mode="after")
    def complete_presentation(self):
        validate_partition(self.presentation, [claim.id for claim in self.claims])
        if self.presentation.format == "prose":
            for claim in self.claims:
                inline_statement(claim.statement)
        return self


class PresentedAnswer(ResearchAnswerV3):
    presentation: Presentation

    @model_validator(mode="after")
    def complete_presentation(self):
        validate_partition(self.presentation, [claim.id for claim in self.claims])
        if self.presentation.format == "prose":
            for claim in self.claims:
                inline_statement(claim.statement)
        return self


def render_presented_answer(answer: PresentedAnswer) -> str:
    by_id = {claim.id: claim.statement for claim in answer.claims}
    if answer.presentation.format == "prose":
        return "\n\n".join(" ".join(f"[{identifier}] {by_id[identifier]}" for identifier in paragraph)
            for paragraph in answer.presentation.paragraphs)
    return "\n\n".join("\n\n".join(f"[{identifier}]\n\n{by_id[identifier]}" for identifier in paragraph)
        for paragraph in answer.presentation.paragraphs)
