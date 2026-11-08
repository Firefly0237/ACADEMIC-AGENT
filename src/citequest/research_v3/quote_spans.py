"""Versioned server-extracted citation spans; model text never becomes a quote.

This is transport granularity, not retrieval chunking. Official locked LangChain
splitting is checked by exact reconstruction, and every selector is bound to
the current payload's permitted source/version/Unicode coordinates.
"""

from __future__ import annotations

import copy
import importlib.metadata
import re
import unicodedata
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Annotated, Literal

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from citequest.agents_v3.contracts import AgentError, digest

from .contracts import Claim, ClaimReview, Edge, ResearchAnswer, Review
from .workset_contracts import DocumentResult, DocumentSummary, Strict, WorksetAnswer

if TYPE_CHECKING:
    from .review_grid import ReviewGrid

VERSION = "citequest-quote-spans/3"
QUOTE_CHARACTERS = 900
Selector = Annotated[str, Field(pattern=r"^Q[1-9][0-9]{0,4}$", max_length=6)]
SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class QuoteProtocolError(AgentError):
    pass


class SpanLocator(Strict):
    source_ref: dict
    source_version_sha256: SHA | None
    version_binding_sha256: SHA
    source_sha256: SHA
    unit_text_sha256: SHA
    quote_sha256: SHA
    span_start: int = Field(ge=0)
    span_end: int = Field(ge=1)
    coordinate_system: str


class SpanEdge(Edge):
    quote_locator: SpanLocator


class SpanClaim(Claim):
    edges: list[SpanEdge] = Field(max_length=5)


class DocumentSummaryV3(DocumentSummary):
    claims: list[SpanClaim] = Field(min_length=1, max_length=8)


class ResearchAnswerV3(ResearchAnswer):
    claims: list[SpanClaim] = Field(min_length=1, max_length=8)


class DocumentResultV3(DocumentResult):
    summary: DocumentSummaryV3 | None


class WorksetAnswerV3(WorksetAnswer):
    documents: list[DocumentResultV3] = Field(min_length=1, max_length=10)
    comparisons: list[SpanClaim] = Field(max_length=8)


class SelectorEdge(Strict):
    quote_id: Selector
    relation: Literal["supports", "contradicts", "context_only"]
    rationale: str = Field(min_length=1, max_length=1000)
    conditions: list[str] = Field(max_length=6)


class SelectorClaim(Claim):
    edges: list[SelectorEdge] = Field(max_length=5)


class SelectorSummary(Strict):
    claims: list[SelectorClaim] = Field(min_length=1, max_length=8)
    gaps: list[str] = Field(max_length=8)
    evidence_status: Literal["supported", "insufficient"]


class SelectorAnswer(ResearchAnswer):
    claims: list[SelectorClaim] = Field(min_length=1, max_length=8)


class SelectorClaimReview(ClaimReview):
    edge_ref_ids: list[Selector] = Field(max_length=5)


class SelectorPlainReview(Review):
    claims: list[SelectorClaimReview] = Field(min_length=1, max_length=8)


class SelectorReview(SelectorPlainReview):
    request_coverage: Literal["complete", "partial"]
    uncovered_requests: list[str] = Field(max_length=12)


@dataclass(frozen=True)
class Unit:
    reference: str
    start: int
    end: int
    text: str


def source_warnings(evidence: list[dict]) -> list[dict]:
    warnings = []
    for source in evidence:
        positions = [source["span_start"] + i for i, c in enumerate(source["text"]) if unicodedata.category(c) == "Co"]
        if positions:
            points = sorted({f"U+{ord(c):04X}" for c in source["text"] if unicodedata.category(c) == "Co"})
            warnings.append(
                {
                    "ref_id": source["ref_id"],
                    "document_id": source.get("document_id"),
                    "code": "unresolved_private_use_unicode",
                    "count": len(positions),
                    "first_positions": positions[:16],
                    "positions_partial": len(positions) > 16,
                    "code_points": points[:16],
                    "code_points_partial": len(points) > 16,
                    "coordinate_system": source["coordinate_system"],
                    "interpretation": "preserved_verbatim_symbol_meaning_not_established",
                }
            )
    return warnings


class QuoteRegistry:
    def __init__(self, evidence: list[dict], *, allow_empty: bool = False) -> None:
        if importlib.metadata.version("langchain-text-splitters") != "1.1.3":
            raise QuoteProtocolError("quote_splitter_lock_mismatch")
        self.sources: dict[str, dict] = {}
        for source in evidence:
            if (
                not isinstance(source.get("text"), str)
                or not source["text"]
                or type(source.get("span_start")) is not int
                or type(source.get("span_end")) is not int
                or digest(source["text"].encode()) != source.get("source_sha256")
                or source.get("span_end", 0) - source.get("span_start", 0) != len(source["text"])
            ):
                raise QuoteProtocolError("quote_source_integrity_failed")
            if (
                not isinstance(source.get("source_ref"), dict)
                or not source["source_ref"]
                or not isinstance(source.get("coordinate_system"), str)
                or not source["coordinate_system"]
                or any(
                    not isinstance(source.get(key), str) or re.fullmatch(r"[a-f0-9]{64}", source[key]) is None
                    for key in ("source_sha256", "unit_text_sha256")
                )
                or (
                    source.get("source_version_sha256") is not None
                    and (
                        not isinstance(source["source_version_sha256"], str)
                        or re.fullmatch(r"[a-f0-9]{64}", source["source_version_sha256"]) is None
                    )
                )
                or (source.get("evidence_type") == "source_window" and source.get("source_version_sha256") is None)
            ):
                raise QuoteProtocolError("quote_source_version_or_coordinates_invalid")
            ref = source.get("ref_id")
            if not isinstance(ref, str) or not ref or (ref in self.sources and self.sources[ref] != source):
                raise QuoteProtocolError("quote_source_reference_changed")
            self.sources[ref] = copy.deepcopy(source)
        if (not self.sources and not allow_empty) or len(self.sources) > 99999:
            raise QuoteProtocolError("quote_bounded_sources_required")
        self.allow_no_evidence = bool(allow_empty and not self.sources)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=QUOTE_CHARACTERS,
            chunk_overlap=0,
            length_function=len,
            keep_separator=True,
            strip_whitespace=False,
            separators=["\n\n", "\n", "。", "；", ". ", "; ", " ", ""],
        )

    def scope(self, contexts: list[dict], *, candidates: list[dict] | None = None) -> QuoteScope:
        return QuoteScope(self, contexts, candidates or [])


class QuoteScope:
    def __init__(self, registry: QuoteRegistry, contexts: list[dict], candidates: list[dict]) -> None:
        self.registry = registry
        validated = []
        for context in contexts:
            source = registry.sources.get(context.get("ref_id"))
            if source is None:
                raise QuoteProtocolError("quote_context_outside_source_scope")
            for key in (
                "document_id",
                "source_ref",
                "source_sha256",
                "unit_text_sha256",
                "source_version_sha256",
                "coverage_kind",
                "coverage",
            ):
                if context.get(key) != source.get(key):
                    raise QuoteProtocolError("quote_context_source_pin_changed")
            start, end = (
                context.get("context_span_start", context.get("span_start")),
                context.get("context_span_end", context.get("span_end")),
            )
            if (
                type(start) is not int
                or type(end) is not int
                or not source["span_start"] <= start < end <= source["span_end"]
                or source["text"][start - source["span_start"] : end - source["span_start"]] != context.get("text")
            ):
                raise QuoteProtocolError("quote_context_coordinates_or_literal_changed")
            validated.append({"context": copy.deepcopy(context), "start": start, "end": end})
        no_cited_claims = bool(candidates) and not any(
            edge for candidate in candidates for claim in candidate.get("claims", []) for edge in claim["edges"]
        )
        if not validated and not no_cited_claims and not registry.allow_no_evidence:
            raise QuoteProtocolError("quote_payload_scope_empty")
        # Union only overlapping/adjacent intervals of the SAME source. Never
        # bridge an unseen gap or repeat a source paragraph once per claim.
        raw_groups = []
        for group in sorted(validated, key=lambda g: (g["context"]["ref_id"], g["start"], g["end"])):
            if (
                raw_groups
                and raw_groups[-1]["context"]["ref_id"] == group["context"]["ref_id"]
                and group["start"] <= raw_groups[-1]["end"]
            ):
                last = raw_groups[-1]
                last["end"] = max(last["end"], group["end"])
                source = registry.sources[last["context"]["ref_id"]]
                last["context"]["text"] = source["text"][
                    last["start"] - source["span_start"] : last["end"] - source["span_start"]
                ]
                last["context"]["review_scope"] = "union_of_current_contexts_without_gaps"
            else:
                raw_groups.append(copy.deepcopy(group))
        self.raw_groups = raw_groups
        # Legacy canonical quotes are admitted only by exact, unique positioning
        # in the actual current contexts. No normalization or favourable-occurrence choice.
        candidate_units = {}
        for candidate in candidates:
            for claim in candidate.get("claims", []):
                for edge in claim["edges"]:
                    ref, start, end = self.locate(edge)
                    candidate_units[(ref, start, end)] = Unit(ref, start, end, edge["quote"])
        units = {}
        self.raw_unit_keys = set()
        for group in raw_groups:
            source = registry.sources[group["context"]["ref_id"]]
            # Candidate boundaries are exact server positions. Partitioning at
            # them avoids copying quote text into a second prompt representation.
            cuts = sorted(
                {
                    group["start"],
                    group["end"],
                    *[
                        position
                        for unit in candidate_units.values()
                        if unit.reference == source["ref_id"]
                        and group["start"] <= unit.start < unit.end <= group["end"]
                        for position in (unit.start, unit.end)
                    ],
                }
            )
            keys = []
            for start, end in pairwise(cuts):
                literal = source["text"][start - source["span_start"] : end - source["span_start"]]
                pieces = registry.splitter.split_text(literal)
                if (
                    not pieces
                    or "".join(pieces) != literal
                    or any(not 0 < len(piece) <= QUOTE_CHARACTERS for piece in pieces)
                ):
                    raise QuoteProtocolError("quote_unit_exact_reconstruction_failed")
                cursor = start
                for piece in pieces:
                    unit = Unit(source["ref_id"], cursor, cursor + len(piece), piece)
                    key = (unit.reference, unit.start, unit.end)
                    units[key] = unit
                    keys.append(key)
                    self.raw_unit_keys.add(key)
                    cursor = unit.end
            if "".join(units[key].text for key in keys) != group["context"]["text"]:
                raise QuoteProtocolError("quote_context_exact_reconstruction_failed")
            group["keys"] = keys
        units.update(candidate_units)
        if len(units) > 99999:
            raise QuoteProtocolError("quote_selector_count_exceeded")
        self.forward = {key: "Q" + str(i) for i, key in enumerate(sorted(units), 1)}
        self.units = {self.forward[key]: unit for key, unit in units.items()}
        self.allowed = {identifier for identifier, unit in self.units.items() if unit.text.strip()}
        if not self.allowed and not no_cited_claims and not registry.allow_no_evidence:
            raise QuoteProtocolError("quote_payload_has_no_citable_text")
        self.claim_selectors: dict[str, set[str]] = {}
        self.binding = {
            "schema_version": VERSION,
            "payload_scope_sha256": digest(
                [
                    {
                        "ref_id": g["context"]["ref_id"],
                        "start": g["start"],
                        "end": g["end"],
                        "text_sha256": digest(g["context"]["text"].encode()),
                    }
                    for g in raw_groups
                ]
            ),
            "allowed_quote_ids": sorted(self.allowed),
        }
        self.binding["binding_sha256"] = digest(
            {
                "scope": self.binding,
                "units": [
                    {"quote_id": identifier, "ref_id": unit.reference, "locator": self.locator(unit)}
                    for identifier, unit in sorted(self.units.items())
                ],
            }
        )
        self.contexts = []
        shown = set()
        for i, group in enumerate(raw_groups, 1):
            shown.update(self.forward[key] for key in group["keys"])
            self.contexts.append(
                {
                    "context_id": "CTX" + str(i),
                    "document_id": group["context"].get("document_id"),
                    "coverage_kind": group["context"].get("coverage_kind", group["context"].get("coverage")),
                    "review_scope": group["context"].get("review_scope", "entire_issued_window_part"),
                    "quote_units": [self.view(self.forward[key]) for key in group["keys"]],
                }
            )
        extra = [self.view(identifier) for identifier in sorted(self.units) if identifier not in shown]
        if extra:
            self.contexts.append(
                {
                    "context_id": "EXACT_CANDIDATE_SPANS",
                    "review_scope": "exact_candidate_spans_within_contexts",
                    "quote_units": extra,
                }
            )
        self.warnings = source_warnings(list(registry.sources.values()))

    def locator(self, unit: Unit) -> dict:
        source = self.registry.sources[unit.reference]
        return {
            "source_ref": source["source_ref"],
            "source_version_sha256": source.get("source_version_sha256"),
            "version_binding_sha256": digest(
                {
                    key: source[key]
                    for key in (
                        "source_ref",
                        "source",
                        "source_version_sha256",
                        "source_pin_sha256",
                        "work_id",
                        "work_version_id",
                        "file_id",
                        "file_revision",
                        "material_version_id",
                        "representation_id",
                        "source_file_sha256",
                        "locator",
                        "missing_ranges",
                        "config_sha256",
                        "artifact_sha256",
                        "outbound_policy",
                        "rights_status",
                        "snapshot_id",
                        "generation_id",
                        "epoch",
                        "source_sha256",
                        "unit_text_sha256",
                        "span_start",
                        "span_end",
                        "coordinate_system",
                        "coverage_kind",
                        "coverage",
                    )
                    if key in source
                }
            ),
            "source_sha256": source["source_sha256"],
            "unit_text_sha256": source["unit_text_sha256"],
            "quote_sha256": digest(unit.text.encode()),
            "span_start": unit.start,
            "span_end": unit.end,
            "coordinate_system": source["coordinate_system"],
        }

    def view(self, identifier: str) -> dict:
        unit = self.units[identifier]
        result = {
            "quote_id": identifier,
            "document_id": self.registry.sources[unit.reference].get("document_id"),
            "coverage_kind": self.registry.sources[unit.reference].get(
                "coverage_kind", self.registry.sources[unit.reference].get("coverage")
            ),
            "source_span_start": unit.start,
            "source_span_end": unit.end,
            "eligible_for_citation": identifier in self.allowed,
            "source_warning_codes": ["unresolved_private_use_unicode"]
            if any(unicodedata.category(c) == "Co" for c in unit.text)
            else [],
        }
        key = (unit.reference, unit.start, unit.end)
        if key in self.raw_unit_keys:
            result["text"] = unit.text
        else:
            contained = [
                (key, self.units[self.forward[key]])
                for key in sorted(self.raw_unit_keys)
                if key[0] == unit.reference and unit.start <= key[1] < key[2] <= unit.end
            ]
            if not contained or "".join(piece.text for _, piece in contained) != unit.text:
                raise QuoteProtocolError("quote_composite_selector_not_contiguous")
            result["covered_quote_units"] = [self.forward[key] for key, _ in contained]
        return result

    def locate(self, edge: dict) -> tuple[str, int, int]:
        ref, quote = edge.get("ref_id"), edge.get("quote")
        source = self.registry.sources.get(ref)
        if source is None or not isinstance(quote, str) or not quote or len(quote) > 1000:
            raise QuoteProtocolError("quote_candidate_source_or_size_invalid")
        locator = edge.get("quote_locator")
        if locator:
            start, end = locator.get("span_start"), locator.get("span_end")
            if type(start) is not int or type(end) is not int:
                raise QuoteProtocolError("quote_candidate_locator_invalid")
            candidate = Unit(ref, start, end, quote)
            if locator != self.locator(candidate):
                raise QuoteProtocolError("quote_candidate_locator_pin_changed")
            positions = [(start, end)]
        else:
            positions = []
            cursor = 0
            while (position := source["text"].find(quote, cursor)) >= 0:
                positions.append((source["span_start"] + position, source["span_start"] + position + len(quote)))
                cursor = position + 1
            if len(positions) != 1:
                raise QuoteProtocolError("quote_candidate_not_exact_unique_literal")
        start, end = positions[0]
        if (
            not source["span_start"] <= start < end <= source["span_end"]
            or source["text"][start - source["span_start"] : end - source["span_start"]] != quote
            or not any(g["context"]["ref_id"] == ref and g["start"] <= start < end <= g["end"] for g in self.raw_groups)
        ):
            raise QuoteProtocolError("quote_candidate_outside_payload_context")
        return ref, start, end

    def schema(self, model: type[BaseModel]) -> dict:
        schema = model.model_json_schema()

        def bind(node: object) -> None:
            if isinstance(node, dict):
                properties = node.get("properties", {})
                if "quote_id" in properties and self.allowed:
                    properties["quote_id"]["enum"] = sorted(self.allowed)
                if "edge_ref_ids" in properties:
                    if self.allowed:
                        properties["edge_ref_ids"]["items"]["enum"] = sorted(self.allowed)
                    else:
                        properties["edge_ref_ids"]["maxItems"] = 0
                if "edges" in properties and not self.allowed:
                    properties["edges"]["maxItems"] = 0
                for child in node.values():
                    bind(child)
            elif isinstance(node, list):
                for child in node:
                    bind(child)

        bind(schema)
        return schema

    def compact_candidate(self, value: dict, *, track_review: bool = False) -> dict:
        result = copy.deepcopy(value)
        for claim in result["claims"]:
            ids = set()
            for edge in claim["edges"]:
                key = self.locate(edge)
                identifier = self.forward.get(key)
                if identifier not in self.allowed:
                    raise QuoteProtocolError("quote_candidate_selector_not_issued")
                for field in ("ref_id", "quote", "quote_locator"):
                    edge.pop(field, None)
                edge["quote_id"] = identifier
                ids.add(identifier)
            if track_review:
                if claim["id"] in self.claim_selectors and self.claim_selectors[claim["id"]] != ids:
                    raise QuoteProtocolError("quote_duplicate_claim_id_changed")
                self.claim_selectors[claim["id"]] = ids
        return result

    def expand(self, value: BaseModel, model: type[BaseModel]) -> BaseModel:
        result = value.model_dump()
        for claim in result["claims"]:
            for edge in claim["edges"]:
                identifier = edge.pop("quote_id")
                if identifier not in self.allowed:
                    raise QuoteProtocolError("quote_output_selector_not_issued")
                unit = self.units[identifier]
                edge.update(ref_id=unit.reference, quote=unit.text, quote_locator=self.locator(unit))
        return model.model_validate(result, strict=True)

    def expand_summary(self, value: SelectorSummary) -> DocumentSummaryV3:
        return self.expand(value, DocumentSummaryV3)

    def expand_answer(self, value: SelectorAnswer) -> ResearchAnswerV3:
        return self.expand(value, ResearchAnswerV3)

    def expand_review(self, value: BaseModel, model: type[BaseModel]) -> BaseModel:
        result = value.model_dump()
        if len({row["claim_id"] for row in result["claims"]}) != len(result["claims"]) or {
            row["claim_id"] for row in result["claims"]
        } != set(self.claim_selectors):
            raise QuoteProtocolError("quote_review_claim_coverage_mismatch")
        for item in result["claims"]:
            if set(item["edge_ref_ids"]) != self.claim_selectors[item["claim_id"]]:
                raise QuoteProtocolError("quote_review_span_coverage_mismatch")
            if any(identifier not in self.allowed for identifier in item["edge_ref_ids"]):
                raise QuoteProtocolError("quote_review_selector_not_issued")
            item["edge_ref_ids"] = [self.units[identifier].reference for identifier in item["edge_ref_ids"]]
        return model.model_validate(result, strict=True)

    def review_grid(self, candidate: dict, *, request_coverage: bool = True) -> ReviewGrid:
        from .review_grid import ReviewGrid

        return ReviewGrid(self, candidate, request_coverage=request_coverage)
