"""Explicit header fixtures: history never preserves a revoked paper title."""
from uuid import UUID

import pytest

from citequest.search_v3.batch_headers import CurrentPaperHeader, locator_key
from citequest.search_v3.schemas import PaperDetail, PaperLocator, SourceCitation
from citequest.v2.domain import Actor
from citequest.workstation_v3.history_projection import project_history


ACTOR = Actor(UUID(int=10))
PAPER = PaperLocator(kind="native_representation", id="a" * 64)
ROW = {"history_id": "one", "target_id": "paper:" + locator_key(PAPER), "kind": "view"}


def header(*, title="Current source title", available=True):
    detail = PaperDetail(paper=PAPER, title=title, abstract=None, authors=None, year=None, work_id=None,
        work_version_id=None, coverage="native_literal", source=SourceCitation(source_namespace="explicit_fixture",
        native_id="one", source_revision=None, projection_sha256="a"*64, source_pointer_sha256="b"*64),
        source_snapshot_id=UUID(int=1), literal_text_available=True,
        coordinate_system="unicode_codepoints_in_native_literal", links_status="not_in_source",
        citation_status="incomplete_metadata") if available else None
    return CurrentPaperHeader(paper=PAPER, detail=detail, source_availability="available" if available else "unavailable",
        code=None if available else "source_access_denied", read_mode="native_scalar")


def test_deduplicated_current_batch_and_no_history_mutation():
    rows = [dict(ROW), {**ROW, "history_id": "two"}]
    calls = []
    def reader(actor, papers):
        calls.append((actor, papers))
        return {locator_key(PAPER): header()}
    actual = project_history(rows, ACTOR, reader)
    assert len(calls) == 1 and calls[0] == (ACTOR, [PAPER])
    assert all(row["display_title"] == "Current source title" for row in actual)
    assert all("display_title" not in row for row in rows)


def test_revocation_hides_old_captured_title():
    old = {**ROW, "display_title": "Old confidential title"}
    result = project_history([old], ACTOR, lambda *_: {locator_key(PAPER): header(available=False)})[0]
    assert result["display_title"] is None and result["source_availability"] == "unavailable"


@pytest.mark.parametrize("reader", [None, lambda *_: {}, lambda *_: {"unexpected": header()},
    lambda *_: {locator_key(PAPER): {"title": "unvalidated"}}])
def test_missing_invalid_or_unconfigured_reader_has_no_title_fallback(reader):
    result = project_history([{**ROW, "display_title": "Do not reuse"}], ACTOR, reader)[0]
    assert result["display_title"] is None and result["source_availability"] == "temporarily_unavailable"


def test_reader_failure_is_explicit_unavailable():
    def failed(*_):
        raise RuntimeError("explicit dependency fault fixture")
    assert project_history([ROW], ACTOR, failed)[0]["source_availability"] == "temporarily_unavailable"


def test_absent_real_title_remains_absent_and_other_activities_are_unchanged():
    other = {"history_id": "file", "target_id": "file:example", "kind": "download_requested"}
    result = project_history([ROW, other], ACTOR, lambda *_: {locator_key(PAPER): header(title=None)})
    assert result[0]["display_title"] is None and result[0]["source_availability"] == "available"
    assert result[1] == other
