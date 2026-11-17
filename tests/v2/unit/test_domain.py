"""Strict inputs and PostgreSQL DDL compilation; no server connection."""

from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateTable

from citequest.v2 import models
from citequest.v2.db import engine
from citequest.v2.domain import SourceInput, WorkInput
from citequest.v2.services import normalize_identifier


def test_unknown_is_null_and_actor_cannot_be_submitted():
    payload = {"title": "A work", "authors": None, "year": None, "client_operation_id": str(uuid4())}
    assert WorkInput.model_validate(payload).year is None
    with pytest.raises(ValidationError):
        WorkInput.model_validate({**payload, "owner_id": str(uuid4())})
    with pytest.raises(ValidationError):
        WorkInput.model_validate({**payload, "year": "2020"})
    with pytest.raises(ValidationError):
        WorkInput.model_validate({**payload, "authors": []})


def test_doi_raw_is_preserved_but_lookup_is_normalized():
    raw = " https://doi.org/10.1234/ABC "
    assert normalize_identifier("doi", raw) == "10.1234/abc"
    assert raw.startswith(" ")
    with pytest.raises(Exception, match="invalid_doi"):
        normalize_identifier("doi", "not-a-doi")


def test_all_business_tables_compile_with_postgres_and_schema_is_separate():
    assert {"cq_app.work_access", "cq_app.idempotency_keys", "cq_app.outbox"} <= set(models.Base.metadata.tables)
    for table in models.Base.metadata.sorted_tables:
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        assert "cq_app." in ddl
        assert table.schema == "cq_app"


def test_product_engine_has_no_sqlite_fallback():
    with pytest.raises(ValueError, match="postgresql"):
        engine("sqlite://")


def test_source_uri_format_is_really_enabled_and_invalid_uri_rejected():
    from datetime import UTC, datetime

    from jsonschema import FormatChecker

    assert {"uri", "date-time", "uuid"} <= set(FormatChecker.checkers)
    value = {
        "source": "synthetic",
        "native_id": "001",
        "source_revision": None,
        "retrieved_at": datetime.now(UTC),
        "raw_sha256": "a" * 64,
        "source_url": "not a URI",
        "rights_status": "unknown",
        "license": None,
    }
    with pytest.raises(ValidationError, match="absolute URI"):
        SourceInput.model_validate(value)
    assert (
        SourceInput.model_validate({**value, "source_url": "https://example.test/a"}).source_url
        == "https://example.test/a"
    )


def test_implemented_openapi_preserves_response_and_request_contracts():
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator
    from sqlalchemy.orm import sessionmaker

    from citequest.v2.api import create_app

    schema = Path(__file__).resolve().parents[3] / "contracts/v2/domain.schema.json"
    value = create_app(sessionmaker(), cursor_key=b"x" * 32, contract_path=schema).openapi()
    Draft202012Validator.check_schema(value)
    assert (
        value["components"]["schemas"]["Work"]["required"]
        == json.loads(schema.read_text(encoding="utf-8"))["$defs"]["Work"]["required"]
    )
    page = value["paths"]["/api/v2/library/items"]["get"]
    assert any(p["name"] == "page_size" for p in page["parameters"])
    assert page["responses"]["200"]["content"]["application/json"]["schema"]["allOf"][1]["properties"]["items"][
        "items"
    ] == {"$ref": "#/components/schemas/LibraryItem"}
    post = value["paths"]["/api/v2/works"]["post"]
    assert "201" in post["responses"] and "422" in post["responses"]
    assert "requestBody" in post and post["security"] == [{"localSession": []}]
    assert value["components"]["schemas"]["WorkInput"]["additionalProperties"] is False
