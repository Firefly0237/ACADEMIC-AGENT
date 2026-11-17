"""CAS pointer reuse and fail-closed cache contracts without a database."""
import copy
import hashlib
import ast
import tempfile
import unittest
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from citequest.research_v3.source_structure import StructurePolicy, StructureSource, StructureUnit
from citequest.research_v3.source_structure_cache import (
    APPLICATION_GRANTS, KEY_FIELDS, SourceStructureCache, cache_structure, native_structure_source,
)
from citequest.v2.domain import DomainError
from citequest.v2.storage import CAS


def fixture():
    text = "Header\nThe complete source includes a condition outside an initial search result."
    return StructureSource(selection={"kind": "native_representation", "id": "a" * 64},
        source_version_sha256="a" * 64, representation_id="a" * 64, security_domain="public", title="Header",
        units=(StructureUnit(reference={"kind": "native_representation", "id": "a" * 64}, text=text,
            source_text_sha256=hashlib.sha256(text.encode()).hexdigest(), coordinate_system="unicode_codepoint",
            annotations={"sectionheader": [(0, 6)], "paragraph": [(7, len(text))]}),))


class Result:
    def __init__(self, value):
        self.value = value

    def scalar_one(self):
        return self.value

    def mappings(self):
        return self

    def one_or_none(self):
        return copy.deepcopy(self.value)


class MemorySession:
    """Explicit SQL contract double; this is not a database integration test."""
    def __init__(self):
        self.ready, self.cache, self.blobs, self.statements = True, {}, {}, []

    def execute(self, statement, values=None):
        sql = " ".join(str(statement).split())
        self.statements.append(sql)
        if "to_regclass(" in sql:
            return Result(self.ready)
        if sql.startswith("SELECT * FROM cq_app.source_structure_cache"):
            return Result(self.cache.get(tuple(values[name] for name in KEY_FIELDS)))
        if sql.startswith("SELECT size_bytes FROM cq_app.blobs"):
            size = self.blobs.get((values["security_domain"], values["sha256"]))
            return Result({"size_bytes": size} if size is not None else None)
        if sql.startswith("INSERT INTO cq_app.blobs"):
            self.blobs.setdefault((values["security_domain"], values["sha256"]), values["size_bytes"])
            return Result(None)
        if sql.startswith("INSERT INTO cq_app.source_structure_cache"):
            self.cache.setdefault(tuple(values[name] for name in KEY_FIELDS), copy.deepcopy(values))
            return Result(None)
        raise AssertionError("unexpected SQL in contract double")

    @contextmanager
    def begin(self):
        yield self


class SourceStructureCacheTests(unittest.TestCase):
    def test_second_lookup_reuses_persisted_pointer_without_rebuilding(self):
        db, current = MemorySession(), fixture()
        with tempfile.TemporaryDirectory() as directory:
            cache = SourceStructureCache(db, CAS(Path(directory)))
            first = cache.get_or_build(current)
            with patch("citequest.research_v3.source_structure_cache.store_structure", side_effect=AssertionError("cache hit rebuilt")):
                second = cache.get_or_build(current)
                observed = cache.get(current)
            self.assertEqual(first.pointer, second.pointer)
            self.assertEqual(observed.pointer, first.pointer)
            self.assertEqual(len(db.cache), 1)
            self.assertFalse(any(sql.startswith(("UPDATE", "DELETE", "CREATE")) for sql in db.statements))

    def test_schema_absence_refuses_before_writing_cas(self):
        db = MemorySession()
        db.ready = False
        with tempfile.TemporaryDirectory() as directory:
            cache = SourceStructureCache(db, CAS(Path(directory)))
            with self.assertRaisesRegex(DomainError, "source_structure_migration_required"):
                cache.get_or_build(fixture())
            self.assertFalse(list(Path(directory).iterdir()))
            self.assertEqual(db.cache, {})

    def test_changed_source_binding_at_same_key_is_not_replaced(self):
        db, current = MemorySession(), fixture()
        with tempfile.TemporaryDirectory() as directory:
            cache = SourceStructureCache(db, CAS(Path(directory)))
            cache.get_or_build(current)
            before = copy.deepcopy(db.cache)
            with self.assertRaisesRegex(DomainError, "cache_binding_conflict"):
                cache.get_or_build(replace(current, title="Mutable display metadata"))
            self.assertEqual(db.cache, before)

    def test_versions_policies_and_domains_have_independent_immutable_keys(self):
        db, current = MemorySession(), fixture()
        with tempfile.TemporaryDirectory() as directory:
            store = CAS(Path(directory))
            first = cache_structure(db, store, current)
            version = cache_structure(db, store, replace(current, source_version_sha256="b" * 64))
            policy = cache_structure(db, store, current, StructurePolicy(maximum_leaf_characters=64))
            domain = cache_structure(db, store, replace(current, security_domain="owner:sample"))
            self.assertEqual(len(db.cache), 4)
            self.assertEqual(len({row.pointer.sha256 for row in (first, version, policy, domain)}), 4)
            self.assertNotEqual(first.pointer.security_domain, domain.pointer.security_domain)

    def test_cas_and_blob_metadata_corruption_are_not_cache_hits(self):
        db, current = MemorySession(), fixture()
        with tempfile.TemporaryDirectory() as directory:
            store = CAS(Path(directory))
            cached = cache_structure(db, store, current)
            db.blobs[("public", cached.pointer.sha256)] += 1
            with self.assertRaisesRegex(DomainError, "cache_blob_conflict"):
                cache_structure(db, store, current)
            db.blobs[("public", cached.pointer.sha256)] -= 1
            store.path("public", cached.pointer.sha256).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "stored_blob_corrupt"):
                cache_structure(db, store, current)

    def test_native_recipe_uses_draft_identity_and_literal_title(self):
        class Locator:
            def model_dump(self, *, mode):
                return {"kind": "s2orc_native_text", "char_start": 0, "char_end": len(Draft.literal_text)}

        class Draft:
            representation_id = "e" * 64
            literal_text = "Literal title\nExact source body"
            source_text_sha256 = hashlib.sha256(literal_text.encode()).hexdigest()
            annotation_spans = {"title": [(0, 13)], "paragraph": [(14, len(literal_text))]}
            annotation_errors = {}

            def validate_identity(self):
                self.checked = True

            def locator(self, start, end):
                return Locator()

        draft = Draft()
        actual = native_structure_source(draft, "public")
        self.assertTrue(draft.checked)
        self.assertEqual(actual.source_version_sha256, draft.representation_id)
        self.assertEqual(actual.title, "Literal title")
        self.assertEqual(actual.units[0].text, draft.literal_text)
        self.assertEqual(actual.selection, actual.units[0].reference)
        self.assertEqual(actual.coverage, "native_literal")

    def test_migration_is_additive_and_forbids_mutation_including_truncate(self):
        path = Path(__file__).resolve().parents[2] / "migrations/app/versions/s17_source_structure_0001.py"
        parsed = ast.parse(path.read_text(encoding="utf-8"))
        constants = {node.targets[0].id: ast.literal_eval(node.value) for node in parsed.body
                     if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in {"down_revision", "DDL"}}
        self.assertEqual(constants["down_revision"], "s16_observed_publication_0001")
        self.assertIn("REFERENCES cq_app.blobs(security_domain,sha256)", constants["DDL"])
        self.assertIn("BEFORE UPDATE OR DELETE", constants["DDL"])
        self.assertIn("BEFORE TRUNCATE", constants["DDL"])
        self.assertNotIn("UPDATE cq_app", constants["DDL"])
        self.assertIn("REVOKE UPDATE,DELETE,TRUNCATE", APPLICATION_GRANTS)
        # Exercise the explicit restore guard without importing a migrator-only dependency.
        downgrade = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "downgrade")
        namespace = {}
        exec(compile(ast.Module(body=[downgrade], type_ignores=[]), str(path), "exec"), namespace)
        with self.assertRaisesRegex(RuntimeError, "explicit_restore"):
            namespace["downgrade"]()


if __name__ == "__main__":
    unittest.main()
