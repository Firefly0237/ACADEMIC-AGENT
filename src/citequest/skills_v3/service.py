"""Private project skill catalog, append-only versions and current-pin guards.

Every method opens a short business transaction. No import reads a local path,
fetches a URL, starts a process, calls a model, or changes a tool allowlist.
"""

from __future__ import annotations

import hashlib
from uuid import UUID, uuid4

from psycopg import Connection
from psycopg.types.json import Jsonb

from citequest.agents_v3.contracts import AgentError, digest
from citequest.agents_v3.database import Database

from .contracts import SkillLifecycle, SkillPin, SkillWrite
from .parser import COMPATIBILITY, MAX_BYTES, NAME, parse_skill


class SkillService:
    MAX_SKILLS = 100
    MAX_VERSIONS = 100

    def __init__(self, database: Database) -> None:
        self.database = database

    @staticmethod
    def _authorize(db: Connection, owner: UUID, project: UUID, *, lock: bool = False) -> None:
        def verify() -> None:
            if not db.execute("SELECT 1 FROM cq_app.projects WHERE owner_id=%s AND id=%s", (owner, project)).fetchone():
                raise AgentError("project_not_found", 404)

        verify()
        if lock:
            # One owner-wide lock serializes same-operation replay even across
            # projects, and bounded project/name creation. No external IO here.
            key = int.from_bytes(hashlib.sha256(("user-skills:" + str(owner)).encode()).digest()[:8], "big", signed=True)
            db.execute("SELECT pg_advisory_xact_lock(%s)", (key,))
            verify()
        if not db.execute("SELECT to_regclass('cq_app.user_skills') AS ready").fetchone()["ready"]:
            raise AgentError("user_skill_migration_required", 503)

    @staticmethod
    def _row(db: Connection, owner: UUID, project: UUID, skill: UUID) -> dict:
        row = db.execute("""SELECT s.*,v.content_sha256,v.manifest,v.compatibility FROM cq_app.user_skills s
            JOIN cq_app.user_skill_versions v ON (v.skill_id,v.id)=(s.id,s.head_version_id)
            WHERE s.owner_id=%s AND s.project_id=%s AND s.id=%s""", (owner, project, skill)).fetchone()
        if row is None:
            raise AgentError("user_skill_not_found", 404)
        return row

    @staticmethod
    def _present(row: dict) -> dict:
        pin = SkillPin(skill_id=row["id"], version_id=row["head_version_id"],
                       content_sha256=row["content_sha256"], head_revision=row["head_revision"])
        return {"pin": pin.model_dump(mode="json"), "name": row["name"],
                "qualified_name": "user:" + row["name"], "description": row["manifest"]["description"],
                "frontmatter": row["manifest"], "deleted": row["deleted"],
                "compatibility": row["compatibility"], "trust": "user_instruction"}

    @staticmethod
    def _replay(db: Connection, owner: UUID, operation: UUID, payload_hash: str) -> bool:
        old = db.execute("SELECT payload_sha256 FROM cq_app.user_skill_operations WHERE owner_id=%s AND operation_id=%s",
                         (owner, operation)).fetchone()
        if old is None:
            return False
        if old["payload_sha256"] != payload_hash:
            raise AgentError("skill_idempotency_payload_conflict")
        return True

    @staticmethod
    def _record(db: Connection, owner: UUID, project: UUID, operation: UUID, payload_hash: str, result: dict) -> None:
        db.execute("INSERT INTO cq_app.user_skill_operations(owner_id,project_id,operation_id,payload_sha256,result) VALUES(%s,%s,%s,%s,%s)",
                   (owner, project, operation, payload_hash, Jsonb({"skill_id": result["pin"]["skill_id"]})))

    def write(self, owner: UUID, project: UUID, skill: UUID, request: SkillWrite) -> dict:
        parsed = parse_skill(request.markdown)
        payload_hash = digest({"kind": "write", "project_id": str(project), "skill_id": str(skill),
                               **request.model_dump(mode="json")})
        with self.database.connect() as db:
            self._authorize(db, owner, project, lock=True)
            replay = self._replay(db, owner, request.operation_id, payload_hash)
            if replay:
                # Replay proves the effect already happened; return today's
                # authorized head, never a stale undeleted historical response.
                return self._present(self._row(db, owner, project, skill))
            old = db.execute("SELECT * FROM cq_app.user_skills WHERE owner_id=%s AND project_id=%s AND id=%s",
                             (owner, project, skill)).fetchone()
            if (old["head_revision"] if old else 0) != request.expected_revision:
                raise AgentError("skill_revision_conflict")
            if old and (old["deleted"] or old["name"] != parsed.name):
                raise AgentError("skill_deleted_or_name_immutable")
            if db.execute("SELECT 1 FROM cq_app.user_skills WHERE owner_id=%s AND project_id=%s AND name=%s AND id<>%s",
                          (owner, project, parsed.name, skill)).fetchone():
                raise AgentError("skill_name_already_exists")
            if old is None:
                count = db.execute("SELECT count(*) AS count FROM cq_app.user_skills WHERE owner_id=%s AND project_id=%s",
                                   (owner, project)).fetchone()["count"]
                if count >= self.MAX_SKILLS:
                    raise AgentError("skill_project_catalog_limit", 413)
            count = (db.execute("SELECT count(*) AS count FROM cq_app.user_skill_versions WHERE owner_id=%s AND project_id=%s AND skill_id=%s",
                                (owner, project, skill)).fetchone()["count"] if old else 0)
            if count >= self.MAX_VERSIONS:
                raise AgentError("skill_version_limit", 413)
            revision, version = request.expected_revision + 1, uuid4()
            if old is None:
                inserted = db.execute("""INSERT INTO cq_app.user_skills(id,owner_id,project_id,name,head_revision,head_version_id)
                    VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING id""",
                                      (skill, owner, project, parsed.name, revision, version)).fetchone()
                if inserted is None:
                    raise AgentError("user_skill_not_found", 404)
            db.execute("""INSERT INTO cq_app.user_skill_versions
                (id,skill_id,owner_id,project_id,created_revision,content_sha256,markdown,manifest,compatibility)
                VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                       (version, skill, owner, project, revision, parsed.content_sha256, request.markdown, Jsonb(parsed.metadata), COMPATIBILITY))
            if old:
                db.execute("UPDATE cq_app.user_skills SET head_revision=%s,head_version_id=%s,updated_at=clock_timestamp() WHERE id=%s",
                           (revision, version, skill))
            result = self._present(self._row(db, owner, project, skill))
            self._record(db, owner, project, request.operation_id, payload_hash, result)
            return result

    def lifecycle(self, owner: UUID, project: UUID, skill: UUID, request: SkillLifecycle) -> dict:
        payload_hash = digest({"kind": "lifecycle", "project_id": str(project), "skill_id": str(skill),
                               **request.model_dump(mode="json")})
        with self.database.connect() as db:
            self._authorize(db, owner, project, lock=True)
            replay = self._replay(db, owner, request.operation_id, payload_hash)
            if replay:
                return self._present(self._row(db, owner, project, skill))
            row = self._row(db, owner, project, skill)
            if row["head_revision"] != request.expected_revision:
                raise AgentError("skill_revision_conflict")
            if row["deleted"] == request.deleted:
                raise AgentError("skill_lifecycle_already_in_requested_state")
            db.execute("UPDATE cq_app.user_skills SET head_revision=head_revision+1,deleted=%s,updated_at=clock_timestamp() WHERE id=%s",
                       (request.deleted, skill))
            result = self._present(self._row(db, owner, project, skill))
            self._record(db, owner, project, request.operation_id, payload_hash, result)
            return result

    def catalog(self, owner: UUID, project: UUID, *, limit: int = 20, after_name: str = "", include_deleted: bool = False) -> dict:
        if type(limit) is not int or not 1 <= limit <= 50 or (after_name and (len(after_name) > 64 or NAME.fullmatch(after_name) is None)):
            raise AgentError("skill_catalog_page_invalid", 422)
        with self.database.connect() as db:
            self._authorize(db, owner, project)
            rows = db.execute("""SELECT s.*,v.content_sha256,v.manifest,v.compatibility FROM cq_app.user_skills s
                JOIN cq_app.user_skill_versions v ON (v.skill_id,v.id)=(s.id,s.head_version_id)
                WHERE s.owner_id=%s AND s.project_id=%s AND s.name>%s AND (%s OR NOT s.deleted)
                ORDER BY s.name LIMIT %s""", (owner, project, after_name, include_deleted, limit + 1)).fetchall()
        return {"skills": [self._present(row) for row in rows[:limit]],
                "next_name": rows[limit - 1]["name"] if len(rows) > limit else None,
                "compatibility": COMPATIBILITY}

    def versions(self, owner: UUID, project: UUID, skill: UUID, *, limit: int = 20, before_revision: int = 2147483647) -> dict:
        if type(limit) is not int or not 1 <= limit <= 50 or type(before_revision) is not int or before_revision < 1:
            raise AgentError("skill_version_page_invalid", 422)
        with self.database.connect() as db:
            self._authorize(db, owner, project)
            self._row(db, owner, project, skill)
            rows = db.execute("""SELECT id AS version_id,created_revision,content_sha256,manifest,compatibility,created_at
                FROM cq_app.user_skill_versions WHERE owner_id=%s AND project_id=%s AND skill_id=%s
                AND created_revision<%s ORDER BY created_revision DESC LIMIT %s""",
                              (owner, project, skill, before_revision, limit + 1)).fetchall()
        return {"versions": rows[:limit], "next_revision": rows[limit - 1]["created_revision"] if len(rows) > limit else None}

    def read_version(self, owner: UUID, project: UUID, skill: UUID, version: UUID) -> dict:
        with self.database.connect() as db:
            self._authorize(db, owner, project)
            head = self._row(db, owner, project, skill)
            if head["deleted"]:
                raise AgentError("skill_deleted", 410)
            row = db.execute("""SELECT id AS version_id,created_revision,content_sha256,markdown,compatibility
                FROM cq_app.user_skill_versions WHERE owner_id=%s AND project_id=%s AND skill_id=%s AND id=%s""",
                             (owner, project, skill, version)).fetchone()
            if row is None:
                raise AgentError("skill_version_not_found", 404)
            if parse_skill(row["markdown"]).content_sha256 != row["content_sha256"]:
                raise AgentError("skill_version_integrity_failed", 503)
            return {**row, "current": head["head_version_id"] == version,
                    "head_revision": head["head_revision"], "virtual_path": head["name"] + "/SKILL.md"}

    @staticmethod
    def _pins(pins: list[SkillPin]) -> None:
        if not isinstance(pins, list) or len(pins) > 3 or any(not isinstance(pin, SkillPin) for pin in pins):
            raise AgentError("at_most_three_typed_skill_pins_required", 422)
        if len({pin.skill_id for pin in pins}) != len(pins):
            raise AgentError("duplicate_selected_skill", 422)

    def _check_pins(self, db: Connection, owner: UUID, project: UUID, pins: list[SkillPin]) -> list[dict]:
        self._pins(pins)
        self._authorize(db, owner, project)
        rows = []
        for pin in pins:
            row = self._row(db, owner, project, pin.skill_id)
            if row["deleted"] or row["head_version_id"] != pin.version_id or row["head_revision"] != pin.head_revision or row["content_sha256"] != pin.content_sha256:
                raise AgentError("selected_skill_changed_or_deleted")
            rows.append(row)
        return rows

    def validate_current(self, owner: UUID, project: UUID, pins: list[SkillPin]) -> None:
        """Use again in the final pre-wire guard and when projecting results."""
        with self.database.connect() as db:
            self._check_pins(db, owner, project, pins)

    def activate(self, owner: UUID, project: UUID, pins: list[SkillPin]) -> dict:
        """Explicit selection order; body loaded only here, never silently truncated."""
        with self.database.connect() as db:
            rows = self._check_pins(db, owner, project, pins)
            instructions, size = [], 0
            for pin, head in zip(pins, rows, strict=True):
                version = db.execute("SELECT markdown FROM cq_app.user_skill_versions WHERE id=%s AND skill_id=%s",
                                     (pin.version_id, pin.skill_id)).fetchone()
                parsed = parse_skill(version["markdown"])
                if parsed.content_sha256 != pin.content_sha256 or parsed.metadata != head["manifest"] or parsed.name != head["name"]:
                    raise AgentError("skill_version_integrity_failed", 503)
                size += parsed.utf8_bytes
                if size > MAX_BYTES:
                    raise AgentError("selected_skill_total_byte_limit", 413)
                instructions.append({"pin": pin.model_dump(mode="json"), "name": parsed.name,
                                     "body": parsed.body, "trust": "user_instruction"})
            # A concurrent update after the first read invalidates this context;
            # the dispatch path must also invoke validate_current after waiting.
            self._check_pins(db, owner, project, pins)
        return {"schema_version": "citequest-user-skills/1", "pins": [pin.model_dump(mode="json") for pin in pins],
                "instructions": instructions, "total_utf8_bytes": size, "compatibility": COMPATIBILITY}
