"""Current SQL ACL/monotone source-denial barriers before and after retrieval."""
from __future__ import annotations
from sqlalchemy import and_, exists, or_, select
from citequest.data_v2.application_members import ApplicationMember
from citequest.index_v2.models import IndexUnit
from citequest.updates_v2.models import SourceDenial
from citequest.v2 import models as core
from citequest.v2.domain import Actor


def visible_unit(actor: Actor, domain: str):
    unit, denial = IndexUnit, SourceDenial
    access = exists(select(core.WorkAccess.work_id).where(
        core.WorkAccess.work_id == core.Work.id, core.WorkAccess.owner_id == actor.owner_id,
        core.WorkAccess.revoked.is_(False)).correlate(core.Work))
    work = or_(unit.work_id.is_(None), exists(select(core.Work.id).where(
        core.Work.id == unit.work_id, core.Work.canonical_redirect_work_id.is_(None),
        or_(core.Work.public.is_(True), access)).correlate(unit)))
    version = or_(unit.work_version_id.is_(None), exists(select(core.WorkVersion.id).where(
        core.WorkVersion.id == unit.work_version_id, core.WorkVersion.work_id == unit.work_id,
        core.WorkVersion.security_domain.in_(["public", actor.security_domain])).correlate(unit)))
    # Literal-source aliases refer to the same immutable corpus record.
    direct = exists(select(denial.native_id).where(
        denial.security_domain == domain, denial.active.is_(True), denial.native_id == unit.native_id,
        or_(denial.source_namespace == unit.source_namespace,
            and_(unit.source_namespace == "litsearch_s2orc", denial.source_namespace == "litsearch")),
        or_(denial.source_revision == "*", denial.source_revision == unit.source_revision)).correlate(unit))
    # ACL Anthology revision and Crossref DOI aliases are independent source
    # identities. Conservative crossref * denial never supplies a fake revision.
    acl_denial = exists(select(ApplicationMember.source_row_id).join(
        core.SourceRecord, core.SourceRecord.id == ApplicationMember.source_record_id).join(
        denial, and_(denial.security_domain == domain, denial.active.is_(True),
                     denial.source_namespace == "acl_anthology", denial.native_id == core.SourceRecord.native_id,
                     or_(denial.source_revision == "*", denial.source_revision == core.SourceRecord.source_revision))).where(
        ApplicationMember.representation_id == unit.native_representation_id).correlate(unit))
    doi_denial = exists(select(core.Identifier.normalized).join(denial, and_(
        denial.security_domain == domain, denial.active.is_(True), denial.source_namespace == "crossref",
        denial.native_id == core.Identifier.normalized, denial.source_revision == "*")).where(
        core.Identifier.work_id == unit.work_id, core.Identifier.namespace == "doi",
        core.Identifier.security_domain.in_(["public", actor.security_domain])).correlate(unit))
    native_member = or_(unit.source_reference_kind != "native_representation", exists(
        select(ApplicationMember.source_row_id).join(core.SourceRecord, core.SourceRecord.id == ApplicationMember.source_record_id).where(
            ApplicationMember.representation_id == unit.native_representation_id,
            ApplicationMember.work_id == unit.work_id, ApplicationMember.work_version_id == unit.work_version_id,
            core.SourceRecord.rights_status == "verified",
            core.SourceRecord.security_domain.in_(["public", actor.security_domain])).correlate(unit)))
    return and_(unit.source_denied.is_(False), work, version, native_member, ~direct, ~acl_denial, ~doi_denial)
