"""Real source readers for the versioned relational source variants.

Only the canonical CAS root is accepted. This reader does not make HTTP/TEI
requests, create Works or pretend an Observation/NativeRepresentation is a
metadata SourceProjection. Admission/adoption remain the source owners' gates.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol
from uuid import UUID

from sqlalchemy import select, text

from citequest.data_v2 import models as data
from citequest.data_v2.application_members import ApplicationMember
from citequest.data_v2.landing import Landing
from citequest.data_v2.native_index import read_for_index
from citequest.updates_v2 import models as updates
from citequest.updates_v2.reader import UpdateReader
from citequest.v2.domain import DomainError
from citequest.v2.services import Catalog, digest
from citequest.v2.storage import CAS
from citequest.v2.storage_config import app_storage

from .source_variants import (
    CanonicalSourceUnit,
    DenialReceipt,
    DeniedSourceUnit,
    MetadataReference,
    NativeReference,
    SourceReference,
    SourceRoot,
    UpdateReference,
    metadata_unit,
    native_unit,
    update_unit,
)

ROOT = Path(__file__).resolve().parents[3]


class GenerationSource(Protocol):
    source_root_kind: str
    snapshot_id: UUID | None
    update_seal_id: UUID | None
    security_domain: str
    source_namespace: str
    catalog_name: str
    representation_profile: str
    source_manifest_sha256: str


class SourceReaders:
    def __init__(self, catalog: Catalog, *, store: CAS, policy_sha256: str, evaluation_sha256: str) -> None:
        app_storage().assert_store(store)
        self.catalog, self.db, self.store = catalog, catalog.db, store
        self.landing = Landing(catalog, policy_sha256=policy_sha256, evaluation_sha256=evaluation_sha256)
        self.updates = UpdateReader(catalog.db, catalog.actor, store)

    def root(self, generation: GenerationSource) -> SourceRoot:
        kind = getattr(generation, "source_root_kind", None)
        if kind not in {"metadata_snapshot", "native_snapshot", "update_seal"}:
            raise DomainError("typed_index_schema_not_loaded", 409)
        if kind in {"metadata_snapshot", "native_snapshot"}:
            source = self.landing.snapshot(generation.snapshot_id)
            expected = {
                "metadata_snapshot": ("standard", "litsearch-clean-exact/1", "standard_metadata"),
                "native_snapshot": ("application", "litsearch-s2orc-native-text/1", "application_papers"),
            }[kind]
            if (
                (source.purpose, source.representation_profile, generation.catalog_name) != expected
                or source.security_domain != generation.security_domain
                or source.source_namespace != generation.source_namespace
                or source.source_manifest_sha256 != generation.source_manifest_sha256
                or source.policy_sha256 != self.landing.policy_sha256
                or source.evaluation_sha256 != self.landing.evaluation_sha256
            ):
                raise DomainError("typed_source_root_identity_or_policy_mismatch", 409)
            if source.seal_sha256 is not None and digest(source.seal) != source.seal_sha256:
                raise DomainError("actual_source_seal_hash_mismatch", 409)
            seal = source.seal or {}
            return SourceRoot(
                kind=kind,
                source_root_id=source.id,
                source_namespace=source.source_namespace,
                security_domain=source.security_domain,
                representation_profile=source.representation_profile,
                source_manifest_sha256=source.source_manifest_sha256,
                source_seal_sha256=source.seal_sha256,
                membership_sha256=seal.get("membership_sha256")
                or seal.get("selection_manifest_sha256")
                or seal.get("native_ID_set_sha256"),
                policy_sha256=source.policy_sha256,
                evaluation_sha256=source.evaluation_sha256,
                state=source.status,
                profile_adopted=source.profile_adopted,
                publication_eligible=bool(source.profile_adopted and source.status == "staged" and source.seal_sha256),
                catalog_name=generation.catalog_name,
            )
        seal = self.db.get(updates.UpdateSeal, generation.update_seal_id)
        if (
            seal is None
            or seal.security_domain != generation.security_domain
            or seal.security_domain not in {"public", self.catalog.actor.security_domain}
        ):
            raise DomainError("not_found", 404)
        if (
            digest(seal.payload) != seal.seal_sha256
            or generation.source_manifest_sha256 != seal.seal_sha256
            or generation.source_namespace != seal.source_namespace
            or generation.catalog_name != "managed_metadata"
        ):
            raise DomainError("update_source_root_seal_or_scope_mismatch", 409)
        # The accepted observed-only source profile never gains a publication
        # permission merely because its metadata has been embedded successfully.
        return SourceRoot(
            kind="update_seal",
            source_root_id=seal.id,
            source_namespace=seal.source_namespace,
            security_domain=seal.security_domain,
            representation_profile="cq.protocol-metadata/1",
            source_manifest_sha256=seal.seal_sha256,
            source_seal_sha256=seal.seal_sha256,
            membership_sha256=seal.membership_sha256,
            policy_sha256=self.landing.policy_sha256,
            evaluation_sha256=self.landing.evaluation_sha256,
            state="staged",
            profile_adopted=False,
            publication_eligible=seal.publication_eligible,
            catalog_name="managed_metadata",
        )

    def references(self, root: SourceRoot, *, limit: int | None = None) -> list[SourceReference]:
        if root.state == "disabled":
            raise DomainError("source_root_disabled", 409)
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 64183):
            raise DomainError("bounded_source_selection_required", 400)
        if root.kind == "metadata_snapshot":
            query = (
                select(data.SourceProjection.id)
                .join(data.SourceRow, data.SourceProjection.source_row_id == data.SourceRow.id)
                .where(
                    data.SourceRow.snapshot_id == root.source_root_id,
                    data.SourceProjection.representation_profile == root.representation_profile,
                )
                .order_by(data.SourceRow.artifact_path, data.SourceRow.row_ordinal)
            )
            if limit is not None:
                query = query.limit(limit)
            return [MetadataReference(metadata_projection_id=identifier) for identifier in self.db.scalars(query)]
        if root.kind == "native_snapshot":
            query = (
                select(data.NativeRepresentation.id)
                .join(data.SourceRow, data.NativeRepresentation.source_row_id == data.SourceRow.id)
                .where(data.SourceRow.snapshot_id == root.source_root_id)
                .order_by(data.SourceRow.artifact_path, data.SourceRow.row_ordinal, data.NativeRepresentation.id)
            )
            source = self.landing.snapshot(root.source_root_id)
            if source.profile_adopted:
                seal = source.seal or {}
                if (
                    seal.get("schema_version") != "cq.application-source-seal/2"
                    or seal.get("membership_table") != "cq_app.application_corpus_members"
                    or root.membership_sha256 != seal.get("membership_sha256")
                ):
                    raise DomainError("formal_application_exact_selected_manifest_required", 409)
                count, actual_digest = self.db.execute(
                    text(
                        "SELECT count(*), encode(sha256(convert_to(string_agg(record_sha256,E'\\n' ORDER BY rank)||E'\\n','UTF8')),'hex') "
                        "FROM cq_app.application_corpus_members WHERE snapshot_id=:snapshot"
                    ),
                    {"snapshot": source.id},
                ).one()
                if count != seal.get("coverage", {}).get("qualified") or actual_digest != seal.get(
                    "membership_records_sha256"
                ):
                    raise DomainError("formal_application_member_set_incomplete", 409)
                query = query.join(
                    ApplicationMember,
                    (ApplicationMember.source_row_id == data.SourceRow.id)
                    & (ApplicationMember.representation_id == data.NativeRepresentation.id)
                    & (ApplicationMember.snapshot_id == source.id),
                )
            if limit is not None:
                query = query.limit(limit)
            refs = [NativeReference(native_representation_id=identifier) for identifier in self.db.scalars(query)]
            if source.profile_adopted and limit is None and len(refs) != count:
                raise DomainError("formal_application_member_set_incomplete", 409)
            return refs
        members = list(
            self.db.scalars(
                select(updates.SealMember)
                .where(updates.SealMember.seal_id == root.source_root_id)
                .order_by(updates.SealMember.source_namespace, updates.SealMember.native_id)
            )
        )
        actual = [
            {
                "source_namespace": member.source_namespace,
                "native_id": member.native_id,
                "observation_id": str(member.observation_id),
                "projection_sha256": member.projection_sha256,
            }
            for member in members
        ]
        if digest(actual) != root.membership_sha256:
            raise DomainError("update_source_member_set_hash_mismatch", 409)
        return [
            UpdateReference(update_observation_id=member.observation_id)
            for member in (members if limit is None else members[:limit])
        ]

    def unit_id(self, reference: SourceReference) -> UUID:
        if isinstance(reference, MetadataReference):
            return reference.metadata_projection_id
        if isinstance(reference, UpdateReference):
            return reference.update_observation_id
        row = self.db.get(data.NativeRepresentation, reference.native_representation_id)
        if row is None:
            raise DomainError("not_found", 404)
        return row.index_projection_id

    def read(self, root: SourceRoot, reference: SourceReference) -> CanonicalSourceUnit | DeniedSourceUnit:
        if root.state == "disabled":
            raise DomainError("source_root_disabled", 409)
        if isinstance(reference, MetadataReference):
            if root.kind != "metadata_snapshot":
                raise DomainError("source_reference_family_mismatch", 409)
            source = self.landing.index_projection(reference.metadata_projection_id)
            unit = metadata_unit(source, root)
        elif isinstance(reference, NativeReference):
            if root.kind != "native_snapshot":
                raise DomainError("source_reference_family_mismatch", 409)
            native = self.db.get(data.NativeRepresentation, reference.native_representation_id)
            row = self.db.get(data.SourceRow, native.source_row_id) if native is not None else None
            if row is None or row.snapshot_id != root.source_root_id or row.security_domain != root.security_domain:
                raise DomainError("native_not_in_actual_source_root", 404)
            snapshot = self.landing.snapshot(row.snapshot_id)
            denied = self.updates.current_denials(
                root.source_namespace, [(row.native_id, snapshot.source_revision)], root.security_domain
            )
            if denied:
                return DeniedSourceUnit(
                    source_unit_id=native.index_projection_id,
                    root=root,
                    source_reference=reference,
                    source_kind="s2orc_native",
                    native_id=row.native_id,
                    source_revision=snapshot.source_revision,
                    projection_sha256=native.id,
                    source_pointer_sha256=row.source_pointer_sha256,
                    source_denials=[DenialReceipt.model_validate_json(json.dumps(item)) for item in denied],
                )
            unit = native_unit(read_for_index(self.landing, self.store, reference.native_representation_id), root)
        else:
            if root.kind != "update_seal":
                raise DomainError("source_reference_family_mismatch", 409)
            observation = self.db.get(updates.UpdateObservation, reference.update_observation_id)
            if observation is None or observation.security_domain != root.security_domain:
                raise DomainError("not_found", 404)
            member = self.db.get(
                updates.SealMember, (root.source_root_id, observation.source_namespace, observation.native_id)
            )
            if (
                member is None
                or member.observation_id != observation.id
                or member.projection_sha256 != observation.projection_sha256
            ):
                raise DomainError("observation_not_in_frozen_root", 409)
            denied = self.updates.current_denials(
                observation.source_namespace,
                [(observation.native_id, observation.source_revision)],
                root.security_domain,
            )
            if denied:
                return DeniedSourceUnit(
                    source_unit_id=observation.id,
                    root=root,
                    source_reference=reference,
                    source_kind="update_title_abstract",
                    native_id=observation.native_id,
                    source_revision=observation.source_revision,
                    projection_sha256=observation.projection_sha256,
                    source_pointer_sha256=digest(observation.locator),
                    source_denials=[DenialReceipt.model_validate_json(json.dumps(item)) for item in denied],
                )
            unit = update_unit(
                self.updates.seal_index_projection(root.source_root_id, reference.update_observation_id), root
            )
        denied = self.updates.current_denials(
            unit.root.source_namespace, [(unit.native_id, unit.source_revision)], unit.root.security_domain
        )
        if denied:
            return DeniedSourceUnit(
                source_unit_id=unit.source_unit_id,
                root=root,
                source_reference=reference,
                source_kind=unit.source_kind,
                native_id=unit.native_id,
                source_revision=unit.source_revision,
                projection_sha256=unit.projection_sha256,
                source_pointer_sha256=unit.source_pointer_sha256,
                source_denials=[DenialReceipt.model_validate_json(json.dumps(item)) for item in denied],
            )
        return unit
