"""Typed source references; a logical index unit is not a metadata table row.

These are internal reader DTOs. Constructing one does not adopt/publish a source;
the SQL reader must re-check the actual source root, members, ACL and denials.
"""

from __future__ import annotations

import hashlib
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from citequest.data_v2.native_index import NativeIndexProjection
from citequest.data_v2.schemas import IndexProjection
from citequest.updates_v2.schemas import UpdateIndexProjection

from .projection import BUILDER_VERSION, metadata_text

SHA = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class MetadataReference(Strict):
    kind: Literal["metadata_projection"] = "metadata_projection"
    metadata_projection_id: UUID


class NativeReference(Strict):
    kind: Literal["native_representation"] = "native_representation"
    native_representation_id: SHA


class UpdateReference(Strict):
    kind: Literal["update_observation"] = "update_observation"
    update_observation_id: UUID


SourceReference = Annotated[MetadataReference | NativeReference | UpdateReference, Field(discriminator="kind")]


class DenialReceipt(Strict):
    source_namespace: str
    native_id: str
    source_revision: str
    reason: str
    scope: Literal["this_source_all_versions", "this_source_version"]
    observation_id: UUID
    blocks_current_default_evidence: Literal[True]


class SourceRoot(Strict):
    kind: Literal["metadata_snapshot", "native_snapshot", "update_seal"]
    source_root_id: UUID
    source_namespace: Annotated[str, Field(min_length=1, max_length=100)]
    security_domain: Annotated[str, Field(min_length=1, max_length=100)]
    representation_profile: Annotated[str, Field(min_length=1, max_length=100)]
    source_manifest_sha256: SHA
    source_seal_sha256: SHA | None
    membership_sha256: SHA | None
    policy_sha256: SHA
    evaluation_sha256: SHA
    state: Literal["landing", "staged", "disabled"]
    profile_adopted: bool
    publication_eligible: bool
    catalog_name: Literal["standard_metadata", "application_papers", "managed_metadata"]

    @model_validator(mode="after")
    def mapping(self) -> SourceRoot:
        expected = {
            "metadata_snapshot": ("standard_metadata", "litsearch-clean-exact/1"),
            "native_snapshot": ("application_papers", "litsearch-s2orc-native-text/1"),
            "update_seal": ("managed_metadata", "cq.protocol-metadata/1"),
        }
        if (self.catalog_name, self.representation_profile) != expected[self.kind]:
            raise ValueError("source_root_profile_catalog_mismatch")
        if self.publication_eligible and (
            not self.profile_adopted or self.state != "staged" or self.source_seal_sha256 is None
        ):
            raise ValueError("source_adoption_seal_required_before_publication")
        return self


class CanonicalSourceUnit(Strict):
    source_unit_id: UUID
    root: SourceRoot
    source_reference: SourceReference
    source_kind: Literal["standard_title_abstract", "s2orc_native", "update_title_abstract"]
    native_id: Annotated[str, Field(min_length=1, max_length=512)]
    source_revision: Annotated[str, Field(min_length=1, max_length=100)] | None
    projection_sha256: SHA
    source_pointer_sha256: SHA
    unit_text_sha256: SHA
    literal_text: str
    searchable_text_present: bool
    year: Annotated[int, Field(ge=1000, le=2200)] | None
    work_id: UUID | None
    work_version_id: UUID | None
    coordinate_system: Literal["unicode_codepoints_in_metadata_recipe", "unicode_codepoints_in_native_literal"]
    builder_version: str
    original_source_dto: dict[str, Any]
    source_denied: bool = False
    index_staging_is_not_QA_permission: Literal[True] = True

    @model_validator(mode="after")
    def bindings(self) -> CanonicalSourceUnit:
        mapping = {
            "standard_title_abstract": (
                "metadata_snapshot",
                "metadata_projection",
                "unicode_codepoints_in_metadata_recipe",
            ),
            "s2orc_native": ("native_snapshot", "native_representation", "unicode_codepoints_in_native_literal"),
            "update_title_abstract": ("update_seal", "update_observation", "unicode_codepoints_in_metadata_recipe"),
        }
        if (self.root.kind, self.source_reference.kind, self.coordinate_system) != mapping[self.source_kind]:
            raise ValueError("typed_source_reference_cannot_be_relabelled")
        if hashlib.sha256(self.literal_text.encode()).hexdigest() != self.unit_text_sha256:
            raise ValueError("source_unit_literal_hash_mismatch")
        if self.searchable_text_present != bool(self.literal_text.strip()):
            raise ValueError("source_unit_text_presence_mismatch")
        if (
            isinstance(self.source_reference, MetadataReference)
            and self.source_reference.metadata_projection_id != self.source_unit_id
        ):
            raise ValueError("metadata_unit_ID_must_be_real_projection_ID")
        if (
            isinstance(self.source_reference, UpdateReference)
            and self.source_reference.update_observation_id != self.source_unit_id
        ):
            raise ValueError("update_unit_ID_must_be_real_observation_ID")
        if self.source_kind == "s2orc_native" and (
            self.work_id is None
            or self.work_version_id is None
            or self.projection_sha256 != self.source_reference.native_representation_id
        ):
            raise ValueError("native_paper_lineage_required")
        return self


class DeniedSourceUnit(Strict):
    """Keep a denied member without inventing empty/verified original text."""

    source_unit_id: UUID
    root: SourceRoot
    source_reference: SourceReference
    source_kind: Literal["standard_title_abstract", "s2orc_native", "update_title_abstract"]
    native_id: str
    source_revision: str | None
    projection_sha256: SHA
    source_pointer_sha256: SHA
    source_denials: Annotated[list[DenialReceipt], Field(min_length=1)]
    source_denied: Literal[True] = True
    source_text_withheld: Literal[True] = True
    source_bytes_not_claimed_reverified: Literal[True] = True

    @model_validator(mode="after")
    def exact_scope(self) -> DeniedSourceUnit:
        expected = {
            "standard_title_abstract": ("metadata_snapshot", "metadata_projection"),
            "s2orc_native": ("native_snapshot", "native_representation"),
            "update_title_abstract": ("update_seal", "update_observation"),
        }
        if (self.root.kind, self.source_reference.kind) != expected[self.source_kind]:
            raise ValueError("denied_reference_family_mismatch")
        if any(
            item.source_namespace != self.root.source_namespace
            or item.native_id != self.native_id
            or item.source_revision not in {"*", self.source_revision}
            for item in self.source_denials
        ):
            raise ValueError("denial_does_not_match_this_exact_source_lineage")
        return self


def metadata_unit(projection: IndexProjection, root: SourceRoot) -> CanonicalSourceUnit:
    if (
        root.kind != "metadata_snapshot"
        or projection.snapshot_id != root.source_root_id
        or projection.security_domain != root.security_domain
        or projection.source_namespace != root.source_namespace
    ):
        raise ValueError("metadata_reader_root_identity_mismatch")
    text = metadata_text(projection.title, projection.abstract)
    return CanonicalSourceUnit(
        source_unit_id=projection.projection_id,
        root=root,
        source_reference=MetadataReference(metadata_projection_id=projection.projection_id),
        source_kind="standard_title_abstract",
        native_id=projection.native_id,
        source_revision=projection.source_revision,
        projection_sha256=projection.projection_sha256,
        source_pointer_sha256=projection.source_pointer_sha256,
        unit_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        literal_text=text,
        searchable_text_present=projection.searchable_text_present,
        year=projection.year,
        work_id=projection.work_id,
        work_version_id=None,
        coordinate_system="unicode_codepoints_in_metadata_recipe",
        builder_version=BUILDER_VERSION,
        original_source_dto=projection.model_dump(mode="json"),
    )


def native_unit(projection: NativeIndexProjection, root: SourceRoot) -> CanonicalSourceUnit:
    if (
        root.kind != "native_snapshot"
        or projection.snapshot_id != root.source_root_id
        or projection.security_domain != root.security_domain
        or projection.source_namespace != root.source_namespace
    ):
        raise ValueError("native_reader_root_identity_mismatch")
    return CanonicalSourceUnit(
        source_unit_id=projection.projection_id,
        root=root,
        source_reference=NativeReference(native_representation_id=projection.representation_id),
        source_kind="s2orc_native",
        native_id=projection.native_id,
        source_revision=projection.source_revision,
        projection_sha256=projection.projection_sha256,
        source_pointer_sha256=projection.source_pointer_sha256,
        unit_text_sha256=projection.source_text_sha256,
        literal_text=projection.literal_text,
        searchable_text_present=bool(projection.literal_text.strip()),
        year=projection.year,
        work_id=projection.work_id,
        work_version_id=projection.work_version_id,
        coordinate_system="unicode_codepoints_in_native_literal",
        builder_version="native-literal-source/1-whitespace-boundary-attachment",
        original_source_dto=projection.model_dump(mode="json"),
    )


def update_unit(projection: UpdateIndexProjection, root: SourceRoot) -> CanonicalSourceUnit:
    if (
        root.kind != "update_seal"
        or projection.seal_id != root.source_root_id
        or projection.security_domain != root.security_domain
        or projection.source_namespace != root.source_namespace
        or projection.seal_sha256 != root.source_seal_sha256
        or projection.membership_sha256 != root.membership_sha256
        or projection.publication_eligible != root.publication_eligible
    ):
        raise ValueError("update_reader_root_identity_mismatch")
    text = metadata_text(projection.title, projection.abstract)
    return CanonicalSourceUnit(
        source_unit_id=projection.unit_id,
        root=root,
        source_reference=UpdateReference(update_observation_id=projection.observation_id),
        source_kind="update_title_abstract",
        native_id=projection.native_id,
        source_revision=projection.source_revision,
        projection_sha256=projection.projection_sha256,
        source_pointer_sha256=projection.source_pointer_sha256,
        unit_text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        literal_text=text,
        searchable_text_present=bool(text.strip()),
        year=projection.year,
        work_id=projection.work_id,
        work_version_id=None,
        coordinate_system="unicode_codepoints_in_metadata_recipe",
        builder_version="protocol-title-newline-abstract/1-whitespace-boundary-attachment",
        original_source_dto=projection.model_dump(mode="json"),
    )
