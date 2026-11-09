"""Server-owned school profiles and verified source-template selection."""

from __future__ import annotations

from .school_contracts import (
    CURRENT_PROFILE_ID,
    CURRENT_SCOPE,
    LEGACY_PROFILE_ID,
    SOURCE_URL_2025,
    SchoolProfile,
)

TEMPLATE_SOURCE_URL_2025 = (
    "https://www.cs.sjtu.edu.cn/upload/ueditor/file/20261009/34435705/"
    "%E8%AE%BA%E6%96%87%E5%BC%80%E9%A2%98%E6%8A%A5%E5%91%8A-Thesis%2BProposal.docx"
)
# The 2025 attachment was fetched from the official notice and is byte-identical
# to this existing asset (104840 bytes). Reuse it without changing either pin.
_TEMPLATE_ASSET = "assets/export/sjtu-proposal.docx"
_PROFILES = {
    LEGACY_PROFILE_ID: SchoolProfile(),
    CURRENT_PROFILE_ID: SchoolProfile(id=CURRENT_PROFILE_ID, source_url=SOURCE_URL_2025, scope=CURRENT_SCOPE),
}
_TITLES = (
    "国内外研究进展、现状、挑战与意义",
    "参考文献",
    "研究目标、主要研究内容和拟解决的关键问题",
    "研究方法、研究方案及其可行性分析",
    "课题创新点",
    "计划进度与预期成果",
    "相关工作积累与已有研究成果",
)


def default_profile_id() -> str:
    return CURRENT_PROFILE_ID


def get_profile(identifier: str | None) -> SchoolProfile:
    """None preserves the historical interpretation of already stored Briefs."""
    key = LEGACY_PROFILE_ID if identifier is None else identifier
    try:
        return _PROFILES[key].model_copy(deep=True)
    except (KeyError, TypeError):
        raise ValueError("unknown_school_profile") from None


def template_asset(identifier: str | None) -> str:
    get_profile(identifier)
    return _TEMPLATE_ASSET


def public_profiles() -> dict:
    profiles = []
    for identifier, year in ((CURRENT_PROFILE_ID, 2025), (LEGACY_PROFILE_ID, 2024)):
        value = get_profile(identifier)
        attachment = TEMPLATE_SOURCE_URL_2025 if year == 2025 else None
        profiles.append({
            "id": value.id,
            "label": f"上海交通大学计算机学院 · {year}级硕士",
            "source_url": value.source_url,
            "template_sha256": value.template_sha256,
            "template_source_url": attachment,
            "literature_requirement_source_url": attachment or value.source_url,
            "section_titles": list(_TITLES),
            "literature_minimum_han": value.literature_minimum_han,
            "demonstration_minimum_han": value.demonstration_minimum_han,
            "demonstration_maximum_han": value.demonstration_maximum_han,
        })
    return {"default_id": default_profile_id(), "profiles": profiles}
