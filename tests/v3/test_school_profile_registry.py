"""Source pins and legacy serialization without a writer or renderer run."""

import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest
from pydantic import ValidationError

from citequest.research_v3.school_contracts import (
    CURRENT_PROFILE_ID,
    LEGACY_PROFILE_ID,
    SOURCE_URL_2025,
    SchoolDocument,
    SchoolProfile,
)
from citequest.research_v3.school_profiles import default_profile_id, get_profile, public_profiles, template_asset

ROOT = Path(__file__).resolve().parents[2]


def test_registry_preserves_legacy_profile_and_selects_new_source():
    historical = json.loads((ROOT / "contracts/v3/school/sjtu-scs-2024-master-cn-capacity-demo-v3.json").read_text("utf8"))
    assert SchoolProfile().model_dump(mode="json") == historical
    assert get_profile(None).model_dump(mode="json") == historical
    assert get_profile(LEGACY_PROFILE_ID).model_dump(mode="json") == historical
    assert default_profile_id() == CURRENT_PROFILE_ID
    current = get_profile(default_profile_id())
    assert current.source_url == SOURCE_URL_2025
    assert not current.demonstration_length_is_school_rule
    assert current.literature_minimum_han == 5000
    with pytest.raises(ValueError, match="unknown_school_profile"):
        get_profile("unknown")
    with pytest.raises(ValidationError):
        SchoolProfile(id=CURRENT_PROFILE_ID)
    with pytest.raises(ValidationError, match="frozen_instance"):
        current.scope = "changed"
    assert get_profile(default_profile_id()) == current
    public = public_profiles()
    assert public["default_id"] == CURRENT_PROFILE_ID
    assert len(public["profiles"]) == 2
    for item in public["profiles"]:
        assert len(item["section_titles"]) == 7
        assert "template_source_url" in item and "literature_requirement_source_url" in item


def test_verified_shared_template_has_original_slots_and_school_word_rule():
    profile = get_profile(CURRENT_PROFILE_ID)
    path = ROOT / template_asset(CURRENT_PROFILE_ID)
    assert sha256(path.read_bytes()).hexdigest() == profile.template_sha256
    assert template_asset(None) == template_asset(CURRENT_PROFILE_ID)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with ZipFile(path) as archive:
        body = ET.fromstring(archive.read("word/document.xml")).find("w:body", ns)
    assert len(body) == 60
    literature = "".join(item.text or "" for item in body[19].findall(".//w:t", ns))
    assert "硕士生不少于5,000汉字" in literature
    for index in (21, 27, 32, 37, 43, 47, 52):
        assert body[index].tag == "{" + ns["w"] + "}p"
        assert not body[index].findall(".//w:permStart", ns)
        assert not body[index].findall(".//w:permEnd", ns)


def test_document_version_remains_bound_to_profile():
    for identifier, version in ((LEGACY_PROFILE_ID, "3"), (CURRENT_PROFILE_ID, "4")):
        value = SimpleNamespace(profile=get_profile(identifier), schema_version="citequest-school-document/" + version)
        assert SchoolDocument.profile_version(value) is value
        value.schema_version = "citequest-school-document/" + ("4" if version == "3" else "3")
        with pytest.raises(ValueError, match="profile_version_mismatch"):
            SchoolDocument.profile_version(value)
