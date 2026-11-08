"""Offline source plans and IP policy; no network or SSRF end-to-end claim."""
import pytest

from citequest.materials_v2.references import plan_reference, validate_resolved_ips
from citequest.materials_v2.text import MaterialError


@pytest.mark.parametrize("raw,adapter,identifier", [
    ("10.12345/ABC", "doi_crossref", "10.12345/abc"),
    ("doi:10.12345/ABC", "doi_crossref", "10.12345/abc"),
    ("https://doi.org/10.12345/ABC", "doi_crossref", "10.12345/abc"),
    ("https://arxiv.org/abs/2007.00808v1", "arxiv", "2007.00808v1"),
    ("https://arxiv.org/pdf/2007.00808v1.pdf", "arxiv", "2007.00808v1"),
    ("https://arxiv.org/abs/cs/9901001", "arxiv", "cs/9901001"),
])
def test_supported_reference_plans_are_metadata_only(raw, adapter, identifier):
    plan = plan_reference(raw)
    assert plan.adapter == adapter and plan.identifier == identifier
    assert plan.raw_input == raw and plan.follow_redirects is False
    assert plan.can_qa is False and plan.rights_status == "unknown" and plan.requires_network_verification is True


@pytest.mark.parametrize("raw", ["", "http://arxiv.org/abs/2007.00808", "https://arxiv.org.evil.invalid/abs/2007.00808",
                                "https://user:secret@arxiv.org/abs/2007.00808", "https://arxiv.org:444/abs/2007.00808",
                                "https://127.0.0.1/private", "https://[::1]/private", "file:///private",
                                "https://arxiv.org/abs/../../private", "https://arxiv.org/abs/%2e%2e/private",
                                "https://arxiv.org/abs/2007.00808?target=http://127.0.0.1", "https://arxiv.org/abs/2007.00808#page=1",
                                "https://doi.org/10.12345/abc\nHost: evil", "10.12345/a b",
                                "https://doi.org/10.12345/%FF", "https://arxiv.org/abs/2007.00808%ZZ"])
def test_unsupported_or_ambiguous_destinations_explicitly_rejected(raw):
    with pytest.raises(MaterialError):
        plan_reference(raw)


@pytest.mark.parametrize("address", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "0.0.0.0", "224.0.0.1",
                                    "::1", "fe80::1", "fc00::1", "::ffff:127.0.0.1", "not-an-ip", "2001:4860:4860::8888%eth0"])
def test_ip_check_rejects_forbidden_destinations(address):
    with pytest.raises(MaterialError):
        validate_resolved_ips([address])


def test_all_resolved_addresses_must_be_allowed():
    assert validate_resolved_ips(["8.8.8.8", "2001:4860:4860::8888"]) == ("8.8.8.8", "2001:4860:4860::8888")
    with pytest.raises(MaterialError):
        validate_resolved_ips(["8.8.8.8", "127.0.0.1"])
    with pytest.raises(MaterialError):
        validate_resolved_ips([])
