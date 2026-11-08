"""Explicit browser origin policy; HTTPS is mandatory outside loopback development."""

from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class OriginPolicy:
    origin: str
    development: bool = False

    def __post_init__(self) -> None:
        parts = urlsplit(self.origin)
        if (
            parts.username
            or parts.password
            or parts.query
            or parts.fragment
            or parts.path
            or not parts.hostname
            or parts.netloc != parts.netloc.lower()
        ):
            raise ValueError("canonical_origin_required")
        if self.development:
            if parts.scheme != "http" or parts.hostname not in {"localhost", "127.0.0.1"} or not parts.port:
                raise ValueError("explicit_loopback_development_origin_required")
        elif parts.scheme != "https" or parts.hostname in {"localhost", "127.0.0.1"}:
            raise ValueError("production_https_origin_required")
