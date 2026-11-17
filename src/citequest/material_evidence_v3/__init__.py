"""Current-ACL evidence from owner-uploaded, genuinely parsed materials."""

from .service import MaterialEvidenceService, service_from_config

__all__ = ["MaterialEvidenceService", "service_from_config"]
