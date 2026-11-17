"""Atomic stance with exact goal identities; original450s resource scope."""
from .stance_v2 import StanceV2


class StanceV3(StanceV2):
    profile_id = "stance-v3"
    strict_review_identity = True
    strict_requirement_identity = True
    use_dialogue_references = True
