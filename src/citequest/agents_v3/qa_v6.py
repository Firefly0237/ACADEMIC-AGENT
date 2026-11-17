"""Explicit multi-turn resource version, sharing the official QA5 protocol."""
from .qa_v5 import QA5


class QA6(QA5):
    profile_id = "qa-v6"
    strict_review_identity = True
    strict_requirement_identity = True
    use_dialogue_references = True
