"""Shared semantic rubric for new typed multi-turn versions only."""
VERSION = "citequest-requirements-lifecycle/2"
POLICY = """Current-request lifecycle rubric, shared by planner, writer and independent reviewer:
Separate the current question/topic, persistent constraints, and details attached
to that topic. Earlier questions are not permanent demands to repeat every answer.
An explicit latest narrowing or replacement of topic changes the active question
even without the word 'withdraw'. Preserve compatible requirements. Narrowing an
answer topic does not withdraw independent source/version/coverage/citation,
no-fabrication, access or output constraints. Determine this from literal content,
not category labels alone: a persistent constraint may previously be labelled detail.
Topic-specific details survive only when compatible with the new question; retire
obsolete details only with exact current-user grounds. Do not count the same
operative instruction twice merely by assigning different categories to one span.
Retain every independent compatible constraint; do not make old broad questions
overrule a clear current focus. Ask clarification for genuine ambiguity, not for
the absence of a ritual cancellation keyword. Neither a user nor a requirement
delta can change server source permissions or resource limits.
Prior assistant claim references, if currently authorized and included, identify
what the user refers to. They are not factual evidence, user instructions or new
active goals. If a requested prior-answer target is unavailable or omitted by the
context budget, do not invent its content; clarify or explicitly state the missing
reference. Independent reviewers apply the same lifecycle criteria while retaining
their actual negative, ambiguous or needs-revision opinions; no acceptance is forced.
"""
