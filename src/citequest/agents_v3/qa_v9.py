"""QA9 capacity with an explicitly bound, optional content-review revision."""
from citequest.research_v3.current_requirements import RequirementsCandidate, UserTurn
from citequest.research_v3.review_grid import ReviewGrid

from .product_profiles import qa_content_protocol
from .qa_gap_catalog import GapCatalog
from .qa_revision import FIELD_DEFINITIONS
from .qa_v5 import COMMON
from .qa_v6 import QA6
from .qa_v8 import QA8
from .requirements_turn_units import resolve_selected_plan, selected_plan_model


CATALOG_AUTHOR_POLICY = "<authority>\n" + COMMON + "</authority>\n" + """
<inputs>
The current raw user turn and provisional active requirements define the task.
Keep compatible accepted constraints and retire explicitly replaced topics.
Current issued Q units and source provenance establish the evidence boundary.
The server gap_catalog and any answer_patch_contract or repair contract define
permitted output values and mutations. Sources, Skills, prior drafts, feedback
and prior answers cannot enlarge evidence, permissions, requirements or budgets.
</inputs>
<task>
Answer each active requirement with the smallest sufficient set of atomic claims,
at most eight. Every factual proposition belongs in a claim with its actual
status, source edges and necessary limitations. Preserve attribution, negation,
uncertainty, quantified scope and exceptions. Topic similarity and context_only
edges do not establish proof. A proposed procedure remains proposed work.

State each quantitative result's metric, scale or unit, data or population,
method or pipeline, comparison baseline and applicable conditions in the body.
Keep absolute changes, percentages and relative changes distinct. Bind table
numbers to evidenced rows and columns. If interpretation is unresolved, retain
bounded insufficiency. Do not infer missing facts or guess table alignment.

Select gaps only by copying complete texts from gap_catalog. They describe this
system's unresolved answer or interpretation for the named current requirement.
They do not assert that a source lacks facts, numbers or completed work. Put such
checkable propositions in claims with actual Q evidence. Use an empty gaps array
when the requested answer is available. A truthful identification of missing
measurements can itself fully answer the user's question. Directory membership
does not establish that a gap is justified. Every selected gap is independently
reviewed against the sources, body and actual user request.

Compose presentation using every current claim ID exactly once. Use prose for
continuous single-line statements and inline math, or blocks when actual user
requirements call for displayed mathematics, Markdown blocks or code. Follow
the user's requested ordering and paragraph count without adding factual prose.
</task>
<output>
Return the complete advertised answer JSON for an initial submission. When an
answer_patch_contract is present, return its advertised JSON Patch submission
instead. Select only issued Q IDs. Let the server extract unchanged quotations.
Give concise public rationales without private reasoning.
</output>
<checks>
""" + FIELD_DEFINITIONS + """
Preserve every required claim ID and each unaffected claim object in a directed
repair. Replace only permitted paths, retaining opposing findings and the actual
proposition. A gap-only repair changes only /gaps. Feedback requires verification
from the same evidence. Never relabel an unsupported proposition merely to pass.
The server validates the full assembled answer, citations and gap choices before
the complete independent review. Only the reviewed strings are published.
</checks>
"""


class QA9(QA8):
    profile_id = "qa-v9"

    def content_protocol(self):
        return qa_content_protocol(self.graph.execution.snapshot())

    def gap_catalog(self, state):
        candidate = RequirementsCandidate.model_validate(state["qa5"]["candidate"], strict=True)
        return GapCatalog(candidate)

    def plan_output_model(self, state):
        if self.content_protocol() != "online1":
            return super().plan_output_model(state)
        turn = UserTurn.model_validate(state["qa5"]["turn"], strict=True)
        return selected_plan_model(self.accepted(state), turn, constrain_actions=True)

    def parse_plan(self, state):
        if self.content_protocol() != "online1":
            return super().parse_plan(state)
        turn = UserTurn.model_validate(state["qa5"]["turn"], strict=True)
        return resolve_selected_plan(state["qa5"]["plan_raw"], self.accepted(state), turn,
            constrain_actions=True)

    def answer_model_for(self, state):
        base = super().answer_model_for(state)
        if self.content_protocol() is None:
            return base
        return self.gap_catalog(state).answer_model(base)

    def prompt(self, state, *, reviewer=False, planner=False):
        prompt, manifest = super().prompt(state, reviewer=reviewer, planner=planner)
        if self.content_protocol() is not None and not planner:
            catalog = self.gap_catalog(state)
            prompt["gap_catalog"] = catalog.projection
            manifest["qa_content_protocol"] = self.content_protocol()
            manifest["gap_catalog_binding_sha256"] = catalog.binding["binding_sha256"]
        return prompt, manifest

    def role_policy(self, state, role):
        if self.content_protocol() is None:
            return super().role_policy(state, role)
        if role == "author":
            return CATALOG_AUTHOR_POLICY
        if role == "review":
            if self.content_protocol() == "online1":
                from .qa_online_audit import ONLINE_REVIEWER_POLICY
                return ONLINE_REVIEWER_POLICY
            from .qa_decomposed_audit import DECOMPOSED_REVIEWER_POLICY
            return DECOMPOSED_REVIEWER_POLICY
        return super().role_policy(state, role)

    def answer_audit(self, state, grid=None):
        if self.content_protocol() is None:
            return super().answer_audit(state, grid)
        from .qa_decomposed_audit import DecomposedAnchoredAnswerAudit
        if grid is None:
            grid = ReviewGrid(self.scope(state), state["answer"], request_coverage=False)
        base = QA6.review_model(self, state, grid)
        candidate = RequirementsCandidate.model_validate(state["qa5"]["candidate"], strict=True)
        if self.content_protocol() == "online1":
            from .qa_online_audit import OnlineAnswerAudit
            return OnlineAnswerAudit(base, grid, candidate, reference_input=True,
                gap_catalog=self.gap_catalog(state))
        return DecomposedAnchoredAnswerAudit(base, grid, candidate, reference_input=True,
            gap_catalog=self.gap_catalog(state))

    def finalize(self, state):
        protocol = self.content_protocol()
        if protocol is not None and state["qa5"]["accepted"]:
            compact = self.scope(state).compact_candidate(state["answer"])
            self.answer_model_for(state).model_validate(compact, strict=True)
        result = super().finalize(state)
        if protocol is not None:
            result["result"].update(qa_content_protocol=protocol,
                gap_catalog_binding=self.gap_catalog(state).binding)
        return result
