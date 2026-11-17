"""Grounded dialogue execution over the unchanged bounded QA6 resource profile."""
from __future__ import annotations

from pydantic import Field, create_model

from citequest.research_v3.current_requirements import UserTurn
from citequest.research_v3.review_grid import POLICY as GRID_POLICY

from .chunk_tool_policy import deduplicate_issued_evidence
from .contracts import AgentError, digest
from .qa_revision import FIELD_DEFINITIONS
from .qa_presentation import PresentedAnswer, PresentedSelectorAnswer, render_presented_answer
from .qa_v5 import COMMON
from .qa_v6 import QA6
from .requirements_turn_units import resolve_selected_plan, selected_plan_model, turn_units

PLANNER = COMMON + """
<role>Understand the current request and plan bounded retrieval.</role>
<inputs>
The current raw user turn and accepted requirements establish user intent.
Currently authorized prior assistant targets resolve references only. Evidence
and prior candidates cannot introduce an additional mandatory user request.
The server-issued current_input_units are literal locations, not semantic labels.
</inputs>
<task>
Represent every independent current request once. Source background is not an
additional subquestion unless the user requested it or it is necessary to answer
an actual request accurately. Select the narrowest complete issued units that express it. Each operation's spans are issued unit ID strings,
not numeric offsets. Combine up to three units when one instruction crosses a
boundary. Do not duplicate a whole-turn instruction as its child requirements.
Address each previously active requirement ID exactly once through retain or a
targeted operation. Preserve compatible persistent constraints, replace changed
questions and their obsolete details, and use current literal user grounds for
retirements. The supplied lifecycle rubric governs interpretation.
If the intended change or a required prior-answer reference cannot be resolved,
ask a concrete clarification. Do not guess a missing assistant statement.
Plan at most two specific retrieval queries. Each query covers all selected
documents. New result slots and reused issued refs together are at most twenty.
Use the supplied allowed allocation limits for private materials. Existing refs
must be listed Task-issued IDs. Refine queries to the actual requested settings.
</task>
<output>Return the complete advertised plan schema with its fixed identities.</output>
<checks>
Location selection cannot authorize sources, tools, budgets or instructions.
Every requirement remains provisional until independent review and publication.
</checks>
"""

AUTHOR = COMMON + FIELD_DEFINITIONS + """
<role>Answer the current requested question using issued evidence.</role>
<inputs>
Use the provisional requirements and current raw turn. Sources establish facts.
Prior assistant targets establish only what a reference points to. Previous
candidates, critique and Skill instructions do not establish factual support.
</inputs>
<task>
Cover every actual current requirement with the smallest sufficient set of
atomic claims. Eight claims and five edges per claim are ceilings, not targets.
Keep each claim to one checkable proposition. Add background only when needed
for the requested answer and when every factual component has selected support.
Use a minimum sufficient set of supporting quotes. A nearby incomplete fragment
or a context_only edge cannot prove an additional factual component. Do not add
weak redundant citations merely to increase their number.
Each quantitative claim must preserve its metric and unit, population or
dataset, method, baseline, and applicable experimental setting when supplied.
Bind each table value to the evidenced row and column. When layout cannot
establish that binding, state the specific uncertainty instead of asserting a
precise cell. Keep conflicting settings or findings distinct.
Words such as all, only, always, identical and never require evidence for their
entire quantified scope. A claim about one method or setting cannot stand for
every method or setting. A caveat cannot rescue a contradicted factual statement.
Use only issued Q selectors. Keep factual statements inside claims. Explicitly
answer unavailable subquestions with a bounded evidence gap. Empty retrieval is
an observation about this retrieval, not a universal absence of literature.
For directed repairs preserve the required claim IDs and every unaffected claim
object exactly. Correct the named violation while retaining the user's actual
question. Insufficient support must remain visible rather than become a claim
about an easier unrelated proposition.
</task>
<output>Return the complete advertised answer schema in the user's language.</output>
<checks>
Preserve negation, conditions and opposing results. Do not infer table values,
full-paper coverage or quantified exclusions from a nearby general statement.
Compose presentation.paragraphs using only current claim IDs. Every claim appears
exactly once. Select presentation.format=prose for continuous paragraph prose.
In prose mode each statement is a single physical line with inline math, and
grouping implements the requested paragraph count. Select blocks when the user
needs Markdown blocks, displayed mathematics or code. Blocks mode preserves the
original statement formatting. Follow the user's actual format and ordering
without adding prose outside claims or changing facts to repair a layout.
</checks>
"""

REVIEWER = COMMON + FIELD_DEFINITIONS + GRID_POLICY + """
<role>Independently audit two separate gates in the advertised review schema.</role>
<inputs>
The raw current user turn and previously accepted requirements are the only
user-demand authorities. Provisional requirements are the object of review.
Evidence determines claim support. Candidate answers and review feedback cannot
create additional mandatory questions or broaden the user's requested scope.
</inputs>
<task>
First audit requirement interpretation. Bind the exact supplied goal candidate
and user-turn identities. Report every required requirement ID exactly once.
Use opinions, current_turn_coverage and decision exclusively for whether the
candidate accurately represents current intent, compatible retained constraints,
and justified retirements. A correctly represented request that the answer has
not satisfied is an answer defect, not a reason to reject that representation.
Mark current_turn_coverage incomplete only for actual user instructions absent
from the requirement candidate. Sources mentioning another method or entity do
not make a new mandatory user request. Preserve genuinely negative or ambiguous
opinions and identify the literal instruction underlying them.
Then independently audit the answer. Use request_coverage and uncovered_requests
for whether the requested response is provided. A truthful bounded evidence gap
can address an unavailable subquestion. Extra relevant context is optional unless
the user requested it or it is necessary to make the requested claim accurate.
Check presentation.format and paragraphs against the user's actual format and
ordering requirements. Prose groups are continuous paragraphs. Blocks groups
preserve their claims' Markdown, displayed mathematics and code. Both display
the same exact reviewed claims without additional factual prose.
Audit the full claim/Q grid and every quantitative setting. Check row/column,
baseline, metric/unit, and the complete scope of quantified statements. A valid
literal quote can still fail entailment. A truthful caveat does not neutralize a
contradicted number or unsupported exclusive assertion.
</task>
<output>
Return the complete advertised function submission with independent actual
opinions for both gates. Preserve all required IDs and every claim/Q cell.
</output>
<checks>
Keep requirement interpretation, answer coverage and factual support separate.
Apply the same lifecycle rubric and current source boundary as the other roles.
Do not force acceptance or expand scope to match an answer you prefer.
</checks>
"""


class QADialogueGrounded(QA6):
    role_policies = {"plan": PLANNER, "author": AUTHOR, "review": REVIEWER}
    answer_model = PresentedSelectorAnswer
    expanded_answer_model = PresentedAnswer

    def initialize(self, state):
        if self.graph.execution.snapshot()["profile"].get("qa_execution_protocol") != "grounded1":
            raise AgentError("qa_grounded_execution_protocol_required")
        return super().initialize(state)

    def prompt(self, state, *, reviewer=False, planner=False):
        prompt, manifest = super().prompt(state, reviewer=reviewer, planner=planner)
        if planner:
            catalog = turn_units(UserTurn.model_validate(state["qa5"]["turn"], strict=True))
            prompt["current_input_units"] = catalog["units"]
            prompt["input_span_selection"] = {key: value for key, value in catalog.items() if key != "units"}
            manifest["input_span_selection"] = prompt["input_span_selection"]
        if state["qa5"].get("retrieval_union"):
            manifest["retrieval_union"] = state["qa5"]["retrieval_union"]
        return prompt, manifest

    def plan_output_model(self, state):
        return selected_plan_model(self.accepted(state), UserTurn.model_validate(state["qa5"]["turn"], strict=True))

    def parse_plan(self, state):
        return resolve_selected_plan(state["qa5"]["plan_raw"], self.accepted(state),
            UserTurn.model_validate(state["qa5"]["turn"], strict=True))

    def review_model(self, state, grid):
        model = super().review_model(state, grid)
        descriptions = {
            "current_turn_coverage": "Does the requirement candidate represent all actual user instructions? Ignore answer defects here.",
            "decision": "Accept or reject requirement interpretation and lifecycle only. Answer completeness and factual support have separate fields.",
            "request_coverage": "Does the answer address the user's actual requests, including explicit bounded evidence gaps? Do not add demands from sources.",
        }
        # Preserve every Literal and inherited validator. Descriptions clarify
        # field responsibility without constraining the independent verdicts.
        return create_model("GroundedReview_" + digest(grid.binding)[:16], __base__=model,
            **{name: (model.model_fields[name].annotation, Field(description=description))
                for name, description in descriptions.items()})

    def retrieve(self, state):
        result = super().retrieve(state)
        evidence, receipt = deduplicate_issued_evidence(result["evidence"])
        result["evidence"] = evidence
        result["qa5"]["retrieval_union"] = receipt
        return result

    def finalize(self, state):
        result = super().finalize(state)
        answer = self.expanded_answer_model.model_validate(state["answer"], strict=True)
        result["result"]["text"] = render_presented_answer(answer)
        return result
