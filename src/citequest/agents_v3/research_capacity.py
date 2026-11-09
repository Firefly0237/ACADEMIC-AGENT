"""Planning-time observations of the existing research/School resource gates.

This does not reserve resources or authorize dispatch. The normal Execution and
post-planner admission remain authoritative; no task limit is enlarged here.
"""
from __future__ import annotations

from .contracts import AgentError

SCHOOL_NORMAL_CALLS = 14 * 2 + 2
RESEARCH_OUTPUT_CAP = 2048


def is_experiment_task(task: dict) -> bool:
    return (task.get("kind") == "research" and task["profile"].get("profile_id") == "research-v6"
        and task["brief"].get("research_action") == "experiment")


def school_model_calls(task: dict) -> int:
    return 0 if is_experiment_task(task) else SCHOOL_NORMAL_CALLS


def literature_result_limit(task: dict) -> int:
    if task["profile"].get("profile_id") not in {"research-v5", "research-v6"}:
        return 2
    from citequest.research_v3.selections import reference_selections
    selected = reference_selections(task["brief"])
    return min(8, len(selected)) if selected else 8


def research_v5_capacity(task: dict, questions: int, experiments: int, *,
        outline_calls: int = 0, remaining_revisions: int = 2) -> dict:
    """Admit the complete new-profile path, including shared repair capacity.

    Each RQ searches once and reads at most L current references. Publication
    and School initial/final admission reread each unique ref; q*L is a safe
    upper bound even when repeated queries issue different source chunks. A shared credit can
    repair an outline/literature pair, a plan target plus all RQ/global reviews,
    or one School section plus its full review and cross review. It is reserved
    once across those stages, never separately replenished for each child.
    """
    if (type(questions) is not int or not 1 <= questions <= 8
            or type(experiments) is not int or not questions <= experiments <= 2 * questions
            or type(outline_calls) is not int or not 0 <= outline_calls <= 2
            or type(remaining_revisions) is not int or not 0 <= remaining_revisions <= 2):
        raise AgentError("research_v5_capacity_dimensions_invalid")
    normal_models, _ = planning_operations(questions, experiments)
    repair_calls = remaining_revisions * max(questions + 2, 3)
    school_calls = school_model_calls(task)
    models = outline_calls + normal_models + school_calls + repair_calls
    result_limit = literature_result_limit(task)
    experiment_only = is_experiment_task(task)
    read_multiplier = 2 if experiment_only else 4
    tools = questions * (1 + read_multiplier * result_limit)
    output = models * RESEARCH_OUTPUT_CAP
    profile = task["profile"]
    reasons = []
    from citequest.research_v3.selections import reference_selections
    selected_count = len(reference_selections(task["brief"]))
    if selected_count > 8:
        reasons.append("research_v5_selected_reference_count_exceeds_explicit_eight_source_capacity")
    for key, required in (("model_calls", models), ("tool_calls", tools), ("output_tokens", output)):
        if profile[key] - task[key] < required:
            reasons.append("research_and_school_remaining_" + key + "_insufficient")
    if profile["per_call_output"] < RESEARCH_OUTPUT_CAP:
        reasons.append("research_and_school_per_call_output_below_2048")
    return {"protocol": "citequest-experiment-admission/1" if experiment_only else "citequest-research-school-admission/1", "questions": questions,
        "experiments": experiments, "outline_calls_remaining": outline_calls,
        "evidence_results_per_question": result_limit,
        "selected_reference_documents": selected_count,
        "maximum_unique_evidence_refs": questions * result_limit,
        "tool_formula": f"questions*(1+{read_multiplier}*evidence_results_per_question)",
        "tool_breakdown": {"search": questions, "initial_reads": questions * result_limit,
            "research_publication_reads": questions * result_limit,
            "school_initial_and_final_reads": 0 if experiment_only else 2 * questions * result_limit,
            "revision_additional_reads": 0},
        "source_guard_operations": "existing_fresh_scope_resource_reads_outside_tool_counter; active_deadline_and_receipts_still_apply",
        "school_normal_model_calls": school_calls,
        "shared_revision_credits_remaining": remaining_revisions,
        "shared_revision_model_calls": repair_calls, "required_model_calls": models,
        "required_tool_calls": tools, "reserved_output_tokens": output,
        "per_call_output_cap": RESEARCH_OUTPUT_CAP, "admitted": not reasons, "reasons": reasons,
        "remaining_checks": ["actual_input_usage", "complete_wire_context_before_dispatch",
            "same_task_deadline", "current_source_Brief_memory_and_skill_guards", "independent_content_review"],
        "quality_or_latency_guaranteed": False}


def reviewed_outline_capacity(task: dict, *, remaining_revisions: int) -> dict:
    pairs = []
    for questions in range(1, 9):
        for experiments in range(questions, 2 * questions + 1):
            candidate = research_v5_capacity(task, questions, experiments,
                outline_calls=2, remaining_revisions=remaining_revisions)
            if candidate["admitted"]:
                pairs.append({"research_questions": questions, "experiments": experiments})
    school_calls = school_model_calls(task)
    return {"protocol": "citequest-experiment-admission/1" if is_experiment_task(task) else "citequest-research-school-admission/1",
        "admissible_model_tool_and_output_scope_pairs": pairs,
        "school_normal_model_calls_reserved": school_calls,
        "school_normal_output_tokens_reserved": school_calls * RESEARCH_OUTPUT_CAP,
        "evidence_results_per_question": literature_result_limit(task),
        "shared_revision_credits_remaining": remaining_revisions,
        "rule": "Choose the smallest complete decomposition. Proposed design can be formulated; never invent user facts. No pair means capacity limitation, not missing user input.",
        "actual_input_wire_context_and_content_checks_remain_required": True}


def planning_operations(questions: int, experiments: int) -> tuple[int, int]:
    # Two literature roles/RQ, one methods role/experiment, full RQ review/RQ,
    # then one cross-plan review. The outline call itself is already consumed.
    return 3 * questions + experiments + 1, 5 * questions


def outline_capacity(task: dict, writing_reserve: int) -> dict:
    profile = task["profile"]
    remaining_models = max(0, profile["model_calls"] - task["model_calls"] - 1)
    remaining_tools = max(0, profile["tool_calls"] - task["tool_calls"])
    pairs = []
    for questions in range(1, 9):
        for experiments in range(questions, 2 * questions + 1):
            models, tools = planning_operations(questions, experiments)
            if models + writing_reserve <= remaining_models and tools <= remaining_tools:
                pairs.append({"research_questions": questions, "experiments": experiments})
    return {
        "scope": "observed_existing_model_and_tool_admission_not_dispatch_authority",
        "remaining_model_calls_after_this_outline": remaining_models,
        "remaining_tool_calls": remaining_tools,
        "remaining_input_tokens": max(0, profile["input_tokens"] - task["input_tokens"]),
        "remaining_output_tokens_before_this_outline": max(0, profile["output_tokens"] - task["output_tokens"]),
        "writing_model_calls_to_leave_unused": writing_reserve,
        "planning_model_formula_after_outline": "3*research_questions + experiments + 1",
        "planning_tool_formula": "5*research_questions",
        "admissible_model_and_tool_scope_pairs": pairs,
        "token_and_context_fit_not_guaranteed_by_scope_pairs": True,
        "rule": "Choose the smallest decomposition that covers all confirmed objectives. If no feasible pair covers the request, explicitly report uncovered requests; do not silently replace the research objective.",
    }
