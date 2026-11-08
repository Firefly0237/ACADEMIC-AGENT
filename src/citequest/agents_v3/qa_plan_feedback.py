"""Read-only, aggregate feedback for the versioned QA planning protocol.

As with server-side form validation, independently check fields before batch
constraints so an invalid child cannot hide another determinable error. This
is diagnostic only. The original full plan and ledger validators remain the
authority, including issued-reference authorization and semantic review.
"""
from __future__ import annotations

from typing import get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError

from citequest.research_v3.current_requirements import RequirementsState, UserTurn
from citequest.research_v3.qa_v5_contracts import RetrievalIntent
from citequest.research_v3.typed_feedback import strict_object

from .contracts import AgentError
from .requirements_turn_units import selected_plan_model

_INVALID = object()
_FIELDS = frozenset({"requirement_delta", "action", "queries", "query", "limit", "existing_ref_ids",
    "version", "parent_sha256", "authority_sha256", "turn_id", "turn_sha256", "retain_ids",
    "operations", "target_ids", "category", "spans", "intent", "clarification"})
_FIELD_ERRORS = frozenset({"missing", "extra_forbidden", "literal_error", "string_type", "int_type",
    "list_type", "model_type", "model_attributes_type", "string_too_short", "string_too_long",
    "string_pattern_mismatch", "too_short", "too_long", "greater_than_equal", "less_than_equal",
    "value_error"})
_RULES = {
    "qa_plan_field_invalid": "Meet the original advertised field schema. Do not invent or substitute server-issued identities or evidence references.",
    "qa_plan_operation_targets_unique": "Each operation must name each target requirement at most once.",
    "qa_plan_operation_target_shape": "An add has no old targets. A supersede or withdraw has at least one old target.",
    "qa_plan_operation_category": "A withdraw must have category=null. Add and supersede must have a declared non-null category.",
    "qa_plan_active_addresses": "Address every currently active requirement exactly once across retain_ids and all operation target_ids. Retired or unknown requirements cannot be targets.",
    "qa_plan_clarification_text": "needs_clarification requires nonempty clarification text. clear requires no nonempty clarification text.",
    "qa_plan_clarification_operations": "A needs_clarification delta cannot mutate requirements.",
    "qa_plan_refs_unique": "existing_ref_ids must be unique. Reuse only references actually issued to this task.",
    "qa_plan_action_intent": "clarify requires needs_clarification. Source work requires clear intent.",
    "qa_plan_clarify_no_evidence": "clarify must have no queries and no existing_ref_ids.",
    "qa_plan_retrieve_queries": "retrieve requires at least one query.",
    "qa_plan_result_slots": "All new queries and reused evidence share one maximum of 20 result slots. Replan explicitly within that bound. The server will not clip results or drop reused evidence.",
    "qa_plan_verify_existing_shape": "verify_existing requires existing issued references and no queries.",
    "requirements_initial_user_request_not_represented": "The initial clear user request must be represented by requirement operations.",
    "requirements_turn_already_processed_or_out_of_order": "Use the new server-issued user turn. A processed or older turn cannot be applied again.",
}


def plan_preflight(raw: str, accepted: RequirementsState, turn: UserTurn) -> dict | None:
    """Collect registered deterministic errors without changing any input.

    None is not approval. The caller must still run its complete original
    parser, transition, source authorization and budget checks. In particular,
    this helper has no issued-reference set and cannot authorize unknown refs.
    """
    try:
        value = strict_object(raw)
    except AgentError:
        # Preserve the existing strict JSON failure and its repair policy.
        return None

    model = selected_plan_model(accepted, turn)
    delta_model = model.model_fields["requirement_delta"].annotation
    operation_model = get_args(delta_model.model_fields["operations"].annotation)[0]
    issues: list[dict] = []

    def add(code: str, path: list, **parameters) -> None:
        issue = {"code": code, "path": path, "rule": _RULES[code]}
        if parameters:
            issue["parameters"] = parameters
        if issue not in issues:
            issues.append(issue)

    def fields(schema: type[BaseModel], candidate, path: list) -> dict:
        if not isinstance(candidate, dict):
            add("qa_plan_field_invalid", path, validation="model_type")
            return {}
        if candidate.keys() - schema.model_fields.keys():
            add("qa_plan_field_invalid", [*path, "unrecognized_field"], validation="extra_forbidden")
        result = {}
        for name, field in schema.model_fields.items():
            if name not in candidate:
                if field.is_required():
                    add("qa_plan_field_invalid", [*path, name], validation="missing")
                continue
            try:
                result[name] = TypeAdapter(field.rebuild_annotation()).validate_python(candidate[name], strict=True)
            except ValidationError as error:
                result[name] = _INVALID
                for row in error.errors(include_input=False, include_url=False, include_context=False):
                    location = [*path, name, *row["loc"]]
                    # Original operation/delta after-validators are aggregated
                    # explicitly below, even when another child field fails.
                    operation_shape = len(location) == 3 and location[:2] == ["requirement_delta", "operations"]
                    if row["type"] == "value_error" and (location == ["requirement_delta"] or operation_shape):
                        continue
                    if row["type"] in _FIELD_ERRORS:
                        safe_path = [part if type(part) is int or part in _FIELDS else "unrecognized_field"
                                     for part in location]
                        add("qa_plan_field_invalid", safe_path, validation=row["type"])
                if get_origin(field.annotation) is list:
                    # A list's own length error must not conceal independent
                    # batch checks. Retain only originally typed elements, not
                    # a coerced or repaired value. The length issue stays.
                    try:
                        result[name] = TypeAdapter(field.annotation).validate_python(candidate[name], strict=True)
                    except ValidationError:
                        pass
        return result

    root = fields(model, value, [])
    delta_raw = value.get("requirement_delta")
    delta = fields(delta_model, delta_raw, ["requirement_delta"])
    operations_raw = delta_raw.get("operations") if isinstance(delta_raw, dict) else None
    operations = []
    if isinstance(operations_raw, list):
        for index, operation_raw in enumerate(operations_raw):
            path = ["requirement_delta", "operations", index]
            operation = fields(operation_model, operation_raw, path)
            operations.append(operation)
            action, targets, category = (operation.get(name, _INVALID) for name in ("action", "target_ids", "category"))
            if targets is not _INVALID and len(targets) != len(set(targets)):
                add("qa_plan_operation_targets_unique", [*path, "target_ids"])
            if action is not _INVALID and targets is not _INVALID and ((action == "add") != (not targets)):
                add("qa_plan_operation_target_shape", [*path, "target_ids"])
            if action is not _INVALID and category is not _INVALID and ((action == "withdraw") != (category is None)):
                add("qa_plan_operation_category", [*path, "category"])

    retain = delta.get("retain_ids", _INVALID)
    target_lists = [operation.get("target_ids", _INVALID) for operation in operations]
    if retain is not _INVALID and isinstance(operations_raw, list) and all(targets is not _INVALID for targets in target_lists):
        addresses = [*retain, *(target for targets in target_lists for target in targets)]
        active = {item.id for item in accepted.requirements if item.state == "active"}
        if len(addresses) != len(set(addresses)) or set(addresses) != active:
            add("qa_plan_active_addresses", ["requirement_delta"], active_count=len(active),
                addressed_count=len(addresses), distinct_address_count=len(set(addresses)),
                missing_active_count=len(active - set(addresses)), inactive_address_count=len(set(addresses) - active))

    intent = delta.get("intent", _INVALID)
    clarification = delta.get("clarification", _INVALID)
    if intent is not _INVALID and clarification is not _INVALID and ((intent == "needs_clarification") != bool(clarification)):
        add("qa_plan_clarification_text", ["requirement_delta", "clarification"])
    if intent == "needs_clarification" and isinstance(operations_raw, list) and operations_raw:
        add("qa_plan_clarification_operations", ["requirement_delta", "operations"])
    if intent == "clear" and accepted.revision == 0 and operations_raw == []:
        add("requirements_initial_user_request_not_represented", ["requirement_delta", "operations"])
    if accepted.turns and int(turn.turn_id[1:]) <= int(accepted.turns[-1].turn_id[1:]):
        add("requirements_turn_already_processed_or_out_of_order", ["requirement_delta", "turn_id"])

    action = root.get("action", _INVALID)
    refs = root.get("existing_ref_ids", _INVALID)
    queries_raw = value.get("queries")
    limits = []
    if isinstance(queries_raw, list):
        for index, query in enumerate(queries_raw):
            # Read the original limit field independently of query text and
            # operation validity, so batch capacity is never masked by either.
            limit = fields(RetrievalIntent, query, ["queries", index]).get("limit", _INVALID)
            limits.append(limit)
    if refs is not _INVALID and len(refs) != len(set(refs)):
        add("qa_plan_refs_unique", ["existing_ref_ids"])
    if action is not _INVALID and intent is not _INVALID and ((action == "clarify") != (intent == "needs_clarification")):
        add("qa_plan_action_intent", ["action"])
    if isinstance(queries_raw, list) and refs is not _INVALID:
        if action == "clarify" and (queries_raw or refs):
            add("qa_plan_clarify_no_evidence", ["action"])
        if action == "retrieve":
            if not queries_raw:
                add("qa_plan_retrieve_queries", ["queries"])
            if all(limit is not _INVALID for limit in limits):
                slots = sum(limits) + len(refs)
                if slots > 20:
                    add("qa_plan_result_slots", ["queries"], query_slots=sum(limits), reused_slots=len(refs), total_slots=slots, maximum=20)
        if action == "verify_existing" and (queries_raw or not refs):
            add("qa_plan_verify_existing_shape", ["action"])

    if not issues:
        return None
    return {"error_codes": list(dict.fromkeys(issue["code"] for issue in issues)),
            "details": {"issues": issues, "checks": "server_deterministic_preflight/1"}}
