"""Deterministic byte admission; hard inputs survive, optional context is explicit."""
from __future__ import annotations
import copy
import json
import httpx2
from .contracts import AgentError, canonical

CONTEXT_VERSION = "citequest-context/2-wire-priority"


def wire_bytes(payload: dict) -> int:
    # Same JSON request encoder as the pinned official SDK HTTP transport. No send.
    return len(httpx2.Request("POST", "https://encoding.invalid", json=payload).content)


def fit_context(payload: dict, maximum: int) -> dict:
    result = copy.deepcopy(payload)
    prompt = json.loads(result["input"][0]["content"])
    original = copy.deepcopy(prompt)
    manifest = prompt.setdefault("context_manifest", {})
    manifest["context_builder"] = CONTEXT_VERSION
    manifest["priority"] = "hard Brief/current question/source/policy, confirmed constraints/decisions, newest prior user questions, newest prior answers, preferences by stable ID"
    manifest["context_included"] = []
    manifest["context_excluded"] = []
    memories = prompt.get("confirmed_user_memory_data", [])
    hard = [item for item in memories if item.get("kind") != "preference"]
    optional_memories = [item for item in memories if item.get("kind") == "preference"]
    if "confirmed_user_memory_data" in prompt: prompt["confirmed_user_memory_data"] = hard
    manifest["memory_refs"] = [{"id": item["memory_id"], "revision": item["revision"]} for item in hard]
    manifest["hard_memory_refs"] = list(manifest["memory_refs"])
    for key in ("user_turns", "prior_turns"):
        if key in prompt: prompt[key] = []
    options = []
    question = prompt.get("canonical_brief", {}).get("question")
    for item in reversed(original.get("user_turns", [])):
        if question is not None and item.get("text") == question:
            manifest["context_excluded"].append({"kind": "user_turn", "id": item["turn_id"], "reason": "same_current_question_already_mandatory"})
        else:
            options.append(("user_turns", item, {"kind": "user_turn", "id": item["turn_id"]}))
    for index in reversed(range(len(original.get("prior_turns", [])))):
        options.append(("prior_turns", original["prior_turns"][index], {"kind": "prior_answer", "id": str(index)}))
    for item in optional_memories:
        options.append(("confirmed_user_memory_data", item, {"kind": "memory", "id": item["memory_id"], "revision": item["revision"]}))
    # Exclusion metadata is mandatory too: no silent removal of older context.
    manifest["context_excluded"] += [{**ref, "reason": "wire_budget"} for _, _, ref in options]

    def encode() -> int:
        result["input"][0]["content"] = canonical(prompt).decode()
        return wire_bytes(result)

    if encode() > maximum:
        raise AgentError("required_context_exceeds_wire_limit", 413)
    for key, item, ref in options:
        omission = {**ref, "reason": "wire_budget"}
        manifest["context_excluded"].remove(omission)
        manifest["context_included"].append(ref)
        prompt.setdefault(key, []).append(item)
        if key == "confirmed_user_memory_data": manifest["memory_refs"].append({"id": item["memory_id"], "revision": item["revision"]})
        if encode() > maximum:
            prompt[key].pop()
            manifest["context_included"].pop()
            manifest["context_excluded"].append(omission)
            if key == "confirmed_user_memory_data": manifest["memory_refs"].pop()
    encode()
    return result
