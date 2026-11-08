"""Trusted new-task release selection; historical Task profiles stay immutable."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .contracts import PROFILES, AgentError, Strict, TaskCreate, digest
from .function_output import PROFILE_BINDING_KEY

TaskKind = Literal["qa", "summary", "compare", "stance", "research"]
FUNCTION_REVIEW_PAIRS = frozenset({("qa", "qa-v5"), ("qa", "qa-v6"), ("qa", "qa-v7"), ("qa", "qa-v8"), ("qa", "qa-v9"), ("stance", "stance-v2"), ("stance", "stance-v3")})
# A resource declaration may precede its implementation. Keep historical
# profiles readable while preventing incomplete candidates from accepting work.
UNIMPLEMENTED_PROFILE_IDS = frozenset()
QA_EXECUTION_BINDING_KEY = "qa_execution_protocol"
REVIEW_API_BINDING_KEY = "review_api_protocol"
STRUCTURED_OUTPUT_BINDING_KEY = "structured_output_protocol"
ANSWER_REVISION_BINDING_KEY = "answer_revision_protocol"
QA_CONTEXT_BINDING_KEY = "qa_context_protocol"
QA_CONTENT_BINDING_KEY = "qa_content_protocol"
EVIDENCE_NAVIGATION_BINDING_KEY = "evidence_navigation_protocol"
QA8_PROTOCOL_METADATA = {
    "review_transport_protocol": "json1", "qa_execution_protocol": "grounded3",
    "review_api_protocol": "chat-json2", "structured_output_protocol": "chat-json2",
    "answer_revision_protocol": "jsonpatch1", "qa_context_protocol": "references1",
}


def require_qa8_protocol(task: dict) -> None:
    profile = task["profile"]
    if (task["kind"] != "qa" or profile.get("profile_id") not in {"qa-v8", "qa-v9"}
            or any(profile.get(key) != value for key, value in QA8_PROTOCOL_METADATA.items())):
        raise AgentError("task_qa8_protocol_not_supported", 503)


def is_profile_implemented(profile_id: str) -> bool:
    return profile_id in PROFILES and profile_id not in UNIMPLEMENTED_PROFILE_IDS


class NewTaskProfile(Strict):
    kind: TaskKind
    profile_id: str = Field(min_length=1, max_length=64)
    validation_state: Literal["development", "accepted"]
    # Omission is intentional: old policy serialization/hashes remain exact.
    review_transport_protocol: Literal["function1", "json1"] | None = Field(default=None, exclude_if=lambda value: value is None)
    qa_execution_protocol: Literal["grounded1", "grounded2", "grounded3"] | None = Field(default=None, exclude_if=lambda value: value is None)
    review_api_protocol: Literal["chat-strict1", "chat-strict2", "chat-json1", "chat-json2"] | None = Field(default=None, exclude_if=lambda value: value is None)
    structured_output_protocol: Literal["chat-functions1", "chat-json2"] | None = Field(default=None, exclude_if=lambda value: value is None)
    answer_revision_protocol: Literal["jsonpatch1"] | None = Field(default=None, exclude_if=lambda value: value is None)
    qa_context_protocol: Literal["references1"] | None = Field(default=None, exclude_if=lambda value: value is None)
    qa_content_protocol: Literal["anchored2", "online1", "isolated1"] | None = Field(default=None, exclude_if=lambda value: value is None)
    evidence_navigation_protocol: Literal["laya1"] | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def known_kind_profile(self) -> NewTaskProfile:
        prefixes = {"qa": "qa-", "summary": "workset-", "compare": "workset-",
                    "stance": "stance-", "research": "research-"}
        if self.profile_id not in PROFILES or not self.profile_id.startswith(prefixes[self.kind]):
            raise ValueError("configured profile does not implement the task kind")
        if not is_profile_implemented(self.profile_id):
            raise ValueError("configured profile implementation is incomplete")
        if self.qa_content_protocol is not None and (self.kind, self.profile_id) != ("qa", "qa-v9"):
            raise ValueError("decomposed content review requires the registered QA9 profile")
        if self.evidence_navigation_protocol is not None and (
                (self.kind, self.profile_id) != ("qa", "qa-v9") or self.qa_content_protocol != "isolated1"):
            raise ValueError("Laya evidence navigation requires the independently reviewed QA9 workflow")
        if self.profile_id in {"qa-v8", "qa-v9"}:
            if self.kind != "qa" or any(getattr(self, key) != value for key, value in QA8_PROTOCOL_METADATA.items()):
                raise ValueError("anchored QA requires its complete explicit reasoning and review protocol")
            return self
        if self.review_transport_protocol is not None and (self.kind, self.profile_id) not in FUNCTION_REVIEW_PAIRS:
            raise ValueError("function1 is registered only for the shared QA5 combined review")
        if self.qa_execution_protocol is not None and (
                self.kind != "qa" or self.profile_id not in {"qa-v6", "qa-v7"} or self.review_transport_protocol not in {"function1", "json1"}):
            raise ValueError("grounded QA requires a registered dialogue profile and transport")
        if self.profile_id == "qa-v7" and (
                self.qa_execution_protocol != "grounded3" or (self.review_transport_protocol, self.review_api_protocol)
                not in {("function1", "chat-strict2"), ("json1", "chat-json1")}):
            raise ValueError("qa-v7 requires an explicitly registered complete review protocol")
        if (self.review_transport_protocol == "json1" or self.review_api_protocol == "chat-json1") and (
                self.profile_id != "qa-v7" or self.review_transport_protocol != "json1"
                or self.review_api_protocol != "chat-json1" or self.structured_output_protocol != "chat-functions1"
                or self.answer_revision_protocol != "jsonpatch1"):
            raise ValueError("JSON review requires the complete explicit Pro typed and patch protocol")
        if self.structured_output_protocol is not None and self.profile_id != "qa-v7":
            raise ValueError("complete structured submission requires the registered Pro QA profile")
        if self.answer_revision_protocol is not None and (
                self.profile_id != "qa-v7" or self.structured_output_protocol != "chat-functions1"):
            raise ValueError("scoped answer patches require complete typed Pro QA submissions")
        if self.qa_context_protocol is not None and (
                self.profile_id != "qa-v7" or self.qa_execution_protocol != "grounded3"
                or self.review_transport_protocol != "json1" or self.review_api_protocol != "chat-json1"
                or self.structured_output_protocol != "chat-functions1" or self.answer_revision_protocol != "jsonpatch1"):
            raise ValueError("reference context requires the complete registered JSON-review QA protocol")
        if self.review_api_protocol is not None and self.qa_execution_protocol not in {"grounded1", "grounded2", "grounded3"}:
            raise ValueError("strict Chat review requires grounded QA execution")
        if self.qa_execution_protocol in {"grounded2", "grounded3"} and self.review_api_protocol not in {"chat-strict1", "chat-strict2", "chat-json1"}:
            raise ValueError("audited QA requires the explicit strict review transport")
        if self.review_api_protocol == "chat-strict2" and self.qa_execution_protocol != "grounded3":
            raise ValueError("the length compiler requires verified QA execution")
        return self


class TaskProfilePolicy(Strict):
    schema_version: Literal["new-task-profile-policy/1"] = "new-task-profile-policy/1"
    release_id: str = Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,79}$")
    profiles: list[NewTaskProfile] = Field(max_length=5)

    @model_validator(mode="after")
    def one_per_kind(self) -> TaskProfilePolicy:
        if len({item.kind for item in self.profiles}) != len(self.profiles):
            raise ValueError("each new task kind must have exactly one configured profile")
        return self

    def admit(self, body: TaskCreate) -> None:
        configured = next((item for item in self.profiles if item.kind == body.kind), None)
        if configured is None:
            raise AgentError("task_kind_not_enabled_for_new_tasks", 503)
        if body.profile_id != configured.profile_id:
            raise AgentError("task_profile_refresh_required", 409)

    def runtime_metadata(self, kind: str, profile_id: str) -> dict:
        configured = next((item for item in self.profiles if (item.kind, item.profile_id) == (kind, profile_id)), None)
        metadata = ({PROFILE_BINDING_KEY: configured.review_transport_protocol}
            if configured is not None and configured.review_transport_protocol is not None else {})
        if configured is not None and configured.qa_execution_protocol is not None:
            metadata[QA_EXECUTION_BINDING_KEY] = configured.qa_execution_protocol
        if configured is not None and configured.review_api_protocol is not None:
            metadata[REVIEW_API_BINDING_KEY] = configured.review_api_protocol
        if configured is not None and configured.structured_output_protocol is not None:
            metadata[STRUCTURED_OUTPUT_BINDING_KEY] = configured.structured_output_protocol
        if configured is not None and configured.answer_revision_protocol is not None:
            metadata[ANSWER_REVISION_BINDING_KEY] = configured.answer_revision_protocol
        if configured is not None and configured.qa_context_protocol is not None:
            metadata[QA_CONTEXT_BINDING_KEY] = configured.qa_context_protocol
        if configured is not None and configured.qa_content_protocol is not None:
            metadata[QA_CONTENT_BINDING_KEY] = configured.qa_content_protocol
        if configured is not None and configured.evidence_navigation_protocol is not None:
            metadata[EVIDENCE_NAVIGATION_BINDING_KEY] = configured.evidence_navigation_protocol
        return metadata


def evidence_navigation_protocol(task: dict) -> str | None:
    value = task["profile"].get(EVIDENCE_NAVIGATION_BINDING_KEY)
    if value is not None and (value != "laya1" or task.get("kind") != "qa"
            or task["profile"].get("profile_id") != "qa-v9"
            or qa_content_protocol(task) != "isolated1"):
        raise AgentError("task_evidence_navigation_not_supported", 503)
    return value


def qa_content_protocol(task: dict) -> str | None:
    value = task["profile"].get(QA_CONTENT_BINDING_KEY)
    if value is not None:
        if value not in {"anchored2", "online1", "isolated1"} or task["profile"].get("profile_id") != "qa-v9":
            raise AgentError("task_qa_content_protocol_not_supported", 503)
        require_qa8_protocol(task)
    return value


def qa_execution_protocol(task: dict) -> str | None:
    if task["profile"].get("profile_id") in {"qa-v8", "qa-v9"}:
        require_qa8_protocol(task)
        return QA8_PROTOCOL_METADATA[QA_EXECUTION_BINDING_KEY]
    value = task["profile"].get(QA_EXECUTION_BINDING_KEY)
    if task["profile"].get("profile_id") == "qa-v7" and (
            task["kind"] != "qa" or value != "grounded3"
            or (task["profile"].get(PROFILE_BINDING_KEY), task["profile"].get(REVIEW_API_BINDING_KEY))
            not in {("function1", "chat-strict2"), ("json1", "chat-json1")}):
        raise AgentError("task_qa_execution_protocol_not_supported", 503)
    if task["profile"].get(REVIEW_API_BINDING_KEY) == "chat-json1" and (
            task["profile"].get("profile_id") != "qa-v7" or task["profile"].get(PROFILE_BINDING_KEY) != "json1"
            or task["profile"].get(STRUCTURED_OUTPUT_BINDING_KEY) != "chat-functions1"
            or task["profile"].get(ANSWER_REVISION_BINDING_KEY) != "jsonpatch1"):
        raise AgentError("task_qa_execution_protocol_not_supported", 503)
    if value is not None and (value not in {"grounded1", "grounded2", "grounded3"} or task["kind"] != "qa" or task["profile"].get("profile_id") not in {"qa-v6", "qa-v7"}
            or task["profile"].get(PROFILE_BINDING_KEY) not in {"function1", "json1"}):
        raise AgentError("task_qa_execution_protocol_not_supported", 503)
    if value in {"grounded2", "grounded3"} and task["profile"].get(REVIEW_API_BINDING_KEY) not in {"chat-strict1", "chat-strict2", "chat-json1"}:
        raise AgentError("task_qa_execution_protocol_not_supported", 503)
    if task["profile"].get(REVIEW_API_BINDING_KEY) == "chat-strict2" and value != "grounded3":
        raise AgentError("task_qa_execution_protocol_not_supported", 503)
    return value


def review_api_protocol(task: dict) -> str | None:
    if task["profile"].get("profile_id") in {"qa-v8", "qa-v9"}:
        require_qa8_protocol(task)
        return QA8_PROTOCOL_METADATA[REVIEW_API_BINDING_KEY]
    value = task["profile"].get(REVIEW_API_BINDING_KEY)
    if value is not None and (value not in {"chat-strict1", "chat-strict2", "chat-json1"} or qa_execution_protocol(task) not in {"grounded1", "grounded2", "grounded3"}):
        raise AgentError("task_review_api_protocol_not_supported", 503)
    return value


def review_transport(task: dict) -> str | None:
    """Only immutable server-created Task metadata selects the new transport."""
    if task["profile"].get("profile_id") in {"qa-v8", "qa-v9"}:
        require_qa8_protocol(task)
        return QA8_PROTOCOL_METADATA[PROFILE_BINDING_KEY]
    value = task["profile"].get(PROFILE_BINDING_KEY)
    if value == "json1":
        if task["profile"].get("profile_id") != "qa-v7" or qa_execution_protocol(task) != "grounded3":
            raise AgentError("task_review_transport_not_supported", 503)
    elif value is not None and (value != "function1" or (task["kind"], task["profile"].get("profile_id")) not in FUNCTION_REVIEW_PAIRS):
        raise AgentError("task_review_transport_not_supported", 503)
    return value


def structured_output_protocol(task: dict) -> str | None:
    if task["profile"].get("profile_id") in {"qa-v8", "qa-v9"}:
        require_qa8_protocol(task)
        return QA8_PROTOCOL_METADATA[STRUCTURED_OUTPUT_BINDING_KEY]
    value = task["profile"].get(STRUCTURED_OUTPUT_BINDING_KEY)
    if value is not None and (value != "chat-functions1"
            or task["kind"] != "qa" or task["profile"].get("profile_id") != "qa-v7"
            or qa_execution_protocol(task) != "grounded3"
            or review_api_protocol(task) not in {"chat-strict2", "chat-json1"}):
        raise AgentError("task_structured_output_protocol_not_supported", 503)
    return value


def answer_revision_protocol(task: dict) -> str | None:
    if task["profile"].get("profile_id") in {"qa-v8", "qa-v9"}:
        require_qa8_protocol(task)
        return QA8_PROTOCOL_METADATA[ANSWER_REVISION_BINDING_KEY]
    value = task["profile"].get(ANSWER_REVISION_BINDING_KEY)
    if value is not None and (value != "jsonpatch1" or structured_output_protocol(task) != "chat-functions1"):
        raise AgentError("task_answer_revision_protocol_not_supported", 503)
    return value


def qa_context_protocol(task: dict) -> str | None:
    if task["profile"].get("profile_id") in {"qa-v8", "qa-v9"}:
        require_qa8_protocol(task)
        return QA8_PROTOCOL_METADATA[QA_CONTEXT_BINDING_KEY]
    value = task["profile"].get(QA_CONTEXT_BINDING_KEY)
    if value is not None and (
            value != "references1" or task["kind"] != "qa"
            or task["profile"].get("profile_id") != "qa-v7" or qa_execution_protocol(task) != "grounded3"
            or review_transport(task) != "json1" or review_api_protocol(task) != "chat-json1"
            or structured_output_protocol(task) != "chat-functions1" or answer_revision_protocol(task) != "jsonpatch1"):
        raise AgentError("task_qa_context_protocol_not_supported", 503)
    return value


def public_profile_policy(policy: TaskProfilePolicy | None) -> dict:
    """No secrets, provider config or permission grants in public capabilities.

    None explicitly describes the original API compatibility mode. It does
    not approve new quality results or upgrade an already persisted Task.
    """
    if policy is None:
        policy = TaskProfilePolicy(release_id="legacy-api-compatibility", profiles=[
            NewTaskProfile(kind=kind, profile_id=profile, validation_state="development")
            for kind, profile in (("qa", "qa-v1"), ("summary", "workset-v2"),
                                  ("compare", "workset-v2"), ("stance", "stance-v1-r2"),
                                  ("research", "research-v1"))])
        mode = "legacy_compatibility"
    else:
        mode = "configured_release"
    value = policy.model_dump(mode="json")
    # Include actual immutable resource values in this fingerprint; the UI
    # cannot silently select a different quota under the same capability.
    value["profile_hashes"] = {item.profile_id: digest(PROFILES[item.profile_id].model_dump(mode="json"))
                               for item in policy.profiles}
    return {**value, "mode": mode, "policy_sha256": digest(value)}
