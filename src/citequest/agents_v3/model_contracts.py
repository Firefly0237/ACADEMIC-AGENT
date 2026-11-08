"""Registered provider identities and conservative billing facts for frozen Tasks."""
from __future__ import annotations

_MODELS = {
    "deepseek-flash": {
        "model_alias": "deepseek-flash",
        "model_version": "DeepSeek-V4.1-Flash-1017",
        "context_window": 1048576,
        "input_miss_ncny": 2000,
        "input_cached_ncny": 40,
        "output_ncny": 8000,
    },
    "deepseek-v4-pro": {
        "model_alias": "deepseek-v4-pro",
        "model_version": "DeepSeek-V4-Pro-0813",
        "context_window": 1048576,
        "input_miss_ncny": 9000,
        "input_cached_ncny": 300,
        "output_ncny": 27000,
    },
}


def model_contract(alias: str) -> dict:
    from .contracts import AgentError

    if not isinstance(alias, str) or alias not in _MODELS:
        raise AgentError("provider_model_not_registered")
    return dict(_MODELS[alias])


def validated_model_alias(profile: dict) -> str:
    """Require the complete registered identity and rates, without model substitution."""
    from .contracts import AgentError

    contract = model_contract(profile.get("model_alias"))
    if any(profile.get(name) != value for name, value in contract.items()):
        raise AgentError("provider_frozen_model_contract_mismatch")
    return contract["model_alias"]
