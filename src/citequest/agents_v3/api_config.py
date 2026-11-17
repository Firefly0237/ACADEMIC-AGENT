"""Trusted API composition; no model key, provider, worker or tool command."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator

from .catalog_scope import CatalogScopePolicy
from .contracts import RuntimeLimits, Strict
from .database import Database
from .product_profiles import TaskProfilePolicy
from .development_campaign import load_campaign


class AgentAPIConfig(Strict):
    limits: RuntimeLimits
    business_dsn_file: Path
    checkpoint_dsn_file: Path
    memory_dsn_file: Path
    material_config_file: Path | None = None
    dispatch_policy_file: Path | None = None
    catalog_scope_policy: CatalogScopePolicy | None = None
    task_profile_policy: TaskProfilePolicy | None = None
    development_campaign_file: Path | None = None
    development_campaign_sha256: Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")] | None = None

    @model_validator(mode="after")
    def paired_campaign_reference(self) -> AgentAPIConfig:
        if (self.development_campaign_file is None) != (self.development_campaign_sha256 is None):
            raise ValueError("development campaign requires both trusted file and exact SHA")
        if self.development_campaign_file is not None and self.task_profile_policy is None:
            raise ValueError("development campaign requires explicit task profile policy")
        return self

    def database(self) -> Database:
        campaign = (load_campaign(self.development_campaign_file, self.development_campaign_sha256)
            if self.development_campaign_file is not None else None)
        if campaign is not None:
            campaign.require_configuration(self.limits, self.catalog_scope_policy, self.task_profile_policy)
        return Database.from_files(self.business_dsn_file, self.checkpoint_dsn_file, self.memory_dsn_file,
            dispatch_policy_file=self.dispatch_policy_file, catalog_scope_policy=self.catalog_scope_policy,
            development_campaign_policy=campaign)
