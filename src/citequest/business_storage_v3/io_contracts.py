"""Fixed internal subprocess job; never a user-selectable command or endpoint."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from citequest.platform_v3.blob_config import BlobStoreConfig
from citequest.platform_v3.blob_store import BlobIdentity

from .contracts import parse_reference


class IOProfile(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    version: Literal["citequest-blob-io/1"] = "citequest-blob-io/1"
    maximum_object_bytes: Annotated[int, Field(ge=1, le=64 * 1024**2)] = 64 * 1024**2
    maximum_job_bytes: Literal[65536] = 65536
    maximum_result_bytes: Literal[16384] = 16384
    address_space_bytes: Literal[536870912] = 512 * 1024**2
    file_descriptor_limit: Literal[64] = 64
    per_process_spool_slots: Literal[2] = 2
    global_io_slots: Literal[2] = 2
    maximum_active_seconds: Literal[120] = 120
    kill_confirmation_seconds: Literal[1.0] = 1.0


class IOJob(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)
    version: Literal["citequest-blob-io-job/1"] = "citequest-blob-io-job/1"
    operation: Literal["put", "read", "reconcile", "probe"]
    expected_parent_pid: Annotated[int, Field(ge=1)]
    expires_at: Annotated[float, Field(allow_inf_nan=False)]
    profile: IOProfile
    storage: BlobStoreConfig
    identity: dict | None = None
    reference: dict | None = None

    @model_validator(mode="after")
    def fixed_shape(self) -> IOJob:
        if self.storage.maximum_object_bytes != self.profile.maximum_object_bytes:
            raise ValueError("io_storage_and_resource_object_caps_must_match")
        if self.operation == "probe":
            if self.identity is not None or self.reference is not None:
                raise ValueError("io_probe_has_no_object")
        elif self.operation in {"put", "reconcile"}:
            if self.identity is None or self.reference is not None:
                raise ValueError("io_identity_operation_shape_invalid")
            identity = BlobIdentity(**self.identity)
            if identity.size_bytes > self.profile.maximum_object_bytes:
                raise ValueError("io_identity_exceeds_resource_profile")
        else:
            if self.identity is not None or self.reference is None:
                raise ValueError("io_read_requires_exact_reference")
            if parse_reference(self.reference).identity.size_bytes > self.profile.maximum_object_bytes:
                raise ValueError("io_reference_exceeds_resource_profile")
        return self
