"""Bounded RFC6902 answer revisions over an immutable server-held candidate."""
from __future__ import annotations

import copy
from typing import Literal, Union, get_args

import jsonpatch
from pydantic import BaseModel, Field, create_model

from citequest.research_v3.contracts import Strict
from citequest.research_v3.typed_feedback import claim_objects, strict_object

from .contracts import AgentError, digest

VERSION = "jsonpatch1"


class AnswerPatch:
    """Advertise only authorized whole-field replacements, then apply a copy.

    The assembled answer still requires the caller's original repair lock,
    complete domain validation, literal-source checks and independent review.
    """

    def __init__(self, baseline: dict, answer_model: type[BaseModel],
                 affected_claim_ids: list[str], *, gap_only: bool = False,
                 required_fields: tuple[str, ...] = ()):
        self.baseline = copy.deepcopy(baseline)
        self.baseline_sha256 = digest(self.baseline)
        claims = claim_objects(self.baseline)
        if (set(self.baseline) - answer_model.model_fields.keys()
                or len(affected_claim_ids) != len(set(affected_claim_ids))
                or not set(affected_claim_ids) <= claims.keys()
                or (gap_only and affected_claim_ids) or (not gap_only and not affected_claim_ids)):
            raise AgentError("qa_answer_patch_baseline_invalid")
        if (not isinstance(required_fields, (tuple, list))
                or any(not isinstance(name, str) for name in required_fields)
                or len(required_fields) != len(set(required_fields))
                or any(name == "claims" or name not in answer_model.model_fields
                    or name not in self.baseline for name in required_fields)
                or (gap_only and set(required_fields) - {"gaps"})):
            raise AgentError("qa_answer_patch_required_fields_invalid")
        self.required_fields = tuple(required_fields)
        self.required_paths = []
        models = []
        suffix = self.baseline_sha256[:12]
        if gap_only:
            if "gaps" not in self.baseline or "gaps" not in answer_model.model_fields:
                raise AgentError("qa_answer_patch_baseline_invalid")
            fields = ["gaps"]
            self.required_paths = ["/gaps"]
        else:
            claim_type = get_args(answer_model.model_fields["claims"].annotation)[0]
            for index, claim in enumerate(self.baseline["claims"]):
                if claim["id"] not in affected_claim_ids:
                    continue
                path = f"/claims/{index}"
                self.required_paths.append(path)
                value_model = create_model(f"PatchedClaim_{suffix}_{index}",
                    __base__=claim_type, id=(Literal[claim["id"]], ...))
                models.append(self._operation(f"ClaimReplace_{suffix}_{index}", path, value_model))
            fields = [name for name in answer_model.model_fields
                if name != "claims" and name in self.baseline]
        for name in self.required_fields:
            path = "/" + name.replace("~", "~0").replace("/", "~1")
            if path not in self.required_paths:
                self.required_paths.append(path)
        for name in fields:
            # Names originate in the server-owned answer schema, never user paths.
            path = "/" + name.replace("~", "~0").replace("/", "~1")
            models.append(self._operation(f"FieldReplace_{suffix}_{name}", path,
                answer_model.model_fields[name].rebuild_annotation()))
        self.allowed_paths = [model.model_fields["path"].annotation.__args__[0] for model in models]
        operation_type = models[0] if len(models) == 1 else Union[tuple(models)]
        self.model = create_model(f"AnswerPatch_{suffix}_{'gaps' if gap_only else 'claims'}",
            __base__=Strict,
            baseline_sha256=(Literal[self.baseline_sha256], ...),
            operations=(list[operation_type], Field(min_length=len(self.required_paths), max_length=len(models))))

    @staticmethod
    def _operation(name, path, value_type):
        return create_model(name, __base__=Strict, op=(Literal["replace"], ...),
            path=(Literal[path], ...), value=(value_type, ...))

    def contract(self) -> dict:
        rule = "Replace each affected claim exactly once. Preserve its ID. Other advertised fields are optional replacements. The server retains all other original values and independently validates the complete assembled answer."
        if self.required_fields:
            rule = "Replace every required path exactly once, including affected claims and reviewed top-level fields. Preserve claim IDs. Other advertised fields are optional replacements. The server retains all other original values and independently validates the complete assembled answer."
        return {"protocol": VERSION, "baseline_sha256": self.baseline_sha256,
            "allowed_operations": ["replace"], "allowed_paths": self.allowed_paths,
            "required_paths_exactly_once": self.required_paths,
            "rule": rule}

    def apply(self, raw_patch: str) -> tuple[dict, dict]:
        patch = self.model.model_validate(strict_object(raw_patch), strict=True).model_dump()
        paths = [operation["path"] for operation in patch["operations"]]
        if (len(paths) != len(set(paths)) or not set(self.required_paths) <= set(paths)
                or not set(paths) <= set(self.allowed_paths)):
            raise AgentError("qa_answer_patch_path_set_invalid")
        merged = jsonpatch.apply_patch(self.baseline, patch["operations"], in_place=False)
        receipt = {"protocol": VERSION, "baseline_sha256": self.baseline_sha256,
            "patch_sha256": digest(patch), "assembled_candidate_sha256": digest(merged),
            "paths": paths}
        return merged, receipt
