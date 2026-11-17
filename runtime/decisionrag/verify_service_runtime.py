"""Verify the composed service role without loading a model or using network."""
import importlib.metadata as metadata
import json
from pathlib import Path

from packaging.requirements import Requirement

manifest = json.loads(Path(__file__).with_name("service-runtime.lock.json").read_bytes())
versions = {}
for name, expected in {**manifest["added_distributions"], **manifest["retained_CPU_distributions"]}.items():
    actual = metadata.version(name)
    if actual != expected:
        raise RuntimeError("service_runtime_version_mismatch:" + name)
    versions[name] = actual
checked = []
for name in manifest["added_distributions"]:
    for raw in metadata.requires(name) or []:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        actual = metadata.version(requirement.name)
        if not requirement.specifier.contains(actual, prereleases=True):
            raise RuntimeError("service_runtime_dependency_conflict:" + str(requirement))
        checked.append({"parent": name, "requirement": str(requirement), "installed": actual})
import fastapi
import starlette
import uvicorn
import torch
from citequest.research_v3.decisionrag_gateway import GatewayConfig
if torch.version.cuda is not None:
    raise RuntimeError("CPU_service_requires_CPU_torch")
print(json.dumps({"status": "verified_import_and_dependency_closure", "versions": versions,
    "dependency_edges": checked, "CUDA_runtime": torch.version.cuda, "model_loaded": False, "forwards": 0}))
