"""The existing source gate plus current migrator adoption for new backends."""

from citequest.index_v2 import models as index
from citequest.v2.domain import DomainError

from .models import BackendAdoption, BackendBinding
from .preparation import FullPreparationRepository


class FullBuildRepository(FullPreparationRepository):
    def _read_ready(self, value: index.IndexGeneration) -> None:
        super()._read_ready(value)
        backend = (value.server_proof or {}).get("backend_profile_sha256")
        binding = self.db.get(BackendBinding, value.id, populate_existing=True)
        if backend is not None or binding is not None:
            adoption = self.db.get(BackendAdoption, value.id, populate_existing=True)
            if (
                binding is None
                or adoption is None
                or binding.backend_profile_sha256 != backend
                or adoption.backend_profile_sha256 != backend
                or adoption.expected_manifest_sha256 != value.expected_manifest_sha256
            ):
                raise DomainError("candidate_backend_not_adopted_for_public_reads", 409)
