"""Full bounded-publication driver reusing the existing verified index pipeline."""
from citequest.index_v2.controller_v2 import VariantIndexController
from citequest.v2.services import Catalog
from citequest.v2.domain import DomainError

from .repository_bounded import BoundedFullBuildRepository


class PublicationIndexController(VariantIndexController):
    def before_native_configuration(self, generation_id):
        from .models import BackendBinding
        backend = getattr(self.native, "backend_sha256", None)
        if backend is not None:
            def admitted(repository):
                generation = repository.generation(generation_id)
                binding = repository.db.get(BackendBinding, generation_id)
                return (binding is not None and binding.backend_profile_sha256 == backend
                    and binding.expected_manifest_sha256 == generation.expected_manifest_sha256)
            if not self.tx(admitted):
                raise DomainError("publication_backend_binding_requires_migrator_maintenance", 409)
    @staticmethod
    def admit_window(kind: str, unit_limit: int | None, seconds: float) -> None:
        if (kind != "publication_snapshot" or unit_limit is not None or type(seconds) not in {int, float}
            or not 0 < seconds <= 3600):
            raise ValueError("explicit_complete_publication_and_bounded_execution_window_required")

    def tx(self, operation):
        with self.factory.begin() as db:
            return operation(BoundedFullBuildRepository(Catalog(db, self.actor), store=self.store,
                policy_sha256=self.policy_sha256, evaluation_sha256=self.evaluation_sha256))
