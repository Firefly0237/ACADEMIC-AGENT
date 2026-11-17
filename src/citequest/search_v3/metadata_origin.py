"""Keep public metadata locators independent of their internal source adapter."""
from types import SimpleNamespace

from sqlalchemy import select

from citequest.data_v2.models import SourceProjection
from citequest.index_v2.models import IndexGeneration, IndexPublishIntent
from citequest.index_v2.source_variants import UpdateReference
from citequest.updates_v2.models import PublicationMember, UpdateObservation, UpdateWindow
from citequest.v2.domain import DomainError


def published_observation(db, repository, identifier):
    legacy, observation = db.get(SourceProjection, identifier), db.get(UpdateObservation, identifier)
    if legacy is not None and observation is not None:
        raise DomainError("ambiguous_metadata_origin_ID", 409)
    if observation is None:
        return None
    generation = db.scalar(select(IndexGeneration).join(PublicationMember,
        PublicationMember.snapshot_id == IndexGeneration.publication_snapshot_id).join(IndexPublishIntent,
        IndexPublishIntent.target_generation_id == IndexGeneration.id).where(
            PublicationMember.source_unit_id == identifier, IndexPublishIntent.state == "committed",
            IndexGeneration.state.in_(["active", "retired"]),
            IndexGeneration.security_domain.in_(["public", repository.actor.security_domain]))
        .order_by(IndexGeneration.created_at.desc(), IndexGeneration.id).limit(1))
    if generation is None:
        raise DomainError("metadata_observation_not_published", 404)
    repository._read_ready(generation)
    current = repository.readers.read(repository.readers.root(generation), UpdateReference(update_observation_id=identifier))
    window = db.get(UpdateWindow, observation.window_id)
    if window is None:
        raise DomainError("metadata_observation_window_missing", 409)
    return current, SimpleNamespace(id=generation.publication_snapshot_id, metadata_observation={
        "publication_snapshot_id": generation.publication_snapshot_id, "observation_id": observation.id,
        "source_namespace": observation.source_namespace, "observed_at": observation.created_at,
        "window_lower_bound": window.lower_bound, "observed_watermark": window.upper_bound,
        "current_status_confirmed": False, "complete_global_or_latest_sync_claimed": False})
