"""Identity mappings only. Passwords and OAuth tokens are never stored here."""

from uuid import UUID

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from citequest.v2.models import SCHEMA, Base, Identity


class OIDCIdentity(Identity, Base):
    __tablename__ = "oidc_identities"
    __table_args__ = (UniqueConstraint("issuer", "subject"),)
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"), index=True)
    issuer: Mapped[str] = mapped_column(String(1000))
    subject: Mapped[str] = mapped_column(String(500))


class GuestOwner(Identity, Base):
    __tablename__ = "guest_owners"
    owner_id: Mapped[UUID] = mapped_column(ForeignKey(f"{SCHEMA}.owners.id"), unique=True)
