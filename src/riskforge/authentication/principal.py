"""Immutable identity representation for authenticated subjects."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Principal(BaseModel):
    """An authenticated subject within the RiskForge system.

    The ``subject_id`` is the stable, opaque identifier used by downstream
    subsystems (e.g., ``ReviewerAuthorizer``, audit writers).  ``roles`` and
    ``scopes`` carry authorization data resolved from verified credential
    claims (e.g., the JWT ``roles``/``scope`` claims); route dependencies
    enforce them at the boundary.  Frozen and extra-forbidden to prevent
    transient authentication state leaking into the domain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Stable, opaque identifier for the authenticated subject.",
    )
    roles: tuple[str, ...] = Field(
        default=(),
        description="Roles granted to this subject; empty means no role.",
    )
    scopes: tuple[str, ...] = Field(
        default=(),
        description="Explicit scopes granted to this subject; empty means none.",
    )

    def has_any_role(self, *required: str) -> bool:
        return any(role in self.roles for role in required)

    def has_scope(self, scope: str) -> bool:
        return scope in self.scopes
