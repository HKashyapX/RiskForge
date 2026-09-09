"""Immutable identity representation for authenticated subjects."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Principal(BaseModel):
    """An authenticated subject within the RiskForge system.

    The ``subject_id`` is the stable, opaque identifier used by downstream
    subsystems (e.g., ``ReviewerAuthorizer``, audit writers).  The principal
    is intentionally minimal: authentication infrastructure does not carry
    roles, permissions, or session state.

    The model is immutable (frozen) and forbids extra fields to prevent
    accidental leakage of transient authentication state into the domain.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Stable, opaque identifier for the authenticated subject.",
    )
