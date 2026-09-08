"""Narrow dependencies consumed by the HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from fastapi import Request


@dataclass(frozen=True)
class ReadinessSnapshot:
    """Transport-safe readiness data supplied by runtime composition."""

    ready: bool
    state: str
    checked_at: datetime
    components: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.state.strip():
            raise ValueError("readiness state must not be empty")
        if len(self.state) > 32:
            raise ValueError("readiness state is too long")
        if any(not component.strip() for component in self.components):
            raise ValueError("readiness component names must not be empty")
        if len(self.components) != len(set(self.components)):
            raise ValueError("readiness component names must be unique")


@runtime_checkable
class ReadinessProvider(Protocol):
    def snapshot(self) -> ReadinessSnapshot:
        """Return the current transport-safe runtime state."""


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """Identity already authenticated by an injected provider."""

    reviewer_id: str

    def __post_init__(self) -> None:
        if not self.reviewer_id.strip() or len(self.reviewer_id) > 128:
            raise ValueError("reviewer identity is invalid")


class UnauthenticatedError(RuntimeError):
    """Raised when a request has no valid authenticated identity."""


@runtime_checkable
class PrincipalResolver(Protocol):
    def resolve(self, request: Request) -> AuthenticatedPrincipal:
        """Resolve an authenticated principal without API-owned auth logic."""
