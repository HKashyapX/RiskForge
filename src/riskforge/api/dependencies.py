"""Narrow dependencies consumed by the HTTP transport."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from riskforge.authentication.principal import Principal


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


def require_principal() -> Principal:
    """Marker dependency for routes requiring authentication.

    When ``create_app`` is called with an ``auth_service``, this dependency
    is overridden to extract and verify credentials from the HTTP request.
    Without an auth service, invoking this dependency raises
    ``MissingCredentialsError``.

    Routes that need an authenticated caller declare::

        principal: Principal = Depends(require_principal)
    """
    from riskforge.authentication.exceptions import MissingCredentialsError

    raise MissingCredentialsError()
