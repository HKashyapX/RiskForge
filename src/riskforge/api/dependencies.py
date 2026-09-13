"""Narrow dependencies consumed by the HTTP transport."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Protocol, runtime_checkable

from fastapi import Depends

from riskforge.authentication.principal import Principal

# ── Capability scopes ────────────────────────────────────────────────────
# Minimal explicit scopes; roles map onto scopes at the JWT issuer, but the
# API enforces scopes (and review roles) at every route boundary.
SCOPE_INCIDENTS_READ = "incidents:read"
SCOPE_ANALYTICS_READ = "analytics:read"
SCOPE_SCORING_WRITE = "scoring:write"
SCOPE_AUDIT_READ = "audit:read"





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

    Routes that also need a capability scope declare::

        principal: Principal = Depends(require_scoring_write)
    """
    from riskforge.authentication.exceptions import MissingCredentialsError

    raise MissingCredentialsError()


def _scope_dependency(
    scope: str,
    fallback_roles: tuple[str, ...],
) -> Callable[..., Principal]:
    """Build a FastAPI dependency enforcing *scope* (or a fallback role).

    The returned dependency resolves the authenticated principal through
    ``require_principal`` (so ``create_app``'s authentication override
    applies) and then checks the capability scope; possession of one of
    ``fallback_roles`` also satisfies the check.  A principal with neither
    is denied with 403 (scope_forbidden).
    """

    def _dependency(
        principal: Annotated[Principal, Depends(require_principal)],
    ) -> Principal:
        if principal.has_scope(scope) or principal.has_any_role(*fallback_roles):
            return principal
        from riskforge.api.app import _ApiFailure
        from riskforge.api.errors import ErrorCode, TranslatedError

        raise _ApiFailure(
            "unavailable",
            TranslatedError(
                403,
                ErrorCode.SCOPE_FORBIDDEN,
                f"principal lacks the required {scope!r} scope",
                False,
            ),
        )

    return _dependency


def require_incidents_read() -> Callable[..., Principal]:
    """Require the incidents:read scope (or a reader role)."""
    return _scope_dependency(
        SCOPE_INCIDENTS_READ, ("reader", "reviewer", "safety_officer", "admin")
    )


def require_analytics_read() -> Callable[..., Principal]:
    """Require the analytics:read scope (or a reader role)."""
    return _scope_dependency(
        SCOPE_ANALYTICS_READ, ("reader", "reviewer", "safety_officer", "admin")
    )


def require_scoring_write() -> Callable[..., Principal]:
    """Require the scoring:write scope (or an ingestor role)."""
    return _scope_dependency(
        SCOPE_SCORING_WRITE, ("ingestor", "safety_officer", "admin")
    )


def require_audit_read() -> Callable[..., Principal]:
    """Require the audit:read scope (or an auditor role)."""
    return _scope_dependency(
        SCOPE_AUDIT_READ, ("auditor", "reviewer", "safety_officer", "admin")
    )
