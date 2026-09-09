"""Authentication boundary for RiskForge.

Provides the public API for identity verification without coupling to
specific authentication mechanisms (JWT, OIDC, API keys, etc.).

The authentication subsystem is intentionally decoupled from persistence:
it does not read or write user tables.  Concrete authentication providers
are injected at application startup through the ``AuthenticationService``
protocol.
"""

from riskforge.authentication.exceptions import (
    AuthenticationError,
    InvalidCredentialsError,
    MissingCredentialsError,
)
from riskforge.authentication.principal import Principal
from riskforge.authentication.protocols import AuthenticationService

__all__ = [
    "AuthenticationError",
    "AuthenticationService",
    "InvalidCredentialsError",
    "MissingCredentialsError",
    "Principal",
]
