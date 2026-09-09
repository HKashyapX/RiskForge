"""Typed authentication failure taxonomy.

All authentication errors inherit from ``AuthenticationError``.  Concrete
error types allow the API layer to map failures to transport-safe responses
without inspecting internal exception text.

Credentials are never stored, logged, or included in exception messages.
"""

from __future__ import annotations


class AuthenticationError(RuntimeError):
    """Base class for all authentication failures."""


class MissingCredentialsError(AuthenticationError):
    """Raised when the request contains no credentials."""


class InvalidCredentialsError(AuthenticationError):
    """Raised when the supplied credentials are invalid or expired."""
