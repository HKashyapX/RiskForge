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

    reason: str = "invalid_credentials"


class MalformedCredentialError(InvalidCredentialsError):
    """Raised when the credential structure is invalid."""

    reason = "malformed"


class ExpiredCredentialError(InvalidCredentialsError):
    """Raised when the credential has expired."""

    reason = "expired"


class InvalidIssuerError(InvalidCredentialsError):
    """Raised when the credential issuer does not match the expected value."""

    reason = "invalid_issuer"


class InvalidAudienceError(InvalidCredentialsError):
    """Raised when the credential audience does not match the expected value."""

    reason = "invalid_audience"


class UnknownKeyError(InvalidCredentialsError):
    """Raised when the credential references an unknown key identifier."""

    reason = "unknown_key"


class InvalidSignatureError(InvalidCredentialsError):
    """Raised when the credential signature verification fails."""

    reason = "invalid_signature"
