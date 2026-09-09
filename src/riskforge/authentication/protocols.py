"""Provider-agnostic authentication boundary protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from riskforge.authentication.principal import Principal


@runtime_checkable
class AuthenticationService(Protocol):
    """Boundary protocol for credential verification.

    Implementations translate raw credentials (bearer tokens, API keys,
    etc.) into a ``Principal`` or raise a typed ``AuthenticationError``.
    The protocol is synchronous to match the existing RiskForge service
    layer convention.

    Providers must:
    - Never log, store, or expose raw credentials.
    - Never depend on persistence subsystem infrastructure.
    - Raise ``AuthenticationError`` subclasses for all failure modes.
    """

    def authenticate(self, credential: str) -> Principal:
        """Verify *credential* and return the authenticated ``Principal``.

        Parameters
        ----------
        credential:
            The raw credential value extracted from the HTTP request
            (typically the ``Authorization`` header value without the
            ``Bearer `` prefix).

        Returns
        -------
        Principal
            The authenticated subject.

        Raises
        ------
        AuthenticationError
            If the credential is missing, invalid, or otherwise unacceptable.
        """
        ...
