"""Development-only authentication provider.

This module exists **exclusively** for local development, integration tests,
and demo environments.  It must never be deployed to production.

The dev provider trusts a single static principal and ignores the credential
value entirely.  This allows developers to exercise authenticated code paths
without standing up a real identity provider.
"""

from __future__ import annotations

import warnings

from riskforge.authentication.principal import Principal


class DevAuthenticationService:
    """Static authentication provider for development and testing.

    Parameters
    ----------
    subject_id:
        The fixed subject identifier returned for every authentication
        attempt.  Defaults to ``"dev-user"``.

    Raises
    ------
    RuntimeError
        If instantiated outside of a ``development`` or ``test`` context
        (detected via the ``RISKFORGE_ENV`` environment variable).
    """

    def __init__(self, subject_id: str = "dev-user") -> None:
        import os

        env = os.environ.get("RISKFORGE_ENV", "development")
        if env not in ("development", "test"):
            raise RuntimeError(
                f"DevAuthenticationService must not be used in '{env}' environment. "
                "Use a production authentication provider."
            )
        if env == "development":
            warnings.warn(
                "DevAuthenticationService is active. "
                "Do not use in production.",
                stacklevel=2,
            )
        self._principal = Principal(subject_id=subject_id)

    def authenticate(self, credential: str) -> Principal:
        """Return the static principal, ignoring the credential value."""
        return self._principal
