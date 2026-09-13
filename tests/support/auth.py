"""Shared test support: deterministic auth doubles and JWT token minting."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from riskforge.authentication.principal import Principal

_SECRET = b"unit-test-signing-secret-0123456789abcdef"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def mint_hs256_token(
    secret: bytes = _SECRET,
    *,
    subject: str = "test-user",
    issuer: str = "riskforge-test",
    audience: str = "riskforge-api",
    roles: list[str] | None = None,
    scopes: list[str] | None = None,
    issued_at: float | None = None,
    expires_after_s: float = 3600.0,
    not_before: float | None = None,
    kid: str = "test-key",
    override_claims: dict[str, Any] | None = None,
) -> str:
    """Mint a deterministic HS256 JWT for authentication tests."""
    now = time.time() if issued_at is None else issued_at
    header = {"alg": "HS256", "typ": "JWT", "kid": kid}
    payload: dict[str, Any] = {
        "sub": subject,
        "iss": issuer,
        "aud": audience,
        "iat": int(now),
        "exp": int(now + expires_after_s),
    }
    if roles is not None:
        payload["roles"] = roles
    if scopes is not None:
        payload["scope"] = " ".join(scopes)
    if not_before is not None:
        payload["nbf"] = int(not_before)
    if override_claims:
        payload.update(override_claims)
    segments = [
        _b64url(json.dumps(header, separators=(",", ":")).encode()),
        _b64url(json.dumps(payload, separators=(",", ":")).encode()),
    ]
    signing_input = ".".join(segments).encode()
    signature = hmac.new(secret, signing_input, hashlib.sha256).digest()
    segments.append(_b64url(signature))
    return ".".join(segments)


class RolePrincipalAuth:
    """Deterministic AuthenticationService double returning a fixed principal."""

    def __init__(self, principal: Principal, valid_credential: str = "valid-token") -> None:
        self._principal = principal
        self._valid = valid_credential
        self.calls: list[str] = []

    def authenticate(self, credential: str) -> Principal:
        self.calls.append(credential)
        if credential == self._valid:
            return self._principal
        from riskforge.authentication.exceptions import InvalidCredentialsError

        raise InvalidCredentialsError()
