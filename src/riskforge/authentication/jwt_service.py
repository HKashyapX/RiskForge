"""Production authentication provider using offline HS256 JWT verification.

This module implements the ``AuthenticationService`` protocol using
HMAC-SHA256 signed JSON Web Tokens (JWTs) with an operator-provisioned
local keyring.  Verification requires exactly zero network access and
is fully compatible with air-gapped deployments.

Keys are loaded from a local directory of ``<kid>.key`` files at
construction time and held in memory for the lifetime of the service.

Operators mint tokens offline using HMAC-SHA256 with the same shared
secret and distribute the signing secrets via a read-only mounted
volume (e.g., Docker Secrets).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Callable

from riskforge.authentication.exceptions import (
    ExpiredCredentialError,
    InvalidAudienceError,
    InvalidCredentialsError,
    InvalidIssuerError,
    InvalidSignatureError,
    MalformedCredentialError,
    UnknownKeyError,
)
from riskforge.authentication.principal import Principal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_KID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
_MIN_KEY_BYTES = 32
_DEFAULT_MAX_TOKEN_BYTES = 16_384

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _b64url_decode(data: str) -> bytes:
    """Decode a base64url-encoded string (no padding required)."""
    padded = data + "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(padded)


def _b64url_encode(data: bytes) -> str:
    """Encode bytes to a base64url string without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _parse_jwt_header(header_b64: str) -> dict[str, object]:
    """Decode and parse the JWT header segment.

    Returns
    -------
    dict
        The parsed header as a JSON object.

    Raises
    ------
    MalformedCredentialError
        If the header cannot be decoded or is not a JSON object.
    """
    try:
        raw = _b64url_decode(header_b64)
    except Exception as exc:
        raise MalformedCredentialError() from exc

    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MalformedCredentialError() from exc

    if not isinstance(decoded, dict):
        raise MalformedCredentialError()

    return decoded


def _parse_jwt_payload(payload_b64: str) -> dict[str, object]:
    """Decode and parse the JWT payload segment.

    Returns
    -------
    dict
        The parsed payload as a JSON object.

    Raises
    ------
    MalformedCredentialError
        If the payload cannot be decoded or is not a JSON object.
    """
    try:
        raw = _b64url_decode(payload_b64)
    except Exception as exc:
        raise MalformedCredentialError() from exc

    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise MalformedCredentialError() from exc

    if not isinstance(decoded, dict):
        raise MalformedCredentialError()

    return decoded


def _load_keyring(keys_dir: Path) -> dict[str, bytes]:
    """Load key material from a directory of ``<kid>.key`` files.

    Parameters
    ----------
    keys_dir:
        Path to the directory containing key files.

    Returns
    -------
    dict
        Mapping of key ID to secret bytes.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    NotADirectoryError
        If the path is not a directory.
    ValueError
        If the directory is empty or any key file is invalid.
    """
    if not keys_dir.exists():
        raise FileNotFoundError(f"Keys directory does not exist: {keys_dir}")

    if not keys_dir.is_dir():
        raise NotADirectoryError(f"Keys path is not a directory: {keys_dir}")

    keyring: dict[str, bytes] = {}

    for entry in sorted(keys_dir.iterdir()):
        if entry.is_dir():
            continue

        stem = entry.stem
        if not _KID_PATTERN.match(stem):
            raise ValueError(
                f"Invalid key filename '{entry.name}': "
                f"kid must match {_KID_PATTERN.pattern}"
            )

        secret = entry.read_bytes()

        # Strip trailing whitespace (standard secret-file provisioning).
        # Handles both Unix (\n) and Windows (\r\n) line endings.
        secret = secret.rstrip(b"\r\n")

        if len(secret) < _MIN_KEY_BYTES:
            raise ValueError(
                f"Key '{stem}' is too short: {len(secret)} bytes "
                f"(minimum {_MIN_KEY_BYTES})"
            )

        keyring[stem] = secret

    if not keyring:
        raise ValueError(f"No valid key files found in {keys_dir}")

    return keyring


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class JwtAuthenticationService:
    """Production ``AuthenticationService`` using offline HS256 JWT verification.

    Parameters
    ----------
    issuer:
        Expected ``iss`` claim value.  Tokens with a different or missing
        issuer are rejected.
    audience:
        Expected ``aud`` claim value.  Tokens with a different or missing
        audience are rejected.
    keyring:
        Mapping of key ID (``kid``) to HMAC shared secret.  The key
        ``kid`` in the token header selects which secret is used for
        signature verification.
    leeway_seconds:
        Allowable clock skew in seconds when checking token expiration.
        The default ``0.0`` is strict (no tolerance).
    max_token_bytes:
        Maximum allowed credential length in bytes.  Tokens exceeding
        this size are rejected before any parsing.
    now_provider:
        Callable returning the current time as a UNIX timestamp.  Inject
        a deterministic clock for testing.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        keyring: Mapping[str, bytes | str],
        leeway_seconds: float = 0.0,
        max_token_bytes: int = _DEFAULT_MAX_TOKEN_BYTES,
        now_provider: Callable[[], float] = time.time,
    ) -> None:
        # --- Configuration validation (fail-fast) ---
        if not issuer:
            raise ValueError("issuer must be non-empty")
        if not audience:
            raise ValueError("audience must be non-empty")
        if not isinstance(keyring, Mapping) or not keyring:
            raise ValueError("keyring must be a non-empty mapping")

        normalized_keyring: dict[str, bytes] = {}
        for kid, secret in keyring.items():
            if not isinstance(kid, str) or not kid:
                raise ValueError("keyring keys must be non-empty strings")
            if isinstance(secret, str):
                secret_bytes = secret.encode("utf-8")
            else:
                secret_bytes = secret
            normalized_keyring[kid] = secret_bytes

        if not math.isfinite(leeway_seconds) or leeway_seconds < 0:
            raise ValueError("leeway_seconds must be a finite number >= 0")

        self._issuer = issuer
        self._audience = audience
        self._keyring = normalized_keyring
        self._leeway = leeway_seconds
        self._max_token_bytes = max_token_bytes
        self._now = now_provider

    @classmethod
    def from_env(cls) -> JwtAuthenticationService:
        """Construct a service from environment variables.

        Reads:

        - ``RISKFORGE_AUTH_ISSUER`` (required)
        - ``RISKFORGE_AUTH_AUDIENCE`` (required)
        - ``RISKFORGE_AUTH_KEYS_DIR`` (required)
        - ``RISKFORGE_AUTH_LEEWAY_SECONDS`` (optional, default ``0``)

        Raises
        ------
        RuntimeError
            If any required variable is missing or the keys directory is
            invalid.
        """
        issuer = os.environ.get("RISKFORGE_AUTH_ISSUER", "")
        if not issuer:
            raise RuntimeError("RISKFORGE_AUTH_ISSUER is not set")

        audience = os.environ.get("RISKFORGE_AUTH_AUDIENCE", "")
        if not audience:
            raise RuntimeError("RISKFORGE_AUTH_AUDIENCE is not set")

        keys_dir_str = os.environ.get("RISKFORGE_AUTH_KEYS_DIR", "")
        if not keys_dir_str:
            raise RuntimeError("RISKFORGE_AUTH_KEYS_DIR is not set")

        keys_dir = Path(keys_dir_str)
        try:
            keyring = _load_keyring(keys_dir)
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            raise RuntimeError(str(exc)) from exc

        leeway_str = os.environ.get("RISKFORGE_AUTH_LEEWAY_SECONDS", "0")
        try:
            leeway = float(leeway_str)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(
                f"Invalid RISKFORGE_AUTH_LEEWAY_SECONDS: {leeway_str!r}"
            ) from exc

        if not math.isfinite(leeway) or leeway < 0:
            raise RuntimeError(
                f"RISKFORGE_AUTH_LEEWAY_SECONDS must be a finite number >= 0: {leeway}"
            )

        return cls(
            issuer=issuer,
            audience=audience,
            keyring=keyring,
            leeway_seconds=leeway,
        )

    # ------------------------------------------------------------------
    # AuthenticationService protocol
    # ------------------------------------------------------------------

    def authenticate(self, credential: str) -> Principal:
        """Verify a JWT credential and return the authenticated ``Principal``.

        Verification is deterministic and requires exactly zero network
        access.  The credential is never logged or stored.

        Parameters
        ----------
        credential:
            The raw JWT token string (without ``Bearer `` prefix).

        Returns
        -------
        Principal
            The authenticated subject.

        Raises
        ------
        MalformedCredentialError
            If the token structure is invalid.
        UnknownKeyError
            If the ``kid`` is not in the keyring.
        InvalidSignatureError
            If the HMAC signature does not match.
        ExpiredCredentialError
            If the token has expired.
        InvalidIssuerError
            If the issuer claim is missing or mismatched.
        InvalidAudienceError
            If the audience claim is missing or mismatched.
        InvalidCredentialsError
            If the subject claim is invalid or missing.
        """
        # --- 1. Credential size ---
        if len(credential.encode("utf-8")) > self._max_token_bytes:
            raise MalformedCredentialError()

        # --- 2. Structure ---
        parts = credential.split(".")
        if len(parts) != 3:
            raise MalformedCredentialError()

        header_b64, payload_b64, signature_b64 = parts

        # --- 3. Header decoding ---
        header = _parse_jwt_header(header_b64)

        # --- 4. Algorithm ---
        alg = header.get("alg")
        if alg != "HS256":
            raise MalformedCredentialError()

        # --- 5. Key ID ---
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise UnknownKeyError()

        secret = self._keyring.get(kid)
        if secret is None:
            raise UnknownKeyError()

        # --- 6. Signature ---
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        try:
            expected_sig = hmac.new(secret, signing_input, hashlib.sha256).digest()
        except Exception as exc:
            raise MalformedCredentialError() from exc

        try:
            claimed_sig = _b64url_decode(signature_b64)
        except Exception as exc:
            raise MalformedCredentialError() from exc

        if not hmac.compare_digest(expected_sig, claimed_sig):
            raise InvalidSignatureError()

        # --- 7. Payload decoding ---
        payload = _parse_jwt_payload(payload_b64)

        # --- 8. Expiration ---
        exp = payload.get("exp")
        if exp is None or isinstance(exp, bool) or not isinstance(exp, (int, float)):
            raise MalformedCredentialError()

        now = self._now()
        if now > exp + self._leeway:
            raise ExpiredCredentialError()

        # --- 9. Issuer ---
        iss = payload.get("iss")
        if not isinstance(iss, str) or iss != self._issuer:
            raise InvalidIssuerError()

        # --- 10. Audience ---
        aud = payload.get("aud")
        if isinstance(aud, str):
            if aud != self._audience:
                raise InvalidAudienceError()
        elif isinstance(aud, list):
            if self._audience not in aud or not all(
                isinstance(item, str) for item in aud
            ):
                raise InvalidAudienceError()
        else:
            raise InvalidAudienceError()

        # --- 11. Subject ---
        sub = payload.get("sub")
        if not isinstance(sub, str) or not sub or len(sub) > 128:
            raise InvalidCredentialsError()

        # --- 12. Return ---
        return Principal(subject_id=sub)
