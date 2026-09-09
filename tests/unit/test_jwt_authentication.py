"""Comprehensive tests for the production JWT authentication service.

Covers token verification, key management, configuration validation,
security invariants, and protocol conformance for offline HS256 JWT
authentication.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import socket
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from riskforge.authentication.exceptions import (
    AuthenticationError,
    ExpiredCredentialError,
    InvalidAudienceError,
    InvalidCredentialsError,
    InvalidIssuerError,
    InvalidSignatureError,
    MalformedCredentialError,
    MissingCredentialsError,
    UnknownKeyError,
)
from riskforge.authentication.jwt_service import (
    JwtAuthenticationService,
    _load_keyring,
    _parse_jwt_header,
    _parse_jwt_payload,
)
from riskforge.authentication.principal import Principal
from riskforge.authentication.protocols import AuthenticationService

# ---------------------------------------------------------------------------
# Test helpers — JWT fixture construction
# ---------------------------------------------------------------------------

_DEFAULT_ISSUER = "riskforge"
_DEFAULT_AUDIENCE = "riskforge-api"
_DEFAULT_SECRET = "test-secret-key-that-is-long-enough-256-bits!"
_DEFAULT_KID = "test-key-1"


def _b64url_encode(data: bytes) -> str:
    """Encode bytes to base64url without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _mint_token(
    *,
    sub: str = "test-user",
    iss: str = _DEFAULT_ISSUER,
    aud: str | list[str] | None = _DEFAULT_AUDIENCE,
    exp: float | None = None,
    secret: str = _DEFAULT_SECRET,
    kid: str = _DEFAULT_KID,
    alg: str = "HS256",
    include_header: dict[str, Any] | None = None,
    include_payload: dict[str, Any] | None = None,
) -> str:
    """Construct a valid (or intentionally malformed) JWT for testing."""
    now = int(time.time())
    if exp is None:
        exp = now + 3600  # 1 hour from now

    header = {"alg": alg, "kid": kid}
    if include_header is not None:
        header = include_header

    payload: dict[str, Any] = {"sub": sub, "exp": exp, "iss": iss}
    if aud is not None:
        payload["aud"] = aud
    if include_payload is not None:
        payload = include_payload

    header_b64 = _b64url_encode(json.dumps(header).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(payload).encode("utf-8"))

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64url_encode(sig)

    return f"{header_b64}.{payload_b64}.{sig_b64}"


def _make_service(**kwargs: Any) -> JwtAuthenticationService:
    """Create a service with sensible defaults for testing."""
    defaults = {
        "issuer": _DEFAULT_ISSUER,
        "audience": _DEFAULT_AUDIENCE,
        "keyring": {_DEFAULT_KID: _DEFAULT_SECRET},
    }
    defaults.update(kwargs)
    return JwtAuthenticationService(**defaults)


# ---------------------------------------------------------------------------
# Principal model tests
# ---------------------------------------------------------------------------


class TestPrincipal:
    def test_principal_is_frozen(self) -> None:
        principal = Principal(subject_id="user-1")
        with pytest.raises(Exception):
            principal.subject_id = "user-2"  # type: ignore[misc]

    def test_principal_rejects_extra_fields(self) -> None:
        with pytest.raises(Exception):
            Principal(subject_id="user-1", role="admin")  # type: ignore[call-arg]

    def test_principal_requires_non_empty_subject_id(self) -> None:
        with pytest.raises(Exception):
            Principal(subject_id="")

    def test_principal_rejects_long_subject_id(self) -> None:
        with pytest.raises(Exception):
            Principal(subject_id="x" * 129)

    def test_principal_accepts_max_length_subject_id(self) -> None:
        principal = Principal(subject_id="x" * 128)
        assert principal.subject_id == "x" * 128

    def test_principal_equality(self) -> None:
        p1 = Principal(subject_id="user-1")
        p2 = Principal(subject_id="user-1")
        assert p1 == p2

    def test_principal_inequality(self) -> None:
        p1 = Principal(subject_id="user-1")
        p2 = Principal(subject_id="user-2")
        assert p1 != p2

    def test_principal_is_hashable(self) -> None:
        principal = Principal(subject_id="user-1")
        assert hash(principal) == hash(Principal(subject_id="user-1"))
        assert len({principal, Principal(subject_id="user-1")}) == 1

    def test_principal_serialization_roundtrip(self) -> None:
        original = Principal(subject_id="user-1")
        data = original.model_dump()
        restored = Principal.model_validate(data)
        assert original == restored


# ---------------------------------------------------------------------------
# Exception hierarchy tests
# ---------------------------------------------------------------------------


class TestAuthenticationExceptionHierarchy:
    def test_missing_credentials_is_authentication_error(self) -> None:
        assert issubclass(MissingCredentialsError, AuthenticationError)

    def test_invalid_credentials_is_authentication_error(self) -> None:
        assert issubclass(InvalidCredentialsError, AuthenticationError)

    def test_authentication_error_is_runtime_error(self) -> None:
        assert issubclass(AuthenticationError, RuntimeError)

    def test_malformed_credential_is_invalid(self) -> None:
        assert issubclass(MalformedCredentialError, InvalidCredentialsError)

    def test_expired_credential_is_invalid(self) -> None:
        assert issubclass(ExpiredCredentialError, InvalidCredentialsError)

    def test_invalid_issuer_is_invalid(self) -> None:
        assert issubclass(InvalidIssuerError, InvalidCredentialsError)

    def test_invalid_audience_is_invalid(self) -> None:
        assert issubclass(InvalidAudienceError, InvalidCredentialsError)

    def test_unknown_key_is_invalid(self) -> None:
        assert issubclass(UnknownKeyError, InvalidCredentialsError)

    def test_invalid_signature_is_invalid(self) -> None:
        assert issubclass(InvalidSignatureError, InvalidCredentialsError)

    def test_exception_messages_do_not_contain_credentials(self) -> None:
        credential = "super-secret-token-12345"
        exceptions = [
            MissingCredentialsError(),
            InvalidCredentialsError(),
            MalformedCredentialError(),
            ExpiredCredentialError(),
            InvalidIssuerError(),
            InvalidAudienceError(),
            UnknownKeyError(),
            InvalidSignatureError(),
        ]
        for exc in exceptions:
            assert credential not in str(exc), (
                f"{type(exc).__name__} exposed credential in str representation"
            )


class TestExceptionReasonAttributes:
    def test_malformed_credential_reason(self) -> None:
        assert MalformedCredentialError.reason == "malformed"

    def test_expired_credential_reason(self) -> None:
        assert ExpiredCredentialError.reason == "expired"

    def test_invalid_issuer_reason(self) -> None:
        assert InvalidIssuerError.reason == "invalid_issuer"

    def test_invalid_audience_reason(self) -> None:
        assert InvalidAudienceError.reason == "invalid_audience"

    def test_unknown_key_reason(self) -> None:
        assert UnknownKeyError.reason == "unknown_key"

    def test_invalid_signature_reason(self) -> None:
        assert InvalidSignatureError.reason == "invalid_signature"


# ---------------------------------------------------------------------------
# Valid token tests
# ---------------------------------------------------------------------------


class TestValidToken:
    def test_valid_token_returns_principal(self) -> None:
        token = _mint_token(sub="alice")
        service = _make_service()
        principal = service.authenticate(token)
        assert principal.subject_id == "alice"

    def test_valid_token_with_different_subject(self) -> None:
        token = _mint_token(sub="bob-123")
        service = _make_service()
        principal = service.authenticate(token)
        assert principal.subject_id == "bob-123"

    def test_valid_token_with_max_length_subject(self) -> None:
        token = _mint_token(sub="x" * 128)
        service = _make_service()
        principal = service.authenticate(token)
        assert len(principal.subject_id) == 128

    def test_returns_principal_instance(self) -> None:
        token = _mint_token()
        service = _make_service()
        result = service.authenticate(token)
        assert isinstance(result, Principal)

    def test_valid_token_with_audience_list(self) -> None:
        token = _mint_token(aud=["riskforge-api", "riskforge-admin"])
        service = _make_service()
        principal = service.authenticate(token)
        assert principal.subject_id == "test-user"


# ---------------------------------------------------------------------------
# Malformed credential tests
# ---------------------------------------------------------------------------


class TestMalformedCredentials:
    def test_empty_credential(self) -> None:
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate("")

    def test_two_segments(self) -> None:
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate("abc.def")

    def test_four_segments(self) -> None:
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate("a.b.c.d")

    def test_one_segment(self) -> None:
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate("not-a-jwt")

    def test_invalid_base64_header(self) -> None:
        payload_b64 = _b64url_encode(json.dumps({"sub": "x"}).encode("utf-8"))
        sig_b64 = _b64url_encode(b"fake-sig")
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(f"!!!invalid!!!.{payload_b64}.{sig_b64}")

    def test_invalid_json_header(self) -> None:
        raw_header = b"{not valid json}"
        header_b64 = _b64url_encode(raw_header)
        payload_b64 = _b64url_encode(json.dumps({"sub": "x"}).encode("utf-8"))
        sig_b64 = _b64url_encode(b"fake-sig")
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(f"{header_b64}.{payload_b64}.{sig_b64}")

    def test_non_object_header(self) -> None:
        header_b64 = _b64url_encode(json.dumps("just-a-string").encode("utf-8"))
        payload_b64 = _b64url_encode(json.dumps({"sub": "x"}).encode("utf-8"))
        sig_b64 = _b64url_encode(b"fake-sig")
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(f"{header_b64}.{payload_b64}.{sig_b64}")

    def test_invalid_json_payload(self) -> None:
        """Invalid JSON payload with a valid signature → MalformedCredentialError.

        Signature check (step 6) happens before payload decoding (step 7),
        so we must sign the malformed payload with the correct key to pass
        the signature check and reach the payload parsing step.
        """
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(b"{bad json")
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        sig_b64 = _b64url_encode(sig)
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(f"{header_b64}.{payload_b64}.{sig_b64}")

    def test_non_object_payload(self) -> None:
        """Non-object payload with valid signature → MalformedCredentialError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(json.dumps("not-an-object").encode("utf-8"))
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        sig_b64 = _b64url_encode(sig)
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(f"{header_b64}.{payload_b64}.{sig_b64}")

    def test_missing_alg(self) -> None:
        header_b64 = _b64url_encode(
            json.dumps({"kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(json.dumps({"sub": "x"}).encode("utf-8"))
        sig_b64 = _b64url_encode(b"fake-sig")
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(f"{header_b64}.{payload_b64}.{sig_b64}")

    def test_alg_none(self) -> None:
        token = _mint_token(alg="none")
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(token)

    def test_alg_rs256(self) -> None:
        token = _mint_token(alg="RS256")
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(token)

    def test_alg_es256(self) -> None:
        token = _mint_token(alg="ES256")
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(token)

    def test_alg_hs384(self) -> None:
        token = _mint_token(alg="HS384")
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(token)

    def test_alg_hs512(self) -> None:
        token = _mint_token(alg="HS512")
        service = _make_service()
        with pytest.raises(MalformedCredentialError):
            service.authenticate(token)

    def test_oversized_token(self) -> None:
        service = _make_service(max_token_bytes=100)
        token = _mint_token()
        oversized = token + "x" * 100
        with pytest.raises(MalformedCredentialError):
            service.authenticate(oversized)

    def test_malformed_exp_missing(self) -> None:
        """Missing exp → MalformedCredentialError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps({"sub": "x", "iss": _DEFAULT_ISSUER, "aud": _DEFAULT_AUDIENCE}).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        sig_b64 = _b64url_encode(sig)
        bad_token = f"{header_b64}.{payload_b64}.{sig_b64}"
        with pytest.raises(MalformedCredentialError):
            _make_service().authenticate(bad_token)

    def test_malformed_exp_boolean(self) -> None:
        """Boolean exp → MalformedCredentialError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {"sub": "x", "exp": True, "iss": _DEFAULT_ISSUER, "aud": _DEFAULT_AUDIENCE}
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        sig_b64 = _b64url_encode(sig)
        bad_token = f"{header_b64}.{payload_b64}.{sig_b64}"
        with pytest.raises(MalformedCredentialError):
            _make_service().authenticate(bad_token)

    def test_malformed_exp_string(self) -> None:
        """String exp → MalformedCredentialError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {"sub": "x", "exp": "not-a-number", "iss": _DEFAULT_ISSUER, "aud": _DEFAULT_AUDIENCE}
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        sig_b64 = _b64url_encode(sig)
        bad_token = f"{header_b64}.{payload_b64}.{sig_b64}"
        with pytest.raises(MalformedCredentialError):
            _make_service().authenticate(bad_token)


# ---------------------------------------------------------------------------
# Signature tests
# ---------------------------------------------------------------------------


class TestSignature:
    def test_invalid_signature(self) -> None:
        """Token signed with a different key → InvalidSignatureError."""
        token = _mint_token(secret="wrong-key-that-is-long-enough-for-testing!")
        with pytest.raises(InvalidSignatureError):
            _make_service().authenticate(token)

    def test_tampered_payload_invalid_signature(self) -> None:
        """Modifying the payload without re-signing → InvalidSignatureError."""
        token = _mint_token(sub="original-user")
        parts = token.split(".")
        # Build a new payload but DON'T re-sign (use original signature)
        tampered_payload = _b64url_encode(
            json.dumps(
                {"sub": "tampered", "exp": 9999999999, "iss": _DEFAULT_ISSUER, "aud": _DEFAULT_AUDIENCE}
            ).encode("utf-8")
        )
        tampered = f"{parts[0]}.{tampered_payload}.{parts[2]}"
        with pytest.raises(InvalidSignatureError):
            _make_service().authenticate(tampered)

    def test_token_signed_with_wrong_key(self) -> None:
        """Token signed with a key not in the keyring → InvalidSignatureError."""
        token = _mint_token(secret="a-completely-different-secret-key-256bit!!!")
        with pytest.raises(InvalidSignatureError):
            _make_service().authenticate(token)

    def test_signature_uses_original_header_payload_bytes(self) -> None:
        """Signature is verified over the original encoded header.payload."""
        token = _mint_token(sub="test-user")
        # Valid token verifies fine
        assert _make_service().authenticate(token).subject_id == "test-user"

        # Modify payload without re-signing → signature fails
        parts = token.split(".")
        new_payload = _b64url_encode(
            json.dumps(
                {"sub": "other", "exp": int(time.time()) + 3600, "iss": _DEFAULT_ISSUER, "aud": _DEFAULT_AUDIENCE}
            ).encode("utf-8")
        )
        tampered = f"{parts[0]}.{new_payload}.{parts[2]}"
        with pytest.raises(InvalidSignatureError):
            _make_service().authenticate(tampered)


# ---------------------------------------------------------------------------
# Expiration tests
# ---------------------------------------------------------------------------


class TestExpiration:
    def test_expired_token(self) -> None:
        """Token with exp in the past → ExpiredCredentialError."""
        token = _mint_token(exp=int(time.time()) - 100)
        with pytest.raises(ExpiredCredentialError):
            _make_service().authenticate(token)

    def test_valid_token_at_expiration_boundary(self) -> None:
        """Token with exp == now (int) should be valid (not expired)."""
        # Use a fixed integer timestamp to avoid float truncation issues
        now = 1700000000
        token = _mint_token(exp=now)
        service = _make_service(now_provider=lambda: float(now))
        principal = service.authenticate(token)
        assert principal.subject_id == "test-user"

    def test_token_one_second_before_expiry(self) -> None:
        """Token with exp == now + 1 should be valid."""
        now = 1700000000
        token = _mint_token(exp=now + 1)
        service = _make_service(now_provider=lambda: float(now))
        principal = service.authenticate(token)
        assert principal.subject_id == "test-user"

    def test_token_one_second_after_expiry(self) -> None:
        """Token with exp == now - 1 should be expired."""
        now = 1700000000
        token = _mint_token(exp=now - 1)
        service = _make_service(now_provider=lambda: float(now))
        with pytest.raises(ExpiredCredentialError):
            service.authenticate(token)

    def test_leeway_allows_recently_expired(self) -> None:
        """Leeway allows recently expired tokens."""
        now = 1700000000
        # Expired 5 seconds ago
        token = _mint_token(exp=now - 5)
        # Without leeway: expired
        service_strict = _make_service(now_provider=lambda: float(now))
        with pytest.raises(ExpiredCredentialError):
            service_strict.authenticate(token)
        # With leeway: valid
        service_leeway = _make_service(leeway_seconds=10, now_provider=lambda: float(now))
        principal = service_leeway.authenticate(token)
        assert principal.subject_id == "test-user"

    def test_injected_clock(self) -> None:
        """now_provider is used for expiration checks."""
        now = 1700000000.0
        token = _mint_token(exp=int(now) + 100)
        service = _make_service(now_provider=lambda: now)
        principal = service.authenticate(token)
        assert principal.subject_id == "test-user"

    def test_clock_advancing_past_expiry(self) -> None:
        """Valid token becomes expired when clock advances past exp."""
        now = 1700000000.0
        token = _mint_token(exp=int(now) + 10)

        call_count = [0]

        def advancing_clock() -> float:
            call_count[0] += 1
            if call_count[0] == 1:
                return now  # First call: valid
            return now + 100  # Second call: expired

        service = _make_service(now_provider=advancing_clock)
        # First call uses now (valid)
        principal = service.authenticate(token)
        assert principal.subject_id == "test-user"
        # Second call: clock advanced past expiry
        with pytest.raises(ExpiredCredentialError):
            service.authenticate(token)


# ---------------------------------------------------------------------------
# Issuer tests
# ---------------------------------------------------------------------------


class TestIssuer:
    def test_valid_issuer(self) -> None:
        token = _mint_token(iss=_DEFAULT_ISSUER)
        assert _make_service().authenticate(token).subject_id == "test-user"

    def test_wrong_issuer(self) -> None:
        token = _mint_token(iss="wrong-issuer")
        with pytest.raises(InvalidIssuerError):
            _make_service().authenticate(token)

    def test_missing_issuer(self) -> None:
        """Token without iss → InvalidIssuerError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {"sub": "x", "exp": int(time.time()) + 3600, "aud": _DEFAULT_AUDIENCE}
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        bad_token = f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"
        with pytest.raises(InvalidIssuerError):
            _make_service().authenticate(bad_token)


# ---------------------------------------------------------------------------
# Audience tests
# ---------------------------------------------------------------------------


class TestAudience:
    def test_valid_audience_string(self) -> None:
        token = _mint_token(aud=_DEFAULT_AUDIENCE)
        assert _make_service().authenticate(token).subject_id == "test-user"

    def test_wrong_audience_string(self) -> None:
        token = _mint_token(aud="wrong-audience")
        with pytest.raises(InvalidAudienceError):
            _make_service().authenticate(token)

    def test_missing_audience(self) -> None:
        """Token without aud → InvalidAudienceError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {"sub": "x", "exp": int(time.time()) + 3600, "iss": _DEFAULT_ISSUER}
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        bad_token = f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"
        with pytest.raises(InvalidAudienceError):
            _make_service().authenticate(bad_token)

    def test_valid_audience_list(self) -> None:
        token = _mint_token(aud=["riskforge-api", "other-api"])
        assert _make_service().authenticate(token).subject_id == "test-user"

    def test_audience_list_without_match(self) -> None:
        token = _mint_token(aud=["wrong-api", "other-api"])
        with pytest.raises(InvalidAudienceError):
            _make_service().authenticate(token)

    def test_audience_invalid_type_int(self) -> None:
        """Non-string/non-list aud → InvalidAudienceError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {"sub": "x", "exp": int(time.time()) + 3600, "iss": _DEFAULT_ISSUER, "aud": 12345}
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        bad_token = f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"
        with pytest.raises(InvalidAudienceError):
            _make_service().authenticate(bad_token)

    def test_audience_list_with_invalid_entries(self) -> None:
        """Audience list containing non-strings → InvalidAudienceError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {"sub": "x", "exp": int(time.time()) + 3600, "iss": _DEFAULT_ISSUER, "aud": [123, "riskforge-api"]}
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        bad_token = f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"
        with pytest.raises(InvalidAudienceError):
            _make_service().authenticate(bad_token)

    def test_audience_empty_string(self) -> None:
        """Empty string aud → InvalidAudienceError."""
        token = _mint_token(aud="")
        with pytest.raises(InvalidAudienceError):
            _make_service().authenticate(token)

    def test_audience_empty_list(self) -> None:
        """Empty list aud → InvalidAudienceError."""
        token = _mint_token(aud=[])
        with pytest.raises(InvalidAudienceError):
            _make_service().authenticate(token)


# ---------------------------------------------------------------------------
# Subject tests
# ---------------------------------------------------------------------------


class TestSubject:
    def test_valid_subject(self) -> None:
        token = _mint_token(sub="valid-user")
        assert _make_service().authenticate(token).subject_id == "valid-user"

    def test_missing_subject(self) -> None:
        """Token without sub → InvalidCredentialsError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {"exp": int(time.time()) + 3600, "iss": _DEFAULT_ISSUER, "aud": _DEFAULT_AUDIENCE}
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        bad_token = f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"
        with pytest.raises(InvalidCredentialsError):
            _make_service().authenticate(bad_token)

    def test_empty_subject(self) -> None:
        """Empty sub → InvalidCredentialsError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {"sub": "", "exp": int(time.time()) + 3600, "iss": _DEFAULT_ISSUER, "aud": _DEFAULT_AUDIENCE}
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        bad_token = f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"
        with pytest.raises(InvalidCredentialsError):
            _make_service().authenticate(bad_token)

    def test_invalid_subject_type(self) -> None:
        """Non-string sub → InvalidCredentialsError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {"sub": 12345, "exp": int(time.time()) + 3600, "iss": _DEFAULT_ISSUER, "aud": _DEFAULT_AUDIENCE}
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        bad_token = f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"
        with pytest.raises(InvalidCredentialsError):
            _make_service().authenticate(bad_token)

    def test_subject_over_128_chars(self) -> None:
        """Sub > 128 chars → InvalidCredentialsError (Pydantic validation caught)."""
        token = _mint_token(sub="z" * 129)
        with pytest.raises(InvalidCredentialsError):
            _make_service().authenticate(token)

    def test_subject_at_128_chars(self) -> None:
        """Sub == 128 chars → valid."""
        token = _mint_token(sub="y" * 128)
        principal = _make_service().authenticate(token)
        assert principal.subject_id == "y" * 128


# ---------------------------------------------------------------------------
# Key management tests
# ---------------------------------------------------------------------------


class TestKeyManagement:
    def test_missing_kid(self) -> None:
        """Token without kid → UnknownKeyError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256"}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {"sub": "x", "exp": int(time.time()) + 3600, "iss": _DEFAULT_ISSUER, "aud": _DEFAULT_AUDIENCE}
            ).encode("utf-8")
        )
        sig_b64 = _b64url_encode(b"fake-sig")
        with pytest.raises(UnknownKeyError):
            _make_service().authenticate(f"{header_b64}.{payload_b64}.{sig_b64}")

    def test_unknown_kid(self) -> None:
        """Token with kid not in keyring → UnknownKeyError."""
        token = _mint_token(kid="nonexistent-key")
        with pytest.raises(UnknownKeyError):
            _make_service().authenticate(token)

    def test_valid_kid(self) -> None:
        token = _mint_token(kid=_DEFAULT_KID)
        assert _make_service().authenticate(token).subject_id == "test-user"

    def test_multiple_keys(self) -> None:
        """Service with multiple keys verifies each independently."""
        secret1 = "secret-key-1-that-is-long-enough-256!!!"
        secret2 = "secret-key-2-that-is-long-enough-256!!!"
        keyring = {"key-1": secret1, "key-2": secret2}
        service = _make_service(keyring=keyring)

        token1 = _mint_token(sub="user-1", kid="key-1", secret=secret1)
        token2 = _mint_token(sub="user-2", kid="key-2", secret=secret2)

        assert service.authenticate(token1).subject_id == "user-1"
        assert service.authenticate(token2).subject_id == "user-2"

    def test_key_rotation(self) -> None:
        """After adding a new key, both old and new tokens verify."""
        secret1 = "secret-key-1-that-is-long-enough-256!!"
        secret2 = "secret-key-2-that-is-long-enough-256!!"

        keyring_v1 = {"key-1": secret1}
        service_v1 = _make_service(keyring=keyring_v1)
        token_old = _mint_token(sub="old-user", kid="key-1", secret=secret1)
        assert service_v1.authenticate(token_old).subject_id == "old-user"

        keyring_v2 = {"key-1": secret1, "key-2": secret2}
        service_v2 = _make_service(keyring=keyring_v2)
        token_new = _mint_token(sub="new-user", kid="key-2", secret=secret2)
        assert service_v2.authenticate(token_old).subject_id == "old-user"
        assert service_v2.authenticate(token_new).subject_id == "new-user"

    def test_removed_key_invalidates_old_token(self) -> None:
        """After removing a key, old tokens → UnknownKeyError."""
        secret1 = "secret-key-1-that-is-long-enough-256!!"
        secret2 = "secret-key-2-that-is-long-enough-256!!"

        keyring_v2 = {"key-1": secret1, "key-2": secret2}
        service_v2 = _make_service(keyring=keyring_v2)
        token_old = _mint_token(sub="old-user", kid="key-1", secret=secret1)
        assert service_v2.authenticate(token_old).subject_id == "old-user"

        keyring_v3 = {"key-2": secret2}
        service_v3 = _make_service(keyring=keyring_v3)
        with pytest.raises(UnknownKeyError):
            service_v3.authenticate(token_old)

    def test_no_fallback_verification(self) -> None:
        """Unknown kid never falls back to other keys."""
        keyring = {"key-1": "secret-key-1-that-is-long-enough-256!!"}
        service = _make_service(keyring=keyring)
        token = _mint_token(
            sub="attacker",
            kid="key-2",
            secret="attacker-secret-key-long-enough-for-testing!!!",
        )
        with pytest.raises(UnknownKeyError):
            service.authenticate(token)


# ---------------------------------------------------------------------------
# Key file loading tests
# ---------------------------------------------------------------------------


class TestKeyFileLoading:
    def test_load_valid_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "key-1.key").write_bytes(b"this-is-a-secret-key-long-enough!!!")
            (keys_dir / "key-2.key").write_bytes(b"another-secret-key-long-enough!!!")
            keyring = _load_keyring(keys_dir)
            assert "key-1" in keyring
            assert "key-2" in keyring
            assert len(keyring["key-1"]) > 0

    def test_missing_directory(self) -> None:
        with pytest.raises(FileNotFoundError):
            _load_keyring(Path("/nonexistent/path"))

    def test_not_a_directory(self) -> None:
        with tempfile.NamedTemporaryFile() as f:
            with pytest.raises(NotADirectoryError):
                _load_keyring(Path(f.name))

    def test_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="No valid key files"):
                _load_keyring(Path(tmpdir))

    def test_invalid_kid_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "invalid kid name.key").write_bytes(
                b"this-is-a-secret-key-long-enough!!!"
            )
            with pytest.raises(ValueError, match="Invalid key filename"):
                _load_keyring(keys_dir)

    def test_short_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "short.key").write_bytes(b"too-short")
            with pytest.raises(ValueError, match="too short"):
                _load_keyring(keys_dir)

    def test_empty_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "empty.key").write_bytes(b"")
            with pytest.raises(ValueError, match="too short"):
                _load_keyring(keys_dir)

    def test_key_strips_trailing_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "newlined.key").write_bytes(
                b"secret-key-with-newline-at-end!!\n"
            )
            keyring = _load_keyring(keys_dir)
            assert keyring["newlined"] == b"secret-key-with-newline-at-end!!"

    def test_key_strips_multiple_trailing_newlines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "multi.key").write_bytes(
                b"secret-key-with-newlines-at-end!!!\n\n\n"
            )
            keyring = _load_keyring(keys_dir)
            assert keyring["multi"] == b"secret-key-with-newlines-at-end!!!"

    def test_key_exactly_32_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "min.key").write_bytes(b"1" * 32)
            keyring = _load_keyring(keys_dir)
            assert len(keyring["min"]) == 32


# ---------------------------------------------------------------------------
# Configuration validation tests
# ---------------------------------------------------------------------------


class TestConfiguration:
    def test_missing_issuer(self) -> None:
        with pytest.raises(ValueError, match="issuer"):
            JwtAuthenticationService(
                issuer="",
                audience=_DEFAULT_AUDIENCE,
                keyring={_DEFAULT_KID: _DEFAULT_SECRET},
            )

    def test_missing_audience(self) -> None:
        with pytest.raises(ValueError, match="audience"):
            JwtAuthenticationService(
                issuer=_DEFAULT_ISSUER,
                audience="",
                keyring={_DEFAULT_KID: _DEFAULT_SECRET},
            )

    def test_empty_keyring(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            JwtAuthenticationService(
                issuer=_DEFAULT_ISSUER,
                audience=_DEFAULT_AUDIENCE,
                keyring={},
            )

    def test_negative_leeway(self) -> None:
        with pytest.raises(ValueError, match="leeway"):
            JwtAuthenticationService(
                issuer=_DEFAULT_ISSUER,
                audience=_DEFAULT_AUDIENCE,
                keyring={_DEFAULT_KID: _DEFAULT_SECRET},
                leeway_seconds=-1.0,
            )

    def test_string_keys_converted_to_bytes(self) -> None:
        service = JwtAuthenticationService(
            issuer=_DEFAULT_ISSUER,
            audience=_DEFAULT_AUDIENCE,
            keyring={_DEFAULT_KID: "string-secret-long-enough-for-testing!"},
        )
        assert isinstance(service._keyring[_DEFAULT_KID], bytes)

    def test_custom_max_token_bytes(self) -> None:
        service = JwtAuthenticationService(
            issuer=_DEFAULT_ISSUER,
            audience=_DEFAULT_AUDIENCE,
            keyring={_DEFAULT_KID: _DEFAULT_SECRET},
            max_token_bytes=100,
        )
        assert service._max_token_bytes == 100

    def test_from_env_missing_issuer(self) -> None:
        env = {
            "RISKFORGE_AUTH_AUDIENCE": _DEFAULT_AUDIENCE,
            "RISKFORGE_AUTH_KEYS_DIR": "/tmp/keys",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="RISKFORGE_AUTH_ISSUER"):
                JwtAuthenticationService.from_env()

    def test_from_env_missing_audience(self) -> None:
        env = {
            "RISKFORGE_AUTH_ISSUER": _DEFAULT_ISSUER,
            "RISKFORGE_AUTH_KEYS_DIR": "/tmp/keys",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="RISKFORGE_AUTH_AUDIENCE"):
                JwtAuthenticationService.from_env()

    def test_from_env_missing_keys_dir(self) -> None:
        env = {
            "RISKFORGE_AUTH_ISSUER": _DEFAULT_ISSUER,
            "RISKFORGE_AUTH_AUDIENCE": _DEFAULT_AUDIENCE,
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="RISKFORGE_AUTH_KEYS_DIR"):
                JwtAuthenticationService.from_env()

    def test_from_env_invalid_leeway(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "key-1.key").write_bytes(b"secret-key-long-enough-for-testing!!!!")
            env = {
                "RISKFORGE_AUTH_ISSUER": _DEFAULT_ISSUER,
                "RISKFORGE_AUTH_AUDIENCE": _DEFAULT_AUDIENCE,
                "RISKFORGE_AUTH_KEYS_DIR": str(keys_dir),
                "RISKFORGE_AUTH_LEEWAY_SECONDS": "not-a-number",
            }
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(RuntimeError, match="Invalid RISKFORGE_AUTH_LEEWAY_SECONDS"):
                    JwtAuthenticationService.from_env()

    def test_from_env_negative_leeway(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "key-1.key").write_bytes(b"secret-key-long-enough-for-testing!!!!")
            env = {
                "RISKFORGE_AUTH_ISSUER": _DEFAULT_ISSUER,
                "RISKFORGE_AUTH_AUDIENCE": _DEFAULT_AUDIENCE,
                "RISKFORGE_AUTH_KEYS_DIR": str(keys_dir),
                "RISKFORGE_AUTH_LEEWAY_SECONDS": "-5",
            }
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(RuntimeError, match="finite"):
                    JwtAuthenticationService.from_env()

    def test_from_env_empty_keys_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            env = {
                "RISKFORGE_AUTH_ISSUER": _DEFAULT_ISSUER,
                "RISKFORGE_AUTH_AUDIENCE": _DEFAULT_AUDIENCE,
                "RISKFORGE_AUTH_KEYS_DIR": str(keys_dir),
            }
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(RuntimeError, match="No valid key files"):
                    JwtAuthenticationService.from_env()

    def test_from_env_success(self) -> None:
        """from_env() with valid config constructs the service."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "prod-key.key").write_bytes(
                b"production-secret-key-long-enough-256-bits!!!!"
            )
            env = {
                "RISKFORGE_AUTH_ISSUER": _DEFAULT_ISSUER,
                "RISKFORGE_AUTH_AUDIENCE": _DEFAULT_AUDIENCE,
                "RISKFORGE_AUTH_KEYS_DIR": str(keys_dir),
                "RISKFORGE_AUTH_LEEWAY_SECONDS": "0",
            }
            with patch.dict(os.environ, env, clear=True):
                service = JwtAuthenticationService.from_env()
                assert isinstance(service, JwtAuthenticationService)

    def test_from_env_default_leeway(self) -> None:
        """from_env() without RISKFORGE_AUTH_LEEWAY_SECONDS uses default 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "prod-key.key").write_bytes(
                b"production-secret-key-long-enough-256-bits!!!!"
            )
            env = {
                "RISKFORGE_AUTH_ISSUER": _DEFAULT_ISSUER,
                "RISKFORGE_AUTH_AUDIENCE": _DEFAULT_AUDIENCE,
                "RISKFORGE_AUTH_KEYS_DIR": str(keys_dir),
            }
            with patch.dict(os.environ, env, clear=True):
                service = JwtAuthenticationService.from_env()
                assert service._leeway == 0.0


# ---------------------------------------------------------------------------
# Security tests
# ---------------------------------------------------------------------------


class TestSecurity:
    def test_no_token_in_exception_str(self) -> None:
        """Exception strings must never contain the raw token."""
        token = _mint_token(sub="test-user")
        service = _make_service()
        try:
            service.authenticate(token)
        except AuthenticationError as exc:
            assert token not in str(exc)
            assert token not in repr(exc)

    def test_no_token_in_exception_args(self) -> None:
        """Exception args must never contain the raw token."""
        token = _mint_token(sub="test-user")
        service = _make_service()
        try:
            service.authenticate(token)
        except AuthenticationError as exc:
            assert token not in str(exc.args)

    def test_no_key_material_in_exceptions(self) -> None:
        """Key material must never appear in exception messages."""
        service = _make_service()
        try:
            service.authenticate("invalid.token.here")
        except AuthenticationError as exc:
            assert _DEFAULT_SECRET not in str(exc)
            assert _DEFAULT_SECRET not in repr(exc)

    def test_no_credential_leakage_in_logs(self, caplog: Any) -> None:
        """Credentials must not appear in log output."""
        token = _mint_token(sub="test-user")
        service = _make_service()
        # Successful auth — token should not appear in logs
        with caplog.at_level(logging.DEBUG):
            service.authenticate(token)
            assert token not in caplog.text

        # Failed auth — tampered token should not leak original token
        with caplog.at_level(logging.DEBUG):
            bad_token = _mint_token(sub="x", secret="wrong-key-long-enough-for-testing!!!")
            with pytest.raises(InvalidSignatureError):
                service.authenticate(bad_token)
            assert token not in caplog.text

    def test_offline_authentication(self) -> None:
        """authenticate() must work with no network access."""
        token = _mint_token(sub="offline-user")
        service = _make_service()

        original_connect = socket.socket.connect

        def blocking_connect(self: socket.socket, address: Any) -> None:
            raise ConnectionRefusedError("Network access blocked in test")

        socket.socket.connect = blocking_connect  # type: ignore[assignment]
        try:
            principal = service.authenticate(token)
            assert principal.subject_id == "offline-user"
        finally:
            socket.socket.connect = original_connect

    def test_offline_auth_rejects_bad_token(self) -> None:
        """authenticate() must reject bad tokens without network access."""
        service = _make_service()
        original_connect = socket.socket.connect

        def blocking_connect(self: socket.socket, address: Any) -> None:
            raise ConnectionRefusedError("Network access blocked in test")

        socket.socket.connect = blocking_connect  # type: ignore[assignment]
        try:
            # Use a valid structure but wrong key → InvalidSignatureError
            bad_token = _mint_token(sub="x", secret="wrong-key-long-enough!!!")
            with pytest.raises(InvalidSignatureError):
                service.authenticate(bad_token)
        finally:
            socket.socket.connect = original_connect


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_authentication_service_protocol(self) -> None:
        """JwtAuthenticationService must satisfy the AuthenticationService protocol."""
        service = _make_service()
        assert isinstance(service, AuthenticationService)

    def test_authenticate_returns_principal(self) -> None:
        """authenticate() must return a Principal."""
        token = _mint_token()
        result = _make_service().authenticate(token)
        assert isinstance(result, Principal)

    def test_class_has_authenticate_method(self) -> None:
        """Service must have the authenticate method."""
        service = _make_service()
        assert hasattr(service, "authenticate")
        assert callable(service.authenticate)


# ---------------------------------------------------------------------------
# Import boundary tests
# ---------------------------------------------------------------------------


class TestImportBoundary:
    def _read_source(self) -> str:
        import importlib

        spec = importlib.util.find_spec("riskforge.authentication.jwt_service")
        assert spec is not None
        assert spec.origin is not None
        with open(spec.origin) as f:
            return f.read()

    def test_no_fastapi_import(self) -> None:
        assert "fastapi" not in self._read_source()

    def test_no_http_client_imports(self) -> None:
        source = self._read_source()
        for forbidden in ("httpx", "requests", "aiohttp", "urllib.request"):
            assert forbidden not in source, f"Forbidden import: {forbidden}"

    def test_no_riskforge_core_import(self) -> None:
        assert "riskforge.core" not in self._read_source()

    def test_no_riskforge_metrics_import(self) -> None:
        assert "riskforge.metrics" not in self._read_source()

    def test_no_riskforge_persistence_import(self) -> None:
        assert "riskforge.persistence" not in self._read_source()

    def test_no_riskforge_runtime_import(self) -> None:
        assert "riskforge.runtime" not in self._read_source()


# ---------------------------------------------------------------------------
# Regression: existing authentication tests remain green
# ---------------------------------------------------------------------------


class TestExistingAuthenticationRegression:
    """Ensure the new typed exceptions do not break existing translations."""

    def test_missing_credentials_error_still_translates(self) -> None:
        from riskforge.api.errors import ErrorCode, translate_application_error

        error = translate_application_error(MissingCredentialsError())
        assert error.status_code == 401
        assert error.code == ErrorCode.MISSING_CREDENTIALS

    def test_invalid_credentials_error_still_translates(self) -> None:
        from riskforge.api.errors import ErrorCode, translate_application_error

        error = translate_application_error(InvalidCredentialsError())
        assert error.status_code == 401
        assert error.code == ErrorCode.INVALID_CREDENTIALS

    def test_generic_auth_error_still_translates(self) -> None:
        from riskforge.api.errors import ErrorCode, translate_application_error

        error = translate_application_error(AuthenticationError())
        assert error.status_code == 401
        assert error.code == ErrorCode.INVALID_CREDENTIALS

    def test_new_subtypes_map_to_invalid_credentials(self) -> None:
        """All new typed failures translate to INVALID_CREDENTIALS."""
        from riskforge.api.errors import ErrorCode, translate_application_error

        new_errors = [
            MalformedCredentialError(),
            ExpiredCredentialError(),
            InvalidIssuerError(),
            InvalidAudienceError(),
            UnknownKeyError(),
            InvalidSignatureError(),
        ]
        for exc in new_errors:
            translated = translate_application_error(exc)
            assert translated.status_code == 401
            assert translated.code == ErrorCode.INVALID_CREDENTIALS
            assert translated.retryable is False


# ---------------------------------------------------------------------------
# Edge-case security regression tests
# ---------------------------------------------------------------------------


class TestSecurityEdgeCases:
    """Focused regression tests for subtle security properties."""

    def test_nan_leeway_rejected(self) -> None:
        """NaN leeway must be rejected — NaN comparisons are always False,
        which would cause tokens to never expire."""
        import math

        with pytest.raises(ValueError, match="finite"):
            JwtAuthenticationService(
                issuer=_DEFAULT_ISSUER,
                audience=_DEFAULT_AUDIENCE,
                keyring={_DEFAULT_KID: _DEFAULT_SECRET},
                leeway_seconds=float("nan"),
            )

    def test_inf_leeway_rejected(self) -> None:
        """Inf leeway must be rejected — tokens would never expire."""
        with pytest.raises(ValueError, match="finite"):
            JwtAuthenticationService(
                issuer=_DEFAULT_ISSUER,
                audience=_DEFAULT_AUDIENCE,
                keyring={_DEFAULT_KID: _DEFAULT_SECRET},
                leeway_seconds=float("inf"),
            )

    def test_neg_inf_leeway_rejected(self) -> None:
        """Negative inf leeway must be rejected."""
        with pytest.raises(ValueError, match="finite"):
            JwtAuthenticationService(
                issuer=_DEFAULT_ISSUER,
                audience=_DEFAULT_AUDIENCE,
                keyring={_DEFAULT_KID: _DEFAULT_SECRET},
                leeway_seconds=float("-inf"),
            )

    def test_from_env_rejects_nan_leeway(self) -> None:
        """from_env must reject NaN leeway."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "key-1.key").write_bytes(
                b"secret-key-long-enough-for-testing!!!!"
            )
            env = {
                "RISKFORGE_AUTH_ISSUER": _DEFAULT_ISSUER,
                "RISKFORGE_AUTH_AUDIENCE": _DEFAULT_AUDIENCE,
                "RISKFORGE_AUTH_KEYS_DIR": str(keys_dir),
                "RISKFORGE_AUTH_LEEWAY_SECONDS": "nan",
            }
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(RuntimeError, match="finite"):
                    JwtAuthenticationService.from_env()

    def test_from_env_rejects_inf_leeway(self) -> None:
        """from_env must reject Inf leeway."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "key-1.key").write_bytes(
                b"secret-key-long-enough-for-testing!!!!"
            )
            env = {
                "RISKFORGE_AUTH_ISSUER": _DEFAULT_ISSUER,
                "RISKFORGE_AUTH_AUDIENCE": _DEFAULT_AUDIENCE,
                "RISKFORGE_AUTH_KEYS_DIR": str(keys_dir),
                "RISKFORGE_AUTH_LEEWAY_SECONDS": "inf",
            }
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(RuntimeError, match="finite"):
                    JwtAuthenticationService.from_env()

    def test_from_env_nonexistent_keys_dir(self) -> None:
        """from_env with non-existent keys directory → RuntimeError."""
        env = {
            "RISKFORGE_AUTH_ISSUER": _DEFAULT_ISSUER,
            "RISKFORGE_AUTH_AUDIENCE": _DEFAULT_AUDIENCE,
            "RISKFORGE_AUTH_KEYS_DIR": "/nonexistent/path/to/keys",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(RuntimeError, match="does not exist"):
                JwtAuthenticationService.from_env()

    def test_exp_as_float_accepted(self) -> None:
        """Token with float exp (e.g., from JSON decoder) is accepted."""
        now = 1700000000
        # Construct a token with float exp directly
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {
                    "sub": "float-exp-user",
                    "exp": float(now + 3600),
                    "iss": _DEFAULT_ISSUER,
                    "aud": _DEFAULT_AUDIENCE,
                }
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        sig_b64 = _b64url_encode(sig)
        token = f"{header_b64}.{payload_b64}.{sig_b64}"
        service = _make_service(now_provider=lambda: float(now))
        principal = service.authenticate(token)
        assert principal.subject_id == "float-exp-user"

    def test_empty_kid_rejected(self) -> None:
        """Token with empty string kid → UnknownKeyError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": ""}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {
                    "sub": "x",
                    "exp": int(time.time()) + 3600,
                    "iss": _DEFAULT_ISSUER,
                    "aud": _DEFAULT_AUDIENCE,
                }
            ).encode("utf-8")
        )
        sig_b64 = _b64url_encode(b"fake-sig")
        with pytest.raises(UnknownKeyError):
            _make_service().authenticate(
                f"{header_b64}.{payload_b64}.{sig_b64}"
            )

    def test_integer_issuer_rejected(self) -> None:
        """Token with integer iss → InvalidIssuerError."""
        header_b64 = _b64url_encode(
            json.dumps({"alg": "HS256", "kid": _DEFAULT_KID}).encode("utf-8")
        )
        payload_b64 = _b64url_encode(
            json.dumps(
                {
                    "sub": "x",
                    "exp": int(time.time()) + 3600,
                    "iss": 12345,
                    "aud": _DEFAULT_AUDIENCE,
                }
            ).encode("utf-8")
        )
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        sig = hmac.new(
            _DEFAULT_SECRET.encode("utf-8"), signing_input, hashlib.sha256
        ).digest()
        sig_b64 = _b64url_encode(sig)
        with pytest.raises(InvalidIssuerError):
            _make_service().authenticate(
                f"{header_b64}.{payload_b64}.{sig_b64}"
            )

    def test_key_with_crlf_line_endings(self) -> None:
        """Key files with Windows-style \\r\\n endings load correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            # Write a key with \r\n line endings
            (keys_dir / "crlf.key").write_bytes(
                b"this-is-a-secret-key-long-enough!!!\r\n"
            )
            keyring = _load_keyring(keys_dir)
            assert keyring["crlf"] == b"this-is-a-secret-key-long-enough!!!"

    def test_key_with_cr_only(self) -> None:
        """Key files with \\r-only endings: \\r is also stripped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            keys_dir = Path(tmpdir)
            (keys_dir / "cr.key").write_bytes(
                b"this-is-a-secret-key-long-enough!!!\r"
            )
            keyring = _load_keyring(keys_dir)
            assert keyring["cr"] == b"this-is-a-secret-key-long-enough!!!"
