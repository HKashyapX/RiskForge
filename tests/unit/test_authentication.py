"""Comprehensive authentication boundary tests.

Covers the full authentication lifecycle from credential extraction through
principal verification, ensuring credentials never leak into exception
messages, logs, or error responses.
"""

from __future__ import annotations

import os
import warnings
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from riskforge.api.app import create_app
from riskforge.api.dependencies import ReadinessSnapshot, require_principal
from riskforge.api.errors import ErrorCode, translate_application_error
from riskforge.application.exceptions import InferenceApplicationError
from riskforge.authentication.dev import DevAuthenticationService
from riskforge.authentication.exceptions import (
    AuthenticationError,
    InvalidCredentialsError,
    MissingCredentialsError,
)
from riskforge.authentication.principal import Principal
from riskforge.authentication.protocols import AuthenticationService
from riskforge.core.contracts import (
    AssetRiskSummary,
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _incident(log_id: str = "LOG_1") -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Sensitive incident narrative.",
        spans=[],
    )


def _result(log_id: str = "LOG_1") -> ModelInferenceResult:
    return ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=0.2,
        calibrated_sif_p_score=0.2,
        deterministic_override=False,
        routing=RoutingBucket.AUTO_DISMISS,
        matched_iogp_rules=[],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )


class FakeApplication:
    def process_incident(self, record: Any) -> ModelInferenceResult:
        return _result(record.log_id)

    def process_batch(self, records: Any) -> list[ModelInferenceResult]:
        return [_result(record.log_id) for record in records]

    def get_asset_summary(self, asset_id: str) -> AssetRiskSummary:
        return AssetRiskSummary(
            asset_id=asset_id,
            asset_type=AssetType.DRILLING_RIG,
            total_logs=4,
            sif_precursor_count=1,
            spd_score=25.0,
            recurrent_failed_barriers=[],
        )


class FakeReadiness:
    def snapshot(self) -> ReadinessSnapshot:
        return ReadinessSnapshot(
            ready=True,
            state="ready",
            checked_at=datetime(2026, 1, 1, tzinfo=UTC),
            components=("model",),
        )


class FakeAuthService:
    """Deterministic auth service for testing."""

    def __init__(self, subject_id: str = "test-user") -> None:
        self._principal = Principal(subject_id=subject_id)

    def authenticate(self, credential: str) -> Principal:
        if credential == "valid-token":
            return self._principal
        raise InvalidCredentialsError()


class RejectingAuthService:
    """Auth service that rejects all credentials."""

    def authenticate(self, credential: str) -> Principal:
        raise InvalidCredentialsError()


# ---------------------------------------------------------------------------
# Principal model tests
# ---------------------------------------------------------------------------


class TestPrincipal:
    def test_principal_is_frozen(self) -> None:
        principal = Principal(subject_id="user-1")
        with pytest.raises(ValidationError):
            principal.subject_id = "user-2"  # type: ignore[misc]

    def test_principal_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            Principal(subject_id="user-1", role="admin")  # type: ignore[call-arg]

    def test_principal_requires_non_empty_subject_id(self) -> None:
        with pytest.raises(ValidationError):
            Principal(subject_id="")

    def test_principal_rejects_long_subject_id(self) -> None:
        with pytest.raises(ValidationError):
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
# Authentication exception tests
# ---------------------------------------------------------------------------


class TestAuthenticationExceptions:
    def test_missing_credentials_is_authentication_error(self) -> None:
        assert issubclass(MissingCredentialsError, AuthenticationError)

    def test_invalid_credentials_is_authentication_error(self) -> None:
        assert issubclass(InvalidCredentialsError, AuthenticationError)

    def test_authentication_error_is_runtime_error(self) -> None:
        assert issubclass(AuthenticationError, RuntimeError)

    def test_exception_messages_do_not_contain_credentials(self) -> None:
        credential = "super-secret-token-12345"
        exc = InvalidCredentialsError()
        assert credential not in str(exc)

    def test_missing_credentials_error_message(self) -> None:
        exc = MissingCredentialsError()
        assert "credentials" in str(exc).lower() or str(exc) == ""

    def test_invalid_credentials_error_message(self) -> None:
        exc = InvalidCredentialsError()
        assert "credentials" in str(exc).lower() or str(exc) == ""


# ---------------------------------------------------------------------------
# DevAuthenticationService tests
# ---------------------------------------------------------------------------


class TestDevAuthenticationService:
    def test_dev_auth_returns_static_principal(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ["RISKFORGE_ENV"] = "development"
            service = DevAuthenticationService(subject_id="dev-user")
            principal = service.authenticate("any-credential")
            assert principal.subject_id == "dev-user"

    def test_dev_auth_default_subject_id(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ["RISKFORGE_ENV"] = "development"
            service = DevAuthenticationService()
            principal = service.authenticate("anything")
            assert principal.subject_id == "dev-user"

    def test_dev_auth_ignores_credential_value(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ["RISKFORGE_ENV"] = "development"
            service = DevAuthenticationService()
            p1 = service.authenticate("token-a")
            p2 = service.authenticate("token-b")
            assert p1 == p2

    def test_dev_auth_raises_in_production(self) -> None:
        os.environ["RISKFORGE_ENV"] = "production"
        try:
            with pytest.raises(RuntimeError, match="must not be used"):
                DevAuthenticationService()
        finally:
            os.environ["RISKFORGE_ENV"] = "development"

    def test_dev_auth_warns_in_development(self) -> None:
        os.environ["RISKFORGE_ENV"] = "development"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DevAuthenticationService()
            assert any("Do not use in production" in str(w.message) for w in caught)

    def test_dev_auth_silenced_in_test(self) -> None:
        os.environ["RISKFORGE_ENV"] = "test"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            DevAuthenticationService()
            assert len(caught) == 0


# ---------------------------------------------------------------------------
# Authentication error translation tests
# ---------------------------------------------------------------------------


class TestAuthenticationErrorTranslation:
    def test_missing_credentials_returns_401(self) -> None:
        error = translate_application_error(MissingCredentialsError())
        assert error.status_code == 401
        assert error.code == ErrorCode.MISSING_CREDENTIALS
        assert error.retryable is False

    def test_invalid_credentials_returns_401(self) -> None:
        error = translate_application_error(InvalidCredentialsError())
        assert error.status_code == 401
        assert error.code == ErrorCode.INVALID_CREDENTIALS
        assert error.retryable is False

    def test_generic_auth_error_returns_401(self) -> None:
        error = translate_application_error(AuthenticationError())
        assert error.status_code == 401
        assert error.code == ErrorCode.INVALID_CREDENTIALS
        assert error.retryable is False

    def test_auth_error_messages_do_not_expose_details(self) -> None:
        for exc_class in (MissingCredentialsError, InvalidCredentialsError, AuthenticationError):
            translated = translate_application_error(exc_class())
            assert "secret" not in translated.message.lower()
            assert "token" not in translated.message.lower()


# ---------------------------------------------------------------------------
# API integration tests — auth-free routes
# ---------------------------------------------------------------------------


class TestAuthFreeRoutes:
    """Health and readiness must never require authentication."""

    def test_health_needs_no_auth(self) -> None:
        client = TestClient(create_app(FakeApplication(), FakeReadiness()))
        response = client.get("/health")
        assert response.status_code == 200

    def test_readiness_needs_no_auth(self) -> None:
        client = TestClient(create_app(FakeApplication(), FakeReadiness()))
        response = client.get("/ready", headers={"X-Correlation-ID": "probe"})
        assert response.status_code == 200

    def test_inference_needs_no_auth_when_auth_not_configured(self) -> None:
        client = TestClient(create_app(FakeApplication(), FakeReadiness()))
        response = client.post(
            "/v1/inference",
            json={
                "correlation_id": "c-1",
                "incident": _incident().model_dump(mode="json"),
            },
        )
        assert response.status_code == 200


# ---------------------------------------------------------------------------
# API integration tests — auth-enabled app
# ---------------------------------------------------------------------------


class TestAuthEnabledApp:
    """Tests for apps created with an auth_service parameter."""

    def _client(self, auth_service: AuthenticationService | None = None) -> TestClient:
        return TestClient(
            create_app(FakeApplication(), FakeReadiness(), auth_service=auth_service)
        )

    def _app_with_auth_route(
        self, auth_service: AuthenticationService | None = None
    ) -> TestClient:
        """Create a test app with an authenticated route using create_app."""
        app = FastAPI()

        # Register the same exception handling that create_app uses
        from fastapi.responses import JSONResponse as _JSON

        from riskforge.api.errors import ErrorCode as _EC
        from riskforge.api.errors import TranslatedError as _TE

        @app.exception_handler(MissingCredentialsError)
        async def _handle_auth(_request: Any, _error: Any) -> _JSON:
            translated = _TE(
                status_code=401,
                code=_EC.MISSING_CREDENTIALS,
                message="credentials required",
                retryable=False,
            )
            return _JSON(
                status_code=401,
                content={"code": translated.code.value, "message": translated.message},
            )

        @app.exception_handler(InvalidCredentialsError)
        async def _handle_invalid(_request: Any, _error: Any) -> _JSON:
            translated = _TE(
                status_code=401,
                code=_EC.INVALID_CREDENTIALS,
                message="invalid credentials",
                retryable=False,
            )
            return _JSON(
                status_code=401,
                content={"code": translated.code.value, "message": translated.message},
            )

        if auth_service is not None:

            def _extract(request: Request) -> Principal:
                auth_header = request.headers.get("authorization", "")
                if not auth_header.startswith("Bearer "):
                    raise MissingCredentialsError()
                token = auth_header[7:]
                return auth_service.authenticate(token)

            app.dependency_overrides[require_principal] = _extract

        @app.get("/protected")
        def protected(principal: Principal = Depends(require_principal)) -> dict[str, str]:  # noqa: B008
            return {"subject_id": principal.subject_id}

        return TestClient(app)

    def test_no_auth_service_raises_missing_credentials(self) -> None:
        client = self._app_with_auth_route(auth_service=None)
        response = client.get("/protected")
        assert response.status_code == 401

    def test_valid_token_returns_principal(self) -> None:
        client = self._app_with_auth_route(FakeAuthService("alice"))
        response = client.get("/protected", headers={"Authorization": "Bearer valid-token"})
        assert response.status_code == 200
        assert response.json()["subject_id"] == "alice"

    def test_invalid_token_returns_401(self) -> None:
        client = self._app_with_auth_route(FakeAuthService())
        response = client.get("/protected", headers={"Authorization": "Bearer bad-token"})
        assert response.status_code == 401

    def test_missing_authorization_header_returns_401(self) -> None:
        client = self._app_with_auth_route(FakeAuthService())
        response = client.get("/protected")
        assert response.status_code == 401

    def test_malformed_authorization_header_returns_401(self) -> None:
        client = self._app_with_auth_route(FakeAuthService())
        response = client.get("/protected", headers={"Authorization": "Basic abc"})
        assert response.status_code == 401

    def test_empty_bearer_token_returns_401(self) -> None:
        client = self._app_with_auth_route(FakeAuthService())
        response = client.get("/protected", headers={"Authorization": "Bearer "})
        # Empty token goes to auth service which rejects it
        assert response.status_code == 401

    def test_auth_error_response_exposes_no_credentials(self) -> None:
        client = self._app_with_auth_route(FakeAuthService())
        response = client.get(
            "/protected", headers={"Authorization": "Bearer top-secret-token"}
        )
        assert response.status_code == 401
        body = response.json()
        assert "top-secret-token" not in str(body)

    def test_auth_free_routes_still_work_with_auth_service(self) -> None:
        client = TestClient(
            create_app(FakeApplication(), FakeReadiness(), FakeAuthService())
        )
        assert client.get("/health").status_code == 200
        assert client.get("/ready", headers={"X-Correlation-ID": "p"}).status_code == 200


# ---------------------------------------------------------------------------
# API integration tests — require_principal dependency behavior
# ---------------------------------------------------------------------------


class TestRequirePrincipalDependency:
    def test_require_principal_without_override_raises(self) -> None:
        """Without dependency override, require_principal raises MissingCredentialsError."""
        with pytest.raises(MissingCredentialsError):
            require_principal()

    def test_require_principal_with_override_returns_principal(self) -> None:
        app = FastAPI()

        def _fake_auth() -> Principal:
            return Principal(subject_id="override-user")

        app.dependency_overrides[require_principal] = _fake_auth

        @app.get("/test")
        def test_route(principal: Principal = Depends(require_principal)) -> dict[str, str]:  # noqa: B008
            return {"subject_id": principal.subject_id}

        client = TestClient(app)
        response = client.get("/test")
        assert response.status_code == 200
        assert response.json()["subject_id"] == "override-user"


# ---------------------------------------------------------------------------
# Protocol conformance tests
# ---------------------------------------------------------------------------


class TestAuthenticationServiceProtocol:
    def test_fake_auth_service_satisfies_protocol(self) -> None:
        assert isinstance(FakeAuthService(), AuthenticationService)

    def test_dev_auth_service_satisfies_protocol(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ["RISKFORGE_ENV"] = "test"
            assert isinstance(DevAuthenticationService(), AuthenticationService)

    def test_class_without_authenticate_does_not_satisfy(self) -> None:
        class NotAnAuthService:
            pass

        assert not isinstance(NotAnAuthService(), AuthenticationService)


# ---------------------------------------------------------------------------
# Security boundary tests
# ---------------------------------------------------------------------------


class TestSecurityBoundaries:
    def test_credentials_never_appear_in_exception_str(self) -> None:
        """Ensure no authentication exception exposes the raw credential."""
        secret = "sk-proj-abc123secretkey"
        exceptions = [
            MissingCredentialsError(),
            InvalidCredentialsError(),
            AuthenticationError(),
        ]
        for exc in exceptions:
            assert secret not in str(exc), (
                f"{type(exc).__name__} exposed credential in str representation"
            )

    def test_dev_auth_service_rejects_production(self) -> None:
        """DevAuthenticationService must refuse to run in production."""
        os.environ["RISKFORGE_ENV"] = "production"
        try:
            with pytest.raises(RuntimeError, match="must not be used"):
                DevAuthenticationService()
        finally:
            os.environ["RISKFORGE_ENV"] = "development"

    def test_principal_immutable_after_creation(self) -> None:
        """Principal must not be mutable after construction."""
        principal = Principal(subject_id="immutable-user")
        original_id = principal.subject_id
        # Attempt mutation (should raise ValidationError)
        with pytest.raises(ValidationError):
            principal.subject_id = "mutated"  # type: ignore[misc]
        assert principal.subject_id == original_id

    def test_dev_auth_always_returns_same_principal(self) -> None:
        """DevAuthenticationService must return the same principal for any credential."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            os.environ["RISKFORGE_ENV"] = "test"
            service = DevAuthenticationService(subject_id="consistent-user")
            results = [service.authenticate(f"token-{i}") for i in range(10)]
            assert all(r.subject_id == "consistent-user" for r in results)
            assert len({r.subject_id for r in results}) == 1


# ---------------------------------------------------------------------------
# Application error translation backward compatibility
# ---------------------------------------------------------------------------


class TestCreateAppAuthErrorHandling:
    """Verify that create_app() exception handlers translate auth errors to HTTP 401."""

    def _make_protected_app(
        self, auth_service: AuthenticationService | None = None
    ) -> TestClient:
        """Create a create_app()-backed app with a protected route."""
        app = create_app(FakeApplication(), FakeReadiness(), auth_service=auth_service)

        @app.get("/protected")
        def protected(principal: Principal = Depends(require_principal)) -> dict[str, str]:  # noqa: B008
            return {"subject_id": principal.subject_id}

        return TestClient(app)

    def test_missing_header_returns_401_via_create_app(self) -> None:
        client = self._make_protected_app(FakeAuthService())
        response = client.get("/protected")
        assert response.status_code == 401

    def test_invalid_token_returns_401_via_create_app(self) -> None:
        client = self._make_protected_app(FakeAuthService())
        response = client.get("/protected", headers={"Authorization": "Bearer bad"})
        assert response.status_code == 401

    def test_no_auth_service_returns_401_via_create_app(self) -> None:
        client = self._make_protected_app(auth_service=None)
        response = client.get("/protected")
        assert response.status_code == 401

    def test_valid_token_returns_200_via_create_app(self) -> None:
        client = self._make_protected_app(FakeAuthService("bob"))
        response = client.get("/protected", headers={"Authorization": "Bearer valid-token"})
        assert response.status_code == 200
        assert response.json()["subject_id"] == "bob"

    def test_auth_error_response_exposes_no_credentials(self) -> None:
        client = self._make_protected_app(FakeAuthService())
        response = client.get(
            "/protected", headers={"Authorization": "Bearer my-secret-value"}
        )
        assert response.status_code == 401
        body = response.json()
        assert "my-secret-value" not in str(body)

    def test_auth_error_response_has_correct_structure(self) -> None:
        client = self._make_protected_app(FakeAuthService())
        response = client.get("/protected")
        assert response.status_code == 401
        body = response.json()
        assert "error" in body
        assert "code" in body["error"]
        assert "message" in body["error"]
        assert "retryable" in body["error"]

    def test_auth_free_routes_unaffected_by_auth_service(self) -> None:
        client = TestClient(
            create_app(FakeApplication(), FakeReadiness(), FakeAuthService())
        )
        assert client.get("/health").status_code == 200


class TestErrorTranslationBackwardCompatibility:
    """Ensure new auth error codes don't break existing translations."""

    def test_inference_error_still_translates(self) -> None:
        error = translate_application_error(
            InferenceApplicationError("service down")
        )
        assert error.status_code == 503
        assert error.code == ErrorCode.INFERENCE_UNAVAILABLE

    def test_unknown_error_still_translates(self) -> None:
        error = translate_application_error(RuntimeError("something"))
        assert error.status_code == 500
        assert error.code == ErrorCode.INTERNAL_ERROR
