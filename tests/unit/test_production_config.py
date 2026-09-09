"""Tests for production configuration, security headers, and deployment readiness."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from riskforge.api.app import create_app

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSnapshot:
    """Minimal RuntimeStatus-like snapshot for testing."""

    def __init__(self) -> None:
        self.ready = True
        self.state = "ready"
        self.checked_at = datetime.now(tz=UTC)
        self.components = ("test-component",)


def _noop_application() -> Any:
    """Return a minimal mock for BackendApplication."""
    return MagicMock()


def _noop_readiness() -> MagicMock:
    """Return a mock ReadinessProvider with a valid snapshot."""
    rp = MagicMock()
    rp.snapshot.return_value = _FakeSnapshot()
    return rp


# ---------------------------------------------------------------------------
# Security Headers Tests
# ---------------------------------------------------------------------------


class TestSecurityHeaders:
    """Verify security headers are present on every response."""

    def test_health_endpoint_has_security_headers(self) -> None:
        client = TestClient(create_app(_noop_application(), _noop_readiness()))
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["X-XSS-Protection"] == "0"
        assert resp.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert resp.headers["Cache-Control"] == "no-store"

    def test_ready_endpoint_has_security_headers(self) -> None:
        client = TestClient(create_app(_noop_application(), _noop_readiness()))
        resp = client.get("/ready", headers={"X-Correlation-ID": "test-456"})
        # May be 200 or 503 depending on readiness state, but headers should be present
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"

    def test_404_response_has_security_headers(self) -> None:
        client = TestClient(create_app(_noop_application(), _noop_readiness()))
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"


# ---------------------------------------------------------------------------
# CORS Configuration Tests
# ---------------------------------------------------------------------------


class TestCORSConfiguration:
    """Verify CORS middleware is configured correctly."""

    def test_cors_wildcard_allows_all_origins(self) -> None:
        app = create_app(
            _noop_application(),
            _noop_readiness(),
            cors_origins=["*"],
        )
        client = TestClient(app)
        resp = client.options(
            "/health",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # With allow_origins=["*"], CORSMiddleware echoes back the requesting origin
        assert resp.headers.get("access-control-allow-origin") is not None

    def test_cors_restricted_origins(self) -> None:
        app = create_app(
            _noop_application(),
            _noop_readiness(),
            cors_origins=["https://app.riskforge.io"],
        )
        client = TestClient(app)
        resp = client.options(
            "/health",
            headers={
                "Origin": "https://evil.example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Evil origin should not be allowed
        assert resp.headers.get("access-control-allow-origin") != "https://evil.example.com"

    def test_cors_empty_list_disables_cors(self) -> None:
        app = create_app(
            _noop_application(),
            _noop_readiness(),
            cors_origins=[],
        )
        client = TestClient(app)
        resp = client.options(
            "/health",
            headers={
                "Origin": "https://app.riskforge.io",
                "Access-Control-Request-Method": "GET",
            },
        )
        # No CORS headers when disabled
        assert "access-control-allow-origin" not in resp.headers

    def test_cors_reads_from_env_variable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("RISKFORGE_CORS_ORIGINS", "https://a.com, https://b.com")
        app = create_app(_noop_application(), _noop_readiness())
        client = TestClient(app)
        resp = client.options(
            "/health",
            headers={
                "Origin": "https://a.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert resp.headers.get("access-control-allow-origin") == "https://a.com"


    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_secure_environment_disables_cors_by_default(
        self, monkeypatch: pytest.MonkeyPatch, environment: str
    ) -> None:
        monkeypatch.setenv("RISKFORGE_ENV", environment)
        monkeypatch.delenv("RISKFORGE_CORS_ORIGINS", raising=False)
        client = TestClient(create_app(_noop_application(), _noop_readiness()))

        response = client.options(
            "/health",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert "access-control-allow-origin" not in response.headers

    @pytest.mark.parametrize("environment", ["staging", "production"])
    def test_secure_environment_rejects_wildcard_cors(
        self, monkeypatch: pytest.MonkeyPatch, environment: str
    ) -> None:
        monkeypatch.setenv("RISKFORGE_ENV", environment)

        with pytest.raises(ValueError, match="wildcard CORS is forbidden"):
            create_app(
                _noop_application(),
                _noop_readiness(),
                cors_origins=["*"],
            )

    def test_development_wildcard_does_not_allow_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("RISKFORGE_ENV", "development")
        client = TestClient(
            create_app(
                _noop_application(),
                _noop_readiness(),
                cors_origins=["*"],
            )
        )

        response = client.options(
            "/health",
            headers={
                "Origin": "https://example.test",
                "Access-Control-Request-Method": "GET",
            },
        )

        assert response.headers["access-control-allow-origin"] == "*"
        assert response.headers.get("access-control-allow-credentials") != "true"


# ---------------------------------------------------------------------------
# Metrics Endpoint Tests
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    """Verify /metrics endpoint is available."""

    def test_metrics_endpoint_returns_prometheus_format(self) -> None:
        client = TestClient(create_app(_noop_application(), _noop_readiness()))
        resp = client.get("/metrics")
        assert resp.status_code == 200
        # prometheus_client returns text/plain with metrics
        assert "riskforge_request_duration_seconds" in resp.text or "HELP" in resp.text

    def test_metrics_endpoint_disabled_when_flag_false(self) -> None:
        client = TestClient(
            create_app(_noop_application(), _noop_readiness(), enable_metrics=False)
        )
        resp = client.get("/metrics")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Application Factory Tests
# ---------------------------------------------------------------------------


class TestApplicationFactory:
    """Verify create_app produces a valid FastAPI application."""

    def test_create_app_returns_fastapi_instance(self) -> None:
        app = create_app(_noop_application(), _noop_readiness())
        assert isinstance(app, FastAPI)
        assert app.title == "RiskForge API"
        assert app.version == "1.0.0"

    def test_create_app_with_all_params(self) -> None:
        app = create_app(
            _noop_application(),
            _noop_readiness(),
            enable_metrics=True,
            cors_origins=["https://example.com"],
        )
        assert isinstance(app, FastAPI)

    def test_health_endpoint_accessible(self) -> None:
        client = TestClient(create_app(_noop_application(), _noop_readiness()))
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "live"

    def test_ready_endpoint_accessible(self) -> None:
        client = TestClient(create_app(_noop_application(), _noop_readiness()))
        resp = client.get("/ready", headers={"X-Correlation-ID": "test-123"})
        assert resp.status_code in (200, 503)


# ---------------------------------------------------------------------------
# Environment Variable Tests
# ---------------------------------------------------------------------------


class TestEnvironmentVariables:
    """Verify critical environment variables are documented and have defaults."""

    def test_riskforge_log_level_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RISKFORGE_LOG_LEVEL", raising=False)
        from riskforge.logging_config import configure_logging
        # Should not raise with default
        configure_logging()

    def test_riskforge_cors_origins_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("RISKFORGE_CORS_ORIGINS", raising=False)
        app = create_app(_noop_application(), _noop_readiness())
        assert isinstance(app, FastAPI)

    def test_pghost_has_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PGHOST", raising=False)
        from riskforge.persistence.postgres.connection import PostgresConfig
        cfg = PostgresConfig()
        assert cfg.host == "localhost"

    def test_pgport_has_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PGPORT", raising=False)
        from riskforge.persistence.postgres.connection import PostgresConfig
        cfg = PostgresConfig()
        assert cfg.port == 5432

    def test_migration_dsn_preserves_database_password(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PGPASSWORD", "migration-test-secret")
        from riskforge.persistence.postgres.migrate import _dsn_from_env

        dsn = _dsn_from_env()

        assert "password=migration-test-secret" in dsn
        assert "password=***" not in dsn
