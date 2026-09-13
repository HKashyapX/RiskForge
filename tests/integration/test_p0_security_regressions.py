"""P0 security regressions: metrics exposure, demo route removal, JWT claims.

These tests encode the non-negotiable P0 behaviors from the security
remediation mandate:

1. ``/metrics`` is mounted exactly once — unauthenticated requests against
   authenticated deployments receive 401, never Prometheus output.
2. Demo mode does not register any operational route — every ``/v1/*``
   surface answers 404, not 401 or synthetic data.
3. JWT ``exp``/``nbf`` claims must be real finite numbers: NaN, Infinity,
   and Boolean values are rejected with a malformed-credential error.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from tests.support.auth import mint_hs256_token

# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def pilot_client(tmp_path, monkeypatch):
    """Pilot deployment: authenticated, metrics enabled, SQLite persistence."""
    monkeypatch.setenv("RISKFORGE_DEPLOYMENT_MODE", "pilot")
    monkeypatch.setenv("RISKFORGE_SQLITE_PATH", str(tmp_path / "p0-pilot.db"))
    monkeypatch.delenv("RISKFORGE_MODEL_PATH", raising=False)
    monkeypatch.delenv("RISKFORGE_MANIFEST_PATH", raising=False)
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir(exist_ok=True)
    (keys_dir / "test-key.key").write_bytes(
        b"ingest-fixture-signing-secret-0123456789abcdef"
    )
    monkeypatch.setenv("RISKFORGE_AUTH_ENABLED", "true")
    monkeypatch.setenv("RISKFORGE_AUTH_ISSUER", "riskforge-test")
    monkeypatch.setenv("RISKFORGE_AUTH_AUDIENCE", "riskforge-api")
    monkeypatch.setenv("RISKFORGE_AUTH_KEYS_DIR", str(keys_dir))
    from riskforge.runtime.production import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


@pytest.fixture()
def demo_client(monkeypatch):
    monkeypatch.setenv("RISKFORGE_MODE", "demo")
    monkeypatch.setenv("RISKFORGE_ENV", "development")
    from riskforge.runtime.demo import create_demo_app

    return TestClient(create_demo_app())


def _pilot_headers() -> dict[str, str]:
    token = mint_hs256_token(
        secret=b"ingest-fixture-signing-secret-0123456789abcdef",
        roles=["reviewer"],
    )
    return {"X-Correlation-ID": "p0-test", "Authorization": f"Bearer {token}"}


# ── P0-1: /metrics mounted exactly once, guarded when auth present ──────────


class TestMetricsExposure:
    def test_unauthenticated_metrics_is_401_in_pilot(self, pilot_client):
        response = pilot_client.get("/metrics")
        assert response.status_code == 401
        assert b"riskforge" not in response.content.lower() or (
            b"authentication" in response.content
        )
        body = response.text
        # A 401 body must never contain Prometheus exposition format.
        assert "# HELP" not in body and "# TYPE" not in body

    def test_authenticated_metrics_serves_prometheus_output(self, pilot_client):
        response = pilot_client.get("/metrics", headers=_pilot_headers())
        assert response.status_code == 200
        assert "# HELP" in response.text or "# TYPE" in response.text

    def test_metrics_mounted_exactly_once(self, pilot_client):
        mounted = [
            route.path
            for route in pilot_client.app.routes
            if getattr(route, "path", None) == "/metrics"
        ]
        assert len(mounted) == 1

    def test_demo_has_no_metrics_endpoint(self, demo_client):
        assert demo_client.get("/metrics").status_code in {404, 405}


# ── P0-2: demo registers no operational routes ───────────────────────────────


OPERATIONAL_ROUTES = [
    ("post", "/v1/inference"),
    ("post", "/v1/inference/batch"),
    ("post", "/v1/ingest"),
    ("get", "/v1/analytics/summary"),
    ("get", "/v1/incidents"),
    ("get", "/v1/incidents/critical"),
    ("get", "/v1/incidents/ANY-1"),
    ("get", "/v1/incidents/ANY-1/audit"),
    ("get", "/v1/incidents/ANY-1/explanation"),
    ("get", "/v1/assets/ANY-1/summary"),
    ("post", "/v1/incidents/ANY-1/reviews"),
]


class TestDemoOperationalRoutesAbsent:
    @pytest.mark.parametrize(("method", "path"), OPERATIONAL_ROUTES)
    def test_operational_route_is_404_in_demo(self, demo_client, method, path):
        response = getattr(demo_client, method)(path)
        assert response.status_code == 404, (
            f"{method.upper()} {path} must not exist in demo mode"
        )

    @pytest.mark.parametrize(
        ("method", "path"), [("post", "/v1/inference"), ("get", "/v1/incidents")]
    )
    def test_operational_route_is_404_even_when_authenticated(
        self, demo_client, method, path
    ):
        # 404 regardless of credentials: the surface is gone, not hidden.
        token = mint_hs256_token(roles=["admin"])
        response = getattr(demo_client, method)(
            path, headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 404

    def test_demo_health_and_ready_identify_the_deployment(self, demo_client):
        health = demo_client.get("/health")
        assert health.status_code == 200
        ready = demo_client.get("/ready", headers={"X-Correlation-ID": "p0-demo"})
        assert ready.status_code == 503
        components = ready.json()["components"]
        assert "deployment-mode:demo" in components

    def test_demo_operational_routes_not_in_route_table(self, demo_client):
        paths = {getattr(route, "path", "") for route in demo_client.app.routes}
        assert "/v1/inference" not in paths
        assert "/v1/ingest" not in paths
        assert "/v1/analytics/summary" not in paths


# ── P0-3: JWT exp/nbf must be finite non-boolean numbers ────────────────────


_SECRET = b"jwt-claim-test-signing-secret-0123456"


class TestJwtClaimValidation:
    @pytest.fixture(autouse=True)
    def _service(self, tmp_path, monkeypatch):
        keys_dir = tmp_path / "keys"
        keys_dir.mkdir(exist_ok=True)
        (keys_dir / "test-key.key").write_bytes(_SECRET)
        monkeypatch.setenv("RISKFORGE_AUTH_ISSUER", "riskforge-test")
        monkeypatch.setenv("RISKFORGE_AUDIENCE", "riskforge-api")
        monkeypatch.setenv("RISKFORGE_AUTH_AUDIENCE", "riskforge-api")
        monkeypatch.setenv("RISKFORGE_AUTH_KEYS_DIR", str(keys_dir))
        from riskforge.authentication.jwt_service import (
            JwtAuthenticationService,
            MalformedCredentialError,
        )

        self.service = JwtAuthenticationService.from_env()
        self.malformed = MalformedCredentialError
        self.expired = None

    def _token(self, claims: dict) -> str:
        def _b64(data: bytes) -> str:
            return base64.urlsafe_b64encode(data).rstrip(b"=").decode()

        header = _b64(
            json.dumps({"alg": "HS256", "typ": "JWT", "kid": "test-key"}).encode()
        )
        payload = _b64(json.dumps(claims).encode())
        signature = _b64(
            hmac.new(
                _SECRET, f"{header}.{payload}".encode(), hashlib.sha256
            ).digest()
        )
        return f"{header}.{payload}.{signature}"

    def _base_claims(self) -> dict:
        now = int(time.time())
        return {
            "iss": "riskforge-test",
            "aud": "riskforge-api",
            "sub": "claim-user",
            "iat": now,
            "exp": now + 3600,
        }

    @pytest.mark.parametrize(
        "bad_exp",
        [float("nan"), float("inf"), float("-inf"), True, False, "now+3600"],
    )
    def test_non_finite_or_boolean_exp_rejected(self, bad_exp):
        claims = {**self._base_claims(), "exp": bad_exp}
        with pytest.raises(self.malformed):
            self.service.authenticate(self._token(claims))

    @pytest.mark.parametrize(
        "bad_nbf",
        [float("nan"), float("inf"), float("-inf"), True, False, "yesterday"],
    )
    def test_non_finite_or_boolean_nbf_rejected(self, bad_nbf):
        claims = {**self._base_claims(), "nbf": bad_nbf}
        with pytest.raises(self.malformed):
            self.service.authenticate(self._token(claims))

    def test_valid_finite_exp_and_nbf_accepted(self):
        now = int(time.time())
        claims = {**self._base_claims(), "nbf": now - 10}
        principal = self.service.authenticate(self._token(claims))
        assert principal.subject_id == "claim-user"

    def test_missing_exp_rejected(self):
        claims = self._base_claims()
        del claims["exp"]
        with pytest.raises(self.malformed):
            self.service.authenticate(self._token(claims))
