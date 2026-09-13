"""Scope enforcement at route boundaries: wrong-scope callers get 403.

Every capability class (read incidents/analytics, score/ingest, read audit,
record reviews) is guarded by an explicit scope or role.  These tests prove
that a principal with *none* of the required grants is refused even though
authentication itself succeeds.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.support.auth import mint_hs256_token

SECRET = b"ingest-fixture-signing-secret-0123456789abcdef"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RISKFORGE_DEPLOYMENT_MODE", "pilot")
    monkeypatch.setenv("RISKFORGE_SQLITE_PATH", str(tmp_path / "scopes.db"))
    monkeypatch.delenv("RISKFORGE_MODEL_PATH", raising=False)
    monkeypatch.delenv("RISKFORGE_MANIFEST_PATH", raising=False)
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir(exist_ok=True)
    (keys_dir / "test-key.key").write_bytes(SECRET)
    monkeypatch.setenv("RISKFORGE_AUTH_ENABLED", "true")
    monkeypatch.setenv("RISKFORGE_AUTH_ISSUER", "riskforge-test")
    monkeypatch.setenv("RISKFORGE_AUTH_AUDIENCE", "riskforge-api")
    monkeypatch.setenv("RISKFORGE_AUTH_KEYS_DIR", str(keys_dir))
    from riskforge.runtime.production import create_app

    with TestClient(create_app()) as test_client:
        yield test_client


def _headers(roles: list[str], scopes: list[str] | None = None) -> dict[str, str]:
    token = mint_hs256_token(secret=SECRET, roles=roles, scopes=scopes)
    return {"X-Correlation-ID": "scope-test", "Authorization": f"Bearer {token}"}


class TestScopeEnforcement:
    @pytest.mark.parametrize(
        ("method", "path", "headers"),
        [
            # scoring:write required
            ("post", "/v1/inference", _headers(["reader"])),
            ("post", "/v1/ingest", _headers(["reader"])),
            # analytics:read required
            ("get", "/v1/analytics/summary", _headers(["ingestor"])),
            # incidents:read required
            ("get", "/v1/incidents", _headers(["ingestor"])),
            ("get", "/v1/incidents/critical", _headers(["ingestor"])),
            # audit:read required
            ("get", "/v1/incidents/ANY-1/audit", _headers(["reader"])),
            # reviews: reviewer role or reviews:write scope required
            ("post", "/v1/incidents/ANY-1/reviews", _headers(["reader"], scopes=[])),
        ],
    )
    def test_wrong_scope_is_403(self, client, method, path, headers):
        json = None
        if path.endswith("/reviews"):
            json = {
                "correlation_id": "scope-review",
                "decision_id": "scope-decision-1",
                "action": "confirm",
                "reason": "scope enforcement probe",
            }
        if path == "/v1/inference":
            json = {
                "correlation_id": "scope-inference",
                "incident": {
                    "log_id": "SCOPE-INC-1",
                    "timestamp": "2026-03-01T08:00:00Z",
                    "asset_id": "RIG_01",
                    "asset_type": "drilling_rig",
                    "raw_narrative": "probe narrative",
                    "spans": [],
                },
            }
        if json is not None:
            response = getattr(client, method)(path, headers=headers, json=json)
        else:
            response = getattr(client, method)(path, headers=headers)
        assert response.status_code == 403, (
            f"{method.upper()} {path} must refuse a principal without the "
            "capability scope or role"
        )

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("post", "/v1/inference"),
            ("get", "/v1/analytics/summary"),
            ("get", "/v1/incidents"),
        ],
    )
    def test_explicit_scope_grants_access_without_roles(self, client, method, path):
        response = getattr(client, method)(
            path, headers=_headers([], scopes=[_scope_for(path)])
        )
        assert response.status_code in {200, 404, 422}

    def test_ingest_with_scoring_scope_succeeds(self, client):
        data = (
            b"log_id,timestamp,asset_id,asset_type,raw_narrative\n"
            b"SCOPE-1,2026-03-01T08:00:00Z,RIG_01,drilling_rig,"
            b"Routine inspection completed with no findings\n"
        )
        response = client.post(
            "/v1/ingest?format=csv",
            content=data,
            headers=_headers([], scopes=["scoring:write"]),
        )
        assert response.status_code == 200

    def test_empty_principal_is_denied_everything(self, client):
        headers = _headers([], scopes=[])
        assert client.get("/v1/incidents", headers=headers).status_code == 403
        assert client.get("/v1/analytics/summary", headers=headers).status_code == 403
        assert client.get("/v1/incidents/A/audit", headers=headers).status_code == 403


def _scope_for(path: str) -> str:
    if path == "/v1/inference":
        return "scoring:write"
    if path == "/v1/analytics/summary":
        return "analytics:read"
    return "incidents:read"
