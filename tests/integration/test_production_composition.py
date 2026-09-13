"""End-to-end tests for the production composer.

Verifies that the real application boots with SQLite persistence and the
deterministic heuristic engine (no model artifacts), scores and stores
reports, and serves them through the workflow routes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from riskforge.core.contracts import RoutingBucket
from riskforge.runtime.contracts import RuntimeSettings
from riskforge.runtime.production import ProductionComposer, create_app
from tests.support.auth import mint_hs256_token

NOW = "2026-03-01T08:00:00Z"
HEADERS = {"X-Correlation-ID": "test-correlation-1"}


def _enable_pilot_auth(monkeypatch, tmp_path) -> None:
    """Configure a real JwtAuthenticationService for pilot-mode fixtures."""
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir(exist_ok=True)
    (keys_dir / "test-key.key").write_bytes(
        b"pilot-fixture-signing-secret-0123456789abcdef"
    )
    monkeypatch.setenv("RISKFORGE_AUTH_ENABLED", "true")
    monkeypatch.setenv("RISKFORGE_AUTH_ISSUER", "riskforge-test")
    monkeypatch.setenv("RISKFORGE_AUTH_AUDIENCE", "riskforge-api")
    monkeypatch.setenv("RISKFORGE_AUTH_KEYS_DIR", str(keys_dir))
    monkeypatch.setenv("RISKFORGE_REVIEWER_SUBJECTS", "test-user")


def _auth_headers(**claims) -> dict[str, str]:
    token = mint_hs256_token(
        secret=b"pilot-fixture-signing-secret-0123456789abcdef",
        roles=["reviewer", "reader", "ingestor", "auditor"],
        scopes=["incidents:read", "reviews:write"],
        **claims,
    )
    return {**HEADERS, "Authorization": f"Bearer {token}"}

WORK_AT_HEIGHT = (
    "Technician working at height on a scaffold near an unprotected edge; "
    "harness missing while the worker changed filters at 12 m."
)
BENIGN = "Site housekeeping completed. Signage replaced near the main gate."


def _normalized(log_id: str, narrative: str) -> dict:
    return {
        "log_id": log_id,
        "timestamp": NOW,
        "asset_id": "RIG_01",
        "asset_type": "drilling_rig",
        "raw_narrative": narrative,
        "spans": [],
    }


@pytest.fixture()
def app_client(tmp_path, monkeypatch):
    monkeypatch.setenv("RISKFORGE_DEPLOYMENT_MODE", "pilot")
    monkeypatch.setenv("RISKFORGE_SQLITE_PATH", str(tmp_path / "riskforge.db"))
    monkeypatch.delenv("RISKFORGE_MODEL_PATH", raising=False)
    monkeypatch.delenv("RISKFORGE_MANIFEST_PATH", raising=False)
    _enable_pilot_auth(monkeypatch, tmp_path)
    app = create_app()
    with TestClient(app) as client:
        yield client


class TestProductionBoot:
    def test_compose_returns_backend_facade(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RISKFORGE_DEPLOYMENT_MODE", "pilot")
        monkeypatch.setenv("RISKFORGE_SQLITE_PATH", str(tmp_path / "riskforge.db"))
        monkeypatch.delenv("RISKFORGE_MODEL_PATH", raising=False)
        monkeypatch.delenv("RISKFORGE_MANIFEST_PATH", raising=False)
        settings = RuntimeSettings(
            model_path=tmp_path / "unused.onnx",
            manifest_path=tmp_path / "unused.yaml",
        )
        assembly = ProductionComposer().compose(settings)
        names = [component.name for component in assembly.components]
        assert "inference-engine" in names
        assert assembly.application is not None

    def test_health_and_ready(self, app_client):
        health = app_client.get("/health", headers=_auth_headers())
        assert health.status_code == 200
        ready = app_client.get("/ready", headers=_auth_headers())
        assert ready.status_code == 200
        assert "inference-engine" in ready.json()["components"]

    def test_score_persists_and_becomes_queryable(self, app_client):
        response = app_client.post(
            "/v1/inference",
            json={"correlation_id": "c-1", "incident": _normalized("LOG_H1", WORK_AT_HEIGHT)},
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        body = response.json()
        result = body["result"]
        assert result["log_id"] == "LOG_H1"
        assert result["routing"] == RoutingBucket.CRITICAL_ESCALATION.value
        assert "work_at_height" in result["matched_iogp_rules"]
        assert result["raw_sif_p_score"] >= 0.65

        listed = app_client.get(
            "/v1/incidents",
            params={"correlation_id": "c-2", "limit": 10},
            headers=_auth_headers(),
        )
        assert listed.status_code == 200
        items = listed.json()["page"]["items"]
        log_ids = [item["incident"]["log_id"] for item in items]
        assert "LOG_H1" in log_ids

        detail = app_client.get("/v1/incidents/LOG_H1", headers=_auth_headers())
        assert detail.status_code == 200
        assert detail.json()["incident"]["incident"]["log_id"] == "LOG_H1"

    def test_benign_report_routes_to_auto_dismiss(self, app_client):
        response = app_client.post(
            "/v1/inference",
            json={"correlation_id": "c-3", "incident": _normalized("LOG_B1", BENIGN)},
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        result = response.json()["result"]
        assert result["routing"] == RoutingBucket.AUTO_DISMISS.value
        assert result["matched_iogp_rules"] == []

    def test_batch_scores_are_persisted(self, app_client):
        response = app_client.post(
            "/v1/inference/batch",
            json={
                "correlation_id": "c-4",
                "incidents": [
                    _normalized("LOG_BA1", WORK_AT_HEIGHT),
                    _normalized("LOG_BA2", BENIGN),
                ],
            },
            headers=_auth_headers(),
        )
        assert response.status_code == 200
        assert len(response.json()["results"]) == 2
        for log_id in ("LOG_BA1", "LOG_BA2"):
            found = app_client.get(f"/v1/incidents/{log_id}", headers=_auth_headers())
            assert found.status_code == 200

    def test_rescoring_is_idempotent(self, app_client):
        for _ in range(2):
            response = app_client.post(
                "/v1/inference",
                json={"correlation_id": "c-5", "incident": _normalized("LOG_ID1", BENIGN)},
                headers=_auth_headers(),
            )
            assert response.status_code == 200
        listed = app_client.get(
            "/v1/incidents",
            params={"correlation_id": "c-6", "limit": 50},
            headers=_auth_headers(),
        )
        log_ids = [
            item["incident"]["log_id"] for item in listed.json()["page"]["items"]
        ]
        assert log_ids.count("LOG_ID1") == 1

    def test_factory_creates_managed_app(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RISKFORGE_DEPLOYMENT_MODE", "pilot")
        monkeypatch.setenv("RISKFORGE_SQLITE_PATH", str(tmp_path / "riskforge.db"))
        monkeypatch.delenv("RISKFORGE_MODEL_PATH", raising=False)
        monkeypatch.delenv("RISKFORGE_MANIFEST_PATH", raising=False)
        _enable_pilot_auth(monkeypatch, tmp_path)
        app = create_app()
        with TestClient(app) as client:
            assert client.get("/health", headers=_auth_headers()).status_code == 200
