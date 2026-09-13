"""Inference audit trail: every scored write appends a governance event."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.support.auth import mint_hs256_token

SECRET = b"ingest-fixture-signing-secret-0123456789abcdef"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RISKFORGE_DEPLOYMENT_MODE", "pilot")
    monkeypatch.setenv("RISKFORGE_SQLITE_PATH", str(tmp_path / "audit.db"))
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


def _auth_headers() -> dict[str, str]:
    token = mint_hs256_token(secret=SECRET, roles=["reviewer"])
    return {"X-Correlation-ID": "audit-test", "Authorization": f"Bearer {token}"}


class TestInferenceAuditTrail:
    def test_inference_scores_append_audit_event_with_actor(self, client):
        payload = {
            "correlation_id": "audit-c-1",
            "incident": {
                "log_id": "AUDIT-1",
                "timestamp": "2026-03-01T08:00:00Z",
                "asset_id": "RIG_01",
                "asset_type": "drilling_rig",
                "raw_narrative": (
                    "Worker on an unprotected edge at 12 m while working at "
                    "height; harness missing"
                ),
                "spans": [],
            },
        }
        response = client.post("/v1/inference", json=payload, headers=_auth_headers())
        assert response.status_code == 200

        audit = client.get(
            "/v1/incidents/AUDIT-1/audit", headers=_auth_headers()
        ).json()["page"]["items"]
        kinds = [event["event_type"] for event in audit]
        assert "inference_recorded" in kinds
        event = next(e for e in audit if e["event_type"] == "inference_recorded")
        assert event["actor_id"] == "test-user"
        assert "scoring_mode=" in (event["reason"] or "")
        # The raw narrative must never appear in governance metadata.
        assert "unprotected edge" not in (event["reason"] or "")

    def test_ingest_appends_audit_events_for_every_scored_row(self, client):
        data = (
            b"log_id,timestamp,asset_id,asset_type,raw_narrative\n"
            b"AUDIT-2,2026-03-01T08:00:00Z,RIG_01,drilling_rig,"
            b"Energy isolation not applied before opening the pressure line\n"
        )
        ingest = client.post(
            "/v1/ingest?format=csv", content=data, headers=_auth_headers()
        )
        assert ingest.status_code == 200

        audit = client.get(
            "/v1/incidents/AUDIT-2/audit", headers=_auth_headers()
        ).json()["page"]["items"]
        assert any(e["event_type"] == "inference_recorded" for e in audit)
