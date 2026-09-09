from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from riskforge.runtime.demo import create_demo_app


def test_demo_requires_explicit_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RISKFORGE_MODE", raising=False)
    with pytest.raises(RuntimeError, match="RISKFORGE_MODE=demo"):
        create_demo_app()


@pytest.mark.parametrize("environment", ["staging", "production"])
def test_demo_is_forbidden_in_secure_environments(
    monkeypatch: pytest.MonkeyPatch, environment: str
) -> None:
    monkeypatch.setenv("RISKFORGE_MODE", "demo")
    monkeypatch.setenv("RISKFORGE_ENV", environment)
    with pytest.raises(RuntimeError, match="forbidden"):
        create_demo_app()


def test_demo_is_live_but_never_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RISKFORGE_MODE", "demo")
    monkeypatch.setenv("RISKFORGE_ENV", "development")
    client = TestClient(create_demo_app())

    assert client.get("/health").status_code == 200
    response = client.get("/ready", headers={"X-Correlation-ID": "demo-check"})
    assert response.status_code == 503
    assert response.json() == {
        "api_version": "v1",
        "correlation_id": "demo-check",
        "ready": False,
        "state": "not_ready",
        "checked_at": response.json()["checked_at"],
        "components": ["model"],
    }


def test_demo_inference_returns_safe_unavailable_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RISKFORGE_MODE", "demo")
    monkeypatch.setenv("RISKFORGE_ENV", "development")
    client = TestClient(create_demo_app(), raise_server_exceptions=False)

    response = client.post(
        "/v1/inference",
        json={
            "correlation_id": "demo-inference",
            "incident": {
                "log_id": "DEMO-1",
                "raw_narrative": "Demo incident",
                "asset_id": "DEMO-ASSET",
                "asset_type": "drilling_rig",
                "timestamp": "2026-09-09T00:00:00Z",
                "spans": [],
            },
        },
    )

    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "inference_unavailable",
        "message": "inference service unavailable",
        "retryable": True,
    }
