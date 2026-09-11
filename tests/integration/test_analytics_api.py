"""Integration tests for the analytics summary and explanation endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from riskforge.runtime.production import create_app

HEADERS = {"X-Correlation-ID": "analytics-test"}

NARRATIVES = [
    ("E-1", "RIG_01", "2026-06-01T08:00:00Z", "Worker on unprotected edge at 15 m; harness missing; working at height."),
    ("E-2", "RIG_01", "2026-06-08T08:00:00Z", "Scaffold fall hazard; fall protection missing at height."),
    ("E-3", "RIG_02", "2026-06-15T08:00:00Z", "Confined space tank entry; no gas test; permit missing."),
    ("E-4", "RIG_02", "2026-06-15T10:00:00Z", "Routine inspection completed; no anomalies."),
    ("E-5", "RIG_03", "2026-06-22T08:00:00Z", "H2s 25 ppm release; detector alarm; technician exposed."),
    ("E-6", "RIG_03", "2026-06-29T08:00:00Z", "Vehicle reversing without a spotter during a site journey."),
]


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("RISKFORGE_SQLITE_PATH", str(tmp_path / "analytics.db"))
    monkeypatch.delenv("RISKFORGE_MODEL_PATH", raising=False)
    monkeypatch.delenv("RISKFORGE_MANIFEST_PATH", raising=False)
    with TestClient(create_app()) as test_client:
        lines = [
            (
                '{"log_id":"' + log_id + '","timestamp":"' + ts
                + '","asset_id":"' + asset + '","asset_type":"drilling_rig","raw_narrative":"'
                + narrative + '"}'
            )
            for log_id, asset, ts, narrative in NARRATIVES
        ]
        response = test_client.post(
            "/v1/ingest?format=jsonl", content=("\n".join(lines) + "\n").encode(), headers=HEADERS
        )
        assert response.status_code == 200
        assert response.json()["normalized"] == len(NARRATIVES)
        yield test_client


class TestAnalyticsSummaryEndpoint:
    def test_summary_shape_and_values(self, client):
        response = client.get("/v1/analytics/summary", headers=HEADERS)
        assert response.status_code == 200
        summary = response.json()["summary"]
        assert summary["total_reports"] == len(NARRATIVES)
        assert summary["sif_precursors"] == 4
        assert summary["sif_precursor_density"] == round(4 / 6, 4)
        # E-1/E-2/E-3/E-5 critical, E-4 dismiss, E-6 borderline driving hazard.
        assert summary["routing_counts"] == {
            "critical_escalation": 4,
            "auto_dismiss": 1,
            "hitl_review": 1,
        }
        assert summary["matched_rule_counts"]["work_at_height"] == 2
        assert len(summary["weekly_trend"]) == 5
        assert {a["asset_id"] for a in summary["assets"]} == {"RIG_01", "RIG_02", "RIG_03"}
        assert "disclaimer" in summary

    def test_summary_deterministic_across_calls(self, client):
        first = client.get("/v1/analytics/summary", headers=HEADERS).json()["summary"]
        second = client.get("/v1/analytics/summary", headers=HEADERS).json()["summary"]
        assert first == second


class TestExplanationEndpoint:
    def test_critical_incident_explains_its_classification(self, client):
        response = client.get("/v1/incidents/E-1/explanation", headers=HEADERS)
        assert response.status_code == 200
        explanation = response.json()["explanation"]
        assert explanation["routing"] == "critical_escalation"
        assert "work_at_height" in explanation["matched_iogp_rules"]
        assert explanation["calibrated_sif_p_score"] >= 0.65
        assert explanation["recommendations"]
        assert "do not replace" in explanation["disclaimer"]
        assert "triad" in explanation

    def test_benign_incident_has_no_recommendations(self, client):
        response = client.get("/v1/incidents/E-4/explanation", headers=HEADERS)
        assert response.status_code == 200
        explanation = response.json()["explanation"]
        assert explanation["routing"] == "auto_dismiss"
        assert explanation["matched_iogp_rules"] == []
        assert explanation["recommendations"] == []

    def test_unknown_incident_returns_404(self, client):
        response = client.get("/v1/incidents/MISSING-1/explanation", headers=HEADERS)
        assert response.status_code == 404
