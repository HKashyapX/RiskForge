"""Ingestion pipeline and end-to-end upload route tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from riskforge.core.contracts import AssetType, IncidentRawRecord, RoutingBucket
from riskforge.ingestion.exceptions import ReportValidationError
from riskforge.ingestion.pipeline import IngestionPipeline
from riskforge.runtime.production import create_app

HEADERS = {"X-Correlation-ID": "ingest-test"}


class _StubEngine:
    def infer(self, record):
        from riskforge.core.contracts import ModelInferenceResult, OperationalTriad

        return ModelInferenceResult(
            log_id=record.log_id,
            raw_sif_p_score=0.1,
            calibrated_sif_p_score=0.1,
            deterministic_override=False,
            routing=RoutingBucket.AUTO_DISMISS,
            matched_iogp_rules=[],
            triad=OperationalTriad(),
            latency_ms=1.0,
        )

    def infer_batch(self, records):
        return [self.infer(record) for record in records]


class TestPipeline:
    def _pipeline(self) -> IngestionPipeline:
        def normalize(record):
            from riskforge.core.contracts import IncidentNormalizedRecord

            return IncidentNormalizedRecord(
                log_id=record.log_id,
                timestamp=record.timestamp,
                asset_id=record.asset_id,
                asset_type=record.asset_type,
                raw_narrative=record.raw_narrative,
                spans=[],
            )

        return IngestionPipeline(normalize, _StubEngine())

    def test_run_scores_each_record(self):
        records_data = (
            b"log_id,timestamp,asset_id,asset_type,raw_narrative\n"
            b"P-1,2026-03-01T08:00:00Z,RIG_01,drilling_rig,text one\n"
            b"P-2,2026-03-02T08:00:00Z,RIG_01,drilling_rig,text two\n"
        )
        outcome = self._pipeline().run(records_data, "csv")
        assert (outcome.received, outcome.normalized, outcome.failed) == (2, 2, 0)
        assert outcome.ok is True
        assert [item.result.log_id for item in outcome.items] == ["P-1", "P-2"]

    def test_normalization_failure_is_isolated_per_report(self):
        def failing_normalize(_record):
            raise ValueError("normalization blew up")

        pipeline = IngestionPipeline(failing_normalize, _StubEngine())
        outcome = pipeline.run(
            (
                b"log_id,timestamp,asset_id,asset_type,raw_narrative\n"
                b"P-1,2026-03-01T08:00:00Z,RIG_01,drilling_rig,text\n"
            ),
            "csv",
        )
        assert (outcome.normalized, outcome.failed) == (0, 1)
        assert "normalization blew up" in outcome.items[0].error

    def test_validate_records_rejects_empty_narratives(self):
        record = IncidentRawRecord(
            log_id="E-1",
            timestamp=datetime(2026, 3, 1, tzinfo=UTC),
            asset_id="A",
            asset_type=AssetType.DRILLING_RIG,
            raw_narrative="   ",
        )
        with pytest.raises(ReportValidationError, match="reports failed validation"):
            self._pipeline().validate_records([record])


@pytest.fixture()
def client(monkeypatch, tmp_path):
    monkeypatch.setenv("RISKFORGE_SQLITE_PATH", str(tmp_path / "ingest.db"))
    monkeypatch.delenv("RISKFORGE_MODEL_PATH", raising=False)
    monkeypatch.delenv("RISKFORGE_MANIFEST_PATH", raising=False)
    with TestClient(create_app()) as test_client:
        yield test_client


class TestIngestRoute:
    def test_csv_upload_scores_and_persists(self, client):
        data = (
            b"log_id,timestamp,asset_id,asset_type,raw_narrative\n"
            b"CSV-1,2026-03-01T08:00:00Z,RIG_01,drilling_rig,Worker on an unprotected edge at 12 m while working at height; harness missing\n"
            b"CSV-2,2026-03-02T09:00:00Z,RIG_01,drilling_rig,Routine housekeeping completed near the gate\n"
        )
        response = client.post("/v1/ingest?format=csv", content=data, headers=HEADERS)
        assert response.status_code == 200
        body = response.json()
        assert (body["received"], body["normalized"], body["failed"]) == (2, 2, 0)
        routings = {item["log_id"]: item["result"]["routing"] for item in body["items"]}
        assert routings["CSV-1"] == RoutingBucket.CRITICAL_ESCALATION.value
        assert routings["CSV-2"] == RoutingBucket.AUTO_DISMISS.value

        queue = client.get("/v1/incidents?limit=10", headers=HEADERS).json()["page"]["items"]
        assert {"CSV-1", "CSV-2"} <= {item["incident"]["log_id"] for item in queue}

    def test_reingesting_same_file_is_idempotent(self, client):
        data = (
            b"log_id,timestamp,asset_id,asset_type,raw_narrative\n"
            b"CSV-DUP,2026-03-01T08:00:00Z,RIG_01,drilling_rig,Repeated upload of the same file\n"
        )
        first = client.post("/v1/ingest?format=csv", content=data, headers=HEADERS)
        second = client.post("/v1/ingest?format=csv", content=data, headers=HEADERS)
        assert first.status_code == 200 and second.status_code == 200
        queue = client.get("/v1/incidents?limit=10", headers=HEADERS).json()["page"]["items"]
        assert [i["incident"]["log_id"] for i in queue].count("CSV-DUP") == 1

    def test_malformed_document_returns_422(self, client):
        response = client.post(
            "/v1/ingest?format=csv",
            content=b"log_id,timestamp\nBROKEN,not-a-date\n",
            headers=HEADERS,
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_request"

    def test_unsupported_format_returns_415(self, client):
        response = client.post(
            "/v1/ingest?format=docx", content=b"whatever", headers=HEADERS
        )
        assert response.status_code == 415

    def test_empty_body_returns_422(self, client):
        response = client.post("/v1/ingest?format=csv", content=b"", headers=HEADERS)
        assert response.status_code == 422
