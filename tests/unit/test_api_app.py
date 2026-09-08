from datetime import UTC, datetime

from fastapi.testclient import TestClient

from riskforge.api.app import create_app
from riskforge.api.dependencies import ReadinessSnapshot
from riskforge.application.exceptions import InferenceApplicationError
from riskforge.core.contracts import (
    AssetRiskSummary,
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)


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
    def __init__(self) -> None:
        self.fail_with: Exception | None = None

    def process_incident(self, record):
        if self.fail_with is not None:
            raise self.fail_with
        return _result(record.log_id)

    def process_batch(self, records):
        if self.fail_with is not None:
            raise self.fail_with
        return [_result(record.log_id) for record in records]

    def get_asset_summary(self, asset_id):
        if self.fail_with is not None:
            raise self.fail_with
        return AssetRiskSummary(
            asset_id=asset_id,
            asset_type=AssetType.DRILLING_RIG,
            total_logs=4,
            sif_precursor_count=1,
            spd_score=25.0,
            recurrent_failed_barriers=[],
        )


class FakeReadiness:
    def __init__(self, ready: bool = True) -> None:
        self.ready = ready

    def snapshot(self) -> ReadinessSnapshot:
        return ReadinessSnapshot(
            ready=self.ready,
            state="ready" if self.ready else "not_ready",
            checked_at=datetime(2026, 1, 1, tzinfo=UTC),
            components=("model", "storage"),
        )


def _client(application=None, readiness=None) -> TestClient:
    return TestClient(create_app(application or FakeApplication(), readiness or FakeReadiness()))


def test_health_and_readiness_are_distinct() -> None:
    client = _client(readiness=FakeReadiness(ready=False))

    assert client.get("/health").json() == {"api_version": "v1", "status": "live"}
    response = client.get("/ready", headers={"X-Correlation-ID": "probe-1"})
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["correlation_id"] == "probe-1"


def test_single_and_batch_inference_use_the_application_boundary() -> None:
    client = _client()
    incident = _incident()

    single = client.post(
        "/v1/inference",
        json={"correlation_id": "request-1", "incident": incident.model_dump(mode="json")},
    )
    assert single.status_code == 200
    assert single.json()["result"]["log_id"] == "LOG_1"

    batch = client.post(
        "/v1/inference/batch",
        json={
            "correlation_id": "batch-1",
            "incidents": [
                _incident("LOG_2").model_dump(mode="json"),
                _incident("LOG_1").model_dump(mode="json"),
            ],
        },
    )
    assert batch.status_code == 200
    assert [item["log_id"] for item in batch.json()["results"]] == ["LOG_2", "LOG_1"]


def test_asset_summary_uses_correlation_header() -> None:
    response = _client().get(
        "/v1/assets/RIG_01/summary",
        headers={"X-Correlation-ID": "asset-1"},
    )
    assert response.status_code == 200
    assert response.json()["correlation_id"] == "asset-1"
    assert response.json()["summary"]["spd_score"] == 25.0


def test_application_failures_are_translated_without_detail_leakage() -> None:
    application = FakeApplication()
    application.fail_with = InferenceApplicationError("Sensitive incident narrative.")
    response = _client(application=application).post(
        "/v1/inference",
        json={
            "correlation_id": "request-1",
            "incident": _incident().model_dump(mode="json"),
        },
    )
    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "inference_unavailable",
        "message": "inference service unavailable",
        "retryable": True,
    }
    assert "Sensitive incident narrative" not in response.text


def test_validation_errors_do_not_echo_invalid_incident_payloads() -> None:
    incident = _incident().model_dump(mode="json")
    incident["asset_type"] = "invalid:Sensitive incident narrative"
    response = _client().post(
        "/v1/inference",
        json={"correlation_id": "request-1", "incident": incident},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"
    assert response.json()["correlation_id"] == "unavailable"
    assert "Sensitive incident narrative" not in response.text
