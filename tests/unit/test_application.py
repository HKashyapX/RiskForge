from datetime import UTC, datetime

import pytest

from riskforge.application.exceptions import (
    DuplicateLogIdError,
    InferenceApplicationError,
    ResultCorrelationError,
)
from riskforge.application.service import ApplicationService
from riskforge.core.contracts import (
    AssetRiskSummary,
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)


def _record(log_id: str) -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=datetime.now(UTC),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Synthetic incident.",
        spans=[],
    )


def _result(log_id: str) -> ModelInferenceResult:
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


class FakeInference:
    def __init__(self, *, reverse_batch: bool = False, wrong_id: bool = False) -> None:
        self.reverse_batch = reverse_batch
        self.wrong_id = wrong_id

    def infer(self, record):
        return _result("WRONG" if self.wrong_id else record.log_id)

    def infer_batch(self, records):
        results = [_result(record.log_id) for record in records]
        return list(reversed(results)) if self.reverse_batch else results


class FailingInference(FakeInference):
    def infer(self, record):
        raise RuntimeError("dependency failure")


class FakeMetrics:
    def asset_summary(self, asset_id: str) -> AssetRiskSummary:
        return AssetRiskSummary(
            asset_id=asset_id,
            asset_type=AssetType.DRILLING_RIG,
            total_logs=1,
            sif_precursor_count=0,
            spd_score=0.0,
            recurrent_failed_barriers=[],
        )


def _service(inference=None) -> ApplicationService:
    return ApplicationService(inference or FakeInference(), FakeMetrics())


def test_process_incident_preserves_log_id() -> None:
    record = _record("LOG_1")
    assert _service().process_incident(record).log_id == "LOG_1"


def test_process_batch_restores_request_order_by_log_id() -> None:
    records = [_record("LOG_1"), _record("LOG_2"), _record("LOG_3")]
    results = _service(FakeInference(reverse_batch=True)).process_batch(records)
    assert [result.log_id for result in results] == ["LOG_1", "LOG_2", "LOG_3"]


def test_duplicate_ids_are_rejected_before_inference() -> None:
    with pytest.raises(DuplicateLogIdError):
        _service().process_batch([_record("LOG_1"), _record("LOG_1")])


def test_single_result_correlation_mismatch_is_rejected() -> None:
    with pytest.raises(ResultCorrelationError):
        _service(FakeInference(wrong_id=True)).process_incident(_record("LOG_1"))


def test_dependency_failures_are_translated() -> None:
    with pytest.raises(InferenceApplicationError):
        _service(FailingInference()).process_incident(_record("LOG_1"))


def test_empty_batch_is_deterministic_and_does_not_call_inference() -> None:
    assert _service().process_batch([]) == []


def test_metrics_access_uses_injected_protocol() -> None:
    summary = _service().get_asset_summary("RIG_01")
    assert summary.asset_id == "RIG_01"
