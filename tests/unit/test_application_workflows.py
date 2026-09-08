from datetime import UTC, datetime

import pytest

from riskforge.application.backend_service import BackendApplicationService
from riskforge.application.exceptions import (
    IncidentNotFoundError,
    QueryApplicationError,
    ReviewPermissionApplicationError,
)
from riskforge.application.workflow_models import (
    AuditEventView,
    IncidentQuery,
    IncidentView,
    Page,
    PageRequest,
    ReviewAction,
    ReviewCommand,
)
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
        raw_narrative="incident",
        spans=[],
    )


def _result(log_id: str = "LOG_1") -> ModelInferenceResult:
    return ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=0.7,
        calibrated_sif_p_score=0.7,
        deterministic_override=False,
        routing=RoutingBucket.CRITICAL_ESCALATION,
        matched_iogp_rules=[],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )


class Core:
    def process_incident(self, record):
        return _result(record.log_id)

    def process_batch(self, records):
        return [_result(record.log_id) for record in records]

    def get_asset_summary(self, asset_id):
        return AssetRiskSummary(
            asset_id=asset_id,
            asset_type=AssetType.DRILLING_RIG,
            total_logs=1,
            sif_precursor_count=1,
            spd_score=100.0,
            recurrent_failed_barriers=[],
        )


class Reader:
    def __init__(self) -> None:
        self.fail = False

    def get_incident(self, log_id):
        if self.fail:
            raise RuntimeError("private")
        if log_id == "missing":
            return None
        return IncidentView(incident=_incident(log_id), result=_result(log_id))

    def list_incidents(self, query, page):
        return Page(items=(self.get_incident("LOG_1"),), offset=page.offset, limit=page.limit, total=1)

    def list_audit_events(self, log_id, page):
        return Page(
            items=(
                AuditEventView(
                    event_id="event-1",
                    log_id=log_id,
                    event_type="inference_recorded",
                    actor_id="system",
                    occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
                ),
            ),
            offset=page.offset,
            limit=page.limit,
            total=1,
        )


class Reviews:
    def decide(self, command):
        raise ReviewPermissionApplicationError("safe")


def test_backend_facade_delegates_core_and_exposes_workflow_views() -> None:
    service = BackendApplicationService(Core(), Reader(), Reviews())
    assert service.process_incident(_incident()).log_id == "LOG_1"
    assert service.get_incident("LOG_2").result.log_id == "LOG_2"
    page = service.list_incidents(IncidentQuery(), PageRequest(limit=10))
    assert [item.result.log_id for item in page.items] == ["LOG_1"]
    audit = service.list_audit_events("LOG_1", PageRequest())
    assert audit.items[0].event_id == "event-1"


def test_backend_facade_translates_query_failures_and_missing_incidents() -> None:
    reader = Reader()
    service = BackendApplicationService(Core(), reader, Reviews())
    with pytest.raises(IncidentNotFoundError):
        service.get_incident("missing")
    reader.fail = True
    with pytest.raises(QueryApplicationError) as caught:
        service.get_incident("LOG_1")
    assert "private" not in str(caught.value)


def test_backend_facade_preserves_specific_review_errors() -> None:
    service = BackendApplicationService(Core(), Reader(), Reviews())
    command = ReviewCommand(
        log_id="LOG_1",
        decision_id="decision-1",
        reviewer_id="reviewer-1",
        action=ReviewAction.CONFIRM,
        reason="verified",
    )
    with pytest.raises(ReviewPermissionApplicationError):
        service.decide_review(command)
