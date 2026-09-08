from datetime import UTC, datetime

from fastapi.testclient import TestClient

from riskforge.api.app import create_app
from riskforge.api.dependencies import ReadinessSnapshot
from riskforge.application.exceptions import IncidentNotFoundError
from riskforge.application.workflow_models import (
    AuditEventView,
    IncidentView,
    Page,
    ReviewAction,
    ReviewDecisionView,
)
from riskforge.authentication.principal import Principal
from riskforge.core.contracts import (
    AssetRiskSummary,
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _incident(log_id: str = "LOG_1") -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=NOW,
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Sensitive incident narrative.",
        spans=[],
    )


def _result(log_id: str = "LOG_1") -> ModelInferenceResult:
    return ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=0.8,
        calibrated_sif_p_score=0.8,
        deterministic_override=False,
        routing=RoutingBucket.CRITICAL_ESCALATION,
        matched_iogp_rules=[],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )


class Application:
    def __init__(self) -> None:
        self.last_query = None
        self.last_review = None

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

    def get_incident(self, log_id):
        if log_id == "missing":
            raise IncidentNotFoundError("private")
        return IncidentView(incident=_incident(log_id), result=_result(log_id))

    def list_incidents(self, query, page):
        self.last_query = query
        return Page(
            items=(IncidentView(incident=_incident(), result=_result()),),
            offset=page.offset,
            limit=page.limit,
            total=1,
        )

    def list_audit_events(self, log_id, page):
        return Page(
            items=(
                AuditEventView(
                    event_id="event-1",
                    log_id=log_id,
                    event_type="inference_recorded",
                    actor_id="system",
                    occurred_at=NOW,
                ),
            ),
            offset=page.offset,
            limit=page.limit,
            total=1,
        )

    def decide_review(self, command):
        self.last_review = command
        return ReviewDecisionView(
            decision_id=command.decision_id,
            log_id=command.log_id,
            reviewer_id=command.reviewer_id,
            action=command.action,
            reason=command.reason,
            decided_at=NOW,
        )


class Readiness:
    def snapshot(self):
        return ReadinessSnapshot(True, "ready", NOW, ("model",))


class TrustedAuth:
    """Auth service that returns a known trusted reviewer."""

    def authenticate(self, credential: str) -> Principal:
        return Principal(subject_id="trusted-reviewer")


class RejectingAuth:
    """Auth service that rejects all credentials."""

    def authenticate(self, credential: str) -> Principal:
        from riskforge.authentication.exceptions import InvalidCredentialsError

        raise InvalidCredentialsError()


def _client(application: Application, auth_service=None) -> TestClient:
    return TestClient(create_app(application, Readiness(), auth_service))


def test_incident_queue_and_critical_routes_use_application_filters() -> None:
    application = Application()
    client = _client(application)
    response = client.get(
        "/v1/incidents?asset_id=RIG_01&limit=10",
        headers={"X-Correlation-ID": "queue-1"},
    )
    assert response.status_code == 200
    assert response.json()["page"]["total"] == 1
    assert application.last_query.asset_id == "RIG_01"

    critical = client.get(
        "/v1/incidents/critical",
        headers={"X-Correlation-ID": "critical-1"},
    )
    assert critical.status_code == 200
    assert application.last_query.routing is RoutingBucket.CRITICAL_ESCALATION


def test_incident_queue_rejects_inverted_time_window_safely() -> None:
    response = _client(Application()).get(
        "/v1/incidents?timestamp_from=2026-02-01T00:00:00Z&timestamp_to=2026-01-01T00:00:00Z",
        headers={"X-Correlation-ID": "queue-1"},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_incident_detail_and_audit_history_are_versioned() -> None:
    client = _client(Application())
    detail = client.get(
        "/v1/incidents/LOG_1", headers={"X-Correlation-ID": "detail-1"}
    )
    assert detail.status_code == 200
    assert detail.json()["incident"]["result"]["log_id"] == "LOG_1"

    audit = client.get(
        "/v1/incidents/LOG_1/audit", headers={"X-Correlation-ID": "audit-1"}
    )
    assert audit.status_code == 200
    assert audit.json()["page"]["items"][0]["event_id"] == "event-1"


def test_missing_incident_uses_safe_not_found_response() -> None:
    response = _client(Application()).get(
        "/v1/incidents/missing", headers={"X-Correlation-ID": "detail-1"}
    )
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "incident_not_found"
    assert "private" not in response.text


def test_review_identity_comes_only_from_injected_principal() -> None:
    application = Application()
    response = _client(application, TrustedAuth()).post(
        "/v1/incidents/LOG_1/reviews",
        json={
            "correlation_id": "review-1",
            "decision_id": "decision-1",
            "action": "confirm",
            "reason": "verified",
        },
        headers={"Authorization": "Bearer valid-token"},
    )
    assert response.status_code == 200
    assert response.json()["decision"]["reviewer_id"] == "trusted-reviewer"
    assert application.last_review.action is ReviewAction.CONFIRM


def test_review_fails_closed_without_authentication_provider() -> None:
    response = _client(Application()).post(
        "/v1/incidents/LOG_1/reviews",
        json={
            "correlation_id": "review-1",
            "decision_id": "decision-1",
            "action": "confirm",
            "reason": "verified",
        },
    )
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "authentication_unavailable"


def test_review_requires_an_authenticated_principal() -> None:
    response = _client(Application(), RejectingAuth()).post(
        "/v1/incidents/LOG_1/reviews",
        json={
            "correlation_id": "review-1",
            "decision_id": "decision-1",
            "action": "confirm",
            "reason": "verified",
        },
        headers={"Authorization": "Bearer some-token"},
    )
    assert response.status_code == 401
    assert "private" not in response.text
