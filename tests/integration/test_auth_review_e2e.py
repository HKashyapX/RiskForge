"""End-to-end authentication and review integration tests.

Verifies the full lifecycle: HTTP request → credential extraction →
principal verification → review command → persistence → audit trail.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from riskforge.api.app import create_app
from riskforge.api.dependencies import ReadinessSnapshot
from riskforge.api.errors import ErrorCode
from riskforge.application.workflow_models import (
    IncidentQuery,
    PageRequest,
    ReviewAction,
    ReviewCommand,
)
from riskforge.authentication.principal import Principal
from riskforge.authentication.protocols import AuthenticationService
from riskforge.core.contracts import (
    AssetRiskSummary,
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)
from riskforge.persistence.models import (
    AuditEventType,
    StoredIncidentResult,
)
from riskforge.persistence.sqlite.repository import (
    SQLiteAuditEventRepository,
    SQLiteIncidentResultRepository,
)
from riskforge.persistence.sqlite.review_audit import SQLiteReviewAuditWriter
from riskforge.review.authorizer import AllowAllReviewerAuthorizer
from riskforge.review.service import ReviewService
from riskforge.runtime.workflow_adapters import (
    PersistenceWorkflowReader,
    ReviewServiceAdapter,
)

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _stored(log_id: str = "LOG_1", routing: RoutingBucket = RoutingBucket.CRITICAL_ESCALATION) -> StoredIncidentResult:
    incident = IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=NOW,
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Synthetic incident.",
        spans=[],
    )
    result = ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=0.8,
        calibrated_sif_p_score=0.8,
        deterministic_override=False,
        routing=routing,
        matched_iogp_rules=[],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )
    return StoredIncidentResult(incident=incident, result=result)


class FakeAuthService:
    """Deterministic auth service for testing."""

    def __init__(self, subject_id: str = "test-user") -> None:
        self._principal = Principal(subject_id=subject_id)
        self.calls: list[str] = []

    def authenticate(self, credential: str) -> Principal:
        self.calls.append(credential)
        if credential == "valid-token":
            return self._principal
        from riskforge.authentication.exceptions import InvalidCredentialsError
        raise InvalidCredentialsError()


class RejectingAuthService:
    """Auth service that rejects all credentials."""

    def authenticate(self, credential: str) -> Principal:
        from riskforge.authentication.exceptions import InvalidCredentialsError
        raise InvalidCredentialsError()


class FakeReadiness:
    def snapshot(self) -> ReadinessSnapshot:
        return ReadinessSnapshot(
            ready=True,
            state="ready",
            checked_at=datetime(2026, 1, 1, tzinfo=UTC),
            components=("model",),
        )


class FakeCoreApp:
    """Minimal core application for testing."""

    def process_incident(self, record: Any) -> ModelInferenceResult:
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

    def process_batch(self, records: Any) -> list[ModelInferenceResult]:
        return [self.process_incident(r) for r in records]

    def get_asset_summary(self, asset_id: str) -> AssetRiskSummary:
        return AssetRiskSummary(
            asset_id=asset_id,
            asset_type=AssetType.DRILLING_RIG,
            total_logs=1,
            sif_precursor_count=0,
            spd_score=0.0,
            recurrent_failed_barriers=[],
        )


class FakeWorkflowReader:
    """Workflow reader backed by SQLite repositories."""

    def __init__(self, incidents: SQLiteIncidentResultRepository, audit: SQLiteAuditEventRepository) -> None:
        self._reader = PersistenceWorkflowReader(incidents, audit)

    def get_incident(self, log_id: str) -> Any:
        return self._reader.get_incident(log_id)

    def list_incidents(self, query: IncidentQuery, page: PageRequest) -> Any:
        return self._reader.list_incidents(query, page)

    def list_audit_events(self, log_id: str, page: PageRequest) -> Any:
        return self._reader.list_audit_events(log_id, page)


class FakeWorkflowWriter:
    """Workflow writer backed by ReviewService."""

    def __init__(self, review_service: ReviewService) -> None:
        self._adapter = ReviewServiceAdapter(review_service)

    def decide(self, command: ReviewCommand) -> Any:
        return self._adapter.decide(command)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def _build_app(
    db_path: Any,
    auth_service: AuthenticationService | None = None,
) -> TestClient:
    """Build a fully wired test client with SQLite persistence."""
    incidents = SQLiteIncidentResultRepository(db_path)
    audit = SQLiteAuditEventRepository(db_path)
    writer = SQLiteReviewAuditWriter(db_path)
    authorizer = AllowAllReviewerAuthorizer()
    review_service = ReviewService(incidents, writer, authorizer)

    reader = FakeWorkflowReader(incidents, audit)
    writer_adapter = FakeWorkflowWriter(review_service)

    # Create BackendApplicationService-like object
    class FakeBackendApp(FakeCoreApp):
        def get_incident(self, log_id: str) -> Any:
            return reader.get_incident(log_id)

        def list_incidents(self, query: IncidentQuery, page: PageRequest) -> Any:
            return reader.list_incidents(query, page)

        def list_audit_events(self, log_id: str, page: PageRequest) -> Any:
            return reader.list_audit_events(log_id, page)

        def decide_review(self, command: ReviewCommand) -> Any:
            return writer_adapter.decide(command)

    app = create_app(FakeBackendApp(), FakeReadiness(), auth_service=auth_service)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests: No auth configured → review returns 401
# ---------------------------------------------------------------------------


class TestReviewWithoutAuthConfigured:
    """When no auth_service is provided, review route should return 401."""

    def test_review_returns_401_without_auth_service(self, tmp_path: Any) -> None:
        db_path = tmp_path / "riskforge.db"
        incidents = SQLiteIncidentResultRepository(db_path)
        incidents.create_idempotent(_stored())

        client = _build_app(db_path, auth_service=None)
        response = client.post(
            "/v1/incidents/LOG_1/reviews",
            json={
                "correlation_id": "c-1",
                "decision_id": "D-1",
                "action": "confirm",
                "reason": "Reviewed.",
            },
        )
        assert response.status_code == 401
        body = response.json()
        assert body["error"]["code"] == ErrorCode.MISSING_CREDENTIALS.value

    def test_auth_free_routes_still_work(self, tmp_path: Any) -> None:
        client = _build_app(tmp_path / "riskforge.db", auth_service=None)
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Tests: Auth configured → review with valid/invalid credentials
# ---------------------------------------------------------------------------


class TestReviewWithAuthConfigured:
    """When auth_service is provided, review route requires valid credentials."""

    def test_valid_token_allows_review(self, tmp_path: Any) -> None:
        db_path = tmp_path / "riskforge.db"
        incidents = SQLiteIncidentResultRepository(db_path)
        incidents.create_idempotent(_stored())

        auth = FakeAuthService("alice")
        client = _build_app(db_path, auth_service=auth)

        response = client.post(
            "/v1/incidents/LOG_1/reviews",
            json={
                "correlation_id": "c-1",
                "decision_id": "D-1",
                "action": "confirm",
                "reason": "Verified.",
            },
            headers={"Authorization": "Bearer valid-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["decision"]["reviewer_id"] == "alice"
        assert data["decision"]["action"] == "confirm"

    def test_invalid_token_returns_401(self, tmp_path: Any) -> None:
        db_path = tmp_path / "riskforge.db"
        incidents = SQLiteIncidentResultRepository(db_path)
        incidents.create_idempotent(_stored())

        auth = FakeAuthService()
        client = _build_app(db_path, auth_service=auth)

        response = client.post(
            "/v1/incidents/LOG_1/reviews",
            json={
                "correlation_id": "c-1",
                "decision_id": "D-1",
                "action": "confirm",
                "reason": "Review.",
            },
            headers={"Authorization": "Bearer bad-token"},
        )
        assert response.status_code == 401
        body = response.json()
        assert body["error"]["code"] == ErrorCode.INVALID_CREDENTIALS.value

    def test_missing_header_returns_401(self, tmp_path: Any) -> None:
        db_path = tmp_path / "riskforge.db"
        incidents = SQLiteIncidentResultRepository(db_path)
        incidents.create_idempotent(_stored())

        auth = FakeAuthService()
        client = _build_app(db_path, auth_service=auth)

        response = client.post(
            "/v1/incidents/LOG_1/reviews",
            json={
                "correlation_id": "c-1",
                "decision_id": "D-1",
                "action": "confirm",
                "reason": "Review.",
            },
        )
        assert response.status_code == 401

    def test_malformed_header_returns_401(self, tmp_path: Any) -> None:
        db_path = tmp_path / "riskforge.db"
        incidents = SQLiteIncidentResultRepository(db_path)
        incidents.create_idempotent(_stored())

        auth = FakeAuthService()
        client = _build_app(db_path, auth_service=auth)

        response = client.post(
            "/v1/incidents/LOG_1/reviews",
            json={
                "correlation_id": "c-1",
                "decision_id": "D-1",
                "action": "confirm",
                "reason": "Review.",
            },
            headers={"Authorization": "Basic abc"},
        )
        assert response.status_code == 401

    def test_auth_error_exposes_no_credentials(self, tmp_path: Any) -> None:
        db_path = tmp_path / "riskforge.db"
        incidents = SQLiteIncidentResultRepository(db_path)
        incidents.create_idempotent(_stored())

        auth = FakeAuthService()
        client = _build_app(db_path, auth_service=auth)

        response = client.post(
            "/v1/incidents/LOG_1/reviews",
            json={
                "correlation_id": "c-1",
                "decision_id": "D-1",
                "action": "confirm",
                "reason": "Review.",
            },
            headers={"Authorization": "Bearer super-secret-key"},
        )
        assert response.status_code == 401
        body_str = str(response.json())
        assert "super-secret-key" not in body_str


# ---------------------------------------------------------------------------
# Tests: Identity propagation through the full stack
# ---------------------------------------------------------------------------


class TestIdentityPropagation:
    """Verify that the authenticated principal's subject_id flows through
    to the persisted review decision and audit event."""

    def test_reviewer_id_matches_principal(self, tmp_path: Any) -> None:
        db_path = tmp_path / "riskforge.db"
        incidents = SQLiteIncidentResultRepository(db_path)
        incidents.create_idempotent(_stored())

        auth = FakeAuthService("reviewer-bob")
        client = _build_app(db_path, auth_service=auth)

        response = client.post(
            "/v1/incidents/LOG_1/reviews",
            json={
                "correlation_id": "c-1",
                "decision_id": "D-1",
                "action": "confirm",
                "reason": "Confirmed after inspection.",
            },
            headers={"Authorization": "Bearer valid-token"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["decision"]["reviewer_id"] == "reviewer-bob"

    def test_different_principals_get_different_reviewer_ids(self, tmp_path: Any) -> None:
        db_path = tmp_path / "riskforge.db"
        incidents = SQLiteIncidentResultRepository(db_path)
        incidents.create_idempotent(_stored("LOG_1"))
        incidents.create_idempotent(_stored("LOG_2"))

        auth_alice = FakeAuthService("alice")
        client_alice = _build_app(db_path, auth_service=auth_alice)

        # Alice reviews LOG_1
        resp1 = client_alice.post(
            "/v1/incidents/LOG_1/reviews",
            json={
                "correlation_id": "c-1",
                "decision_id": "D-1",
                "action": "confirm",
                "reason": "Alice confirms.",
            },
            headers={"Authorization": "Bearer valid-token"},
        )
        assert resp1.status_code == 200
        assert resp1.json()["decision"]["reviewer_id"] == "alice"

        # Bob reviews LOG_2
        auth_bob = FakeAuthService("bob")
        client_bob = _build_app(db_path, auth_service=auth_bob)

        resp2 = client_bob.post(
            "/v1/incidents/LOG_2/reviews",
            json={
                "correlation_id": "c-2",
                "decision_id": "D-2",
                "action": "dismiss",
                "reason": "Bob dismisses.",
            },
            headers={"Authorization": "Bearer valid-token"},
        )
        assert resp2.status_code == 200
        assert resp2.json()["decision"]["reviewer_id"] == "bob"


# ---------------------------------------------------------------------------
# Tests: Audit trail records the authenticated actor
# ---------------------------------------------------------------------------


class TestAuditTrailIdentity:
    """Verify that audit events record the authenticated principal as actor_id."""

    def test_audit_event_actor_matches_principal(self, tmp_path: Any) -> None:
        db_path = tmp_path / "riskforge.db"
        incidents = SQLiteIncidentResultRepository(db_path)
        incidents.create_idempotent(_stored())

        auth = FakeAuthService("auditor-carol")
        client = _build_app(db_path, auth_service=auth)

        # Perform a review
        resp = client.post(
            "/v1/incidents/LOG_1/reviews",
            json={
                "correlation_id": "c-1",
                "decision_id": "D-1",
                "action": "escalate",
                "reason": "Escalating for safety review.",
            },
            headers={"Authorization": "Bearer valid-token"},
        )
        assert resp.status_code == 200

        # Verify audit trail
        audit_repo = SQLiteAuditEventRepository(db_path)
        events = audit_repo.list_for_incident("LOG_1", page=PageRequest())
        assert events.total == 1
        event = events.items[0]
        assert event.event_type == AuditEventType.REVIEW_DECISION_RECORDED
        assert event.actor_id == "auditor-carol"
        assert event.reason == "Escalating for safety review."


# ---------------------------------------------------------------------------
# Tests: Auth free routes unaffected by auth configuration
# ---------------------------------------------------------------------------


class TestAuthFreeRoutesUnaffected:
    """Health, readiness, and read-only routes must work regardless of auth config."""

    def test_health_works_with_auth_service(self, tmp_path: Any) -> None:
        client = _build_app(tmp_path / "riskforge.db", auth_service=FakeAuthService())
        assert client.get("/health").status_code == 200

    def test_readiness_works_with_auth_service(self, tmp_path: Any) -> None:
        client = _build_app(tmp_path / "riskforge.db", auth_service=FakeAuthService())
        resp = client.get("/ready", headers={"X-Correlation-ID": "probe"})
        assert resp.status_code == 200

    def test_health_works_without_auth_service(self, tmp_path: Any) -> None:
        client = _build_app(tmp_path / "riskforge.db", auth_service=None)
        assert client.get("/health").status_code == 200


# ---------------------------------------------------------------------------
# Tests: AllowAllReviewerAuthorizer protocol conformance
# ---------------------------------------------------------------------------


class TestAllowAllAuthorizer:
    """Verify the default authorizer satisfies the protocol and allows all actions."""

    def test_satisfies_protocol(self) -> None:
        from riskforge.review.protocols import ReviewerAuthorizer
        assert isinstance(AllowAllReviewerAuthorizer(), ReviewerAuthorizer)

    @pytest.mark.parametrize("action", list(ReviewAction))
    def test_allows_all_actions(self, action: ReviewAction) -> None:
        auth = AllowAllReviewerAuthorizer()
        assert auth.can_decide("any-reviewer", "any-log-id", action) is True


# ---------------------------------------------------------------------------
# Tests: Full end-to-end review lifecycle with auth
# ---------------------------------------------------------------------------


class TestFullReviewLifecycle:
    """End-to-end test: create incident → authenticate → review → verify persistence."""

    def test_complete_lifecycle(self, tmp_path: Any) -> None:
        db_path = tmp_path / "riskforge.db"
        incidents = SQLiteIncidentResultRepository(db_path)
        incidents.create_idempotent(_stored())

        auth = FakeAuthService("inspector-dave")
        client = _build_app(db_path, auth_service=auth)

        # 1. Review the incident
        resp = client.post(
            "/v1/incidents/LOG_1/reviews",
            json={
                "correlation_id": "c-1",
                "decision_id": "D-1",
                "action": "confirm",
                "reason": "Inspected and confirmed.",
            },
            headers={"Authorization": "Bearer valid-token"},
        )
        assert resp.status_code == 200
        decision = resp.json()["decision"]
        assert decision["reviewer_id"] == "inspector-dave"
        assert decision["action"] == "confirm"
        assert decision["log_id"] == "LOG_1"

        # 2. Verify the audit trail
        audit_repo = SQLiteAuditEventRepository(db_path)
        events = audit_repo.list_for_incident("LOG_1", page=PageRequest())
        assert events.total == 1
        assert events.items[0].actor_id == "inspector-dave"

        # 3. Verify the original automated result is unchanged
        incident = incidents.get("LOG_1")
        assert incident is not None
        assert incident.result.routing is RoutingBucket.CRITICAL_ESCALATION
        assert incident.result.calibrated_sif_p_score == 0.8
