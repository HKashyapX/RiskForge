from datetime import UTC, datetime

import pytest

from riskforge.core.contracts import (
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)
from riskforge.persistence.exceptions import PersistenceConflictError
from riskforge.persistence.models import (
    AuditEvent,
    AuditEventType,
    ReviewAction,
    ReviewDecision,
    StoredIncidentResult,
)
from riskforge.review.exceptions import (
    ReviewConflictError,
    ReviewNotFoundError,
    ReviewPermissionError,
    ReviewPersistenceError,
)
from riskforge.review.service import ReviewService

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _stored(log_id: str = "LOG_1") -> StoredIncidentResult:
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
        routing=RoutingBucket.CRITICAL_ESCALATION,
        matched_iogp_rules=[],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )
    return StoredIncidentResult(incident=incident, result=result)


class FakeReader:
    def __init__(self, record: StoredIncidentResult | None = None) -> None:
        self.record = record

    def get(self, log_id: str) -> StoredIncidentResult | None:
        if self.record is None or self.record.log_id != log_id:
            return None
        return self.record


class FakeAuthorizer:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[tuple[str, str, ReviewAction]] = []

    def can_decide(self, reviewer_id: str, log_id: str, action: ReviewAction) -> bool:
        self.calls.append((reviewer_id, log_id, action))
        return self.allowed


class FakeWriter:
    def __init__(self) -> None:
        self.calls: list[tuple[ReviewDecision, AuditEvent]] = []

    def append_review_atomically(
        self, decision: ReviewDecision, event: AuditEvent
    ) -> tuple[ReviewDecision, AuditEvent]:
        self.calls.append((decision, event))
        return decision, event


class FailingWriter(FakeWriter):
    def append_review_atomically(
        self, decision: ReviewDecision, event: AuditEvent
    ) -> tuple[ReviewDecision, AuditEvent]:
        raise RuntimeError("writer unavailable")


class ConflictWriter(FakeWriter):
    def append_review_atomically(
        self, decision: ReviewDecision, event: AuditEvent
    ) -> tuple[ReviewDecision, AuditEvent]:
        raise PersistenceConflictError("duplicate")


def _service(*, authorizer=True, writer=None, record=None):
    return ReviewService(
        FakeReader(record or _stored()),
        writer or FakeWriter(),
        FakeAuthorizer(authorizer),
    )


@pytest.mark.parametrize("action", list(ReviewAction))
def test_all_review_actions_are_recorded(action: ReviewAction) -> None:
    writer = FakeWriter()
    service = _service(writer=writer)
    result = service.decide(
        log_id="LOG_1",
        decision_id=f"D-{action.value}",
        reviewer_id="reviewer-1",
        action=action,
        reason="Reviewed.",
        decided_at=NOW,
    )
    assert result.action is action
    assert len(writer.calls) == 1
    decision, event = writer.calls[0]
    assert decision == result
    assert event.log_id == "LOG_1"
    assert event.event_type is AuditEventType.REVIEW_DECISION_RECORDED
    assert event.actor_id == "reviewer-1"


def test_permission_denial_prevents_write() -> None:
    writer = FakeWriter()
    service = _service(authorizer=False, writer=writer)
    with pytest.raises(ReviewPermissionError):
        service.decide(
            log_id="LOG_1",
            decision_id="D1",
            reviewer_id="reviewer-1",
            action=ReviewAction.CONFIRM,
            reason="Rejected.",
            decided_at=NOW,
        )
    assert writer.calls == []


def test_missing_incident_is_rejected_before_permission_check() -> None:
    service = _service(record=None)
    with pytest.raises(ReviewNotFoundError):
        service.decide(
            log_id="MISSING",
            decision_id="D1",
            reviewer_id="reviewer-1",
            action=ReviewAction.CONFIRM,
            reason="Missing.",
            decided_at=NOW,
        )


def test_original_automated_result_is_not_mutated() -> None:
    record = _stored()
    original = record.model_copy(deep=True)
    service = _service(record=record)
    service.decide(
        log_id="LOG_1",
        decision_id="D1",
        reviewer_id="reviewer-1",
        action=ReviewAction.DISMISS,
        reason="Reviewed.",
        decided_at=NOW,
    )
    assert record == original
    assert record.result.routing is RoutingBucket.CRITICAL_ESCALATION
    assert record.result.calibrated_sif_p_score == 0.8


def test_conflict_is_translated() -> None:
    with pytest.raises(ReviewConflictError):
        _service(writer=ConflictWriter()).decide(
            log_id="LOG_1",
            decision_id="D1",
            reviewer_id="reviewer-1",
            action=ReviewAction.CONFIRM,
            reason="Reviewed.",
            decided_at=NOW,
        )


def test_persistence_failure_is_translated() -> None:
    with pytest.raises(ReviewPersistenceError):
        _service(writer=FailingWriter()).decide(
            log_id="LOG_1",
            decision_id="D1",
            reviewer_id="reviewer-1",
            action=ReviewAction.CONFIRM,
            reason="Reviewed.",
            decided_at=NOW,
        )


def test_invalid_identifiers_and_reason_are_rejected() -> None:
    service = _service()
    with pytest.raises(ValueError):
        service.decide(
            log_id=" ",
            decision_id="D1",
            reviewer_id="reviewer-1",
            action=ReviewAction.CONFIRM,
            reason="Reviewed.",
            decided_at=NOW,
        )
    with pytest.raises(ValueError):
        service.decide(
            log_id="LOG_1",
            decision_id="D1",
            reviewer_id="reviewer-1",
            action=ReviewAction.CONFIRM,
            reason=" ",
            decided_at=NOW,
        )
