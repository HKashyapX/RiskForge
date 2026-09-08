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
from riskforge.persistence.sqlite.repository import (
    SQLiteAuditEventRepository,
    SQLiteIncidentResultRepository,
    SQLiteReviewDecisionRepository,
)
from riskforge.persistence.sqlite.review_audit import SQLiteReviewAuditWriter
from riskforge.review.protocols import ReviewAuditWriter
from riskforge.review.service import ReviewService

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _stored() -> StoredIncidentResult:
    incident = IncidentNormalizedRecord(
        log_id="LOG_1",
        timestamp=NOW,
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Synthetic incident.",
        spans=[],
    )
    result = ModelInferenceResult(
        log_id="LOG_1",
        raw_sif_p_score=0.8,
        calibrated_sif_p_score=0.8,
        deterministic_override=False,
        routing=RoutingBucket.CRITICAL_ESCALATION,
        matched_iogp_rules=[],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )
    return StoredIncidentResult(incident=incident, result=result)


def _decision(
    decision_id: str = "D1",
    action: ReviewAction = ReviewAction.CONFIRM,
    reason: str = "Confirmed.",
) -> ReviewDecision:
    return ReviewDecision(
        decision_id=decision_id,
        log_id="LOG_1",
        action=action,
        reviewer_id="reviewer-1",
        decided_at=NOW,
        reason=reason,
    )


def _event(event_id: str = "review:D1", reason: str = "Confirmed.") -> AuditEvent:
    return AuditEvent(
        event_id=event_id,
        log_id="LOG_1",
        event_type=AuditEventType.REVIEW_DECISION_RECORDED,
        actor_id="reviewer-1",
        occurred_at=NOW,
        reason=reason,
    )


class AllowAll:
    def can_decide(self, reviewer_id: str, log_id: str, action: ReviewAction) -> bool:
        return True


def test_atomic_review_writer_implements_protocol_and_survives_restart(tmp_path) -> None:
    db_path = tmp_path / "riskforge.db"
    assert isinstance(SQLiteReviewAuditWriter(db_path), ReviewAuditWriter)
    SQLiteIncidentResultRepository(db_path).create_idempotent(_stored())

    service = ReviewService(
        SQLiteIncidentResultRepository(db_path),
        SQLiteReviewAuditWriter(db_path),
        AllowAll(),
    )
    decision = service.decide(
        log_id="LOG_1",
        decision_id="D1",
        reviewer_id="reviewer-1",
        action=ReviewAction.CONFIRM,
        reason="Confirmed.",
        decided_at=NOW,
    )
    assert decision == _decision()

    decisions = SQLiteReviewDecisionRepository(db_path).list_for_incident("LOG_1")
    events = SQLiteAuditEventRepository(db_path).list_for_incident("LOG_1")
    assert decisions.items == (_decision(),)
    assert events.items == (_event(),)


def test_atomic_review_writer_rolls_back_both_records_on_conflict(tmp_path) -> None:
    db_path = tmp_path / "riskforge.db"
    writer = SQLiteReviewAuditWriter(db_path)
    writer.append_review_atomically(_decision(), _event())
    with pytest.raises(PersistenceConflictError):
        writer.append_review_atomically(
            _decision("D2", ReviewAction.DISMISS, "Dismissed."),
            _event(),
        )

    decisions = SQLiteReviewDecisionRepository(db_path).list_for_incident("LOG_1")
    events = SQLiteAuditEventRepository(db_path).list_for_incident("LOG_1")
    assert [item.decision_id for item in decisions.items] == ["D1"]
    assert [item.event_id for item in events.items] == ["review:D1"]


def test_review_service_persists_original_automated_result_unchanged(tmp_path) -> None:
    db_path = tmp_path / "riskforge.db"
    incident_repo = SQLiteIncidentResultRepository(db_path)
    original = _stored()
    incident_repo.create_idempotent(original)

    ReviewService(incident_repo, SQLiteReviewAuditWriter(db_path), AllowAll()).decide(
        log_id="LOG_1",
        decision_id="D1",
        reviewer_id="reviewer-1",
        action=ReviewAction.ESCALATE,
        reason="Escalated after review.",
        decided_at=NOW,
    )

    assert incident_repo.get("LOG_1") == original
