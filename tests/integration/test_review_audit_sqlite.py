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
from riskforge.persistence.sqlite.repository import SQLiteIncidentResultRepository
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


def _decision() -> ReviewDecision:
    return ReviewDecision(
        decision_id="D1",
        log_id="LOG_1",
        action=ReviewAction.CONFIRM,
        reviewer_id="reviewer-1",
        decided_at=NOW,
        reason="Confirmed.",
    )


def _event() -> AuditEvent:
    return AuditEvent(
        event_id="review:D1",
        log_id="LOG_1",
        event_type=AuditEventType.REVIEW_DECISION_RECORDED,
        actor_id="reviewer-1",
        occurred_at=NOW,
        reason="Confirmed.",
    )


def test_atomic_review_writer_implements_protocol_and_survives_restart(tmp_path) -> None:
    db_path = tmp_path / "riskforge.db"
    assert isinstance(SQLiteReviewAuditWriter(db_path), ReviewAuditWriter)
    SQLiteIncidentResultRepository(db_path).create_idempotent(_stored())

    class AllowAll:
        def can_decide(self, reviewer_id, log_id, action):
            return True

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
    assert SQLiteReviewAuditWriter(db_path) is not None

    restarted_db = tmp_path / "riskforge.db"
    from riskforge.persistence.sqlite.repository import SQLiteReviewDecisionRepository
    from riskforge.persistence.sqlite.repository import SQLiteAuditEventRepository

    decisions = SQLiteReviewDecisionRepository(restarted_db).list_for_incident("LOG_1")
    events = SQLiteAuditEventRepository(restarted_db).list_for_incident("LOG_1")
    assert decisions.items == (_decision(),)
    assert events.items == (_event(),)


def test_atomic_review_writer_rolls_back_when_audit_conflicts(tmp_path) -> None:
    db_path = tmp_path / "riskforge.db"
    writer = SQLiteReviewAuditWriter(db_path)
    writer.append_review_atomically(_decision(), _event())
    with pytest.raises(PersistenceConflictError):
        writer.append_review_atomically(
            ReviewDecision(
                decision_id="D2",
                log_id="LOG_1",
                action=ReviewAction.DISMISS,
                reviewer_id="reviewer-1",
                decided_at=NOW,
                reason="Dismissed.",
            ),
            _event(),
        )

    assert SQLiteIncidentResultRepository(db_path).get("LOG_1") is None
    from riskforge.persistence.sqlite.repository import SQLiteReviewDecisionRepository

    assert SQLiteReviewDecisionRepository(db_path).list_for_incident("LOG_1").items == (_decision(),)
