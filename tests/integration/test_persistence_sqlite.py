from datetime import UTC, datetime, timedelta

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
    IncidentResultFilter,
    PageRequest,
    ReviewAction,
    ReviewDecision,
    StoredIncidentResult,
)
from riskforge.persistence.protocols import (
    AuditEventRepository,
    IncidentResultRepository,
    ReviewDecisionRepository,
)
from riskforge.persistence.sqlite.repository import (
    SQLiteAuditEventRepository,
    SQLiteIncidentResultRepository,
    SQLiteReviewDecisionRepository,
)


NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _stored(
    log_id: str,
    *,
    timestamp: datetime = NOW,
    asset_id: str = "RIG_01",
    score: float = 0.2,
) -> StoredIncidentResult:
    incident = IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=timestamp,
        asset_id=asset_id,
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Synthetic incident.",
        spans=[],
    )
    result = ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=score,
        calibrated_sif_p_score=score,
        deterministic_override=False,
        routing=(
            RoutingBucket.CRITICAL_ESCALATION
            if score >= 0.65
            else RoutingBucket.AUTO_DISMISS
        ),
        matched_iogp_rules=[],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )
    return StoredIncidentResult(incident=incident, result=result)


def _decision(
    decision_id: str,
    log_id: str = "LOG_1",
    *,
    when: datetime = NOW,
) -> ReviewDecision:
    return ReviewDecision(
        decision_id=decision_id,
        log_id=log_id,
        action=ReviewAction.CONFIRM,
        reviewer_id="reviewer-1",
        decided_at=when,
        reason="Confirmed.",
    )


def _event(event_id: str, log_id: str = "LOG_1", *, when: datetime = NOW) -> AuditEvent:
    return AuditEvent(
        event_id=event_id,
        log_id=log_id,
        event_type=AuditEventType.INFERENCE_RECORDED,
        actor_id="system",
        occurred_at=when,
        reason=None,
    )


def test_sqlite_repositories_satisfy_protocols(tmp_path) -> None:
    db_path = tmp_path / "riskforge.db"
    assert isinstance(SQLiteIncidentResultRepository(db_path), IncidentResultRepository)
    assert isinstance(SQLiteReviewDecisionRepository(db_path), ReviewDecisionRepository)
    assert isinstance(SQLiteAuditEventRepository(db_path), AuditEventRepository)


def test_sqlite_incident_result_idempotency_and_deterministic_queries(tmp_path) -> None:
    db_path = tmp_path / "riskforge.db"
    repo = SQLiteIncidentResultRepository(db_path)
    first = _stored("LOG_1")
    repo.create_idempotent(
        _stored(
            "LOG_2",
            timestamp=NOW + timedelta(seconds=2),
            asset_id="RIG_02",
            score=0.9,
        )
    )
    repo.create_idempotent(first)
    assert repo.get("LOG_1") == first
    assert repo.list().items[0].log_id == "LOG_1"
    filtered = repo.list(
        IncidentResultFilter(asset_id="RIG_02"),
        page=PageRequest(limit=1),
    )
    assert filtered.total == 1
    assert filtered.items[0].log_id == "LOG_2"
    with pytest.raises(PersistenceConflictError):
        repo.create_idempotent(_stored("LOG_1", score=0.9))

    restarted = SQLiteIncidentResultRepository(db_path)
    assert restarted.get("LOG_1") == first


def test_sqlite_review_history_is_append_only(tmp_path) -> None:
    repo = SQLiteReviewDecisionRepository(tmp_path / "riskforge.db")
    repo.append(_decision("D2", when=NOW + timedelta(seconds=1)))
    repo.append(_decision("D1"))
    page = repo.list_for_incident("LOG_1")
    assert [item.decision_id for item in page.items] == ["D1", "D2"]
    with pytest.raises(PersistenceConflictError):
        repo.append(_decision("D1"))


def test_sqlite_audit_append_many_rolls_back_on_conflict(tmp_path) -> None:
    repo = SQLiteAuditEventRepository(tmp_path / "riskforge.db")
    repo.append(_event("E1"))
    with pytest.raises(PersistenceConflictError):
        repo.append_many((_event("E2"), _event("E1")))
    assert repo.list_for_incident("LOG_1").items == (_event("E1"),)


def test_sqlite_audit_history_is_deterministic(tmp_path) -> None:
    repo = SQLiteAuditEventRepository(tmp_path / "riskforge.db")
    repo.append(_event("E2", when=NOW))
    repo.append(_event("E1", when=NOW))
    page = repo.list_for_incident("LOG_1")
    assert [item.event_id for item in page.items] == ["E1", "E2"]
