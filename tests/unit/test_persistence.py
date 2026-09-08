from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from riskforge.core.contracts import (
    AssetType,
    IncidentNormalizedRecord,
    LifeSavingRule,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)
from riskforge.persistence.exceptions import PersistenceConflictError
from riskforge.persistence.models import (
    AuditEvent,
    AuditEventType,
    IncidentResultFilter,
    Page,
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

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _incident(log_id: str, *, asset_id: str = "RIG_01", timestamp: datetime = NOW) -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=timestamp,
        asset_id=asset_id,
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Synthetic incident.",
        spans=[],
    )


def _result(log_id: str, *, score: float = 0.2) -> ModelInferenceResult:
    return ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=score,
        calibrated_sif_p_score=score,
        deterministic_override=False,
        routing=(
            RoutingBucket.CRITICAL_ESCALATION
            if score >= 0.65
            else RoutingBucket.AUTO_DISMISS
        ),
        matched_iogp_rules=[LifeSavingRule.LINE_OF_FIRE] if score >= 0.65 else [],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )


def _stored(
    log_id: str,
    *,
    score: float = 0.2,
    asset_id: str = "RIG_01",
    timestamp: datetime = NOW,
) -> StoredIncidentResult:
    return StoredIncidentResult(
        incident=_incident(log_id, asset_id=asset_id, timestamp=timestamp),
        result=_result(log_id, score=score),
    )


class FakeIncidentResults:
    def __init__(self) -> None:
        self._items: dict[str, StoredIncidentResult] = {}

    def create_idempotent(self, record: StoredIncidentResult) -> StoredIncidentResult:
        existing = self._items.get(record.log_id)
        if existing is None:
            self._items[record.log_id] = record
            return record
        if existing != record:
            raise PersistenceConflictError("log_id already contains a different result")
        return existing

    def get(self, log_id: str) -> StoredIncidentResult | None:
        return self._items.get(log_id)

    def list(self, filters=None, *, page=None) -> Page[StoredIncidentResult]:
        active_filter = filters or IncidentResultFilter()
        request = page or PageRequest()
        values = sorted(self._items.values(), key=lambda item: (item.timestamp, item.log_id))
        values = [
            item
            for item in values
            if active_filter.asset_id is None or item.asset_id == active_filter.asset_id
        ]
        values = [
            item
            for item in values
            if active_filter.routing is None or item.result.routing is active_filter.routing
        ]
        items = tuple(values[request.offset : request.offset + request.limit])
        return Page(items=items, offset=request.offset, limit=request.limit, total=len(values))


class FakeReviewDecisions:
    def __init__(self) -> None:
        self._items: dict[str, ReviewDecision] = {}

    def append(self, decision: ReviewDecision) -> ReviewDecision:
        if decision.decision_id in self._items:
            raise PersistenceConflictError("decision_id already exists")
        self._items[decision.decision_id] = decision
        return decision

    def list_for_incident(self, log_id: str, *, page=None) -> Page[ReviewDecision]:
        request = page or PageRequest()
        values = sorted(
            (item for item in self._items.values() if item.log_id == log_id),
            key=lambda item: (item.decided_at, item.decision_id),
        )
        items = tuple(values[request.offset : request.offset + request.limit])
        return Page(items=items, offset=request.offset, limit=request.limit, total=len(values))


class FakeAudit:
    def __init__(self) -> None:
        self._items: dict[str, AuditEvent] = {}

    def append(self, event: AuditEvent) -> AuditEvent:
        if event.event_id in self._items:
            raise PersistenceConflictError("event_id already exists")
        self._items[event.event_id] = event
        return event

    def append_many(self, events: tuple[AuditEvent, ...]) -> tuple[AuditEvent, ...]:
        ids = [event.event_id for event in events]
        if len(ids) != len(set(ids)) or any(event_id in self._items for event_id in ids):
            raise PersistenceConflictError("audit event IDs must be unique")
        for event in events:
            self._items[event.event_id] = event
        return events

    def list_for_incident(self, log_id: str, *, page=None) -> Page[AuditEvent]:
        request = page or PageRequest()
        values = sorted(
            (item for item in self._items.values() if item.log_id == log_id),
            key=lambda item: (item.occurred_at, item.event_id),
        )
        items = tuple(values[request.offset : request.offset + request.limit])
        return Page(items=items, offset=request.offset, limit=request.limit, total=len(values))


def test_repository_protocols_accept_fake_implementations() -> None:
    assert isinstance(FakeIncidentResults(), IncidentResultRepository)
    assert isinstance(FakeReviewDecisions(), ReviewDecisionRepository)
    assert isinstance(FakeAudit(), AuditEventRepository)


def test_incident_result_creation_is_idempotent_and_conflict_safe() -> None:
    repo = FakeIncidentResults()
    record = _stored("LOG_1")
    assert repo.create_idempotent(record) == record
    assert repo.create_idempotent(record) == record
    with pytest.raises(PersistenceConflictError):
        repo.create_idempotent(_stored("LOG_1", score=0.9))


def test_incident_query_filters_and_paginates_deterministically() -> None:
    repo = FakeIncidentResults()
    repo.create_idempotent(
        _stored("LOG_2", timestamp=NOW + timedelta(seconds=1), score=0.8)
    )
    repo.create_idempotent(_stored("LOG_1", timestamp=NOW))
    page = repo.list(
        IncidentResultFilter(routing=RoutingBucket.CRITICAL_ESCALATION),
        page=PageRequest(limit=1),
    )
    assert page.total == 1
    assert [item.log_id for item in page.items] == ["LOG_2"]


def test_page_request_rejects_invalid_values() -> None:
    with pytest.raises(ValidationError):
        PageRequest(offset=-1)
    with pytest.raises(ValidationError):
        PageRequest(limit=0)
    with pytest.raises(ValidationError):
        PageRequest(limit=501)


def test_filter_rejects_inverted_ranges() -> None:
    with pytest.raises(ValueError):
        IncidentResultFilter(min_calibrated_sif_p_score=0.8, max_calibrated_sif_p_score=0.2)
    with pytest.raises(ValueError):
        IncidentResultFilter(timestamp_from=NOW, timestamp_to=NOW - timedelta(seconds=1))


def test_review_history_is_append_only_and_deterministically_ordered() -> None:
    repo = FakeReviewDecisions()
    repo.append(
        ReviewDecision(
            decision_id="D2",
            log_id="LOG_1",
            action=ReviewAction.DISMISS,
            reviewer_id="reviewer-a",
            decided_at=NOW,
            reason="Insufficient evidence.",
        )
    )
    repo.append(
        ReviewDecision(
            decision_id="D1",
            log_id="LOG_1",
            action=ReviewAction.CONFIRM,
            reviewer_id="reviewer-b",
            decided_at=NOW,
            reason="Confirmed barrier failure.",
        )
    )
    assert [item.decision_id for item in repo.list_for_incident("LOG_1").items] == ["D1", "D2"]
    with pytest.raises(PersistenceConflictError):
        repo.append(
            ReviewDecision(
                decision_id="D1",
                log_id="LOG_1",
                action=ReviewAction.ESCALATE,
                reviewer_id="reviewer-c",
                decided_at=NOW,
                reason="Do not replace history.",
            )
        )


def test_audit_append_many_is_atomic_and_history_is_deterministic() -> None:
    repo = FakeAudit()
    events = (
        AuditEvent(
            event_id="E2",
            log_id="LOG_1",
            event_type=AuditEventType.REVIEW_DECISION_RECORDED,
            actor_id="reviewer-a",
            occurred_at=NOW + timedelta(seconds=1),
            reason="Dismissed after review.",
        ),
        AuditEvent(
            event_id="E1",
            log_id="LOG_1",
            event_type=AuditEventType.INFERENCE_RECORDED,
            actor_id="system",
            occurred_at=NOW,
            reason=None,
        ),
    )
    assert repo.append_many(events) == events
    assert [item.event_id for item in repo.list_for_incident("LOG_1").items] == ["E1", "E2"]
    with pytest.raises(PersistenceConflictError):
        repo.append_many((events[0],))


def test_review_model_is_immutable() -> None:
    decision = ReviewDecision(
        decision_id="D1",
        log_id="LOG_1",
        action=ReviewAction.CONFIRM,
        reviewer_id="reviewer-a",
        decided_at=NOW,
        reason="Confirmed.",
    )
    with pytest.raises(ValidationError):
        decision.reason = "Changed"
