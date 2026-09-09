"""Parameterized repository protocol tests.

Run the same behavioral expectations against every concrete backend:
  - Fake (in-memory)
  - SQLite
  - PostgreSQL (requires a running database)

PostgreSQL tests are skipped automatically when the PG database is
unreachable, so the suite remains green in environments without PG.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

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

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

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
    action: ReviewAction = ReviewAction.CONFIRM,
    reason: str = "Confirmed.",
) -> ReviewDecision:
    return ReviewDecision(
        decision_id=decision_id,
        log_id=log_id,
        action=action,
        reviewer_id="reviewer-1",
        decided_at=when,
        reason=reason,
    )


def _event(
    event_id: str,
    log_id: str = "LOG_1",
    *,
    when: datetime = NOW,
    event_type: AuditEventType = AuditEventType.INFERENCE_RECORDED,
) -> AuditEvent:
    return AuditEvent(
        event_id=event_id,
        log_id=log_id,
        event_type=event_type,
        actor_id="system",
        occurred_at=when,
        reason=None,
    )


# ---------------------------------------------------------------------------
# Fake (in-memory) implementations
# ---------------------------------------------------------------------------

class FakeIncidentResults:
    def __init__(self) -> None:
        self._items: dict[str, StoredIncidentResult] = {}

    def create_idempotent(self, record: StoredIncidentResult) -> StoredIncidentResult:
        existing = self._items.get(record.log_id)
        if existing is None:
            self._items[record.log_id] = record
            return record
        if existing != record:
            raise PersistenceConflictError(
                "log_id already contains a different persisted result"
            )
        return existing

    def get(self, log_id: str) -> StoredIncidentResult | None:
        return self._items.get(log_id)

    def list(
        self,
        filters: IncidentResultFilter | None = None,
        *,
        page: PageRequest | None = None,
    ) -> Any:
        from riskforge.persistence.models import Page

        active_filter = filters or IncidentResultFilter()
        request = page or PageRequest()
        values = sorted(
            self._items.values(), key=lambda item: (item.timestamp, item.log_id)
        )
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

    def list_for_incident(self, log_id: str, *, page: PageRequest | None = None) -> Any:
        from riskforge.persistence.models import Page

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

    def append_many(self, events: Sequence[AuditEvent]) -> tuple[AuditEvent, ...]:
        ids = [event.event_id for event in events]
        if len(ids) != len(set(ids)) or any(event_id in self._items for event_id in ids):
            raise PersistenceConflictError("audit event IDs must be unique")
        for event in events:
            self._items[event.event_id] = event
        return tuple(events)

    def list_for_incident(self, log_id: str, *, page: PageRequest | None = None) -> Any:
        from riskforge.persistence.models import Page

        request = page or PageRequest()
        values = sorted(
            (item for item in self._items.values() if item.log_id == log_id),
            key=lambda item: (item.occurred_at, item.event_id),
        )
        items = tuple(values[request.offset : request.offset + request.limit])
        return Page(items=items, offset=request.offset, limit=request.limit, total=len(values))


# ---------------------------------------------------------------------------
# Backend fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_incident_repo() -> FakeIncidentResults:
    return FakeIncidentResults()


@pytest.fixture()
def fake_review_repo() -> FakeReviewDecisions:
    return FakeReviewDecisions()


@pytest.fixture()
def fake_audit_repo() -> FakeAudit:
    return FakeAudit()


@pytest.fixture()
def sqlite_incident_repo(tmp_path: Any) -> Any:
    from riskforge.persistence.sqlite.repository import SQLiteIncidentResultRepository

    return SQLiteIncidentResultRepository(tmp_path / "test.db")


@pytest.fixture()
def sqlite_review_repo(tmp_path: Any) -> Any:
    from riskforge.persistence.sqlite.repository import SQLiteReviewDecisionRepository

    return SQLiteReviewDecisionRepository(tmp_path / "test.db")


@pytest.fixture()
def sqlite_audit_repo(tmp_path: Any) -> Any:
    from riskforge.persistence.sqlite.repository import SQLiteAuditEventRepository

    return SQLiteAuditEventRepository(tmp_path / "test.db")


def _pg_available() -> bool:
    """Check whether the PG test database is reachable."""
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    import socket

    try:
        with socket.create_connection((host, int(port)), timeout=2):
            return True
    except OSError:
        return False


def _pg_pool() -> Any:
    from riskforge.persistence.postgres.connection import PostgresConfig, PostgresConnectionPool
    from riskforge.persistence.postgres.migrate import run_migrations

    config = PostgresConfig(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "riskforge_test"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
        connect_timeout=5,
        min_pool_size=1,
        max_pool_size=2,
    )
    run_migrations(dsn=config.dsn())
    pool = PostgresConnectionPool(config=config)
    # Truncate all tables to ensure test isolation
    conn = pool.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE incident_results, review_decisions, audit_events CASCADE")
        conn.commit()
    except Exception:  # noqa: BLE001
        conn.rollback()
    finally:
        pool.putconn(conn)
    return pool


@pytest.fixture()
def pg_incident_repo() -> Any:
    if not _pg_available():
        pytest.skip("PostgreSQL not available")
    pool = _pg_pool()
    from riskforge.persistence.postgres.repository import PostgresIncidentResultRepository

    repo = PostgresIncidentResultRepository(pool)
    yield repo
    pool.close()


@pytest.fixture()
def pg_review_repo() -> Any:
    if not _pg_available():
        pytest.skip("PostgreSQL not available")
    pool = _pg_pool()
    from riskforge.persistence.postgres.repository import PostgresReviewDecisionRepository

    repo = PostgresReviewDecisionRepository(pool)
    yield repo
    pool.close()


@pytest.fixture()
def pg_audit_repo() -> Any:
    if not _pg_available():
        pytest.skip("PostgreSQL not available")
    pool = _pg_pool()
    from riskforge.persistence.postgres.repository import PostgresAuditEventRepository

    repo = PostgresAuditEventRepository(pool)
    yield repo
    pool.close()


# ===========================================================================
# INCIDENT RESULT REPOSITORY TESTS
# ===========================================================================

class TestIncidentResultProtocol:
    """Verify all backends satisfy the IncidentResultRepository protocol."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_incident_repo", "sqlite_incident_repo", "pg_incident_repo"],
    )
    def test_satisfies_protocol(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        assert isinstance(repo, IncidentResultRepository)


class TestIncidentResultIdempotency:
    """Idempotent create and conflict detection."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_incident_repo", "sqlite_incident_repo", "pg_incident_repo"],
    )
    def test_idempotent_same_content(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        record = _stored("LOG_1")
        first = repo.create_idempotent(record)
        second = repo.create_idempotent(record)
        assert first == second
        assert repo.get("LOG_1") == record

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_incident_repo", "sqlite_incident_repo", "pg_incident_repo"],
    )
    def test_conflict_on_different_content(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        repo.create_idempotent(_stored("LOG_1", score=0.2))
        with pytest.raises(PersistenceConflictError):
            repo.create_idempotent(_stored("LOG_1", score=0.9))

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_incident_repo", "sqlite_incident_repo", "pg_incident_repo"],
    )
    def test_get_returns_none_for_missing(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        assert repo.get("NONEXISTENT") is None


class TestIncidentResultOrdering:
    """Deterministic ordering: (timestamp ASC, log_id ASC)."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_incident_repo", "sqlite_incident_repo", "pg_incident_repo"],
    )
    def test_order_by_timestamp_then_log_id(
        self, fixture_name: str, request: Any
    ) -> None:
        repo = request.getfixturevalue(fixture_name)
        repo.create_idempotent(
            _stored("LOG_B", timestamp=NOW + timedelta(seconds=2))
        )
        repo.create_idempotent(_stored("LOG_A", timestamp=NOW))
        repo.create_idempotent(
            _stored("LOG_C", timestamp=NOW + timedelta(seconds=1))
        )
        page = repo.list()
        log_ids = [item.log_id for item in page.items]
        assert log_ids == ["LOG_A", "LOG_C", "LOG_B"]


class TestIncidentResultFiltering:
    """Filter by asset_id and routing."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_incident_repo", "sqlite_incident_repo", "pg_incident_repo"],
    )
    def test_filter_by_routing(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        repo.create_idempotent(_stored("LOG_1", score=0.2))
        repo.create_idempotent(
            _stored("LOG_2", timestamp=NOW + timedelta(seconds=1), score=0.8)
        )
        page = repo.list(
            IncidentResultFilter(routing=RoutingBucket.CRITICAL_ESCALATION)
        )
        assert page.total == 1
        assert page.items[0].log_id == "LOG_2"

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_incident_repo", "sqlite_incident_repo", "pg_incident_repo"],
    )
    def test_filter_by_asset_id(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        repo.create_idempotent(_stored("LOG_1", asset_id="RIG_A"))
        repo.create_idempotent(
            _stored("LOG_2", timestamp=NOW + timedelta(seconds=1), asset_id="RIG_B")
        )
        page = repo.list(IncidentResultFilter(asset_id="RIG_B"))
        assert page.total == 1
        assert page.items[0].log_id == "LOG_2"


class TestIncidentResultPagination:
    """Pagination with offset and limit."""

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_incident_repo", "sqlite_incident_repo", "pg_incident_repo"],
    )
    def test_pagination(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        for i in range(5):
            repo.create_idempotent(
                _stored(f"LOG_{i}", timestamp=NOW + timedelta(seconds=i))
            )
        page1 = repo.list(page=PageRequest(offset=0, limit=2))
        assert len(page1.items) == 2
        assert page1.total == 5
        page2 = repo.list(page=PageRequest(offset=2, limit=2))
        assert len(page2.items) == 2
        assert page2.items[0].log_id != page1.items[0].log_id


# ===========================================================================
# REVIEW DECISION REPOSITORY TESTS
# ===========================================================================

class TestReviewDecisionProtocol:
    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_review_repo", "sqlite_review_repo", "pg_review_repo"],
    )
    def test_satisfies_protocol(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        assert isinstance(repo, ReviewDecisionRepository)


class TestReviewDecisionAppend:
    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_review_repo", "sqlite_review_repo", "pg_review_repo"],
    )
    def test_append_and_retrieve(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        d = _decision("D1")
        repo.append(d)
        page = repo.list_for_incident("LOG_1")
        assert len(page.items) == 1
        assert page.items[0].decision_id == "D1"

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_review_repo", "sqlite_review_repo", "pg_review_repo"],
    )
    def test_duplicate_decision_id_raises(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        repo.append(_decision("D1"))
        with pytest.raises(PersistenceConflictError):
            repo.append(_decision("D1"))

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_review_repo", "sqlite_review_repo", "pg_review_repo"],
    )
    def test_deterministic_ordering(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        repo.append(_decision("D2", when=NOW + timedelta(seconds=1)))
        repo.append(_decision("D1", when=NOW))
        page = repo.list_for_incident("LOG_1")
        assert [d.decision_id for d in page.items] == ["D1", "D2"]


# ===========================================================================
# AUDIT EVENT REPOSITORY TESTS
# ===========================================================================

class TestAuditEventProtocol:
    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_audit_repo", "sqlite_audit_repo", "pg_audit_repo"],
    )
    def test_satisfies_protocol(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        assert isinstance(repo, AuditEventRepository)


class TestAuditEventAppend:
    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_audit_repo", "sqlite_audit_repo", "pg_audit_repo"],
    )
    def test_append_and_retrieve(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        e = _event("E1")
        repo.append(e)
        page = repo.list_for_incident("LOG_1")
        assert len(page.items) == 1
        assert page.items[0].event_id == "E1"

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_audit_repo", "sqlite_audit_repo", "pg_audit_repo"],
    )
    def test_duplicate_event_id_raises(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        repo.append(_event("E1"))
        with pytest.raises(PersistenceConflictError):
            repo.append(_event("E1"))

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_audit_repo", "sqlite_audit_repo", "pg_audit_repo"],
    )
    def test_append_many_atomic(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        events = (_event("E2", when=NOW + timedelta(seconds=1)), _event("E1"))
        result = repo.append_many(events)
        assert len(result) == 2
        page = repo.list_for_incident("LOG_1")
        assert [e.event_id for e in page.items] == ["E1", "E2"]

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_audit_repo", "sqlite_audit_repo", "pg_audit_repo"],
    )
    def test_append_many_rolls_back_on_conflict(
        self, fixture_name: str, request: Any
    ) -> None:
        repo = request.getfixturevalue(fixture_name)
        repo.append(_event("E1"))
        with pytest.raises(PersistenceConflictError):
            repo.append_many((_event("E2"), _event("E1")))
        # E1 should still be there; E2 should not
        page = repo.list_for_incident("LOG_1")
        assert len(page.items) == 1

    @pytest.mark.parametrize(
        "fixture_name",
        ["fake_audit_repo", "sqlite_audit_repo", "pg_audit_repo"],
    )
    def test_append_many_empty(self, fixture_name: str, request: Any) -> None:
        repo = request.getfixturevalue(fixture_name)
        assert repo.append_many(()) == ()
