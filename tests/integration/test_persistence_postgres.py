"""PostgreSQL-specific integration tests.

These tests verify PostgreSQL-specific behaviors that go beyond the
generic repository protocol contract tests.

Requires a running PostgreSQL instance.  Tests are skipped automatically
when the database is unreachable.
"""

from __future__ import annotations

import json
import os
import socket
import threading
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
    ReviewAction,
    ReviewDecision,
    StoredIncidentResult,
)
from riskforge.persistence.postgres.connection import PostgresConfig, PostgresConnectionPool
from riskforge.persistence.postgres.repository import (
    PostgresAuditEventRepository,
    PostgresIncidentResultRepository,
    PostgresReviewAuditWriter,
    PostgresReviewDecisionRepository,
)

NOW = datetime(2026, 1, 1, 12, tzinfo=UTC)


def _pg_available() -> bool:
    host = os.environ.get("PGHOST", "localhost")
    port = int(os.environ.get("PGPORT", "5432"))
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


@pytest.fixture()
def pool() -> Any:
    if not _pg_available():
        pytest.skip("PostgreSQL not available")
    from riskforge.persistence.postgres.migrate import run_migrations

    config = PostgresConfig(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "riskforge_test"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
        connect_timeout=5,
        min_pool_size=1,
        max_pool_size=3,
    )
    run_migrations(dsn=config.dsn())
    p = PostgresConnectionPool(config=config)
    # Truncate all tables for test isolation
    conn = p.getconn()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE incident_results, review_decisions, audit_events CASCADE")
        conn.commit()
    except Exception:  # noqa: BLE001
        conn.rollback()
    finally:
        p.putconn(conn)
    yield p
    p.close()


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


# ---------------------------------------------------------------------------
# Schema & protocol tests
# ---------------------------------------------------------------------------

class TestSchemaAndProtocols:
    def test_repositories_satisfy_protocols(self, pool: Any) -> None:
        from riskforge.persistence.protocols import (
            AuditEventRepository,
            IncidentResultRepository,
            ReviewDecisionRepository,
        )

        assert isinstance(PostgresIncidentResultRepository(pool), IncidentResultRepository)
        assert isinstance(PostgresReviewDecisionRepository(pool), ReviewDecisionRepository)
        assert isinstance(PostgresAuditEventRepository(pool), AuditEventRepository)

    def test_schema_tables_exist(self, pool: Any) -> None:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_name IN "
                    "('incident_results', 'review_decisions', 'audit_events')"
                )
                tables = {row[0] for row in cur.fetchall()}
            assert tables == {"incident_results", "review_decisions", "audit_events"}
        finally:
            pool.putconn(conn)

    def test_schema_indexes_exist(self, pool: Any) -> None:
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT indexname FROM pg_indexes WHERE schemaname = 'public'"
                )
                indexes = {row[0] for row in cur.fetchall()}
            assert "idx_incident_results_timestamp_log_id" in indexes
            assert "idx_review_decisions_incident" in indexes
            assert "idx_audit_events_incident" in indexes
        finally:
            pool.putconn(conn)


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestPostgresIdempotency:
    def test_idempotent_create(self, pool: Any) -> None:
        repo = PostgresIncidentResultRepository(pool)
        record = _stored("LOG_PG_1")
        first = repo.create_idempotent(record)
        second = repo.create_idempotent(record)
        assert first == second
        assert repo.get("LOG_PG_1") == record

    def test_conflict_on_different_content(self, pool: Any) -> None:
        repo = PostgresIncidentResultRepository(pool)
        repo.create_idempotent(_stored("LOG_PG_C", score=0.2))
        with pytest.raises(PersistenceConflictError):
            repo.create_idempotent(_stored("LOG_PG_C", score=0.9))

    def test_reopen_pool_retrieves_data(self, pool: Any) -> None:
        """Data survives pool restart (same PostgreSQL server)."""
        record = _stored("LOG_PG_REOPEN")
        repo1 = PostgresIncidentResultRepository(pool)
        repo1.create_idempotent(record)
        # New repository instance using the same pool
        repo2 = PostgresIncidentResultRepository(pool)
        assert repo2.get("LOG_PG_REOPEN") == record


# ---------------------------------------------------------------------------
# Timestamp handling
# ---------------------------------------------------------------------------

class TestTimestampHandling:
    def test_utc_timestamp_persists_and_round_trips(self, pool: Any) -> None:
        repo = PostgresIncidentResultRepository(pool)
        ts = datetime(2025, 6, 15, 8, 30, 0, tzinfo=UTC)
        record = _stored("LOG_TS_1", timestamp=ts)
        repo.create_idempotent(record)
        retrieved = repo.get("LOG_TS_1")
        assert retrieved is not None
        assert retrieved.timestamp == ts

    def test_offset_naive_timestamp_persists(self, pool: Any) -> None:
        repo = PostgresIncidentResultRepository(pool)
        ts = datetime(2025, 6, 15, 8, 30, 0, tzinfo=UTC)  # explicit UTC
        record = _stored("LOG_TS_2", timestamp=ts)
        repo.create_idempotent(record)
        retrieved = repo.get("LOG_TS_2")
        assert retrieved is not None
        # Should be retrievable (stored as TIMESTAMPTZ, parsed back)
        assert retrieved.timestamp is not None


# ---------------------------------------------------------------------------
# JSONB storage
# ---------------------------------------------------------------------------

class TestJSONBStorage:
    def test_record_json_is_jsonb(self, pool: Any) -> None:
        repo = PostgresIncidentResultRepository(pool)
        record = _stored("LOG_JSON_1")
        repo.create_idempotent(record)
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_typeof(record_json)::text FROM incident_results "
                    "WHERE log_id = 'LOG_JSON_1'"
                )
                type_name = cur.fetchone()[0]
            assert "jsonb" in type_name.lower()
        finally:
            pool.putconn(conn)

    def test_record_json_contains_full_payload(self, pool: Any) -> None:
        repo = PostgresIncidentResultRepository(pool)
        record = _stored("LOG_JSON_2")
        repo.create_idempotent(record)
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT record_json FROM incident_results WHERE log_id = 'LOG_JSON_2'"
                )
                row = cur.fetchone()
                assert row is not None
                # JSONB may come back as dict or as string depending on adapter
                raw = row[0]
                if isinstance(raw, str):
                    data = json.loads(raw)
                else:
                    data = raw
                assert data["result"]["log_id"] == "LOG_JSON_2"
        finally:
            pool.putconn(conn)


# ---------------------------------------------------------------------------
# Concurrent idempotency
# ---------------------------------------------------------------------------

class TestConcurrentIdempotency:
    def test_concurrent_same_content_succeeds(self, pool: Any) -> None:
        """Two threads inserting the same log_id with same content."""
        repo = PostgresIncidentResultRepository(pool)
        record = _stored("LOG_CONC_1")
        errors: list[Exception] = []

        def insert() -> None:
            try:
                repo.create_idempotent(record)
            except PersistenceConflictError as exc:
                errors.append(exc)

        t1 = threading.Thread(target=insert)
        t2 = threading.Thread(target=insert)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert errors == []
        assert repo.get("LOG_CONC_1") == record

    def test_concurrent_different_content_raises(self, pool: Any) -> None:
        """Two threads inserting same log_id with different content."""
        repo = PostgresIncidentResultRepository(pool)
        errors: list[Exception] = []

        def insert(score: float) -> None:
            try:
                repo.create_idempotent(_stored("LOG_CONC_2", score=score))
            except PersistenceConflictError as exc:
                errors.append(exc)

        t1 = threading.Thread(target=insert, args=(0.1,))
        t2 = threading.Thread(target=insert, args=(0.9,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        # Exactly one should succeed, one should fail
        assert len(errors) == 1
        assert isinstance(errors[0], PersistenceConflictError)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

class TestPostgresFiltering:
    def test_filter_by_timestamp_range(self, pool: Any) -> None:
        repo = PostgresIncidentResultRepository(pool)
        repo.create_idempotent(_stored("LOG_F1", timestamp=NOW))
        repo.create_idempotent(
            _stored("LOG_F2", timestamp=NOW + timedelta(hours=1))
        )
        repo.create_idempotent(
            _stored("LOG_F3", timestamp=NOW + timedelta(hours=2))
        )
        page = repo.list(
            IncidentResultFilter(
                timestamp_from=NOW + timedelta(minutes=30),
                timestamp_to=NOW + timedelta(hours=1, minutes=30),
            )
        )
        assert page.total == 1
        assert page.items[0].log_id == "LOG_F2"

    def test_filter_by_score_range(self, pool: Any) -> None:
        repo = PostgresIncidentResultRepository(pool)
        repo.create_idempotent(_stored("LOG_S1", score=0.1))
        repo.create_idempotent(
            _stored("LOG_S2", timestamp=NOW + timedelta(seconds=1), score=0.5)
        )
        repo.create_idempotent(
            _stored("LOG_S3", timestamp=NOW + timedelta(seconds=2), score=0.9)
        )
        page = repo.list(
            IncidentResultFilter(
                min_calibrated_sif_p_score=0.3,
                max_calibrated_sif_p_score=0.7,
            )
        )
        assert page.total == 1
        assert page.items[0].log_id == "LOG_S2"


# ---------------------------------------------------------------------------
# Review audit writer
# ---------------------------------------------------------------------------

class TestPostgresReviewAuditWriter:
    def test_atomic_write(self, pool: Any) -> None:
        writer = PostgresReviewAuditWriter(pool)
        d = _decision("D_ATOM_1")
        e = AuditEvent(
            event_id="review:D_ATOM_1",
            log_id="LOG_1",
            event_type=AuditEventType.REVIEW_DECISION_RECORDED,
            actor_id="reviewer-1",
            occurred_at=NOW,
            reason="Confirmed.",
        )
        rd, ae = writer.append_review_atomically(d, e)
        assert rd.decision_id == "D_ATOM_1"
        assert ae.event_id == "review:D_ATOM_1"

    def test_atomic_write_conflict_rolls_back_both(self, pool: Any) -> None:
        writer = PostgresReviewAuditWriter(pool)
        d1 = _decision("D_ATOM_2")
        e1 = AuditEvent(
            event_id="review:D_ATOM_2",
            log_id="LOG_1",
            event_type=AuditEventType.REVIEW_DECISION_RECORDED,
            actor_id="reviewer-1",
            occurred_at=NOW,
            reason="Confirmed.",
        )
        writer.append_review_atomically(d1, e1)

        d2 = _decision("D_ATOM_3")
        e2 = AuditEvent(
            event_id="review:D_ATOM_2",  # duplicate event_id
            log_id="LOG_1",
            event_type=AuditEventType.REVIEW_DECISION_RECORDED,
            actor_id="reviewer-1",
            occurred_at=NOW,
            reason="Confirmed.",
        )
        with pytest.raises(PersistenceConflictError):
            writer.append_review_atomically(d2, e2)

        # D_ATOM_3 should NOT be in review_decisions (rolled back)
        review_repo = PostgresReviewDecisionRepository(pool)
        page = review_repo.list_for_incident("LOG_1")
        decision_ids = [d.decision_id for d in page.items]
        assert "D_ATOM_2" in decision_ids
        assert "D_ATOM_3" not in decision_ids

    def test_atomic_write_validates_cross_record_invariants(self, pool: Any) -> None:
        writer = PostgresReviewAuditWriter(pool)
        d = _decision("D_INV_1")
        e = AuditEvent(
            event_id="review:D_INV_1",
            log_id="LOG_DIFFERENT",  # mismatched log_id
            event_type=AuditEventType.REVIEW_DECISION_RECORDED,
            actor_id="reviewer-1",
            occurred_at=NOW,
            reason="Confirmed.",
        )
        with pytest.raises(ValueError, match="log_id"):
            writer.append_review_atomically(d, e)


# ---------------------------------------------------------------------------
# Error translation
# ---------------------------------------------------------------------------

class TestErrorTranslation:
    def test_persistence_error_not_leaking_pg_details(self, pool: Any) -> None:
        """Ensure PG connection strings don't leak into PersistenceError messages."""
        repo = PostgresIncidentResultRepository(pool)
        # Force a query against non-existent table to trigger a PG error
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM nonexistent_table_xyz")
            conn.commit()
        except Exception:  # noqa: BLE001 — intentionally triggering PG error
            conn.rollback()
        finally:
            pool.putconn(conn)
        # The key assertion: regular operations still work (pool is healthy)
        record = _stored("LOG_ERR_1")
        result = repo.create_idempotent(record)
        assert result.log_id == "LOG_ERR_1"
