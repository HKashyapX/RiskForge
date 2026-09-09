"""PostgreSQL repository implementations for production persistence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime

from riskforge.persistence.exceptions import PersistenceConflictError, PersistenceError
from riskforge.persistence.models import (
    AuditEvent,
    IncidentResultFilter,
    Page,
    PageRequest,
    ReviewDecision,
    StoredIncidentResult,
)
from riskforge.persistence.postgres.connection import PostgresConnectionPool

# ---------------------------------------------------------------------------
# Helpers (shared with SQLite where possible, but kept self-contained)
# ---------------------------------------------------------------------------

def _canonical_json(model: object) -> str:
    """Deterministic JSON serialization for idempotency comparison."""
    try:
        payload = model.model_dump(mode="json")  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError("persistence payload must be a Pydantic model") from error
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _iso_timestamp(value: datetime) -> str:
    """Convert datetime to ISO-8601 string suitable for TIMESTAMPTZ."""
    if value.tzinfo is not None and value.utcoffset() is not None:
        return value.astimezone(UTC).isoformat()
    return value.isoformat()


def _parse_datetime(value: object) -> datetime:
    """Parse a TIMESTAMPTZ value from PostgreSQL into a datetime object."""
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value))


def _validate_identifier(value: str, field_name: str) -> None:
    """Reject blank identifiers at the repository boundary."""
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _PostgresRepositoryBase:
    """Shared pool reference for all Postgres repositories."""

    def __init__(self, pool: PostgresConnectionPool) -> None:
        self._pool = pool


# ---------------------------------------------------------------------------
# Incident Result Repository
# ---------------------------------------------------------------------------

class PostgresIncidentResultRepository(_PostgresRepositoryBase):
    """PostgreSQL implementation of immutable, idempotent incident-result storage."""

    def create_idempotent(self, record: StoredIncidentResult) -> StoredIncidentResult:
        _validate_identifier(record.log_id, "log_id")
        payload = _canonical_json(record)
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT record_json FROM incident_results WHERE log_id = %s",
                    (record.log_id,),
                )
                row = cur.fetchone()
                if row is not None:
                    return self._validate_existing(row[0], payload)
                try:
                    cur.execute(
                        """
                        INSERT INTO incident_results(
                            log_id, timestamp, asset_id, asset_type, routing,
                            calibrated_sif_p_score, record_json
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                        """,
                        (
                            record.log_id,
                            _iso_timestamp(record.timestamp),
                            record.asset_id,
                            record.asset_type.value,
                            record.result.routing.value,
                            record.result.calibrated_sif_p_score,
                            payload,
                        ),
                    )
                except Exception as insert_exc:
                    if "duplicate key" not in str(insert_exc):
                        raise
                    # Race: another thread inserted between our SELECT and INSERT.
                    # Re-read and compare content.
                    conn.rollback()
                    with conn.cursor() as cur2:
                        cur2.execute(
                            "SELECT record_json FROM incident_results WHERE log_id = %s",
                            (record.log_id,),
                        )
                        retry_row = cur2.fetchone()
                    if retry_row is None:
                        raise PersistenceError(
                            "cannot persist incident inference result"
                        ) from insert_exc
                    return self._validate_existing(retry_row[0], payload)
            conn.commit()
            return record
        except PersistenceConflictError:
            conn.rollback()
            raise
        except Exception as exc:
            conn.rollback()
            raise PersistenceError(
                "cannot persist incident inference result"
            ) from exc
        finally:
            self._pool.putconn(conn)

    @staticmethod
    def _validate_existing(
        existing_json: object, expected_payload: str
    ) -> StoredIncidentResult:
        """Compare stored JSONB against expected canonical payload."""
        if isinstance(existing_json, dict):
            normalized = json.dumps(existing_json, sort_keys=True, separators=(",", ":"))
            stored_str = json.dumps(existing_json)
        else:
            normalized = json.dumps(
                json.loads(str(existing_json)),
                sort_keys=True,
                separators=(",", ":"),
            )
            stored_str = str(existing_json)
        if normalized != expected_payload:
            raise PersistenceConflictError(
                "log_id already contains a different persisted result"
            )
        return StoredIncidentResult.model_validate_json(stored_str)

    def get(self, log_id: str) -> StoredIncidentResult | None:
        _validate_identifier(log_id, "log_id")
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT record_json FROM incident_results WHERE log_id = %s",
                    (log_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                value = row[0]
                if isinstance(value, dict):
                    return StoredIncidentResult.model_validate_json(
                        json.dumps(value)
                    )
                return StoredIncidentResult.model_validate_json(value)
        except Exception as exc:
            raise PersistenceError("cannot read incident inference result") from exc
        finally:
            self._pool.putconn(conn)

    def list(
        self,
        filters: IncidentResultFilter | None = None,
        *,
        page: PageRequest | None = None,
    ) -> Page[StoredIncidentResult]:
        active_filter = filters or IncidentResultFilter()
        request = page or PageRequest()
        clauses: list[str] = []
        parameters: list[object] = []
        if active_filter.asset_id is not None:
            clauses.append("asset_id = %s")
            parameters.append(active_filter.asset_id)
        if active_filter.asset_type is not None:
            clauses.append("asset_type = %s")
            parameters.append(active_filter.asset_type.value)
        if active_filter.routing is not None:
            clauses.append("routing = %s")
            parameters.append(active_filter.routing.value)
        if active_filter.min_calibrated_sif_p_score is not None:
            clauses.append("calibrated_sif_p_score >= %s")
            parameters.append(active_filter.min_calibrated_sif_p_score)
        if active_filter.max_calibrated_sif_p_score is not None:
            clauses.append("calibrated_sif_p_score <= %s")
            parameters.append(active_filter.max_calibrated_sif_p_score)
        if active_filter.timestamp_from is not None:
            clauses.append("timestamp >= %s")
            parameters.append(_iso_timestamp(active_filter.timestamp_from))
        if active_filter.timestamp_to is not None:
            clauses.append("timestamp <= %s")
            parameters.append(_iso_timestamp(active_filter.timestamp_to))
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM incident_results{where}",
                    parameters,
                )
                total = int(cur.fetchone()[0])
                cur.execute(
                    f"""
                    SELECT record_json
                    FROM incident_results
                    {where}
                    ORDER BY timestamp ASC, log_id ASC
                    LIMIT %s OFFSET %s
                    """,
                    [*parameters, request.limit, request.offset],
                )
                rows = cur.fetchall()
            items: tuple[StoredIncidentResult, ...] = tuple(
                StoredIncidentResult.model_validate_json(
                    json.dumps(row[0]) if isinstance(row[0], dict) else row[0]
                )
                for row in rows
            )
            return Page(
                items=items,
                offset=request.offset,
                limit=request.limit,
                total=total,
            )
        except Exception as exc:
            raise PersistenceError("cannot query incident inference results") from exc
        finally:
            self._pool.putconn(conn)


# ---------------------------------------------------------------------------
# Review Decision Repository
# ---------------------------------------------------------------------------

class PostgresReviewDecisionRepository(_PostgresRepositoryBase):
    """PostgreSQL implementation of append-only human review history."""

    def append(self, decision: ReviewDecision) -> ReviewDecision:
        _validate_identifier(decision.decision_id, "decision_id")
        _validate_identifier(decision.log_id, "log_id")
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO review_decisions(
                        decision_id, log_id, action, reviewer_id, decided_at, reason
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        decision.decision_id,
                        decision.log_id,
                        decision.action.value,
                        decision.reviewer_id,
                        _iso_timestamp(decision.decided_at),
                        decision.reason,
                    ),
                )
            conn.commit()
            return decision
        except Exception as exc:
            conn.rollback()
            if "duplicate key" in str(exc):
                raise PersistenceConflictError(
                    "decision_id already exists"
                ) from exc
            raise PersistenceError("cannot append review decision") from exc
        finally:
            self._pool.putconn(conn)

    def list_for_incident(
        self,
        log_id: str,
        *,
        page: PageRequest | None = None,
    ) -> Page[ReviewDecision]:
        _validate_identifier(log_id, "log_id")
        request = page or PageRequest()
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM review_decisions WHERE log_id = %s",
                    (log_id,),
                )
                total = int(cur.fetchone()[0])
                cur.execute(
                    """
                    SELECT decision_id, log_id, action, reviewer_id, decided_at, reason
                    FROM review_decisions
                    WHERE log_id = %s
                    ORDER BY decided_at ASC, decision_id ASC
                    LIMIT %s OFFSET %s
                    """,
                    (log_id, request.limit, request.offset),
                )
                rows = cur.fetchall()
            items: tuple[ReviewDecision, ...] = tuple(
                ReviewDecision(
                    decision_id=row[0],
                    log_id=row[1],
                    action=row[2],
                    reviewer_id=row[3],
                    decided_at=_parse_datetime(row[4]),
                    reason=row[5],
                )
                for row in rows
            )
            return Page(
                items=items,
                offset=request.offset,
                limit=request.limit,
                total=total,
            )
        except Exception as exc:
            raise PersistenceError("cannot query review history") from exc
        finally:
            self._pool.putconn(conn)


# ---------------------------------------------------------------------------
# Audit Event Repository
# ---------------------------------------------------------------------------

class PostgresAuditEventRepository(_PostgresRepositoryBase):
    """PostgreSQL implementation of append-only audit events."""

    def append(self, event: AuditEvent) -> AuditEvent:
        _validate_identifier(event.event_id, "event_id")
        _validate_identifier(event.log_id, "log_id")
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, log_id, event_type, actor_id, occurred_at, reason
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.event_id,
                        event.log_id,
                        event.event_type.value,
                        event.actor_id,
                        _iso_timestamp(event.occurred_at),
                        event.reason,
                    ),
                )
            conn.commit()
            return event
        except Exception as exc:
            conn.rollback()
            if "duplicate key" in str(exc):
                raise PersistenceConflictError("event_id already exists") from exc
            raise PersistenceError("cannot append audit event") from exc
        finally:
            self._pool.putconn(conn)

    def append_many(self, events: Sequence[AuditEvent]) -> tuple[AuditEvent, ...]:
        values = tuple(events)
        if not values:
            return ()
        if len({event.event_id for event in values}) != len(values):
            raise PersistenceConflictError("audit event IDs must be unique")
        for event in values:
            _validate_identifier(event.event_id, "event_id")
            _validate_identifier(event.log_id, "log_id")
        conn = self._pool.getconn()
        try:
            with conn.transaction(), conn.cursor() as cur:
                for event in values:
                    cur.execute(
                        """
                        INSERT INTO audit_events(
                            event_id, log_id, event_type, actor_id, occurred_at, reason
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            event.event_id,
                            event.log_id,
                            event.event_type.value,
                            event.actor_id,
                            _iso_timestamp(event.occurred_at),
                            event.reason,
                        ),
                    )
            return values
        except PersistenceConflictError:
            raise
        except Exception as exc:
            if "duplicate key" in str(exc):
                raise PersistenceConflictError(
                    "audit event ID already exists"
                ) from exc
            raise PersistenceError("cannot append audit events") from exc
        finally:
            self._pool.putconn(conn)

    def list_for_incident(
        self,
        log_id: str,
        *,
        page: PageRequest | None = None,
    ) -> Page[AuditEvent]:
        _validate_identifier(log_id, "log_id")
        request = page or PageRequest()
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM audit_events WHERE log_id = %s",
                    (log_id,),
                )
                total = int(cur.fetchone()[0])
                cur.execute(
                    """
                    SELECT event_id, log_id, event_type, actor_id, occurred_at, reason
                    FROM audit_events
                    WHERE log_id = %s
                    ORDER BY occurred_at ASC, event_id ASC
                    LIMIT %s OFFSET %s
                    """,
                    (log_id, request.limit, request.offset),
                )
                rows = cur.fetchall()
            items: tuple[AuditEvent, ...] = tuple(
                AuditEvent(
                    event_id=row[0],
                    log_id=row[1],
                    event_type=row[2],
                    actor_id=row[3],
                    occurred_at=_parse_datetime(row[4]),
                    reason=row[5],
                )
                for row in rows
            )
            return Page(
                items=items,
                offset=request.offset,
                limit=request.limit,
                total=total,
            )
        except Exception as exc:
            raise PersistenceError("cannot query audit history") from exc
        finally:
            self._pool.putconn(conn)


# ---------------------------------------------------------------------------
# Atomic Review + Audit Writer
# ---------------------------------------------------------------------------

class PostgresReviewAuditWriter(_PostgresRepositoryBase):
    """Persist a review decision and its audit event in one PostgreSQL transaction."""

    def append_review_atomically(
        self, decision: ReviewDecision, event: AuditEvent
    ) -> tuple[ReviewDecision, AuditEvent]:
        _validate_identifier(decision.decision_id, "decision_id")
        _validate_identifier(decision.log_id, "log_id")
        _validate_identifier(decision.reviewer_id, "reviewer_id")
        _validate_identifier(event.event_id, "event_id")
        _validate_identifier(event.log_id, "log_id")
        _validate_identifier(event.actor_id, "actor_id")
        if event.log_id != decision.log_id:
            raise ValueError("review and audit log_id values must match")
        if event.reason != decision.reason:
            raise ValueError("review and audit reasons must match")
        if event.occurred_at != decision.decided_at:
            raise ValueError("review and audit timestamps must match")
        if event.actor_id != decision.reviewer_id:
            raise ValueError("audit actor must match reviewer identity")
        if event.event_type.value != "review_decision_recorded":
            raise ValueError("audit event must record a review decision")

        conn = self._pool.getconn()
        try:
            with conn.transaction(), conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO review_decisions(
                        decision_id, log_id, action, reviewer_id, decided_at, reason
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        decision.decision_id,
                        decision.log_id,
                        decision.action.value,
                        decision.reviewer_id,
                        _iso_timestamp(decision.decided_at),
                        decision.reason,
                    ),
                )
                cur.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, log_id, event_type, actor_id, occurred_at, reason
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        event.event_id,
                        event.log_id,
                        event.event_type.value,
                        event.actor_id,
                        _iso_timestamp(event.occurred_at),
                        event.reason,
                    ),
                )
            return decision, event
        except Exception as exc:
            if "duplicate key" in str(exc):
                raise PersistenceConflictError(
                    "review or audit record already exists"
                ) from exc
            raise PersistenceError(
                "cannot atomically persist review decision and audit event"
            ) from exc
        finally:
            self._pool.putconn(conn)
