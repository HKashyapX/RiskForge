"""SQLite repository implementations for isolated development and integration tests."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from riskforge.persistence.exceptions import PersistenceConflictError, PersistenceError
from riskforge.persistence.models import (
    AuditEvent,
    IncidentResultFilter,
    Page,
    PageRequest,
    ReviewDecision,
    StoredIncidentResult,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incident_results (
    log_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    routing TEXT NOT NULL,
    calibrated_sif_p_score REAL NOT NULL,
    record_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_incident_results_timestamp_log_id
    ON incident_results(timestamp, log_id);
CREATE INDEX IF NOT EXISTS idx_incident_results_asset_id
    ON incident_results(asset_id);
CREATE INDEX IF NOT EXISTS idx_incident_results_asset_type
    ON incident_results(asset_type);
CREATE INDEX IF NOT EXISTS idx_incident_results_routing
    ON incident_results(routing);
CREATE INDEX IF NOT EXISTS idx_incident_results_score
    ON incident_results(calibrated_sif_p_score);

CREATE TABLE IF NOT EXISTS review_decisions (
    decision_id TEXT PRIMARY KEY,
    log_id TEXT NOT NULL,
    action TEXT NOT NULL,
    reviewer_id TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    reason TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_review_decisions_incident
    ON review_decisions(log_id, decided_at, decision_id);

CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY,
    log_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    actor_id TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_events_incident
    ON audit_events(log_id, occurred_at, event_id);
"""


def _canonical_json(model: object) -> str:
    try:
        payload = model.model_dump(mode="json")  # type: ignore[attr-defined]
    except AttributeError as error:
        raise TypeError("persistence payload must be a Pydantic model") from error
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _validate_identifier(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be empty")


class _SQLiteRepositoryBase:
    def __init__(self, database_path: str | Path) -> None:
        path_text = str(database_path).strip()
        if not path_text:
            raise ValueError("database_path must not be empty")
        self.database_path = Path(database_path)
        if self.database_path.parent != Path("."):
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                self.database_path,
                timeout=5.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.Error as error:
            raise PersistenceError("cannot open SQLite persistence database") from error

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(_SCHEMA)
        except sqlite3.Error as error:
            raise PersistenceError("cannot initialize SQLite persistence schema") from error
        finally:
            connection.close()

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass


class SQLiteIncidentResultRepository(_SQLiteRepositoryBase):
    """SQLite implementation of immutable, idempotent incident-result storage."""

    def create_idempotent(self, record: StoredIncidentResult) -> StoredIncidentResult:
        _validate_identifier(record.log_id, "log_id")
        payload = _canonical_json(record)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT record_json FROM incident_results WHERE log_id = ?",
                (record.log_id,),
            ).fetchone()
            if existing is not None:
                if existing["record_json"] != payload:
                    raise PersistenceConflictError(
                        "log_id already contains a different persisted result"
                    )
                connection.execute("COMMIT")
                return StoredIncidentResult.model_validate_json(existing["record_json"])
            connection.execute(
                """
                INSERT INTO incident_results(
                    log_id, timestamp, asset_id, asset_type, routing,
                    calibrated_sif_p_score, record_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.log_id,
                    record.timestamp.isoformat(),
                    record.asset_id,
                    record.asset_type.value,
                    record.result.routing.value,
                    record.result.calibrated_sif_p_score,
                    payload,
                ),
            )
            connection.execute("COMMIT")
            return record
        except PersistenceConflictError:
            self._rollback(connection)
            raise
        except sqlite3.IntegrityError as error:
            self._rollback(connection)
            raise PersistenceConflictError("log_id already exists") from error
        except sqlite3.Error as error:
            self._rollback(connection)
            raise PersistenceError("cannot persist incident inference result") from error
        finally:
            connection.close()

    def get(self, log_id: str) -> StoredIncidentResult | None:
        _validate_identifier(log_id, "log_id")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT record_json FROM incident_results WHERE log_id = ?",
                (log_id,),
            ).fetchone()
            if row is None:
                return None
            return StoredIncidentResult.model_validate_json(row["record_json"])
        except sqlite3.Error as error:
            raise PersistenceError("cannot read incident inference result") from error
        finally:
            connection.close()

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
            clauses.append("asset_id = ?")
            parameters.append(active_filter.asset_id)
        if active_filter.asset_type is not None:
            clauses.append("asset_type = ?")
            parameters.append(active_filter.asset_type.value)
        if active_filter.routing is not None:
            clauses.append("routing = ?")
            parameters.append(active_filter.routing.value)
        if active_filter.min_calibrated_sif_p_score is not None:
            clauses.append("calibrated_sif_p_score >= ?")
            parameters.append(active_filter.min_calibrated_sif_p_score)
        if active_filter.max_calibrated_sif_p_score is not None:
            clauses.append("calibrated_sif_p_score <= ?")
            parameters.append(active_filter.max_calibrated_sif_p_score)
        if active_filter.timestamp_from is not None:
            clauses.append("timestamp >= ?")
            parameters.append(active_filter.timestamp_from.isoformat())
        if active_filter.timestamp_to is not None:
            clauses.append("timestamp <= ?")
            parameters.append(active_filter.timestamp_to.isoformat())
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        connection = self._connect()
        try:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM incident_results{where}", parameters
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT record_json
                FROM incident_results
                {where}
                ORDER BY timestamp ASC, log_id ASC
                LIMIT ? OFFSET ?
                """,
                [*parameters, request.limit, request.offset],
            ).fetchall()
            return Page(
                items=tuple(
                    StoredIncidentResult.model_validate_json(row["record_json"])
                    for row in rows
                ),
                offset=request.offset,
                limit=request.limit,
                total=total,
            )
        except sqlite3.Error as error:
            raise PersistenceError("cannot query incident inference results") from error
        finally:
            connection.close()


class SQLiteReviewDecisionRepository(_SQLiteRepositoryBase):
    """SQLite implementation of append-only human review history."""

    def append(self, decision: ReviewDecision) -> ReviewDecision:
        _validate_identifier(decision.decision_id, "decision_id")
        _validate_identifier(decision.log_id, "log_id")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO review_decisions(
                    decision_id, log_id, action, reviewer_id, decided_at, reason
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.log_id,
                    decision.action.value,
                    decision.reviewer_id,
                    decision.decided_at.isoformat(),
                    decision.reason,
                ),
            )
            connection.execute("COMMIT")
            return decision
        except sqlite3.IntegrityError as error:
            self._rollback(connection)
            raise PersistenceConflictError("decision_id already exists") from error
        except sqlite3.Error as error:
            self._rollback(connection)
            raise PersistenceError("cannot append review decision") from error
        finally:
            connection.close()

    def list_for_incident(
        self,
        log_id: str,
        *,
        page: PageRequest | None = None,
    ) -> Page[ReviewDecision]:
        _validate_identifier(log_id, "log_id")
        return self._list_history(
            table="review_decisions",
            timestamp_column="decided_at",
            tie_column="decision_id",
            log_id=log_id,
            page=page,
            factory=lambda row: ReviewDecision(
                decision_id=row["decision_id"],
                log_id=row["log_id"],
                action=row["action"],
                reviewer_id=row["reviewer_id"],
                decided_at=_parse_datetime(row["decided_at"]),
                reason=row["reason"],
            ),
        )

    def _list_history(
        self,
        *,
        table: str,
        timestamp_column: str,
        tie_column: str,
        log_id: str,
        page: PageRequest | None,
        factory: Callable[[sqlite3.Row], object],
    ) -> Page:
        request = page or PageRequest()
        connection = self._connect()
        try:
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE log_id = ?",
                    (log_id,),
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"""
                SELECT * FROM {table}
                WHERE log_id = ?
                ORDER BY {timestamp_column} ASC, {tie_column} ASC
                LIMIT ? OFFSET ?
                """,
                (log_id, request.limit, request.offset),
            ).fetchall()
            return Page(
                items=tuple(factory(row) for row in rows),
                offset=request.offset,
                limit=request.limit,
                total=total,
            )
        except sqlite3.Error as error:
            raise PersistenceError("cannot query persistence history") from error
        finally:
            connection.close()


class SQLiteAuditEventRepository(_SQLiteRepositoryBase):
    """SQLite implementation of append-only audit events."""

    def append(self, event: AuditEvent) -> AuditEvent:
        _validate_identifier(event.event_id, "event_id")
        _validate_identifier(event.log_id, "log_id")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO audit_events(
                    event_id, log_id, event_type, actor_id, occurred_at, reason
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.log_id,
                    event.event_type.value,
                    event.actor_id,
                    event.occurred_at.isoformat(),
                    event.reason,
                ),
            )
            connection.execute("COMMIT")
            return event
        except sqlite3.IntegrityError as error:
            self._rollback(connection)
            raise PersistenceConflictError("event_id already exists") from error
        except sqlite3.Error as error:
            self._rollback(connection)
            raise PersistenceError("cannot append audit event") from error
        finally:
            connection.close()

    def append_many(self, events: Sequence[AuditEvent]) -> tuple[AuditEvent, ...]:
        values = tuple(events)
        if not values:
            return ()
        if len({event.event_id for event in values}) != len(values):
            raise PersistenceConflictError("audit event IDs must be unique")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            for event in values:
                _validate_identifier(event.event_id, "event_id")
                _validate_identifier(event.log_id, "log_id")
                connection.execute(
                    """
                    INSERT INTO audit_events(
                        event_id, log_id, event_type, actor_id, occurred_at, reason
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.event_id,
                        event.log_id,
                        event.event_type.value,
                        event.actor_id,
                        event.occurred_at.isoformat(),
                        event.reason,
                    ),
                )
            connection.execute("COMMIT")
            return values
        except sqlite3.IntegrityError as error:
            self._rollback(connection)
            raise PersistenceConflictError("audit event ID already exists") from error
        except sqlite3.Error as error:
            self._rollback(connection)
            raise PersistenceError("cannot append audit events") from error
        except ValueError:
            self._rollback(connection)
            raise
        finally:
            connection.close()

    def list_for_incident(
        self,
        log_id: str,
        *,
        page: PageRequest | None = None,
    ) -> Page[AuditEvent]:
        _validate_identifier(log_id, "log_id")
        request = page or PageRequest()
        connection = self._connect()
        try:
            total = int(
                connection.execute(
                    "SELECT COUNT(*) FROM audit_events WHERE log_id = ?",
                    (log_id,),
                ).fetchone()[0]
            )
            rows = connection.execute(
                """
                SELECT * FROM audit_events
                WHERE log_id = ?
                ORDER BY occurred_at ASC, event_id ASC
                LIMIT ? OFFSET ?
                """,
                (log_id, request.limit, request.offset),
            ).fetchall()
            return Page(
                items=tuple(
                    AuditEvent(
                        event_id=row["event_id"],
                        log_id=row["log_id"],
                        event_type=row["event_type"],
                        actor_id=row["actor_id"],
                        occurred_at=_parse_datetime(row["occurred_at"]),
                        reason=row["reason"],
                    )
                    for row in rows
                ),
                offset=request.offset,
                limit=request.limit,
                total=total,
            )
        except sqlite3.Error as error:
            raise PersistenceError("cannot query audit history") from error
        finally:
            connection.close()
