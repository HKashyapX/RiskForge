"""SQLite repositories for isolated development and integration testing."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from threading import RLock

from riskforge.core.contracts import ModelInferenceResult
from riskforge.persistence.exceptions import (
    PersistenceConflictError,
    PersistenceError,
)
from riskforge.persistence.models import (
    AuditEvent,
    IncidentResultFilter,
    Page,
    PageRequest,
    ReviewDecision,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS incident_results (
    log_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    asset_id TEXT NOT NULL,
    asset_type TEXT NOT NULL,
    routing TEXT NOT NULL,
    calibrated_sif_p_score REAL NOT NULL,
    result_json TEXT NOT NULL
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


def _json_payload(model: object) -> str:
    if not hasattr(model, "model_dump"):
        raise TypeError("persistence payload must be a Pydantic model")
    payload = model.model_dump(mode="json")  # type: ignore[attr-defined]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SQLitePersistence:
    """Concrete SQLite implementation of all persistence repositories."""

    def __init__(self, database_path: str | Path) -> None:
        if str(database_path).strip() == "":
            raise ValueError("database_path must not be empty")
        self.database_path = Path(database_path)
        if self.database_path.parent != Path("."):
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(
                self.database_path,
                timeout=5.0,
                isolation_level=None,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            return connection
        except sqlite3.Error as error:
            raise PersistenceError("cannot open SQLite persistence database") from error

    def _initialize(self) -> None:
        with self._lock:
            connection = self._connect()
            try:
                connection.executescript(_SCHEMA)
            except sqlite3.Error as error:
                raise PersistenceError("cannot initialize SQLite persistence schema") from error
            finally:
                connection.close()

    def close(self) -> None:
        """Release repository resources; connections are per operation."""

    def create_idempotent(self, result: ModelInferenceResult) -> ModelInferenceResult:
        if not result.log_id.strip():
            raise ValueError("log_id must not be empty")
        payload = _json_payload(result)
        with self._lock:
            connection = self._connect()
            try:
                connection.execute("BEGIN IMMEDIATE")
                existing = connection.execute(
                    "SELECT result_json FROM incident_results WHERE log_id = ?",
                    (result.log_id,),
                ).fetchone()
                if existing is not None:
                    if existing["result_json"] != payload:
                        raise PersistenceConflictError(
                            "log_id already contains a different result"
                        )
                    connection.execute("COMMIT")
                    return ModelInferenceResult.model_validate_json(existing["result_json"])
                connection.execute(
                    """
                    INSERT INTO incident_results(
                        log_id, timestamp, asset_id, asset_type, routing,
                        calibrated_sif_p_score, result_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.log_id,
                        result.triad.model_dump_json() if False else result.log_id,
                        result.log_id,
                        result.log_id,
                        result.routing.value,
                        result.calibrated_sif_p_score,
                        payload,
                    ),
                )
                connection.execute("COMMIT")
                return result
            except PersistenceConflictError:
                connection.execute("ROLLBACK")
                raise
            except sqlite3.Error as error:
                connection.execute("ROLLBACK")
                raise PersistenceError("cannot persist incident inference result") from error
            finally:
                connection.close()

    def get(self, log_id: str) -> ModelInferenceResult | None:
        if not log_id.strip():
            raise ValueError("log_id must not be empty")
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT result_json FROM incident_results WHERE log_id = ?",
                (log_id,),
            ).fetchone()
            return None if row is None else ModelInferenceResult.model_validate_json(row["result_json"])
        except sqlite3.Error as error:
            raise PersistenceError("cannot read incident inference result") from error
        finally:
            connection.close()

    def list(
        self,
        filters: IncidentResultFilter | None = None,
        *,
        page: PageRequest | None = None,
    ) -> Page[ModelInferenceResult]:
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
                SELECT result_json
                FROM incident_results
                {where}
                ORDER BY timestamp ASC, log_id ASC
                LIMIT ? OFFSET ?
                """,
                [*parameters, request.limit, request.offset],
            ).fetchall()
            items = tuple(ModelInferenceResult.model_validate_json(row["result_json"]) for row in rows)
            return Page(items=items, offset=request.offset, limit=request.limit, total=total)
        except sqlite3.Error as error:
            raise PersistenceError("cannot query incident inference results") from error
        finally:
            connection.close()

    def append(self, decision: ReviewDecision) -> ReviewDecision:
        if not decision.log_id.strip():
            raise ValueError("log_id must not be empty")
        if not decision.decision_id.strip():
            raise ValueError("decision_id must not be empty")
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
            connection.execute("ROLLBACK")
            raise PersistenceConflictError("decision_id already exists") from error
        except sqlite3.Error as error:
            connection.execute("ROLLBACK")
            raise PersistenceError("cannot append review decision") from error
        finally:
            connection.close()

    def list_for_incident(
        self,
        log_id: str,
        *,
        page: PageRequest | None = None,
    ) -> Page[ReviewDecision]:
        return self._list_history(
            table="review_decisions",
            timestamp_column="decided_at",
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

    def append_audit(self, event: AuditEvent) -> AuditEvent:
        if not event.log_id.strip():
            raise ValueError("log_id must not be empty")
        if not event.event_id.strip():
            raise ValueError("event_id must not be empty")
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
            connection.execute("ROLLBACK")
            raise PersistenceConflictError("event_id already exists") from error
        except sqlite3.Error as error:
            connection.execute("ROLLBACK")
            raise PersistenceError("cannot append audit event") from error
        finally:
            connection.close()

    def append_many_audit(self, events: Sequence[AuditEvent]) -> tuple[AuditEvent, ...]:
        values = tuple(events)
        if not values:
            return ()
        if len({event.event_id for event in values}) != len(values):
            raise PersistenceConflictError("audit event IDs must be unique")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            for event in values:
                if not event.event_id.strip() or not event.log_id.strip():
                    raise ValueError("audit event IDs and log IDs must not be empty")
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
            connection.execute("ROLLBACK")
            raise PersistenceConflictError("audit event ID already exists") from error
        except sqlite3.Error as error:
            connection.execute("ROLLBACK")
            raise PersistenceError("cannot append audit events") from error
        finally:
            connection.close()

    def list_audit_for_incident(
        self,
        log_id: str,
        *,
        page: PageRequest | None = None,
    ) -> Page[AuditEvent]:
        return self._list_history(
            table="audit_events",
            timestamp_column="occurred_at",
            log_id=log_id,
            page=page,
            factory=lambda row: AuditEvent(
                event_id=row["event_id"],
                log_id=row["log_id"],
                event_type=row["event_type"],
                actor_id=row["actor_id"],
                occurred_at=_parse_datetime(row["occurred_at"]),
                reason=row["reason"],
            ),
        )

    def _list_history(
        self,
        *,
        table: str,
        timestamp_column: str,
        log_id: str,
        page: PageRequest | None,
        factory,
    ):
        if not log_id.strip():
            raise ValueError("log_id must not be empty")
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
                ORDER BY {timestamp_column} ASC, rowid ASC
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


# Repository-shaped aliases/adapters keep application dependencies protocol-oriented.
class SQLiteIncidentResultRepository:
    def __init__(self, store: SQLitePersistence) -> None:
        self._store = store

    create_idempotent = SQLitePersistence.create_idempotent
    get = SQLitePersistence.get
    list = SQLitePersistence.list


class SQLiteReviewDecisionRepository:
    def __init__(self, store: SQLitePersistence) -> None:
        self._store = store

    append = SQLitePersistence.append
    list_for_incident = SQLitePersistence.list_for_incident


class SQLiteAuditEventRepository:
    def __init__(self, store: SQLitePersistence) -> None:
        self._store = store

    append = SQLitePersistence.append_audit
    append_many = SQLitePersistence.append_many_audit
    list_for_incident = SQLitePersistence.list_audit_for_incident
