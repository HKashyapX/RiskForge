"""Atomic SQLite persistence for review decisions and audit events."""

from __future__ import annotations

import sqlite3

from riskforge.persistence.exceptions import PersistenceConflictError, PersistenceError
from riskforge.persistence.models import AuditEvent, AuditEventType, ReviewDecision
from riskforge.persistence.sqlite.repository import (
    _iso_timestamp,
    _SQLiteRepositoryBase,
    _validate_identifier,
)


class SQLiteReviewAuditWriter(_SQLiteRepositoryBase):
    """Persist a review decision and its audit event in one SQLite transaction."""

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
        if event.event_type is not AuditEventType.REVIEW_DECISION_RECORDED:
            raise ValueError("audit event must record a review decision")

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
                    _iso_timestamp(decision.decided_at),
                    decision.reason,
                ),
            )
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
                    _iso_timestamp(event.occurred_at),
                    event.reason,
                ),
            )
            connection.execute("COMMIT")
            return decision, event
        except sqlite3.IntegrityError as error:
            self._rollback(connection)
            raise PersistenceConflictError("review or audit record already exists") from error
        except sqlite3.Error as error:
            self._rollback(connection)
            raise PersistenceError(
                "cannot atomically persist review decision and audit event"
            ) from error
        except ValueError:
            self._rollback(connection)
            raise
        finally:
            connection.close()
