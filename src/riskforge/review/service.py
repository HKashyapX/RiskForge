"""Human review orchestration with immutable automated-result preservation."""

from __future__ import annotations

from datetime import UTC, datetime

from riskforge.persistence.models import AuditEvent, AuditEventType, ReviewAction, ReviewDecision
from riskforge.persistence.exceptions import PersistenceConflictError, PersistenceError
from riskforge.review.exceptions import (
    ReviewConflictError,
    ReviewNotFoundError,
    ReviewPermissionError,
    ReviewPersistenceError,
)
from riskforge.review.protocols import IncidentResultReader, ReviewAuditWriter, ReviewerAuthorizer


class ReviewService:
    """Record authorized human decisions without mutating automated inference results."""

    def __init__(
        self,
        result_reader: IncidentResultReader,
        audit_writer: ReviewAuditWriter,
        authorizer: ReviewerAuthorizer,
    ) -> None:
        self._result_reader = result_reader
        self._audit_writer = audit_writer
        self._authorizer = authorizer

    def decide(
        self,
        *,
        log_id: str,
        decision_id: str,
        reviewer_id: str,
        action: ReviewAction,
        reason: str,
        decided_at: datetime | None = None,
    ) -> ReviewDecision:
        """Record one authorized review action and its audit event atomically."""
        self._validate_identifier(log_id, "log_id")
        self._validate_identifier(decision_id, "decision_id")
        self._validate_identifier(reviewer_id, "reviewer_id")
        if not reason.strip():
            raise ValueError("reason must not be empty")

        original = self._result_reader.get(log_id)
        if original is None:
            raise ReviewNotFoundError("incident automated result was not found")
        if original.log_id != log_id or original.result.log_id != log_id:
            raise ReviewNotFoundError("incident automated result correlation failed")

        try:
            authorized = self._authorizer.can_decide(reviewer_id, log_id, action)
        except Exception as error:
            raise ReviewPermissionError("reviewer authorization failed") from error
        if not authorized:
            raise ReviewPermissionError("reviewer is not authorized for this action")

        timestamp = decided_at or datetime.now(UTC)
        decision = ReviewDecision(
            decision_id=decision_id,
            log_id=log_id,
            action=action,
            reviewer_id=reviewer_id,
            decided_at=timestamp,
            reason=reason,
        )
        event = AuditEvent(
            event_id=f"review:{decision_id}",
            log_id=log_id,
            event_type=AuditEventType.REVIEW_DECISION_RECORDED,
            actor_id=reviewer_id,
            occurred_at=timestamp,
            reason=reason,
        )
        try:
            persisted_decision, _ = self._audit_writer.append_review_atomically(decision, event)
        except PersistenceConflictError as error:
            raise ReviewConflictError("review decision already exists") from error
        except PersistenceError as error:
            raise ReviewPersistenceError("review decision could not be persisted") from error
        except Exception as error:
            raise ReviewPersistenceError("review decision could not be persisted") from error

        return persisted_decision

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        if not value.strip():
            raise ValueError(f"{name} must not be empty")
