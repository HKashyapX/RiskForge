"""Narrow review-workflow dependencies."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from riskforge.persistence.models import (
    AuditEvent,
    ReviewAction,
    ReviewDecision,
    StoredIncidentResult,
)


@runtime_checkable
class ReviewerAuthorizer(Protocol):
    """Permission boundary for reviewer actions."""

    def can_decide(self, reviewer_id: str, log_id: str, action: ReviewAction) -> bool:
        """Return whether the reviewer may record the requested action."""


@runtime_checkable
class ReviewAuditWriter(Protocol):
    """Atomic writer for a review decision and its audit event."""

    def append_review_atomically(
        self, decision: ReviewDecision, event: AuditEvent
    ) -> tuple[ReviewDecision, AuditEvent]:
        """Persist both records or persist neither."""


@runtime_checkable
class IncidentResultReader(Protocol):
    """Read-only access to persisted automated results."""

    def get(self, log_id: str) -> StoredIncidentResult | None:
        """Return the original automated result and incident context."""
