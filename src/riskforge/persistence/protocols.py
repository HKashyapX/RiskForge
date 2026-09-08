"""Repository interfaces for application and workflow services."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from riskforge.core.contracts import ModelInferenceResult
from riskforge.persistence.models import (
    AuditEvent,
    IncidentResultFilter,
    Page,
    PageRequest,
    ReviewDecision,
)


class IncidentResultRepository(Protocol):
    """Durable repository for immutable automated inference results."""

    def create_idempotent(self, result: ModelInferenceResult) -> ModelInferenceResult:
        """Atomically create a result or return the identical existing result for `log_id`.

        Implementations must raise `PersistenceConflictError` when the same `log_id`
        already exists with different content.
        """

    def get(self, log_id: str) -> ModelInferenceResult | None:
        """Return the immutable automated result for a log ID, if present."""

    def list(
        self,
        filters: IncidentResultFilter | None = None,
        *,
        page: PageRequest | None = None,
    ) -> Page[ModelInferenceResult]:
        """Return results ordered by `(timestamp ASC, log_id ASC)` deterministically."""


class ReviewDecisionRepository(Protocol):
    """Repository for append-only human review decisions."""

    def append(self, decision: ReviewDecision) -> ReviewDecision:
        """Append a decision; decision IDs are immutable and unique."""

    def list_for_incident(
        self,
        log_id: str,
        *,
        page: PageRequest | None = None,
    ) -> Page[ReviewDecision]:
        """Return decisions for one incident in `(decided_at ASC, decision_id ASC)` order."""


class AuditEventRepository(Protocol):
    """Append-only audit event repository."""

    def append(self, event: AuditEvent) -> AuditEvent:
        """Append one governance event without allowing replacement or deletion."""

    def append_many(self, events: Sequence[AuditEvent]) -> tuple[AuditEvent, ...]:
        """Append events atomically while preserving supplied order."""

    def list_for_incident(
        self,
        log_id: str,
        *,
        page: PageRequest | None = None,
    ) -> Page[AuditEvent]:
        """Return events in `(occurred_at ASC, event_id ASC)` order."""
