"""Ports for incident browsing, human review, and audit history."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from riskforge.application.workflow_models import (
    AuditEventView,
    IncidentQuery,
    IncidentView,
    Page,
    PageRequest,
    ReviewCommand,
    ReviewDecisionView,
)


@runtime_checkable
class IncidentWorkflowReader(Protocol):
    def get_incident(self, log_id: str) -> IncidentView | None:
        """Return one incident with its immutable automated result."""

    def list_incidents(self, query: IncidentQuery, page: PageRequest) -> Page[IncidentView]:
        """Return a deterministic incident page."""

    def list_audit_events(self, log_id: str, page: PageRequest) -> Page[AuditEventView]:
        """Return append-only audit history in deterministic order."""


@runtime_checkable
class ReviewWorkflowWriter(Protocol):
    def decide(self, command: ReviewCommand) -> ReviewDecisionView:
        """Record an authorized decision while preserving the automated result."""
