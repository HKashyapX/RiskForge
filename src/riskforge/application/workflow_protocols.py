"""Ports for incident browsing, human review, and audit history."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from riskforge.application.protocols import RiskForgeApplication
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


@runtime_checkable
class BackendApplication(RiskForgeApplication, Protocol):
    """Complete application use cases exposed to external transports."""

    def get_incident(self, log_id: str) -> IncidentView:
        """Return one incident or raise an application not-found error."""

    def list_incidents(self, query: IncidentQuery, page: PageRequest) -> Page[IncidentView]:
        """Return a deterministic incident page."""

    def list_audit_events(self, log_id: str, page: PageRequest) -> Page[AuditEventView]:
        """Return deterministic append-only audit history."""

    def decide_review(self, command: ReviewCommand) -> ReviewDecisionView:
        """Record one authorized human-review command."""
