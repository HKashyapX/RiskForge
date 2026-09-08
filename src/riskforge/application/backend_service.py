"""Unified application facade for HTTP-facing backend use cases."""

from __future__ import annotations

from collections.abc import Sequence

from riskforge.application.exceptions import (
    ApplicationError,
    IncidentNotFoundError,
    QueryApplicationError,
    ReviewApplicationError,
)
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
from riskforge.application.workflow_protocols import IncidentWorkflowReader, ReviewWorkflowWriter
from riskforge.core.contracts import (
    AssetRiskSummary,
    IncidentNormalizedRecord,
    ModelInferenceResult,
)


class BackendApplicationService:
    """Compose established application use cases behind one transport facade."""

    def __init__(
        self,
        core: RiskForgeApplication,
        incidents: IncidentWorkflowReader,
        reviews: ReviewWorkflowWriter,
    ) -> None:
        self._core = core
        self._incidents = incidents
        self._reviews = reviews

    def process_incident(self, record: IncidentNormalizedRecord) -> ModelInferenceResult:
        return self._core.process_incident(record)

    def process_batch(
        self, records: Sequence[IncidentNormalizedRecord]
    ) -> tuple[ModelInferenceResult, ...]:
        return tuple(self._core.process_batch(records))

    def get_asset_summary(self, asset_id: str) -> AssetRiskSummary:
        return self._core.get_asset_summary(asset_id)

    def get_incident(self, log_id: str) -> IncidentView:
        try:
            incident = self._incidents.get_incident(log_id)
        except Exception as error:
            raise QueryApplicationError("incident lookup failed") from error
        if incident is None:
            raise IncidentNotFoundError("incident was not found")
        return incident

    def list_incidents(self, query: IncidentQuery, page: PageRequest) -> Page[IncidentView]:
        try:
            return self._incidents.list_incidents(query, page)
        except Exception as error:
            raise QueryApplicationError("incident query failed") from error

    def list_audit_events(self, log_id: str, page: PageRequest) -> Page[AuditEventView]:
        try:
            return self._incidents.list_audit_events(log_id, page)
        except Exception as error:
            raise QueryApplicationError("audit query failed") from error

    def decide_review(self, command: ReviewCommand) -> ReviewDecisionView:
        try:
            return self._reviews.decide(command)
        except ApplicationError:
            raise
        except Exception as error:
            raise ReviewApplicationError("review decision failed") from error
