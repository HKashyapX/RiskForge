"""Composition adapters for existing persistence and review subsystems."""

from riskforge.application.exceptions import (
    ReviewApplicationError,
    ReviewConflictApplicationError,
    ReviewNotFoundApplicationError,
    ReviewPermissionApplicationError,
)
from riskforge.application.workflow_models import (
    AuditEventView,
    IncidentQuery,
    IncidentView,
    Page,
    PageRequest,
    ReviewCommand,
    ReviewDecisionView,
)
from riskforge.persistence.models import (
    IncidentResultFilter,
)
from riskforge.persistence.models import (
    PageRequest as PersistencePageRequest,
)
from riskforge.persistence.models import (
    ReviewAction as PersistenceReviewAction,
)
from riskforge.persistence.protocols import (
    AuditEventRepository,
    IncidentResultRepository,
)
from riskforge.review.exceptions import (
    ReviewConflictError,
    ReviewNotFoundError,
    ReviewPermissionError,
    ReviewWorkflowError,
)
from riskforge.review.service import ReviewService


class PersistenceWorkflowReader:
    """Translate persistence values into application-owned workflow values."""

    def __init__(
        self,
        incidents: IncidentResultRepository,
        audit_events: AuditEventRepository,
    ) -> None:
        self._incidents = incidents
        self._audit_events = audit_events

    def get_incident(self, log_id: str) -> IncidentView | None:
        stored = self._incidents.get(log_id)
        if stored is None:
            return None
        return IncidentView(incident=stored.incident, result=stored.result)

    def list_incidents(self, query: IncidentQuery, page: PageRequest) -> Page[IncidentView]:
        result = self._incidents.list(
            IncidentResultFilter(
                asset_id=query.asset_id,
                asset_type=query.asset_type,
                routing=query.routing,
                timestamp_from=query.timestamp_from,
                timestamp_to=query.timestamp_to,
            ),
            page=PersistencePageRequest(offset=page.offset, limit=page.limit),
        )
        return Page(
            items=tuple(
                IncidentView(incident=item.incident, result=item.result) for item in result.items
            ),
            offset=result.offset,
            limit=result.limit,
            total=result.total,
        )

    def list_audit_events(self, log_id: str, page: PageRequest) -> Page[AuditEventView]:
        result = self._audit_events.list_for_incident(
            log_id,
            page=PersistencePageRequest(offset=page.offset, limit=page.limit),
        )
        return Page(
            items=tuple(
                AuditEventView(
                    event_id=item.event_id,
                    log_id=item.log_id,
                    event_type=item.event_type.value,
                    actor_id=item.actor_id,
                    occurred_at=item.occurred_at,
                    reason=item.reason,
                )
                for item in result.items
            ),
            offset=result.offset,
            limit=result.limit,
            total=result.total,
        )


class ReviewServiceAdapter:
    """Translate application review commands into the established review workflow."""

    def __init__(self, review_service: ReviewService) -> None:
        self._review_service = review_service

    def decide(self, command: ReviewCommand) -> ReviewDecisionView:
        try:
            decision = self._review_service.decide(
                log_id=command.log_id,
                decision_id=command.decision_id,
                reviewer_id=command.reviewer_id,
                action=PersistenceReviewAction(command.action.value),
                reason=command.reason,
                decided_at=command.decided_at,
            )
        except ReviewNotFoundError as error:
            raise ReviewNotFoundApplicationError("review target was not found") from error
        except ReviewPermissionError as error:
            raise ReviewPermissionApplicationError("review action is not permitted") from error
        except ReviewConflictError as error:
            raise ReviewConflictApplicationError("review decision conflicts with history") from error
        except ReviewWorkflowError as error:
            raise ReviewApplicationError("review workflow failed") from error
        return ReviewDecisionView(
            decision_id=decision.decision_id,
            log_id=decision.log_id,
            reviewer_id=decision.reviewer_id,
            action=command.action,
            reason=decision.reason,
            decided_at=decision.decided_at,
        )
