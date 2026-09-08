"""Persistence-neutral value objects for backend workflow use cases."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from riskforge.core.contracts import (
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    RoutingBucket,
)


class WorkflowModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ReviewAction(str, Enum):
    CONFIRM = "confirm"
    DISMISS = "dismiss"
    ESCALATE = "escalate"
    REQUEST_MORE_INFORMATION = "request_more_information"


class IncidentQuery(WorkflowModel):
    asset_id: str | None = Field(default=None, min_length=1, max_length=128)
    asset_type: AssetType | None = None
    routing: RoutingBucket | None = None
    timestamp_from: datetime | None = None
    timestamp_to: datetime | None = None

    def model_post_init(self, __context: object, /) -> None:
        if (
            self.timestamp_from is not None
            and self.timestamp_to is not None
            and self.timestamp_from > self.timestamp_to
        ):
            raise ValueError("timestamp_from cannot be later than timestamp_to")


class PageRequest(WorkflowModel):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)


T = TypeVar("T")


class Page(WorkflowModel, Generic[T]):
    items: tuple[T, ...]
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    total: int = Field(ge=0)


class IncidentView(WorkflowModel):
    incident: IncidentNormalizedRecord
    result: ModelInferenceResult

    def model_post_init(self, __context: object, /) -> None:
        if self.incident.log_id != self.result.log_id:
            raise ValueError("incident and result log_id values must match")


class ReviewCommand(WorkflowModel):
    log_id: str = Field(min_length=1, max_length=128)
    decision_id: str = Field(min_length=1, max_length=128)
    reviewer_id: str = Field(min_length=1, max_length=128)
    action: ReviewAction
    reason: str = Field(min_length=1, max_length=2_000)
    decided_at: datetime | None = None


class ReviewDecisionView(WorkflowModel):
    decision_id: str
    log_id: str
    reviewer_id: str
    action: ReviewAction
    reason: str
    decided_at: datetime


class AuditEventView(WorkflowModel):
    event_id: str
    log_id: str
    event_type: str
    actor_id: str
    occurred_at: datetime
    reason: str | None = None
