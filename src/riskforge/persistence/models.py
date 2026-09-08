"""Transport-neutral persistence value objects."""

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


class ReviewAction(str, Enum):
    """Human review outcomes supported by the workflow."""

    CONFIRM = "confirm"
    DISMISS = "dismiss"
    ESCALATE = "escalate"
    REQUEST_MORE_INFORMATION = "request_more_information"


class AuditEventType(str, Enum):
    """Stable event kinds written to the append-only audit trail."""

    INFERENCE_RECORDED = "inference_recorded"
    REVIEW_DECISION_RECORDED = "review_decision_recorded"


class PageRequest(BaseModel):
    """Validated zero-based pagination parameters."""

    model_config = ConfigDict(frozen=True)

    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=50, ge=1, le=500)


T = TypeVar("T")


class Page(BaseModel, Generic[T]):
    """Deterministically ordered page of repository results."""

    model_config = ConfigDict(frozen=True)

    items: tuple[T, ...]
    offset: int = Field(ge=0)
    limit: int = Field(ge=1, le=500)
    total: int = Field(ge=0)


class IncidentResultFilter(BaseModel):
    """Explicit incident-result query constraints."""

    model_config = ConfigDict(frozen=True)

    asset_id: str | None = Field(default=None, min_length=1)
    asset_type: AssetType | None = None
    routing: RoutingBucket | None = None
    min_calibrated_sif_p_score: float | None = Field(default=None, ge=0.0, le=1.0)
    max_calibrated_sif_p_score: float | None = Field(default=None, ge=0.0, le=1.0)
    timestamp_from: datetime | None = None
    timestamp_to: datetime | None = None

    def model_post_init(self, __context: object, /) -> None:
        if (
            self.min_calibrated_sif_p_score is not None
            and self.max_calibrated_sif_p_score is not None
            and self.min_calibrated_sif_p_score > self.max_calibrated_sif_p_score
        ):
            raise ValueError("minimum calibrated SIF-P score cannot exceed maximum")
        if (
            self.timestamp_from is not None
            and self.timestamp_to is not None
            and self.timestamp_from > self.timestamp_to
        ):
            raise ValueError("timestamp_from cannot be later than timestamp_to")


class ReviewDecision(BaseModel):
    """Immutable human decision linked to an automated inference result."""

    model_config = ConfigDict(frozen=True)

    decision_id: str = Field(min_length=1)
    log_id: str = Field(min_length=1)
    action: ReviewAction
    reviewer_id: str = Field(min_length=1)
    decided_at: datetime
    reason: str = Field(min_length=1)


class AuditEvent(BaseModel):
    """Append-only governance event."""

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(min_length=1)
    log_id: str = Field(min_length=1)
    event_type: AuditEventType
    actor_id: str = Field(min_length=1)
    occurred_at: datetime
    reason: str | None = None


class StoredIncidentResult(BaseModel):
    """Immutable persisted incident context and automated inference result."""

    model_config = ConfigDict(frozen=True)

    incident: IncidentNormalizedRecord
    result: ModelInferenceResult

    @property
    def log_id(self) -> str:
        return self.result.log_id

    @property
    def timestamp(self) -> datetime:
        return self.incident.timestamp

    @property
    def asset_id(self) -> str:
        return self.incident.asset_id

    @property
    def asset_type(self) -> AssetType:
        return self.incident.asset_type
