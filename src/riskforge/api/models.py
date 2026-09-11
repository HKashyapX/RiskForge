"""Versioned, transport-only request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from riskforge.application.workflow_models import (
    AuditEventView,
    IncidentView,
    Page,
    ReviewDecisionView,
)
from riskforge.core.contracts import (
    AssetRiskSummary,
    IncidentNormalizedRecord,
    ModelInferenceResult,
)

API_VERSION = "v1"
CorrelationId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]


class ApiModel(BaseModel):
    """Strict immutable base for public transport contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class InferenceRequest(ApiModel):
    correlation_id: CorrelationId
    incident: IncidentNormalizedRecord


class BatchInferenceRequest(ApiModel):
    correlation_id: CorrelationId
    incidents: tuple[IncidentNormalizedRecord, ...] = Field(min_length=1, max_length=32)


class IngestionItemResponse(ApiModel):
    """Per-report outcome of an ingestion request."""

    log_id: str
    status: str
    error: str | None = None
    result: ModelInferenceResult | None = None


class IngestionRunResponse(ApiModel):
    """Aggregate outcome of an ingestion request."""

    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    fmt: str
    received: int
    normalized: int
    failed: int
    items: tuple[IngestionItemResponse, ...]


class AnalyticsSummaryResponse(ApiModel):
    """Operational analytics summary for dashboard consumption."""

    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    summary: dict[str, Any]


class ReviewDecisionRequest(ApiModel):
    correlation_id: CorrelationId
    decision_id: Identifier
    action: Literal["confirm", "dismiss", "escalate", "request_more_information"]
    reason: str = Field(min_length=1, max_length=2_000)


class InferenceResponse(ApiModel):
    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    result: ModelInferenceResult


class BatchInferenceResponse(ApiModel):
    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    results: tuple[ModelInferenceResult, ...]


class AssetSummaryResponse(ApiModel):
    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    summary: AssetRiskSummary


class IncidentResponse(ApiModel):
    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    incident: IncidentView


class IncidentPageResponse(ApiModel):
    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    page: Page[IncidentView]


class AuditPageResponse(ApiModel):
    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    page: Page[AuditEventView]


class ReviewDecisionResponse(ApiModel):
    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    decision: ReviewDecisionView


class ErrorDetail(ApiModel):
    code: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9_]+$")
    message: str = Field(min_length=1, max_length=500)
    retryable: bool = False


class ErrorResponse(ApiModel):
    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    error: ErrorDetail


class HealthResponse(ApiModel):
    api_version: Literal["v1"] = API_VERSION
    status: Literal["live"] = "live"


class ReadinessResponse(ApiModel):
    api_version: Literal["v1"] = API_VERSION
    correlation_id: CorrelationId
    ready: bool
    state: str = Field(min_length=1, max_length=32)
    checked_at: datetime
    components: tuple[str, ...] = ()
