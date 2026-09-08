"""Transport contracts for the RiskForge HTTP API boundary."""

from riskforge.api.app import create_app
from riskforge.api.dependencies import (
    AuthenticatedPrincipal,
    PrincipalResolver,
    ReadinessProvider,
    ReadinessSnapshot,
    UnauthenticatedError,
)
from riskforge.api.errors import ErrorCode, TranslatedError, translate_application_error
from riskforge.api.models import (
    API_VERSION,
    AssetSummaryResponse,
    AuditPageResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    IncidentPageResponse,
    IncidentResponse,
    InferenceRequest,
    InferenceResponse,
    ReadinessResponse,
    ReviewDecisionRequest,
    ReviewDecisionResponse,
)

__all__ = [
    "API_VERSION",
    "AssetSummaryResponse",
    "AuditPageResponse",
    "AuthenticatedPrincipal",
    "BatchInferenceRequest",
    "BatchInferenceResponse",
    "ErrorCode",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "IncidentPageResponse",
    "IncidentResponse",
    "InferenceRequest",
    "InferenceResponse",
    "PrincipalResolver",
    "ReadinessProvider",
    "ReadinessResponse",
    "ReadinessSnapshot",
    "ReviewDecisionRequest",
    "ReviewDecisionResponse",
    "TranslatedError",
    "UnauthenticatedError",
    "create_app",
    "translate_application_error",
]
