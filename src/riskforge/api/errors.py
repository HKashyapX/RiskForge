"""Safe translation from application failures to transport-neutral API errors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from riskforge.application.exceptions import (
    DuplicateLogIdError,
    IncidentNotFoundError,
    InferenceApplicationError,
    MetricsApplicationError,
    QueryApplicationError,
    ResultCorrelationError,
    ReviewApplicationError,
    ReviewConflictApplicationError,
    ReviewNotFoundApplicationError,
    ReviewPermissionApplicationError,
)


class ErrorCode(str, Enum):
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION_REQUIRED = "authentication_required"
    AUTHENTICATION_UNAVAILABLE = "authentication_unavailable"
    DUPLICATE_LOG_ID = "duplicate_log_id"
    INCIDENT_NOT_FOUND = "incident_not_found"
    INFERENCE_UNAVAILABLE = "inference_unavailable"
    INVALID_INFERENCE_RESULT = "invalid_inference_result"
    METRICS_UNAVAILABLE = "metrics_unavailable"
    QUERY_UNAVAILABLE = "query_unavailable"
    REVIEW_CONFLICT = "review_conflict"
    REVIEW_FORBIDDEN = "review_forbidden"
    REVIEW_UNAVAILABLE = "review_unavailable"
    INTERNAL_ERROR = "internal_error"


@dataclass(frozen=True)
class TranslatedError:
    status_code: int
    code: ErrorCode
    message: str
    retryable: bool


def translate_application_error(error: Exception) -> TranslatedError:
    """Map known application errors without exposing internal exception text."""
    if isinstance(error, DuplicateLogIdError):
        return TranslatedError(409, ErrorCode.DUPLICATE_LOG_ID, "duplicate log identifiers", False)
    if isinstance(error, ResultCorrelationError):
        return TranslatedError(
            502, ErrorCode.INVALID_INFERENCE_RESULT, "invalid inference result", False
        )
    if isinstance(error, InferenceApplicationError):
        return TranslatedError(
            503, ErrorCode.INFERENCE_UNAVAILABLE, "inference service unavailable", True
        )
    if isinstance(error, MetricsApplicationError):
        return TranslatedError(
            503, ErrorCode.METRICS_UNAVAILABLE, "metrics service unavailable", True
        )
    if isinstance(error, (IncidentNotFoundError, ReviewNotFoundApplicationError)):
        return TranslatedError(404, ErrorCode.INCIDENT_NOT_FOUND, "incident was not found", False)
    if isinstance(error, ReviewPermissionApplicationError):
        return TranslatedError(403, ErrorCode.REVIEW_FORBIDDEN, "review action is forbidden", False)
    if isinstance(error, ReviewConflictApplicationError):
        return TranslatedError(409, ErrorCode.REVIEW_CONFLICT, "review decision conflicts", False)
    if isinstance(error, QueryApplicationError):
        return TranslatedError(503, ErrorCode.QUERY_UNAVAILABLE, "query service unavailable", True)
    if isinstance(error, ReviewApplicationError):
        return TranslatedError(503, ErrorCode.REVIEW_UNAVAILABLE, "review service unavailable", True)
    return TranslatedError(500, ErrorCode.INTERNAL_ERROR, "internal server error", False)
