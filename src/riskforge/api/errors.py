"""Safe translation from application failures to transport-neutral API errors."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from riskforge.application.exceptions import (
    DuplicateLogIdError,
    InferenceApplicationError,
    MetricsApplicationError,
    ResultCorrelationError,
)


class ErrorCode(str, Enum):
    DUPLICATE_LOG_ID = "duplicate_log_id"
    INFERENCE_UNAVAILABLE = "inference_unavailable"
    INVALID_INFERENCE_RESULT = "invalid_inference_result"
    METRICS_UNAVAILABLE = "metrics_unavailable"
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
    return TranslatedError(500, ErrorCode.INTERNAL_ERROR, "internal server error", False)
