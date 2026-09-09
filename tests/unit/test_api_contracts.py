from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from riskforge.api.errors import ErrorCode, translate_application_error
from riskforge.api.models import (
    BatchInferenceRequest,
    ErrorDetail,
    ErrorResponse,
    InferenceRequest,
    InferenceResponse,
    ReviewDecisionRequest,
)
from riskforge.application.exceptions import (
    DuplicateLogIdError,
    IncidentNotFoundError,
    InferenceApplicationError,
    MetricsApplicationError,
    QueryApplicationError,
    ResultCorrelationError,
    ReviewConflictApplicationError,
    ReviewPermissionApplicationError,
)
from riskforge.core.contracts import (
    AssetType,
    IncidentNormalizedRecord,
    ModelInferenceResult,
    OperationalTriad,
    RoutingBucket,
)


def _incident(log_id: str = "LOG_1") -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=datetime.now(UTC),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative="Sensitive incident narrative.",
        spans=[],
    )


def _result(log_id: str = "LOG_1") -> ModelInferenceResult:
    return ModelInferenceResult(
        log_id=log_id,
        raw_sif_p_score=0.2,
        calibrated_sif_p_score=0.2,
        deterministic_override=False,
        routing=RoutingBucket.AUTO_DISMISS,
        matched_iogp_rules=[],
        triad=OperationalTriad(),
        latency_ms=1.0,
    )


def test_inference_contracts_are_strict_versioned_and_immutable() -> None:
    request = InferenceRequest(correlation_id="request:1", incident=_incident())
    response = InferenceResponse(correlation_id=request.correlation_id, result=_result())

    assert response.api_version == "v1"
    assert response.result.log_id == request.incident.log_id
    with pytest.raises(ValidationError):
        InferenceRequest(correlation_id="contains spaces", incident=_incident())
    with pytest.raises(ValidationError):
        InferenceRequest(correlation_id="request:1", incident=_incident(), extra="forbidden")
    with pytest.raises(ValidationError):
        response.correlation_id = "changed"


def test_batch_request_enforces_transport_bound_without_hiding_application_validation() -> None:
    request = BatchInferenceRequest(
        correlation_id="batch-1", incidents=tuple(_incident(f"LOG_{i}") for i in range(32))
    )
    assert len(request.incidents) == 32
    with pytest.raises(ValidationError):
        BatchInferenceRequest(
            correlation_id="batch-2", incidents=tuple(_incident(str(i)) for i in range(33))
        )


def test_review_request_excludes_reviewer_identity_from_untrusted_body() -> None:
    request = ReviewDecisionRequest(
        correlation_id="review-1",
        decision_id="decision-1",
        action="confirm",
        reason="Evidence verified",
    )
    assert request.action == "confirm"
    with pytest.raises(ValidationError):
        ReviewDecisionRequest(
            correlation_id="review-1",
            decision_id="decision-1",
            action="confirm",
            reason="Evidence verified",
            reviewer_id="untrusted",
        )


@pytest.mark.parametrize(
    ("error", "status", "code", "retryable"),
    [
        (DuplicateLogIdError("private"), 409, ErrorCode.DUPLICATE_LOG_ID, False),
        (ResultCorrelationError("private"), 502, ErrorCode.INVALID_INFERENCE_RESULT, False),
        (InferenceApplicationError("private"), 503, ErrorCode.INFERENCE_UNAVAILABLE, True),
        (MetricsApplicationError("private"), 503, ErrorCode.METRICS_UNAVAILABLE, True),
        (IncidentNotFoundError("private"), 404, ErrorCode.INCIDENT_NOT_FOUND, False),
        (QueryApplicationError("private"), 503, ErrorCode.QUERY_UNAVAILABLE, True),
        (ReviewPermissionApplicationError("private"), 403, ErrorCode.REVIEW_FORBIDDEN, False),
        (ReviewConflictApplicationError("private"), 409, ErrorCode.REVIEW_CONFLICT, False),
        (RuntimeError("private"), 500, ErrorCode.INTERNAL_ERROR, False),
    ],
)
def test_error_translation_is_stable_and_does_not_leak_details(
    error, status, code, retryable
) -> None:
    translated = translate_application_error(error)
    assert translated.status_code == status
    assert translated.code is code
    assert translated.retryable is retryable
    assert "private" not in translated.message


def test_error_response_is_transport_safe() -> None:
    response = ErrorResponse(
        correlation_id="request-1",
        error=ErrorDetail(code="internal_error", message="internal server error"),
    )
    assert "Sensitive incident narrative" not in response.model_dump_json()
