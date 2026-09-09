"""Thin FastAPI application factory over RiskForge application interfaces."""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, Header, Path, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from riskforge.api.dependencies import ReadinessProvider
from riskforge.api.errors import ErrorCode, TranslatedError, translate_application_error
from riskforge.api.models import (
    AssetSummaryResponse,
    BatchInferenceRequest,
    BatchInferenceResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    InferenceRequest,
    InferenceResponse,
    ReadinessResponse,
)
from riskforge.application.protocols import RiskForgeApplication

CorrelationHeader = Annotated[
    str,
    Header(
        alias="X-Correlation-ID",
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9._:-]+$",
    ),
]
AssetIdPath = Annotated[str, Path(min_length=1, max_length=128)]


class _ApiFailure(Exception):
    def __init__(self, correlation_id: str, translated: TranslatedError) -> None:
        super().__init__(translated.code.value)
        self.correlation_id = correlation_id
        self.translated = translated


def _error_response(correlation_id: str, translated: TranslatedError) -> ErrorResponse:
    return ErrorResponse(
        correlation_id=correlation_id,
        error=ErrorDetail(
            code=translated.code.value,
            message=translated.message,
            retryable=translated.retryable,
        ),
    )


def create_app(
    application: RiskForgeApplication,
    readiness: ReadinessProvider,
) -> FastAPI:
    """Create the HTTP shell without constructing concrete infrastructure."""
    app = FastAPI(title="RiskForge API", version="1.0.0")

    @app.exception_handler(_ApiFailure)
    async def handle_api_failure(_request: Request, error: _ApiFailure) -> JSONResponse:
        payload = _error_response(error.correlation_id, error.translated)
        return JSONResponse(
            status_code=error.translated.status_code,
            content=payload.model_dump(mode="json"),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_failure(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        translated = TranslatedError(
            status_code=422,
            code=ErrorCode.INVALID_REQUEST,
            message="request validation failed",
            retryable=False,
        )
        payload = _error_response("unavailable", translated)
        return JSONResponse(status_code=422, content=payload.model_dump(mode="json"))

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.get(
        "/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
    )
    def ready(correlation_id: CorrelationHeader) -> ReadinessResponse | JSONResponse:
        snapshot = readiness.snapshot()
        response = ReadinessResponse(
            correlation_id=correlation_id,
            ready=snapshot.ready,
            state=snapshot.state,
            checked_at=snapshot.checked_at,
            components=snapshot.components,
        )
        if snapshot.ready:
            return response
        return JSONResponse(status_code=503, content=response.model_dump(mode="json"))

    @app.post("/v1/inference", response_model=InferenceResponse)
    def infer(request: InferenceRequest) -> InferenceResponse:
        try:
            result = application.process_incident(request.incident)
        except Exception as error:
            raise _ApiFailure(request.correlation_id, translate_application_error(error)) from error
        return InferenceResponse(correlation_id=request.correlation_id, result=result)

    @app.post("/v1/inference/batch", response_model=BatchInferenceResponse)
    def infer_batch(request: BatchInferenceRequest) -> BatchInferenceResponse:
        try:
            results = tuple(application.process_batch(request.incidents))
        except Exception as error:
            raise _ApiFailure(request.correlation_id, translate_application_error(error)) from error
        return BatchInferenceResponse(correlation_id=request.correlation_id, results=results)

    @app.get("/v1/assets/{asset_id}/summary", response_model=AssetSummaryResponse)
    def asset_summary(
        asset_id: AssetIdPath, correlation_id: CorrelationHeader
    ) -> AssetSummaryResponse:
        try:
            summary = application.get_asset_summary(asset_id)
        except Exception as error:
            raise _ApiFailure(correlation_id, translate_application_error(error)) from error
        return AssetSummaryResponse(
            correlation_id=correlation_id,
            summary=summary,
        )

    return app
