"""Thin FastAPI application factory over RiskForge application interfaces."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.responses import Response

from riskforge.api.dependencies import ReadinessProvider, require_principal
from riskforge.api.errors import ErrorCode, TranslatedError, translate_application_error
from riskforge.api.models import (
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
from riskforge.api.request_controls import RequestControlsMiddleware
from riskforge.application.workflow_models import (
    IncidentQuery,
    PageRequest,
    ReviewAction,
    ReviewCommand,
)
from riskforge.application.workflow_protocols import BackendApplication
from riskforge.authentication.exceptions import (
    AuthenticationError,
    MissingCredentialsError,
)
from riskforge.authentication.principal import Principal
from riskforge.authentication.protocols import AuthenticationService
from riskforge.core.contracts import AssetType, RoutingBucket
from riskforge.observability.middleware import RequestMetricsMiddleware

logger = logging.getLogger("riskforge.api.app")

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
LogIdPath = Annotated[str, Path(min_length=1, max_length=128)]
PageOffset = Annotated[int, Query(ge=0)]
PageLimit = Annotated[int, Query(ge=1, le=500)]


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


def _extract_principal(auth_service: AuthenticationService, request: Request) -> Principal:
    """Extract and verify credentials from the HTTP Authorization header."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise MissingCredentialsError()
    token = auth_header[7:]  # len("Bearer ") == 7
    return auth_service.authenticate(token)


def create_app(
    application: BackendApplication,
    readiness: ReadinessProvider,
    auth_service: AuthenticationService | None = None,
    enable_metrics: bool = True,
    cors_origins: list[str] | None = None,
    max_request_bytes: int = 1_048_576,
    request_timeout_seconds: float = 30.0,
    max_batch_size: int = 32,
) -> FastAPI:
    """Create the HTTP shell without constructing concrete infrastructure.

    Parameters
    ----------
    application:
        Application-facing orchestration boundary.
    readiness:
        Runtime readiness provider.
    auth_service:
        Optional authentication provider.  When provided, routes that
        declare ``Depends(require_principal)`` will extract and verify
        credentials from the ``Authorization: Bearer <token>`` header.
        When ``None``, no routes require authentication.
    enable_metrics:
        When True (default), mount a ``/metrics`` endpoint for Prometheus
        scraping and add request instrumentation middleware.
    cors_origins:
        List of allowed CORS origins.  When ``None``, reads from the
        ``RISKFORGE_CORS_ORIGINS`` environment variable (comma-separated).
        Set to ``["*"]`` for development.  Pass ``[]`` to disable CORS.
    """
    import os

    from fastapi.middleware.cors import CORSMiddleware

    if max_batch_size < 1 or max_batch_size > 32:
        raise ValueError("max_batch_size must be between 1 and 32")

    app = FastAPI(title="RiskForge API", version="1.0.0")
    app.add_middleware(
        RequestControlsMiddleware,
        max_request_bytes=max_request_bytes,
        request_timeout_seconds=request_timeout_seconds,
    )

    # ── Security headers middleware ────────────────────────────────────
    @app.middleware("http")
    async def _security_headers(request: Request, call_next: Any) -> Response:  # type: ignore[type-arg]
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "0"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Cache-Control"] = "no-store"
        return response

    # ── CORS configuration ────────────────────────────────────────────
    if cors_origins is None:
        raw = os.environ.get("RISKFORGE_CORS_ORIGINS", "*")
        cors_origins = [o.strip() for o in raw.split(",") if o.strip()]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        logger.info("cors configured", extra={"origins": cors_origins})

    # Add request-level metrics middleware
    app.add_middleware(RequestMetricsMiddleware)

    # Mount Prometheus /metrics endpoint
    if enable_metrics:
        try:
            from prometheus_client import make_asgi_app

            metrics_app = make_asgi_app()
            app.mount("/metrics", metrics_app)
        except ImportError:
            logger.warning("prometheus-client not installed; /metrics endpoint disabled")

    # Wire the authentication dependency when a service is provided.
    if auth_service is not None:
        def _authenticated_principal(request: Request) -> Principal:
            return _extract_principal(auth_service, request)

        app.dependency_overrides[require_principal] = _authenticated_principal

    @app.exception_handler(_ApiFailure)
    async def handle_api_failure(_request: Request, error: _ApiFailure) -> JSONResponse:
        payload = _error_response(error.correlation_id, error.translated)
        logger.error(
            "api failure",
            extra={
                "error_code": error.translated.code.value,
                "status_code": error.translated.status_code,
                "correlation_id": error.correlation_id,
            },
        )
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

    @app.exception_handler(AuthenticationError)
    async def handle_authentication_error(
        _request: Request, error: AuthenticationError
    ) -> JSONResponse:
        translated = translate_application_error(error)
        logger.warning(
            "authentication error",
            extra={"error_code": translated.code.value},
        )
        payload = _error_response("unavailable", translated)
        return JSONResponse(status_code=401, content=payload.model_dump(mode="json"))

    @app.exception_handler(Exception)
    async def handle_unhandled_exception(
        _request: Request, error: Exception
    ) -> JSONResponse:
        """Global handler for unhandled exceptions — returns safe 500 JSON."""
        logger.error("unhandled exception", exc_info=error)
        translated = TranslatedError(
            status_code=500,
            code=ErrorCode.INTERNAL_ERROR,
            message="internal server error",
            retryable=False,
        )
        payload = _error_response("unavailable", translated)
        return JSONResponse(status_code=500, content=payload.model_dump(mode="json"))

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
        logger.debug(
            "inference request",
            extra={
                "correlation_id": request.correlation_id,
                "endpoint": "/v1/inference",
            },
        )
        try:
            result = application.process_incident(request.incident)
        except Exception as error:
            raise _ApiFailure(request.correlation_id, translate_application_error(error)) from error
        return InferenceResponse(correlation_id=request.correlation_id, result=result)

    @app.post("/v1/inference/batch", response_model=BatchInferenceResponse)
    def infer_batch(request: BatchInferenceRequest) -> BatchInferenceResponse:
        logger.debug(
            "batch inference request",
            extra={
                "correlation_id": request.correlation_id,
                "endpoint": "/v1/inference/batch",
                "batch_size": len(request.incidents),
            },
        )
        if len(request.incidents) > max_batch_size:
            raise _ApiFailure(request.correlation_id, _invalid_request())
        try:
            results = tuple(application.process_batch(request.incidents))
        except Exception as error:
            raise _ApiFailure(request.correlation_id, translate_application_error(error)) from error
        return BatchInferenceResponse(correlation_id=request.correlation_id, results=results)

    @app.get("/v1/assets/{asset_id}/summary", response_model=AssetSummaryResponse)
    def asset_summary(
        asset_id: AssetIdPath, correlation_id: CorrelationHeader
    ) -> AssetSummaryResponse:
        logger.debug(
            "asset summary request",
            extra={"correlation_id": correlation_id, "asset_id": asset_id},
        )
        try:
            summary = application.get_asset_summary(asset_id)
        except Exception as error:
            raise _ApiFailure(correlation_id, translate_application_error(error)) from error
        return AssetSummaryResponse(
            correlation_id=correlation_id,
            summary=summary,
        )

    @app.get("/v1/incidents", response_model=IncidentPageResponse)
    def incident_queue(
        correlation_id: CorrelationHeader,
        offset: PageOffset = 0,
        limit: PageLimit = 50,
        asset_id: Annotated[str | None, Query(min_length=1, max_length=128)] = None,
        asset_type: AssetType | None = None,
        routing: RoutingBucket | None = None,
        timestamp_from: datetime | None = None,
        timestamp_to: datetime | None = None,
    ) -> IncidentPageResponse:
        try:
            query = IncidentQuery(
                asset_id=asset_id,
                asset_type=asset_type,
                routing=routing,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
            )
        except ValidationError as error:
            raise _ApiFailure(correlation_id, _invalid_request()) from error
        try:
            page = application.list_incidents(query, PageRequest(offset=offset, limit=limit))
        except Exception as error:
            raise _ApiFailure(correlation_id, translate_application_error(error)) from error
        return IncidentPageResponse(correlation_id=correlation_id, page=page)

    @app.get("/v1/incidents/critical", response_model=IncidentPageResponse)
    def critical_incidents(
        correlation_id: CorrelationHeader,
        offset: PageOffset = 0,
        limit: PageLimit = 50,
    ) -> IncidentPageResponse:
        try:
            page = application.list_incidents(
                IncidentQuery(routing=RoutingBucket.CRITICAL_ESCALATION),
                PageRequest(offset=offset, limit=limit),
            )
        except Exception as error:
            raise _ApiFailure(correlation_id, translate_application_error(error)) from error
        return IncidentPageResponse(correlation_id=correlation_id, page=page)

    @app.get("/v1/incidents/{log_id}", response_model=IncidentResponse)
    def incident_detail(
        log_id: LogIdPath, correlation_id: CorrelationHeader
    ) -> IncidentResponse:
        try:
            incident = application.get_incident(log_id)
        except Exception as error:
            raise _ApiFailure(correlation_id, translate_application_error(error)) from error
        return IncidentResponse(correlation_id=correlation_id, incident=incident)

    @app.get("/v1/incidents/{log_id}/audit", response_model=AuditPageResponse)
    def audit_history(
        log_id: LogIdPath,
        correlation_id: CorrelationHeader,
        offset: PageOffset = 0,
        limit: PageLimit = 50,
    ) -> AuditPageResponse:
        try:
            page = application.list_audit_events(
                log_id, PageRequest(offset=offset, limit=limit)
            )
        except Exception as error:
            raise _ApiFailure(correlation_id, translate_application_error(error)) from error
        return AuditPageResponse(correlation_id=correlation_id, page=page)

    @app.post("/v1/incidents/{log_id}/reviews", response_model=ReviewDecisionResponse)
    def review_decision(
        log_id: LogIdPath,
        body: ReviewDecisionRequest,
        principal: Principal = Depends(require_principal),  # noqa: B008
    ) -> ReviewDecisionResponse:
        logger.info(
            "review decision request",
            extra={
                "correlation_id": body.correlation_id,
                "log_id": log_id,
                "reviewer_id": principal.subject_id,
                "action": body.action,
            },
        )
        command = ReviewCommand(
            log_id=log_id,
            decision_id=body.decision_id,
            reviewer_id=principal.subject_id,
            action=ReviewAction(body.action),
            reason=body.reason,
        )
        try:
            decision = application.decide_review(command)
        except Exception as error:
            raise _ApiFailure(body.correlation_id, translate_application_error(error)) from error
        return ReviewDecisionResponse(correlation_id=body.correlation_id, decision=decision)

    return app


def _invalid_request() -> TranslatedError:
    return TranslatedError(
        422,
        ErrorCode.INVALID_REQUEST,
        "request validation failed",
        False,
    )
