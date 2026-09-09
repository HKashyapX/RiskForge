"""ASGI middleware for request-level metrics and logging.

The :class:`RequestMetricsMiddleware` instruments every inbound HTTP
request with duration histograms, request counters, error rate counters,
and structured JSON logging via the :mod:`riskforge.logging_config` module.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from riskforge.logging_config import correlation_id_var
from riskforge.observability.metrics import (
    ERRORS_TOTAL,
    REQUEST_DURATION,
    REQUESTS_TOTAL,
)

logger = logging.getLogger("riskforge.api.middleware")

# Endpoints that should not be instrumented (health probes)
_SKIP_LOGGING = frozenset({"/health", "/ready"})


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Collect per-request Prometheus metrics and emit structured logs."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        # Extract or generate correlation ID
        corr_id = request.headers.get("x-correlation-id")
        if corr_id:
            correlation_id_var.set(corr_id)

        method = request.method
        path = request.url.path

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - start

            # Update Prometheus metrics (skip health endpoints to reduce noise)
            if path not in _SKIP_LOGGING:
                REQUEST_DURATION.labels(
                    method=method, endpoint=path, status_code=status_code
                ).observe(duration)
                REQUESTS_TOTAL.labels(
                    method=method, endpoint=path, status_code=status_code
                ).inc()

                # Count errors
                if status_code >= 400:
                    ERRORS_TOTAL.labels(
                        error_code=f"http_{status_code}"
                    ).inc()

            # Structured log for non-health endpoints
            if path not in _SKIP_LOGGING:
                log_data: dict[str, Any] = {
                    "method": method,
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": round(duration * 1000, 2),
                }
                if corr_id:
                    log_data["correlation_id"] = corr_id

                if status_code >= 500:
                    logger.error("request completed", extra=log_data)
                elif status_code >= 400:
                    logger.warning("request completed", extra=log_data)
                elif duration > 0.035:
                    # Log slow requests (> 35ms SLO) at WARNING
                    logger.warning("slow request", extra=log_data)
                else:
                    logger.debug("request completed", extra=log_data)
