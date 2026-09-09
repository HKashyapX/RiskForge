"""Tests for Prometheus metrics and request instrumentation middleware."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from riskforge.observability.middleware import RequestMetricsMiddleware


class TestMetricsExist:
    """Verify that all expected Prometheus metrics are registered."""

    def test_request_duration_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_request_duration_seconds")
        assert metric is not None

    def test_requests_total_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_requests_total")
        assert metric is not None

    def test_errors_total_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_errors_total")
        assert metric is not None

    def test_inference_latency_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_inference_latency_seconds")
        assert metric is not None

    def test_inference_total_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_inferences_total")
        assert metric is not None

    def test_batch_size_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_batch_size")
        assert metric is not None

    def test_inference_queue_depth_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_inference_queue_depth")
        assert metric is not None

    def test_inference_failures_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_inference_failures_total")
        assert metric is not None

    def test_component_ready_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_component_ready")
        assert metric is not None

    def test_startup_duration_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_startup_duration_seconds")
        assert metric is not None

    def test_auth_failures_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_auth_failures_total")
        assert metric is not None

    def test_uptime_registered(self) -> None:
        metric = REGISTRY._names_to_collectors.get("riskforge_uptime_seconds")
        assert metric is not None


class TestRequestMetricsMiddleware:
    def test_instruments_successful_request(self) -> None:
        app = FastAPI()
        app.add_middleware(RequestMetricsMiddleware)

        @app.get("/test")
        def test_endpoint() -> dict[str, str]:
            return {"status": "ok"}

        client = TestClient(app)
        response = client.get("/test", headers={"X-Correlation-ID": "test-corr-123"})
        assert response.status_code == 200

    def test_instruments_error_request(self) -> None:
        app = FastAPI()
        app.add_middleware(RequestMetricsMiddleware)

        @app.get("/error")
        def error_endpoint() -> None:
            raise HTTPException(status_code=500, detail="test error")

        client = TestClient(app)
        response = client.get("/error", headers={"X-Correlation-ID": "err-corr-456"})
        assert response.status_code == 500

    def test_skips_health_endpoint(self) -> None:
        app = FastAPI()
        app.add_middleware(RequestMetricsMiddleware)

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "live"}

        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_correlation_id_propagation(self) -> None:
        app = FastAPI()
        app.add_middleware(RequestMetricsMiddleware)

        @app.get("/test")
        def test_endpoint() -> dict[str, str]:
            from riskforge.logging_config import correlation_id_var
            return {"correlation_id": correlation_id_var.get() or "none"}

        client = TestClient(app)
        response = client.get("/test", headers={"X-Correlation-ID": "my-corr-id"})
        assert response.status_code == 200
        assert response.json()["correlation_id"] == "my-corr-id"


class TestMetricsEndpoint:
    def test_metrics_endpoint_available(self) -> None:
        from riskforge.api.app import create_app
        from riskforge.runtime.contracts import LifecycleState, ReadinessState, RuntimeStatus

        class _FakeBackend:
            pass

        class _FakeReadiness:
            def snapshot(self) -> RuntimeStatus:
                return RuntimeStatus(
                    lifecycle=LifecycleState.READY,
                    readiness=ReadinessState.READY,
                    checked_at=None,
                    components=(),
                )

        app = create_app(
            application=_FakeBackend(),  # type: ignore[arg-type]
            readiness=_FakeReadiness(),  # type: ignore[arg-type]
            enable_metrics=True,
        )

        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
        # Prometheus metrics endpoint returns text/plain by default
        assert "riskforge_" in response.text or "python_" in response.text

    def test_metrics_endpoint_disabled(self) -> None:
        from riskforge.api.app import create_app
        from riskforge.runtime.contracts import LifecycleState, ReadinessState, RuntimeStatus

        class _FakeBackend:
            pass

        class _FakeReadiness:
            def snapshot(self) -> RuntimeStatus:
                return RuntimeStatus(
                    lifecycle=LifecycleState.READY,
                    readiness=ReadinessState.READY,
                    checked_at=None,
                    components=(),
                )

        app = create_app(
            application=_FakeBackend(),  # type: ignore[arg-type]
            readiness=_FakeReadiness(),  # type: ignore[arg-type]
            enable_metrics=False,
        )

        client = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code in (404, 405)  # endpoint not mounted
