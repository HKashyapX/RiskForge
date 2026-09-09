"""Prometheus metric definitions for RiskForge operational observability.

All metrics follow the ``riskforge_`` namespace prefix and are registered
with the default Prometheus registry.  Import this module after calling
``configure_logging()`` to ensure metrics are available for scraping.

The middleware module (:mod:`riskforge.observability.middleware`) updates
these metrics automatically.  The serving engine and runtime lifecycle
update their respective metrics directly.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# ── Request-level metrics ──────────────────────────────────────────────

REQUEST_DURATION = Histogram(
    "riskforge_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint", "status_code"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

REQUESTS_TOTAL = Counter(
    "riskforge_requests_total",
    "Total HTTP requests processed",
    ["method", "endpoint", "status_code"],
)

ERRORS_TOTAL = Counter(
    "riskforge_errors_total",
    "Total errors by error code",
    ["error_code"],
)

# ── Inference metrics ──────────────────────────────────────────────────

INFERENCE_LATENCY = Histogram(
    "riskforge_inference_latency_seconds",
    "ONNX Runtime inference latency per batch",
    ["mode"],
    buckets=(0.005, 0.01, 0.015, 0.020, 0.025, 0.030, 0.035, 0.050, 0.100),
)

INFERENCE_TOTAL = Counter(
    "riskforge_inferences_total",
    "Total inference requests processed",
    ["routing_bucket"],
)

BATCH_SIZE = Histogram(
    "riskforge_batch_size",
    "Batch size distribution for batch inference",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128),
)

INFERENCE_QUEUE_DEPTH = Gauge(
    "riskforge_inference_queue_depth",
    "Current number of pending inference requests in the batcher queue",
)

INFERENCE_FAILURES = Counter(
    "riskforge_inference_failures_total",
    "Total inference failures by error type",
    ["error_type"],
)

# ── Component / readiness metrics ─────────────────────────────────────

COMPONENT_READY = Gauge(
    "riskforge_component_ready",
    "Whether a runtime component is ready (1) or not (0)",
    ["component"],
)

STARTUP_DURATION = Gauge(
    "riskforge_startup_duration_seconds",
    "Time taken to start all runtime components",
)

# ── Authentication metrics ────────────────────────────────────────────

AUTH_FAILURES = Counter(
    "riskforge_auth_failures_total",
    "Total authentication failures",
    ["reason"],
)

# ── Process metrics ───────────────────────────────────────────────────

UPTIME = Gauge(
    "riskforge_uptime_seconds",
    "Process uptime in seconds since last start",
)
