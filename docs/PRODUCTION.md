# RiskForge Production Configuration and Operations Guide

## Table of Contents

1. [Configuration Hierarchy](#1-configuration-hierarchy)
2. [Logging](#2-logging)
3. [Metrics](#3-metrics)
4. [Health and Readiness Probes](#4-health-and-readiness-probes)
5. [Resilience Patterns](#5-resilience-patterns)
6. [Resource Requirements](#6-resource-requirements)
7. [Backup and Recovery](#7-backup-and-recovery)
8. [Scaling Considerations](#8-scaling-considerations)
9. [Security Hardening Checklist](#9-security-hardening-checklist)

---

## 1. Configuration Hierarchy

RiskForge resolves configuration in this order (highest priority first):

1. **Explicit constructor arguments** — passed directly to `RuntimeSettings` or component constructors
2. **Environment variables** — `RISKFORGE_*` and `PG*` prefixed variables
3. **Defaults** — hardcoded in `RuntimeSettings`, `PostgresConfig`, and component factories

### RuntimeSettings model

All non-secret settings are validated through a frozen Pydantic `RuntimeSettings` model:

| Field                       | Type    | Default      | Constraints              | Description                           |
|-----------------------------|---------|--------------|--------------------------|---------------------------------------|
| `environment`               | `enum`  | `development`| One of: `development`, `test`, `staging`, `production` | Runtime environment |
| `api_host`                  | `str`   | `127.0.0.1`  | 1–255 chars              | API bind address                      |
| `api_port`                  | `int`   | `8000`       | 1–65535                  | API listen port                       |
| `model_path`                | `Path`  | —            | —                        | Path to ONNX model file               |
| `manifest_path`             | `Path`  | —            | —                        | Path to model manifest YAML           |
| `max_request_bytes`         | `int`   | `1048576`    | 1–16777216               | Maximum request body size             |
| `max_batch_size`            | `int`   | `32`         | 1–32                     | Maximum inference batch size          |
| `request_timeout_seconds`   | `float` | `30.0`       | >0, ≤300                 | Per-request timeout                   |
| `shutdown_timeout_seconds`  | `float` | `30.0`       | >0, ≤300                 | Graceful shutdown timeout             |
| `inference_queue_size`      | `int`   | `256`        | 1–10000                  | Batched request queue capacity        |
| `inference_queue_delay_ms`  | `float` | `5.0`        | 0–1000                   | Max wait time before batch executes   |

### Config files

Three YAML configuration files in `config/`:

| File                        | Purpose                                         |
|-----------------------------|-------------------------------------------------|
| `gazetteer_rules.yaml`      | Entity normalization patterns (barriers, assets, hazards) |
| `model_hyperparams.yaml`    | Model backbone, sequence length, calibration settings   |
| `threshold_policy.yaml`     | Decision boundaries, critical energy thresholds, runtime constraints |

These are loaded at application startup and are read-only during runtime.

---

## 2. Logging

### 2.1 Structured JSON format

All log output is structured JSON written to stderr. Each entry contains:

```json
{
  "timestamp": "2026-09-09T14:32:01.123456+00:00",
  "level": "INFO",
  "logger": "riskforge.api.app",
  "message": "runtime started",
  "correlation_id": "req-abc-123",
  "extra": {
    "components": 4,
    "startup_duration_s": 1.234
  }
}
```

### 2.2 Log levels

Controlled by `RISKFORGE_LOG_LEVEL` (default: `INFO`).

| Level     | Usage                                                      |
|-----------|------------------------------------------------------------|
| `DEBUG`   | Inference request details, per-request completion logs     |
| `INFO`    | Component startup/shutdown, migration progress, pool creation |
| `WARNING` | Slow requests (>35ms), circuit breaker opens, auth failures|
| `ERROR`   | Unhandled exceptions, inference failures, shutdown failures|

### 2.3 Correlation IDs

Every request can carry an `X-Correlation-ID` header (1–128 chars, pattern `^[A-Za-z0-9._:-]+$`). The middleware:

1. Extracts the header value (or generates one if absent)
2. Sets it on a `ContextVar` (`correlation_id_var`)
3. The JSON formatter injects it into every log entry for that request

This enables end-to-end tracing across a single request lifecycle.

### 2.4 Sensitive field masking

The logging config automatically redacts these keys in all log entries:

- `password`, `dsn`, `authorization`, `token`, `secret`, `api_key`, `apikey`

Values matching `password=\S+` or `Bearer\s+\S+` patterns are masked with `***`.

### 2.5 Suppressed loggers

The following third-party loggers are set to `WARNING` to reduce noise:

- `httpcore`, `httpx`, `uvicorn.access`, `psycopg_pool`

---

## 3. Metrics

### 3.1 Endpoint

Prometheus metrics are exposed at `GET /metrics` (enabled by default via `enable_metrics=True` in `create_app()`).

### 3.2 Request metrics

| Metric                                    | Type      | Labels                      | Description                        |
|-------------------------------------------|-----------|-----------------------------|------------------------------------|
| `riskforge_request_duration_seconds`      | Histogram | `method`, `endpoint`, `status_code` | HTTP request latency (seconds) |
| `riskforge_requests_total`                | Counter   | `method`, `endpoint`, `status_code` | Total HTTP requests           |
| `riskforge_errors_total`                  | Counter   | `error_code`                | Total errors by error code         |

Latency histogram buckets: 5ms, 10ms, 25ms, 50ms, 100ms, 250ms, 500ms, 1s, 2.5s, 5s, 10s.

### 3.3 Inference metrics

| Metric                                    | Type      | Labels             | Description                        |
|-------------------------------------------|-----------|--------------------|------------------------------------|
| `riskforge_inference_latency_seconds`     | Histogram | `mode`             | ONNX Runtime inference latency     |
| `riskforge_inferences_total`              | Counter   | `routing_bucket`   | Total inference requests           |
| `riskforge_batch_size`                    | Histogram | —                  | Batch size distribution            |
| `riskforge_inference_queue_depth`         | Gauge     | —                  | Pending requests in batcher queue  |
| `riskforge_inference_failures_total`      | Counter   | `error_type`       | Inference failures by type         |

### 3.4 Component metrics

| Metric                                    | Type      | Labels         | Description                        |
|-------------------------------------------|-----------|----------------|------------------------------------|
| `riskforge_component_ready`               | Gauge     | `component`    | Component readiness (1 or 0)       |
| `riskforge_startup_duration_seconds`      | Gauge     | —              | Total startup time                 |
| `riskforge_uptime_seconds`                | Gauge     | —              | Process uptime                     |

### 3.5 Authentication metrics

| Metric                                    | Type      | Labels         | Description                        |
|-------------------------------------------|-----------|----------------|------------------------------------|
| `riskforge_auth_failures_total`           | Counter   | `reason`       | Authentication failures by reason  |

---

## 4. Health and Readiness Probes

### 4.1 Liveness: `GET /health`

- **Always returns 200** with `{"status": "ok"}`
- Used by orchestrators to detect if the process is alive
- Does not check backend dependencies
- Not instrumented by metrics middleware (excluded from `/metrics` counters)

### 4.2 Readiness: `GET /ready`

- Returns **200** when the runtime is fully operational
- Returns **503** when any component reports not-ready
- Requires `X-Correlation-ID` header
- Response includes per-component status:

```json
{
  "correlation_id": "probe-001",
  "ready": true,
  "state": "ready",
  "checked_at": "2026-09-09T14:32:01.123456+00:00",
  "components": ["postgres_pool", "onnx_engine", "batcher"]
}
```

### 4.3 Readiness composition

Readiness is `true` only when:
1. The `RuntimeManager` lifecycle state is `READY`
2. Every registered `LifecycleComponent` reports `ready=True`

If any component throws during its readiness check, it is treated as not-ready with a safe error detail.

---

## 5. Resilience Patterns

### 5.1 Circuit breaker (PostgreSQL)

| Parameter               | Value     | Description                                |
|-------------------------|-----------|--------------------------------------------|
| Failure threshold       | 5         | Consecutive failures before opening        |
| Cooldown period         | 30s       | Time before half-open retry                |
| Retry attempts          | 2         | Exponential backoff: 0.1s, 0.2s            |

When the circuit is open, all connection requests fail immediately with `PersistenceConnectionError`. After the cooldown, the next request probes the connection (half-open state).

### 5.2 Inference timeout

| Parameter               | Default   | Configurable via                          |
|-------------------------|-----------|-------------------------------------------|
| Inference timeout       | 30s       | `RISKFORGE_INFERENCE_TIMEOUT_S` or `inference_timeout_s` |

Inference runs in a dedicated `ThreadPoolExecutor` (single thread). If it exceeds the timeout, the request fails with `InferenceFailureError` and the consecutive failure counter increments.

### 5.3 Degradation tracking

The ONNX inference engine tracks consecutive failures. After 5 consecutive failures (`_DEGRADED_THRESHOLD`), `engine.is_degraded` returns `True`. A single successful inference resets the counter.

### 5.4 Graceful shutdown

The `RuntimeManager` handles `SIGTERM` and `SIGINT`:

1. Stops accepting new work
2. Shuts down components in reverse start order
3. Each component gets `shutdown_timeout_seconds` (default 30s) to clean up
4. If a component times out, it is logged and the next component is attempted
5. The process exits with code 1 if any component fails to stop cleanly

### 5.5 Request-level resilience

| Pattern             | Implementation                                    |
|---------------------|---------------------------------------------------|
| Timeout             | Per-request timeout in `RuntimeSettings`          |
| Retry               | Connection-level exponential backoff (0.1s, 0.2s) |
| Circuit breaker     | PostgreSQL pool with 5-failure threshold          |
| Bounded queue       | `inference_queue_size` limits pending requests    |
| Batch coalescing    | `ConcurrentRequestBatcher` groups requests by shape |

### 5.6 Batch request queue

The `ConcurrentRequestBatcher` coalesces concurrent single-record requests:

- **Queue capacity:** `inference_queue_size` (default 256)
- **Max wait:** `inference_queue_delay_ms` (default 5ms) — batch executes after this delay or when full
- **Shape matching:** Only requests with identical tensor shapes are batched together
- **Shutdown:** Drains the queue, optionally cancelling pending requests

When the queue is full, new requests receive `RequestQueueFullError` immediately.

---

## 6. Resource Requirements

### 6.1 CPU

| Component             | Cores | Notes                                      |
|-----------------------|-------|--------------------------------------------|
| ONNX Runtime          | 4     | `intra_op_num_threads=4`, `inter_op_num_threads=1` |
| API server            | 0.5   | Uvicorn with single worker                 |
| Connection pool       | 0.1   | `psycopg_pool` background threads          |
| **Total recommended** | **2** | Matches Docker Compose resource limit      |

### 6.2 Memory

| Component             | Typical | Notes                                      |
|-----------------------|---------|--------------------------------------------|
| ONNX model (INT8)     | 200–500MB | Depends on backbone (deberta-v3-base)   |
| Python runtime        | 200MB  | FastAPI + dependencies                     |
| Connection pool       | 50MB   | 5 connections × ~10MB each                 |
| Request buffers       | 100MB  | Batched tensor data                        |
| **Total recommended** | **1–2GB** | Docker Compose limit: 2G              |

### 6.3 Disk

| Path                  | Size    | Notes                                      |
|-----------------------|---------|--------------------------------------------|
| ONNX model file       | 100–300MB | INT8 quantized DeBERTa-v3-base           |
| Model manifest + checksums | 1KB | YAML artifact manifest                     |
| PostgreSQL data       | Varies | Depends on incident volume and retention   |
| Container image       | 500MB  | Python 3.11-slim + ONNX Runtime            |

### 6.4 ONNX Runtime session configuration

```python
options.intra_op_num_threads = 4
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
providers = ["CPUExecutionProvider"]
```

Do not change these values without re-validating the 35ms latency SLO.

---

## 7. Backup and Recovery

### 7.1 PostgreSQL backup

```bash
# Logical backup (recommended for small-to-medium databases)
docker compose exec postgres pg_dump -U postgres riskforge > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore
cat backup_20260909_143200.sql | docker compose exec -T postgres psql -U postgres riskforge
```

### 7.2 Automated backups

For production, schedule regular backups:

```bash
# Cron job: daily backup at 2 AM
0 2 * * * docker compose exec -T postgres pg_dump -U postgres riskforge | gzip > /backups/riskforge_$(date +\%Y\%m\%d).sql.gz
```

### 7.3 Backup retention

| Retention   | Strategy                                       |
|-------------|------------------------------------------------|
| Daily       | Keep 7 days                                    |
| Weekly      | Keep 4 weeks                                   |
| Monthly     | Keep 12 months                                 |

### 7.4 Recovery procedure

1. Stop the API service: `docker compose stop api`
2. Restore the database from backup
3. Run migrations: `docker compose --profile setup run --rm migrations`
4. Start the API: `docker compose start api`
5. Verify readiness: `curl http://localhost:8000/ready`

### 7.5 Point-in-time recovery

For WAL-based PITR, configure PostgreSQL archiving in your deployment:

```bash
# Enable WAL archiving in postgresql.conf
archive_mode = on
archive_command = 'cp %p /archive/%f'
```

---

## 8. Scaling Considerations

### 8.1 Horizontal scaling

RiskForge is a stateful monolith due to:
- In-process ONNX Runtime session (not cluster-safe)
- Connection pool bound to a single PostgreSQL instance
- Batch request coalescer with local queue

**Recommended approach:** Scale vertically first. If horizontal scaling is required:

1. **Multiple replicas** behind a load balancer — each replica runs its own ONNX session
2. **Shared PostgreSQL** — all replicas connect to the same database
3. **Sticky sessions** — route batch requests to the same replica when possible
4. **Read replicas** — PostgreSQL read replicas for query-heavy endpoints (`/v1/incidents`, `/v1/assets/{id}/summary`)

### 8.2 Vertical scaling

| Resource        | Current | Max Recommended | Notes                        |
|-----------------|---------|-----------------|------------------------------|
| CPU             | 2 cores | 8 cores         | Must update `intra_op_num_threads` |
| Memory          | 2GB     | 8GB             | Larger batch sizes possible  |
| Connection pool | 5       | 20              | Increase `PGMAXPOOL`         |

### 8.3 PostgreSQL tuning

```ini
# postgresql.conf for production
shared_buffers = 512MB
effective_cache_size = 1536MB
work_mem = 16MB
maintenance_work_mem = 128MB
max_connections = 100
```

### 8.4 Connection pool sizing

| Pool Size | Use Case                               |
|-----------|----------------------------------------|
| 1–5       | Single replica, moderate load          |
| 5–10      | Single replica, high concurrency       |
| 10–20     | Multiple replicas, shared PostgreSQL   |

Formula: `PGMAXPOOL = (number_of_replicas × concurrent_requests_per_replica) + headroom`

---

## 9. Security Hardening Checklist

### 9.1 Container security

- [ ] Run as non-root user (Dockerfile sets `USER riskforge`, UID 1000)
- [ ] Read-only root filesystem (`readOnlyRootFilesystem: true`)
- [ ] Drop all Linux capabilities (`capabilities.drop: ["ALL"]`)
- [ ] No privilege escalation (`allowPrivilegeEscalation: false`)
- [ ] Use specific base image tags, not `latest`
- [ ] Scan images for vulnerabilities (Trivy, Snyk)

### 9.2 Network security

- [ ] PostgreSQL not exposed to public internet (bind to container network only)
- [ ] API behind TLS-terminating reverse proxy
- [ ] CORS restricted to known origins
- [ ] Rate limiting at reverse proxy level
- [ ] Network policies in Kubernetes (if applicable)

### 9.3 Credential management

- [ ] `PGPASSWORD` set via secrets manager, not hardcoded in files
- [ ] Auth tokens rotated regularly
- [ ] No credentials in logs (automatically masked by logging config)
- [ ] No credentials in Docker Compose files — use `.env` or Docker secrets
- [ ] `RISKFORGE_ENV=production` set in all production environments

### 9.4 Data security

- [ ] PostgreSQL encryption at rest (provider-managed or LUKS)
- [ ] TLS for PostgreSQL connections (`sslmode=require` in DSN)
- [ ] Backup encryption
- [ ] Incident narratives not exposed in API error responses (enforced by architecture)

### 9.5 Runtime security

- [ ] `DevAuthenticationService` blocked in production (`RISKFORGE_ENV=production`)
- [ ] Authentication enabled on protected routes
- [ ] Audit trail enabled (append-only `audit_events` table)
- [ ] Metrics endpoint not publicly accessible (or behind auth)
- [ ] Health endpoints not rate-limited (needed for orchestrator probes)

### 9.6 Supply chain security

- [ ] Pin Python dependency versions in `pyproject.toml`
- [ ] Verify ONNX model checksum against manifest before loading (`validate_model_checksum`)
- [ ] Use `pip install --no-cache-dir` in Docker builds
- [ ] Multi-stage Docker builds to reduce attack surface
- [ ] No development dependencies in production image
