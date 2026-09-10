# RiskForge Production Deployment Guide

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Quick Start with Docker Compose](#2-quick-start-with-docker-compose)
3. [Environment Variables Reference](#3-environment-variables-reference)
4. [Database Setup and Migration](#4-database-setup-and-migration)
5. [Authentication Configuration](#5-authentication-configuration)
6. [CORS Configuration](#6-cors-configuration)
7. [Monitoring Setup](#7-monitoring-setup)
8. [Kubernetes Deployment Notes](#8-kubernetes-deployment-notes)
9. [Troubleshooting](#9-troubleshooting)
10. [Security Checklist](#10-security-checklist)

---

## 1. Prerequisites

| Requirement     | Minimum Version | Notes                                      |
|-----------------|-----------------|--------------------------------------------|
| Docker          | 24.0+           | With Compose V2 plugin                     |
| Docker Compose  | 2.20+           | `docker compose` (V2 syntax)               |
| PostgreSQL      | 16              | `postgres:16-alpine` recommended           |
| Python          | 3.11+           | Required only for manual (non-Docker) setup |
| ONNX Runtime    | 1.17+           | CPU-only (`CPUExecutionProvider`)           |
| OS              | Linux (amd64)   | Container runtime; ARM64 not yet tested    |

## 2. Quick Start with Docker Compose

### 2.1 Clone and configure

```bash
git clone <repository-url> RiskForge
cd RiskForge
cp .env.example .env
```

Edit `.env` and set at minimum:

```bash
PGPASSWORD=<strong-random-password>
RISKFORGE_AUTH_ENABLED=true
RISKFORGE_CORS_ORIGINS=https://your-frontend.example.com
```

### 2.2 Start services

```bash
docker compose up -d postgres
docker compose --profile setup run --rm migrations
docker compose up -d api
```

### 2.3 Verify

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -H "X-Correlation-ID: test-001" http://localhost:8000/ready
# {"correlation_id":"test-001","ready":true,"state":"ready",...}
```

### 2.4 Enable monitoring (optional)

```bash
docker compose --profile monitoring up -d
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000  (default password: admin)
```

### 2.5 docker-compose.yml profiles

| Profile    | Services                        | Purpose                          |
|------------|--------------------------------|----------------------------------|
| (default)  | `api`, `postgres`              | Core application stack           |
| `setup`    | `migrations`                   | One-shot database migration      |
| `monitoring` | `prometheus`, `grafana`      | Observability stack              |

---

## 3. Environment Variables Reference

### PostgreSQL

| Variable            | Default     | Description                                   |
|---------------------|-------------|-----------------------------------------------|
| `PGHOST`            | `localhost` | PostgreSQL host                               |
| `PGPORT`            | `5432`      | PostgreSQL port                               |
| `PGDATABASE`        | `riskforge` | Database name                                 |
| `PGUSER`            | `postgres`  | Database user                                 |
| `PGPASSWORD`        | (empty)     | Database password (**must set in production**) |
| `PGCONNECT_TIMEOUT` | `10`        | Connection timeout in seconds                 |
| `PGSSLMODE`         | `prefer`    | TLS mode: `disable`, `allow`, `prefer`, `require`, `verify-ca`, `verify-full` |
| `PGMINPOOL`         | `1`         | Minimum connection pool size                  |
| `PGMAXPOOL`         | `5`         | Maximum connection pool size                  |
| `PGPOOL_TIMEOUT`    | `30`        | Seconds to wait for a pooled connection       |
| `PGSTATEMENT_TIMEOUT_MS` | (unset) | Per-statement timeout in ms (unset = disabled) |

### API Server

| Variable                | Default     | Description                                |
|-------------------------|-------------|--------------------------------------------|
| `RISKFORGE_API_HOST`    | `127.0.0.1` | Bind address (`0.0.0.0` in containers)     |
| `RISKFORGE_API_PORT`    | `8000`      | Listen port                                |
| `RISKFORGE_LOG_LEVEL`   | `INFO`      | Logging level (`DEBUG`, `INFO`, `WARNING`)  |
| `RISKFORGE_ENV`         | `development` | Runtime environment (`development`, `test`, `staging`, `production`) |

### CORS

| Variable                 | Default | Description                                      |
|--------------------------|---------|--------------------------------------------------|
| `RISKFORGE_CORS_ORIGINS` | `*`     | Comma-separated allowed origins, or `*` for all  |

### Authentication

| Variable                 | Default | Description                                      |
|--------------------------|---------|--------------------------------------------------|
| `RISKFORGE_AUTH_ENABLED` | (unset) | Set to any value to enable auth on protected routes |

### ML Serving

| Variable                       | Default | Description                                      |
|--------------------------------|---------|--------------------------------------------------|
| `RISKFORGE_MODEL_PATH`         | —       | Path to the quantized ONNX model file            |
| `RISKFORGE_MANIFEST_PATH`      | —       | Path to model artifact manifest YAML             |
| `RISKFORGE_MAX_BATCH_SIZE`     | `32`    | Maximum inference batch size                     |
| `RISKFORGE_INFERENCE_TIMEOUT_S`| `30`    | Inference timeout in seconds                     |

### Grafana (monitoring profile)

| Variable                       | Default  | Description                           |
|--------------------------------|----------|---------------------------------------|
| `GF_SECURITY_ADMIN_PASSWORD`   | `admin`  | Grafana admin password                |

---

## 4. Database Setup and Migration

### 4.1 Automated (Docker Compose)

```bash
# Run all pending migrations against the running PostgreSQL instance
docker compose --profile setup run --rm migrations
```

This executes `python -m riskforge.persistence.postgres.migrate` which:

- Discovers `.sql` files in `src/riskforge/persistence/postgres/migrations/` (sorted lexicographically)
- Creates a `schema_migrations` tracking table on first run
- Applies each pending migration in a single transaction
- Records a SHA-256 checksum and timestamp for each applied migration
- Skips already-applied migrations on re-run

### 4.2 Manual (non-Docker)

```bash
export PGHOST=localhost PGPORT=5432 PGDATABASE=riskforge
export PGUSER=postgres PGPASSWORD=<password>

python -m riskforge.persistence.postgres.migrate
```

Or programmatically:

```python
from riskforge.persistence.postgres.migrate import run_migrations

applied = run_migrations(dsn="host=localhost dbname=riskforge user=postgres password=secret")
print(f"Applied {applied} migrations")
```

### 4.3 Schema tables

| Table              | Purpose                                          |
|--------------------|--------------------------------------------------|
| `incident_results` | Automated inference results (immutable after creation) |
| `review_decisions` | Human review decisions (append-only)             |
| `audit_events`     | Append-only audit trail                          |

### 4.4 Creating a new migration

1. Add a new `.sql` file to `src/riskforge/persistence/postgres/migrations/`
2. Use a lexicographically sortable prefix: `002_add_table.sql`
3. Use idempotent DDL (`IF NOT EXISTS`, `IF EXISTS`) where possible
4. The migration runner will detect and apply it on the next run

---

## 5. Authentication Configuration

### 5.1 Overview

RiskForge uses a pluggable `AuthenticationService` protocol. Authentication is optional — when no auth service is provided, protected routes are inaccessible.

Protected endpoint: `POST /v1/incidents/{id}/reviews`

### 5.2 Development mode (no real auth)

Set `RISKFORGE_ENV=development` (default) and pass `DevAuthenticationService` to `create_app()`. This trusts a single static principal (`dev-user`) and ignores the credential value.

```python
from riskforge.authentication.dev import DevAuthenticationService
from riskforge.api.app import create_app

auth = DevAuthenticationService(subject_id="dev-user")
app = create_app(application, readiness, auth_service=auth)
```

> **Warning:** `DevAuthenticationService` raises `RuntimeError` if `RISKFORGE_ENV` is `production`.

### 5.3 Implementing a custom auth provider

Implement the `AuthenticationService` protocol:

```python
from riskforge.authentication.protocols import AuthenticationService
from riskforge.authentication.principal import Principal
from riskforge.authentication.exceptions import InvalidCredentialsError

class JWTAuthenticationService:
    def __init__(self, jwks_url: str, audience: str) -> None:
        self._jwks_url = jwks_url
        self._audience = audience

    def authenticate(self, credential: str) -> Principal:
        """Verify a JWT bearer token and return the authenticated Principal."""
        try:
            payload = verify_jwt(credential, self._jwks_url, self._audience)
        except Exception:
            raise InvalidCredentialsError("invalid or expired token")
        return Principal(subject_id=payload["sub"])
```

Wire it into the application:

```python
auth = JWTAuthenticationService(jwks_url="https://idp.example.com/.well-known/jwks.json", audience="riskforge")
app = create_app(application, readiness, auth_service=auth)
```

### 5.4 Principal model

The `Principal` is an immutable Pydantic model with a single field:

| Field        | Type   | Constraints          | Description                        |
|--------------|--------|----------------------|------------------------------------|
| `subject_id` | `str`  | 1–128 chars          | Opaque authenticated subject ID    |

### 5.5 Authentication errors

| Exception                | HTTP Status | Description                         |
|--------------------------|-------------|-------------------------------------|
| `MissingCredentialsError`| 401         | No `Authorization` header present   |
| `InvalidCredentialsError`| 401         | Token is invalid or expired         |

---

## 6. CORS Configuration

Set `RISKFORGE_CORS_ORIGINS` to a comma-separated list of allowed origins:

```bash
# Allow specific origins
RISKFORGE_CORS_ORIGINS=https://app.example.com,https://admin.example.com

# Allow all origins (development only)
RISKFORGE_CORS_ORIGINS=*
```

In production, always specify explicit origins rather than `*`.

---

## 7. Monitoring Setup

### 7.1 Prometheus

Start the monitoring profile:

```bash
docker compose --profile monitoring up -d prometheus
```

Prometheus will scrape `http://api:8000/metrics` by default. A sample `config/prometheus.yml` should be mounted into the Prometheus container.

**Key metrics exposed at `/metrics`:**

| Metric                                    | Type      | Description                              |
|-------------------------------------------|-----------|------------------------------------------|
| `riskforge_request_duration_seconds`      | Histogram | HTTP request latency                     |
| `riskforge_requests_total`                | Counter   | Total HTTP requests                      |
| `riskforge_errors_total`                  | Counter   | Total errors by error code               |
| `riskforge_inference_latency_seconds`     | Histogram | ONNX Runtime inference latency           |
| `riskforge_inferences_total`              | Counter   | Total inference requests                 |
| `riskforge_batch_size`                    | Histogram | Batch size distribution                  |
| `riskforge_inference_queue_depth`         | Gauge     | Pending requests in batcher queue        |
| `riskforge_inference_failures_total`      | Counter   | Inference failures by error type         |
| `riskforge_component_ready`               | Gauge     | Component readiness (1=ready, 0=not)     |
| `riskforge_startup_duration_seconds`      | Gauge     | Total startup time                       |
| `riskforge_auth_failures_total`           | Counter   | Authentication failures                  |
| `riskforge_uptime_seconds`                | Gauge     | Process uptime                           |

### 7.2 Grafana

```bash
docker compose --profile monitoring up -d grafana
```

Access at `http://localhost:3000`. Default credentials: `admin` / `admin` (change via `GF_SECURITY_ADMIN_PASSWORD`).

Add the Prometheus data source pointing to `http://prometheus:9090`.

### 7.3 Health endpoints

| Endpoint  | Method | Auth Required | Purpose                    |
|-----------|--------|---------------|----------------------------|
| `/health` | GET    | No            | Liveness probe (always 200)|
| `/ready`  | GET    | No            | Readiness probe (503 if not ready) |
| `/metrics`| GET    | No            | Prometheus scrape target   |

---

## 8. Kubernetes Deployment Notes

### 8.1 Resource limits

Based on the Dockerfile and runtime configuration:

```yaml
resources:
  requests:
    cpu: "1.0"
    memory: "1Gi"
  limits:
    cpu: "2.0"
    memory: "2Gi"
```

The ONNX Runtime session is configured with `intra_op_num_threads=4` and `inter_op_num_threads=1`. Do not allocate more than 4 CPU cores unless the ONNX session options are adjusted.

### 8.2 Health probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
    httpHeaders:
      - name: X-Correlation-ID
        value: "readiness-probe"
  initialDelaySeconds: 15
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3
```

### 8.3 Signal handling

RiskForge registers `SIGTERM` and `SIGINT` handlers for graceful shutdown. The `RuntimeManager` stops components in reverse start order with a configurable timeout (`shutdown_timeout_seconds`, default 30s).

Ensure your Kubernetes deployment uses `terminationGracePeriodSeconds: 60` or higher to allow the process to drain in-flight requests and close the database pool.

### 8.4 Security context

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: ["ALL"]
```

### 8.5 Pod disruption budget

For production workloads, define a PDB:

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: riskforge-pdb
spec:
  minAvailable: 1
  selector:
    matchLabels:
      app: riskforge
```

---

## 9. Troubleshooting

### 9.1 Model not found

**Symptom:** `ArtifactLoadingError: ONNX model not found: /data/models/sif_int8.onnx`

**Fix:**
- Verify the model file exists at the path specified in `RISKFORGE_MODEL_PATH`
- Ensure the manifest file (`RISKFORGE_MANIFEST_PATH`) is in the same directory
- Check file permissions — the container runs as user `riskforge` (UID 1000)
- Mount the model directory as a volume in `docker-compose.yml`:
  ```yaml
  volumes:
    - ./models:/data/models:ro
  ```

### 9.2 Database connection refused

**Symptom:** `PersistenceConnectionError: cannot create PostgreSQL connection pool`

**Fix:**
- Verify PostgreSQL is running: `docker compose ps postgres`
- Check health status: `docker compose exec postgres pg_isready`
- Verify `PGHOST`, `PGPORT`, `PGDATABASE`, `PGUSER`, `PGPASSWORD` match
- Ensure the `migrations` profile has been run at least once
- Check circuit breaker state in logs — after 5 consecutive failures, connections are refused for 30 seconds

### 9.3 Authentication failures

**Symptom:** 401 responses on `/v1/incidents/{id}/reviews`

**Fix:**
- Ensure the `Authorization: Bearer <token>` header is present
- If using `DevAuthenticationService`, verify `RISKFORGE_ENV` is not `production`
- If using a custom provider, verify the token is valid and not expired
- Check logs for `authentication error` entries with the `error_code` field

### 9.4 Readiness probe returns 503

**Symptom:** `/ready` returns HTTP 503 with `ready: false`

**Fix:**
- Check which components report not-ready in the `components` array
- Common causes: PostgreSQL pool not initialized, ONNX model not loaded
- Review startup logs for component initialization errors
- Ensure migrations have been applied before starting the API service

### 9.5 Slow inference

**Symptom:** Inference latency exceeds 35ms SLO

**Fix:**
- Verify `intra_op_num_threads=4` in ONNX session options (do not exceed CPU limit)
- Check `riskforge_inference_queue_depth` metric — high values indicate saturation
- Reduce `RISKFORGE_MAX_BATCH_SIZE` if memory is constrained
- Verify the model is INT8 quantized, not FP32

### 9.6 Circuit breaker open

**Symptom:** Logs show `circuit breaker opened after N consecutive failures`

**Fix:**
- The PostgreSQL circuit breaker opens after 5 consecutive failures and waits 30 seconds
- Check PostgreSQL health and connectivity
- Review connection pool settings (`PGMINPOOL`, `PGMAXPOOL`)
- After the cooldown period, the breaker transitions to half-open and retries automatically

---

## 10. Security Checklist

Before deploying to production, verify:

- [ ] **Change default passwords** — Set a strong `PGPASSWORD` (not the default `postgres`)
- [ ] **Enable authentication** — Set `RISKFORGE_AUTH_ENABLED=true` and provide a production `AuthenticationService`
- [ ] **Restrict CORS** — Set `RISKFORGE_CORS_ORIGINS` to explicit allowed origins (never `*`)
- [ ] **Non-root container** — The Dockerfile already runs as user `riskforge` (UID 1000); verify Kubernetes `securityContext.runAsNonRoot: true`
- [ ] **Read-only filesystem** — Set `readOnlyRootFilesystem: true` in container security context
- [ ] **No secrets in logs** — The logging config automatically masks `password`, `dsn`, `authorization`, `token`, `secret`, `api_key` fields
- [ ] **Enable TLS** — Use a reverse proxy (nginx, Traefik) or cloud load balancer for TLS termination
- [ ] **Restrict database access** — PostgreSQL should only be accessible from the API service network
- [ ] **Rotate credentials** — Implement a credential rotation schedule for `PGPASSWORD` and auth tokens
- [ ] **Set `RISKFORGE_ENV=production`** — Disables development-only features (`DevAuthenticationService`)
- [ ] **Review audit logs** — Verify `audit_events` table is being populated
- [ ] **Monitor metrics** — Set up alerts on `riskforge_errors_total`, `riskforge_auth_failures_total`, and `riskforge_inference_failures_total`
