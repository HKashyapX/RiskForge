# RiskForge

**AI/NLP engine to detect Serious Injury & Fatality (SIF) precursors in OIL's Unsafe-Act/Unsafe-Condition and Near-Miss reports** — SIH26165, Oil India Limited, Smart Automation.

RiskForge converts unstructured safety narratives into actionable SIF intelligence: normalization of oilfield terminology, SIF-precursor probability scoring, IOGP Life-Saving-Rule matching, activity/asset/failed-barrier extraction, human-in-the-loop review, risk aggregation, and an operational dashboard.

## Architecture

```
Report (CSV/JSON/JSONL/XLSX/PDF/API)
  └─► ingestion          parse + validate raw reports
  └─► normalization      SpanPreservingGazetteer (config/gazetteer_rules.yaml)
  └─► serving            ONNX Runtime SIF scoring + postprocessing (CPU-only)
       fallback: deterministic HeuristicRuleEngine (weak-supervision LFs)
  └─► application        orchestration: inference, workflow, review
  └─► metrics            SPD scoring, recurrent barrier analysis
  └─► persistence        SQLite (default) / PostgreSQL (resilient pool)
  └─► api                FastAPI: inference, incidents, reviews, ingest, analytics
  └─► runtime            lifecycle composition, readiness, graceful shutdown
  └─► observability      Prometheus metrics, request correlation
```

Subsystems enforce import boundaries (verified in CI by `scripts/lint_boundaries.py`). Core contracts in `src/riskforge/core/contracts.py` are frozen; design decisions live in `docs/decisions/`.

## Honest capability statement

| Capability | Status |
|---|---|
| Report ingestion (CSV/JSON/JSONL/XLSX/PDF) | Real — `riskforge.ingestion` |
| Oilfield gazetteer normalization (span-preserving) | Real, config-driven |
| SIF precursor scoring | Real via ONNX Runtime **when a trained artifact is present**; otherwise a deterministic, explainable heuristic engine (no fabricated ML) |
| IOGP Life-Saving-Rule matching, triad extraction | Real (deterministic) |
| Risk routing + explainability | Real — every result carries evidence spans and matched rules |
| Human-in-the-loop review + audit trail | Real |
| Aggregation (SPD, recurring barriers) | Real |
| Trends / emerging risk / patterns | Real — analytics service |
| Preventive recommendations | Deterministic mapping from hazards + failed barriers (supports, never replaces, HSE judgement) |
| Dashboard | React + TypeScript (`Website/frontend`); live API adapter or demo fixture data |
| Trained DeBERTa-v3 model artifact | **Not in repo** — training/eval scripts in `src/riskforge/modeling/`; export with `export_onnx.py` |

## Run

### Deployment modes (mandatory)

`RISKFORGE_DEPLOYMENT_MODE` is required and must be one of:

| Mode | Auth | Persistence | Scoring | Notes |
|---|---|---|---|---|
| `demo` | public | none needed | labelled synthetic-heuristic | operational routes disabled |
| `pilot` | **required** | SQLite/Postgres | model or explicitly-labelled heuristic | authenticated field trial |
| `production` | **required** | **PostgreSQL required** | validated ONNX artifact + encoder **required** | fails closed on any missing piece |

Production refuses to start without: authentication configured, PostgreSQL reachable, and a valid model artifact + manifest (matching checksum, schema, ordered IOGP labels) + tokenizer encoder matching the manifest backbone. There is **no** silent heuristic fallback in production. Pilot may run the deterministic heuristic engine, but every result and the readiness endpoint label it `heuristic`.

### Backend (API)

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src

# Pilot mode (SQLite, heuristic scoring, mandatory auth)
export RISKFORGE_DEPLOYMENT_MODE=pilot
export RISKFORGE_AUTH_ENABLED=true
export RISKFORGE_AUTH_ISSUER=riskforge-test
export RISKFORGE_AUTH_AUDIENCE=riskforge-api
export RISKFORGE_AUTH_KEYS_DIR=/run/secrets/riskforge-keys   # <kid>.key files
export RISKFORGE_SQLITE_PATH=data/riskforge.db
uvicorn riskforge.runtime.production:create_app --factory --host 0.0.0.0 --port 8000
```

Tokens are HS256 JWTs; mint them offline with the shared key (`kid` selects the key). Requests need `Authorization: Bearer <token>`; review decisions additionally require a `reviewer` role claim.

> The old `riskforge.runtime.main` entrypoint no longer exists; the ASGI factory is `riskforge.runtime.production:create_app` (matches the Dockerfile CMD).

All scoring, ingest, analytics, incident, audit, and explanation endpoints require authentication; `/health` is public and `/ready` reports the deployment mode and scoring label. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/ready` | liveness / readiness (mode + scoring labels) |
| POST | `/v1/inference`, `/v1/inference/batch` | score normalized records (auth) |
| POST | `/v1/ingest` | ingest CSV/JSON/JSONL/XLSX/PDF reports (auth, resource-limited) |
| GET | `/v1/incidents`, `/v1/incidents/{log_id}` | query stored results (auth) |
| GET | `/v1/analytics/summary` | SPD, trends, de-identified patterns (auth) |
| POST | `/v1/incidents/{log_id}/reviews` | HITL review decisions, reviewer role required (audited) |

### Dashboard (frontend)

```bash
cd Website/frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # static build in dist/
```

Set `VITE_API_BASE_URL` to point the dashboard at a running API; without it the dashboard uses bundled demo data. For authenticated deployments the host application obtains a short-lived token from your identity provider and registers it via `setAuthToken()` (`Website/frontend/src/api/adapters/live/apiClient.ts`) — no credentials are ever placed in `VITE_*` variables, because Vite inlines them into the shipped bundle.

### Docker

```bash
cp .env.example .env   # provision PGUSER/PGPASSWORD/GF_SECURITY_ADMIN_PASSWORD etc.
docker compose up --build          # api + postgres (pilot/production)
docker compose --profile setup up migrations
docker compose --profile monitoring up   # + prometheus + grafana
```

Demo needs no PostgreSQL: run the API with `RISKFORGE_DEPLOYMENT_MODE=demo` and the frontend in its default demo mode instead of the compose stack. Compose images are pinned to version tags; production releases must additionally record and substitute image digests (see `docs/decisions/0006-deployment-modes.md`).

## Tests

```bash
python -m pip install -r requirements-dev.txt
PYTHONPATH=src python -m pytest -q tests/
python -m ruff check src/riskforge/api src/riskforge/runtime src/riskforge/application \
  src/riskforge/persistence src/riskforge/review src/riskforge/supervision src/riskforge/serving \
  src/riskforge/logging_config.py src/riskforge/observability src/riskforge/authentication \
  src/riskforge/ingestion src/riskforge/analytics
python scripts/lint_boundaries.py
```

CI (`.github/workflows/ci.yml`) runs lint, boundary checks, unit + integration tests on Linux/macOS/Windows, and a PostgreSQL integration job against a real server.

## Security

- **Deployment modes are fail-closed** (`src/riskforge/runtime/deployment.py`): demo is public but exposes no operational routes; pilot/production require authentication; production additionally requires PostgreSQL and a validated model artifact — missing pieces refuse startup rather than degrading silently.
- **Authentication** (HS256 JWT, `src/riskforge/authentication/jwt_service.py`): strict algorithm allow-list, validated `kid` against a local keyring, required finite `exp`, issuer/audience checks, `nbf` handling, bounded token size, roles/scope claims mapped onto the principal. Credentials are never logged.
- **Authorization**: capability scopes (`incidents:read`, `analytics:read`, `scoring:write`, `audit:read`) enforced at every route boundary, with documented fallback roles; review decisions additionally need an explicit reviewer role/`reviews:write` scope and the deny-by-default `RISKFORGE_REVIEWER_SUBJECTS` allow-list.
- **Ingestion resource limits** (`src/riskforge/ingestion/limits.py`): declared-format signature verification, XLSX ZIP metadata inspection before extraction, document size, record counts, line sizes, JSON depth, PDF pages, and spreadsheet-expansion ratios are bounded before parsing; operators may tighten limits via `RISKFORGE_INGEST_LIMIT_*` but never disable them. Duplicate `log_id`s are reported explicitly as per-record outcomes.
- **Audit trail**: upload, scoring, idempotent replay, review, and failure actions append `inference_recorded`/`review_decision_recorded` events carrying actor, timestamp, correlation id, scoring mode, and model version — never narrative text.
- Request size caps, timeouts, CORS allowlist, correlation IDs, security headers, and structured errors.
- Docker defaults carry **no credentials**: `PGUSER`/`PGPASSWORD`/`GF_SECURITY_ADMIN_PASSWORD` must be provisioned in `.env`, PostgreSQL is not published to the host, and monitoring ports bind to loopback only.
- Never commit `.env` or key material.

See `docs/decisions/0006-deployment-modes.md` for the mode-boundary rationale.
