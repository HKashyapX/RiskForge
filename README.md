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

### Backend (API)

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src

# SQLite persistence (default, zero-config)
RISKFORGE_ENV=development \
RISKFORGE_MODEL_PATH=data/models/sif.onnx \
RISKFORGE_MANIFEST_PATH=data/models/manifest.yaml \
uvicorn riskforge.runtime.main:create_app --factory --host 0.0.0.0 --port 8000
```

Without model artifacts the API starts with the deterministic heuristic engine and reports its mode at `GET /health` and `GET /ready`. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/ready` | liveness / readiness (component detail) |
| POST | `/v1/inference`, `/v1/inference/batch` | score normalized records |
| POST | `/v1/ingest` | ingest CSV/JSON/JSONL/XLSX/PDF reports |
| GET | `/v1/incidents`, `/v1/incidents/{log_id}` | query stored results |
| GET | `/v1/analytics/summary` | SPD, trends, patterns, recommendations |
| POST | `/v1/reviews` | HITL review decisions (audited) |

### Dashboard (frontend)

```bash
cd Website/frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # static build in dist/
```

Set `VITE_API_BASE_URL` to point the dashboard at a running API; without it the dashboard uses bundled demo data.

### Docker

```bash
cp .env.example .env   # edit credentials
docker compose up --build          # api + postgres
docker compose --profile setup up migrations
docker compose --profile monitoring up   # + prometheus + grafana
```

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

- JWT (HS256) authentication on protected routes when `RISKFORGE_AUTH_ENABLED` is set; secrets via environment only (`src/riskforge/authentication/jwt_service.py`).
- Request size caps, timeouts, CORS allowlist, correlation IDs, and structured errors (`src/riskforge/api/request_controls.py`).
- Never commit `.env`; `PGPASSWORD=CHANGE_ME_IN_PRODUCTION` must be replaced before deployment.
