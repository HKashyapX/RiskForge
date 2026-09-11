# RiskForge Project State

## Objective
Complete SIH26165 RiskForge: convert OIL unstructured safety reports (UA/UC, near-miss) into SIF intelligence — ingestion → NLP normalization → SIF precursor detection → risk scoring → explanation → pattern/trend analytics → recommendations → dashboard — honestly labeled (deterministic engine, no fabricated ML).

## Current Phase
Stage 5 of 5 (frontend ↔ backend wiring) + final verification/review.

## Architecture (actual)
- **Backend** (Python 3.14, FastAPI, src-layout): `core` frozen contracts → `normalization` (config-driven gazetteer, span-preserving) → `serving` (ONNX engine protocol + deterministic `HeuristicRuleEngine` fallback, clearly labeled) → `supervision` (LF gate) → `application` workflow → `api` (JWT auth, correlation IDs, `/v1/inference`, `/v1/incidents`, `/v1/ingest`, `/v1/analytics/summary`, `/v1/explanations/{id}`) → `runtime.production.create_production_app` composer (SQLite default, Postgres optional).
- **Ingestion** (`riskforge.ingestion`): CSV/TSV/JSON/JSONL/XLSX/PDF parsers → validation → gazetteer via injected port → persisted.
- **Analytics** (`riskforge.analytics`): weekly trends, emerging-risk z-score, pattern clusters (canonical entity pairs), barrier recurrence, recommendations.
- **Frontend** (React 19 + Vite + Tailwind 4): adapter pattern; `src/api/adapters/index.ts` selects demo (baked JSONL) vs live (FastAPI via `VITE_API_MODE`/`VITE_API_BASE_URL`). Live adapters: `adapters/live/*` (apiClient with X-Correlation-ID, mappers, LiveIncidentAdapter, LiveOverviewAdapter).
- Boundaries enforced by `scripts/lint_boundaries.py`; CI lint scope in `.github/workflows/ci.yml`.

## Completed (commits on main)
- `6c43441`/`0d6ce37` JWT auth restored + packaging fixed
- `deb7bf7` production composer + deterministic fallback engine (Dockerfile CMD fixed)
- `79c9e9a` ingestion subsystem + POST /v1/ingest
- `cd4e3c1` analytics + recommendations + explanation endpoints
- `7db0540` CI green (openpyxl/pypdf in test deps)
- UNCOMMITTED (this checkpoint): expanded gazetteer vocabulary (config/gazetteer_rules.yaml, ~140 surface forms); gazetteer matcher rewritten O(text) with first-token index (p99 0.654ms vs 5ms SLA, previously 7ms failing); CSV parser header/alias normalization; frontend live adapters + page rewire; frontend .env.example.

## In Progress
Stage 5 E2E: boot backend + frontend dev server, verify dashboard renders live API data (preview registration requested by environment).

## Blocked
None.

## Agent Assignments
Single agent (no subagent tooling in this environment); parallel tool batching used instead. Reviewer pass = self-review checklist + full test gate before each push.

## Decisions
- Deterministic `HeuristicRuleEngine` labeled as rule-based fallback (no trained model artifact exists) — honesty per mandate.
- Gazetteer matcher rewritten as token-index dispatch, semantics preserved (flexible `[\s\-\.]` separators, longest-first overlap resolution) — perf SLA restored without weakening vocabulary.
- Frontend keeps demo adapter as default so the site works without backend; live mode via env.
- `.freebuff/` stays uncommitted (local tool state).

## Known Issues
- Repo-wide `ruff` flags legacy style in frozen packages outside CI scope (pre-existing; not touched).
- Frontend pages were built against demo JSONL; live mappers cover the overview/incident surfaces; some detail pages may need live-path verification.

## Verification
- Full suite: 1036 passed, 43 skipped (before this checkpoint's gazetteer fix); targeted battery now 130/130 incl. benchmark.
- CI on main: green at `7db0540`.

## Current Checkpoint
"Stage 5: frontend live adapters + gazetteer perf/vocabulary + CSV parser aliases" (this commit).

## Next Exact Actions
1. Push checkpoint; confirm CI green.
2. Boot backend (`create_production_app`, uvicorn) + seed via `/v1/ingest`; boot frontend dev server with `VITE_API_MODE=live`; verify dashboard in preview (snapshot/evaluate/logs).
3. Fix any live-mapping mismatches found in browser.
4. Reviewer pass over stage-4/5 diffs; fix CRITICAL/HIGH.
5. Final full gate (tests + CI lint scope + boundaries + tsc build), commit, push.
