# ADR 0006: Explicit demo / pilot / production deployment modes

## Status

Accepted

## Context

Before this decision, a single composition path served every environment.
Heuristic scoring engaged silently whenever a model artifact was absent,
authentication was opt-in, review authorization defaulted to allow-all,
and Docker defaults shipped well-known credentials with PostgreSQL
published to the host. The result was an ambiguous "production-looking"
deployment that was actually running demo-grade behavior — the single
largest integrity risk for a safety-intelligence product.

## Decision

RiskForge has exactly three deployment modes, selected once through
`RISKFORGE_DEPLOYMENT_MODE` and validated at composition time
(`src/riskforge/runtime/deployment.py`). Every safety-relevant default
derives from the mode; composition **fails closed** when a requirement is
unmet.

| | demo | pilot | production |
|---|---|---|---|
| Authentication | none (public) | mandatory | mandatory |
| Operational routes | disabled | enabled | enabled |
| Persistence | SQLite | SQLite/Postgres | PostgreSQL only |
| Scoring | heuristic, labelled `synthetic-heuristic` | model or heuristic, always labelled | validated ONNX artifact + manifest + encoder, never heuristic |
| Metrics endpoint | disabled | private (auth) | private (auth) |
| Review authorization | deny-by-default | deny-by-default (`RISKFORGE_REVIEWER_SUBJECTS`) | deny-by-default |

Key properties:

- **No silent fallback.** A configured-but-missing artifact is a hard error
  in every mode. Production cannot serve heuristic output — the engine
  wrapper refuses results whose scoring mode contradicts the deployment.
- **Honest labels.** Every result carries `scoring_mode`, `engine_name`,
  `model_version`, and `calibration_version`; readiness reports the
  deployment mode and scoring label. Rule scores can never be presented as
  model probabilities (enforced by the contract validator).
- **Separate environments.** Demo and pilot/production deployments use
  different databases, secrets, identities, storage, and monitoring
  credentials; there is no shared default.
- **Fail-closed startup.** Production refuses to start without
  authentication, PostgreSQL, and a fully validated artifact bundle
  (checksum, manifest schema, I/O names, ordered IOGP labels,
  max sequence length, calibration metadata, tokenizer↔backbone match).

## Consequences

- Operators must provision credentials and artifacts explicitly; the
  compose file guides them through required `.env` values and refuses to
  start with unset ones.
- Heuristic pilot trials remain possible but are permanently visible as
  heuristic in readiness, results, and logs.
- Future model upgrades are auditable: every stored score names the exact
  artifact and calibration that produced it.
