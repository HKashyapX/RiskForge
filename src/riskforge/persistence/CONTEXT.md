# Subsystem: Persistence

## Boundary
- Authority: Durable storage and deterministic querying of incident results, human review decisions, and append-only audit events.
- Forbidden: Model inference, score modification, HTTP transport, normalization, or training.

## Design
- Define repository protocols before infrastructure implementations.
- Production storage is PostgreSQL-oriented; isolated development and integration may use SQLite.
- Repository methods are use-case-oriented rather than generic CRUD abstractions.

## Invariants
- `log_id` is the incident idempotency and correlation key.
- Persisted automated inference results are immutable after successful creation.
- Review decisions append history and never mutate the original automated result.
- Audit events are append-only.
- Queries must provide deterministic ordering and explicit pagination semantics.
- External query values are validated at the repository boundary.
