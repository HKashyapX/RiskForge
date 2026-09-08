# ADR-0004: Persistence Boundary and Immutable Review History

## Status
Accepted

## Context
RiskForge needs durable incident-result storage, review history, and an append-only audit trail without coupling application services to a specific database engine.

## Decision
Define use-case-oriented repository protocols outside `core/contracts.py`. Automated inference results are immutable and idempotent by `log_id`: an identical repeat is a no-op, while different content for an existing `log_id` is a conflict. Human review decisions and audit events are append-only. Repository queries must apply explicit pagination and deterministic ordering.

SQLite may be used as an isolated development and integration implementation; production deployments remain PostgreSQL-oriented.

## Consequences
- Application and workflow code remain independent of database technology.
- Historical automated decisions cannot be silently overwritten.
- Idempotent processing is enforceable at the persistence boundary.
- Deterministic query behavior is testable across storage implementations.
