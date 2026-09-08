# Subsystem: Review & Audit Workflow

## Boundary
- Authority: Human review decisions and append-only governance events for persisted automated inference results.
- Forbidden: HTTP transport, model execution, training, normalization, metrics implementation, and direct mutation of automated results.

## Inputs / Outputs
- Consumes: persisted `StoredIncidentResult`, reviewer identity, action, timestamp, and reason.
- Produces: immutable `ReviewDecision` plus an append-only `AuditEvent` through an injected atomic writer.

## Invariants
- A reviewer must be authorized before a decision is written.
- The original automated inference result is never modified or replaced.
- Review decisions are append-only and ordered deterministically by `(decided_at ASC, decision_id ASC)`.
- A persisted review decision and its corresponding audit event are committed atomically.
- Workflow errors must not expose incident narratives.
