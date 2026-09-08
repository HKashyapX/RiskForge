# Subsystem: Application Orchestration

## Boundary
- Authority: Coordinate normalized incidents, inference, application-facing metrics, incident
  browsing, review commands, audit history, and error translation.
- Forbidden: Direct storage, HTTP transport, model training, normalization mutation, or metrics implementation details.

## Inputs / Outputs
- Consumes: `IncidentNormalizedRecord`.
- Produces: inference results and application-owned workflow views through injected protocols.

## Invariants
- Preserve `log_id` correlation exactly.
- Reject duplicate `log_id` values in a batch before inference.
- Preserve caller batch order deterministically.
- Do not log or expose incident narratives from orchestration errors.
- Depend on narrow protocols rather than concrete subsystem implementations.
