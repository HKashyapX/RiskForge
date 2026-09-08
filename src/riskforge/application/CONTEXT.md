# Subsystem: Application Orchestration

## Boundary
- Authority: Coordinate normalized incidents, inference, application-facing metrics access, and error translation.
- Forbidden: Direct storage, HTTP transport, model training, normalization mutation, or metrics implementation details.

## Inputs / Outputs
- Consumes: `IncidentNormalizedRecord`.
- Produces: `ModelInferenceResult` and application-facing metrics values through injected protocols.

## Invariants
- Preserve `log_id` correlation exactly.
- Reject duplicate `log_id` values in a batch before inference.
- Preserve caller batch order deterministically.
- Do not log or expose incident narratives from orchestration errors.
- Depend on narrow protocols rather than concrete subsystem implementations.
