# Subsystem: HTTP API

## Boundary
- Authority: external transport for RiskForge application use cases.
- Allowed: HTTP request/response DTOs, request validation, authentication handoff, HTTP status mapping, correlation metadata, and route registration.
- Forbidden: business rules, direct SQL, direct ONNX Runtime calls, model training, and direct mutation of core contracts.

## Inputs / Outputs
- Consumes: validated transport requests and authenticated principals.
- Produces: versioned HTTP responses and transport-safe error payloads.

## Invariants
- Routes call application-facing interfaces rather than infrastructure implementations.
- Internal exceptions and sensitive incident narratives must not leak to clients.
- API semantics must preserve application-level idempotency and correlation behavior.
- Health and readiness are distinct: liveness reports process health; readiness reports usable backend state.
