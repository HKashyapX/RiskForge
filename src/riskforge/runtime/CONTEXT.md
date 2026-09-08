# Subsystem: Runtime composition

## Boundary
- Authority: deterministic construction and lifecycle wiring of application-facing RiskForge dependencies.
- Allowed: configuration loading, dependency construction, startup validation, readiness state, and graceful shutdown coordination.
- Forbidden: business rules, SQL queries, HTTP route handlers, model training, and direct mutation of core contracts.

## Inputs / Outputs
- Consumes: typed runtime configuration and concrete infrastructure implementations.
- Produces: a fully wired application service and runtime lifecycle state.

## Invariants
- Application behavior depends on protocols, not infrastructure-specific implementations.
- Model artifact validation/warm-up must complete before the service is reported ready.
- Runtime configuration must not require OS-specific paths or shell behavior.
- Startup failures must remain distinguishable from request-time application failures.
