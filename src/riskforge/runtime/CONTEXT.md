# Subsystem: Runtime Composition

## Boundary
- Authority: configuration, dependency construction, startup validation, readiness composition, lifecycle management, and graceful shutdown coordination.
- Allowed: concrete dependency wiring through existing subsystem interfaces.
- Forbidden: business rules, HTTP route behavior, SQL implementation, model training, and direct ONNX Runtime execution.

## Inputs / Outputs
- Consumes: validated runtime settings and dependency composers.
- Produces: a validated dependency assembly and transport-neutral lifecycle/readiness state.

## Invariants
- Runtime settings contain no committed credentials.
- Configuration and paths remain portable across Linux, Windows, and macOS.
- Components start in declared order and shut down in reverse order.
- Readiness is false unless the lifecycle is ready and every required component is ready.
- Error details exposed through readiness must not contain incident narratives or secrets.
