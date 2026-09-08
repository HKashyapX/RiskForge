# Decision 0005: API and runtime composition boundaries

## Status
Accepted as an implementation boundary for the backend completion phase.

## Decision
RiskForge will separate external HTTP transport from application orchestration and runtime dependency composition.

The intended dependency direction is:

HTTP API → Application interfaces/services → subsystem protocols → concrete infrastructure

Runtime composition is responsible for constructing concrete dependencies, validating startup configuration, and managing readiness/lifecycle. It does not contain business rules.

The HTTP API is responsible for transport concerns only: DTO validation, authentication handoff, HTTP status/error mapping, correlation metadata, and route registration. It does not issue SQL or call ONNX Runtime directly.

## Rationale
The repository already has stable subsystem boundaries for serving, persistence, review, metrics, normalization, and application orchestration. This decision keeps the new service layer from bypassing those boundaries while allowing infrastructure implementations to change independently.

## Scope
This decision establishes structure and ownership only. It does not implement the HTTP server, runtime bootstrap, authentication provider, or production database adapter.
