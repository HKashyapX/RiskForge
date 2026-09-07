# Subsystem: Core Contracts & Domain Types

## Boundary
- Authority: Single Source of Truth for shared domain models, enums, and exceptions.
- Forbidden: No ML frameworks, no torch, no external network calls, no business logic.

## Dependencies
- pydantic >= 2.7.0

## Invariants
- Models are immutable during execution.
- Modifications require explicit team consensus and RFC PR.
