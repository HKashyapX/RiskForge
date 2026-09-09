# Authentication Boundary — Architectural Context

## Scope

The authentication subsystem handles **inbound** credential verification for HTTP
clients calling the RiskForge API. It does NOT handle service-to-service (S2S)
authentication because RiskForge is a self-contained monolithic API server with
no outbound service calls.

## Architectural Decision: No Service-to-Service Authentication

**Status:** Accepted  
**Date:** 2026-09-09

### Context

RiskForge is a monolithic API server exposing a REST interface for incident
inference, review workflows, and audit queries. The system's external
dependencies are:

- **PostgreSQL** — data persistence (via psycopg wire protocol, not HTTP)
- **Local filesystem** — ONNX model artifacts, configuration files

There are **zero outbound HTTP calls** anywhere in the production source code.
No HTTP client libraries (`httpx`, `requests`, `aiohttp`) appear in production
dependencies. The `httpx` library is dev-only (for `TestClient`).

All internal component communication is synchronous, in-process Python method
calls:

```
HTTP Client → FastAPI routes → ApplicationService → [Serving|Metrics|Review] → Persistence
```

### Decision

Service-to-service authentication (JWT/OIDC/service tokens, mTLS, HMAC
signatures) is **not implemented** because:

1. There are no outbound service calls to secure.
2. PostgreSQL connections are secured at the database level (connection string
   credentials, network ACLs), not via application-level S2S auth.
3. Adding S2S auth infrastructure would introduce unused complexity with no
   corresponding security benefit.

### Invariants

The following architectural invariants are enforced by tests:

- **No outbound HTTP client libraries** in production source code
- **No URL/endpoint configuration** for external services in production code
- **No message queue, event bus, or gRPC dependencies** in production code

### Re-evaluation Triggers

This decision should be re-evaluated if RiskForge adds:

- Outbound webhook dispatch (e.g., notifying external systems)
- External API integrations (e.g., pulling threat intelligence feeds)
- Message queue producers/consumers (e.g., Kafka, RabbitMQ)
- gRPC service meshes
- Multi-service orchestration (microservices decomposition)
