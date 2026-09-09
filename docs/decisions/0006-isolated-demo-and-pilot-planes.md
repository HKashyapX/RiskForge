# ADR 0006: Isolate public demo and authenticated pilot planes

## Status

Accepted.

## Context

RiskForge needs a publicly accessible demonstration and a future operational
pilot that may process sensitive incident narratives. A public demo is an
untrusted environment. Allowing it to share storage, credentials, model
configuration, or fallback behavior with the pilot would create an avoidable
route from an Internet-facing showcase into operational data.

## Decision

RiskForge will deploy two independently configured environments:

- **Demo plane:** public, synthetic data only, no upload endpoint, no
  operational database, and no pilot credentials.
- **Pilot plane:** authenticated, allow-listed origins, encrypted managed
  PostgreSQL, immutable automated results, append-only review/audit events,
  and a validated model artifact.

The two planes must use separate databases, service identities, secrets,
deployment projects, object-storage prefixes, and observability credentials.
Pilot startup must fail closed if authentication, the database, or the model
artifact is unavailable. It must never fall back to demo data.

The frontend must make the active plane visible. Demo responses must be
clearly marked as synthetic and must not be accepted by pilot API routes.

## Consequences

- A public portfolio can remain easy to access without weakening pilot
  controls.
- Environment drift must be checked in CI and deployment configuration.
- Features must be evaluated explicitly for demo, pilot, or both.
- Deployment has additional cost and configuration, but the security boundary
  is simpler to reason about and test.

## Verification

- Contract tests reject demo-mode composition in a production environment.
- Deployment tests verify distinct origins and storage configuration.
- End-to-end tests verify that pilot failures do not trigger demo fallback.
- Synthetic fixtures contain no copied or transformed operational records.
