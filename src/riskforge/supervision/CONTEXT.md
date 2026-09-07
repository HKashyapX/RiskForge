# Subsystem: Weak Supervision & Heuristics

## Boundary
- Authority: Programmatic labeling functions and EM label aggregation for cold-start datasets.
- Forbidden: PyTorch imports; hardcoded majority voting.

## Inputs / Outputs
- Consumes: `IncidentNormalizedRecord`
- Produces: Weak label posterior probability distributions `[P(NON_SIF), P(SIF_P)]`

## Invariants
- Zero external runtime dependencies outside NumPy and standard library.
- Heuristic functions must enforce negation checking within token windows.
