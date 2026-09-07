# Subsystem: Metrics & Asset Risk Analytics

## Boundary
- Authority: Aggregation of SIF Precursor Density (SPD) and barrier degradation counters.
- Forbidden: Modifying incident scores; executing ML inference.

## Inputs / Outputs
- Consumes: `ModelInferenceResult`
- Produces: `AssetRiskSummary`

## Invariants
- Formula: SPD = (Count(SIF-P Incidents) / Total Logs) * 100.
