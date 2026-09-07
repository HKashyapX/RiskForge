# Subsystem: Normalization & Gazetteer

## Boundary
- Authority: Ingest raw text and extract domain entity spans with character offset mappings.
- Forbidden: String substitution or mutation of raw text narrative.

## Inputs / Outputs
- Consumes: `IncidentRawRecord`
- Produces: `IncidentNormalizedRecord` with `List[EntitySpan]`

## Invariants
- Must preserve exact character start and end indices against raw text.
- Must execute on CPU in under 5 ms per incident record.
