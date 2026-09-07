# ADR-0003: Span Preservation Over Regex String Replacement

## Status
Accepted

## Context
Applying regex substitutions directly to narrative text invalidates original character index offsets, preventing the HITL dashboard from highlighting extracted entities in the raw narrative.

## Decision
Retain raw text immutability. The normalization layer extracts canonicalized `EntitySpan` objects tracking original start and end character offsets.

## Consequences
- All subsequent modules must reference original character spans via index offsets.
- Downstream HITL UI can highlight entities directly against raw logs.
