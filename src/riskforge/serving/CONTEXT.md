# Subsystem: Air-Gapped ONNX Runtime Serving

## Boundary
- Authority: Execution of INT8 quantized models via ONNX Runtime and deterministic barrier gate.
- Forbidden: PyTorch imports; model training routines; outbound network requests.

## Inputs / Outputs
- Consumes: `IncidentNormalizedRecord`, token IDs, attention masks.
- Produces: `ModelInferenceResult`

## Invariants
- CPU execution provider strictly constrained to 4 vCPUs.
- End-to-end inference latency must remain under 35 ms.
