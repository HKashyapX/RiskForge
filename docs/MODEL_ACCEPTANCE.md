# Production Model Acceptance

RiskForge treats model quality and serving performance as measured acceptance results, not
properties guaranteed by the architecture or loss function.

## Required artifact

The candidate must be the exact versioned ONNX bundle intended for deployment. Its manifest must
identify the model checksum, backbone, quantization, maximum sequence length, calibration
temperature, input and output names, and exact IOGP rule order.

## Serving acceptance

Run `scripts/benchmark_serving.py` on the named reference CPU with enough iterations to produce
stable percentiles. Preserve the generated JSON report with the release evidence.

The current product targets are:

- INT8 artifact validation passes.
- Warm batch-1 p95 model execution and postprocessing latency is at most 35 ms.
- Peak total process RSS is at most 1,200,000,000 bytes.
- Batch sizes 1, 8, 16, and 32 complete successfully.
- Repeated and concurrent inference show bounded memory behavior.

The serving report excludes tokenization and HTTP transport. End-to-end latency must be reported
separately before making an end-to-end performance claim.

## Quality acceptance owned by modeling

The modeling evaluation must use an expert-reviewed held-out set that was not used for training,
weak-label fitting, calibration, or threshold selection. The report must identify the dataset
version, model checksum, evaluation timestamp, and sample counts.

Required SIF-P results:

- Confusion matrix with true positive, false positive, true negative, and false negative counts.
- Recall, precision, false-negative rate, and false-positive rate.
- Results at the calibrated 0.40 and 0.65 routing thresholds.
- Results before and after deterministic safety overrides.
- Confidence intervals and the number of positive examples.

Required IOGP results:

- Per-rule precision, recall, and F1 for all nine Life-Saving Rules.
- Macro and micro averages.
- Support count for every rule.

The team may state a 95% recall result only when this evaluation demonstrates it. Focal loss is a
training method and must never be described as guaranteeing recall. Dataset ratios, alert
suppression rates, and operational impact figures require evidence from the actual evaluation or
deployment population.

## Blocked acceptance

Final acceptance remains blocked until both inputs exist:

1. The production INT8 ONNX artifact.
2. The versioned expert-reviewed held-out evaluation set.

Synthetic ONNX graphs remain suitable for integration tests but cannot substantiate production
latency, memory, or accuracy claims.
