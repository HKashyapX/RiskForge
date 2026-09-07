# Subsystem: Multi-Task Neural Modeling

## Boundary
- Authority: Training and evaluation of multi-task DeBERTa-v3 with Asymmetric Focal Loss.
- Forbidden: Production inference serving logic; reliance on GPU during serving.

## Inputs / Outputs
- Consumes: Tokenized inputs mapped to weak labels and IOGP gold annotations.
- Produces: PyTorch model checkpoints and exported ONNX computation graphs.

## Invariants
- Asymmetric Focal Loss must use gamma=2.0, alpha=0.75.
- Logits must pass through temperature scaling calibration before deployment.
