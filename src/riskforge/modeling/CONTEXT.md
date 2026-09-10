# Subsystem: Multi-Task Neural Modeling

## Boundary
- Authority: Training and evaluation of multi-task DeBERTa-v3 with Asymmetric Focal Loss.
- Forbidden: Production inference serving logic; reliance on GPU during serving.

## Inputs / Outputs
- Consumes: Tokenized inputs mapped to weak labels and IOGP gold annotations.
- Produces: PyTorch model checkpoints and exported ONNX computation graphs.

## Portable local data
- Modeling commands read `RISKFORGE_MODELING_DATA_DIR` when set.
- The default is `data/processed` relative to the repository working directory.
- Private datasets and generated model artifacts must remain outside Git history.
- Inspect JSONL/CSV shape without printing values using:
  `python -m riskforge.modeling.dataset_inspection <path>`.

## Invariants
- Asymmetric Focal Loss must use gamma=2.0, alpha=0.75.
- Logits must pass through temperature scaling calibration before deployment.
