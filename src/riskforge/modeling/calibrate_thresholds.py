"""
RiskForge calibration-set threshold selection.

IMPORTANT:
- Does NOT train or modify model weights.
- Uses ONLY the calibration split.
- NEVER uses the held-out test set.
- SIF target: choose the highest threshold that maintains Recall >= 0.95.
- IOGP: evaluate thresholds independently for each rule.
- Saves calibration results for later review/export.

Expected files:
    C:\\data\\processed\\iogp_calibration.csv
    C:\\data\\processed\\deberta_multitask_checkpoints\\best.pt

Outputs:
    C:\\data\\processed\\deberta_multitask_checkpoints\\calibration\\
        calibration_predictions.csv
        calibration_metrics.json
        selected_thresholds.json
"""

from __future__ import annotations

import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import AutoTokenizer

from riskforge.modeling.config import modeling_data_dir


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "microsoft/deberta-v3-base"

DATA_DIR = modeling_data_dir()

CALIBRATION_FILE = DATA_DIR / "iogp_calibration.csv"

CHECKPOINT_DIR = DATA_DIR / "deberta_multitask_checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "best.pt"

OUTPUT_DIR = CHECKPOINT_DIR / "calibration"

PREDICTIONS_FILE = OUTPUT_DIR / "calibration_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "calibration_metrics.json"
THRESHOLDS_FILE = OUTPUT_DIR / "selected_thresholds.json"

MAX_LENGTH = 128
BATCH_SIZE = 16

# SIF requirement from project specification.
SIF_TARGET_RECALL = 0.95

# Threshold search resolution.
THRESHOLD_STEP = 0.005

# Exact IOGP order required by the project.
IOGP_RULES = [
    "bypassing_safety_controls",
    "confined_space",
    "driving",
    "energy_isolation",
    "hot_work",
    "line_of_fire",
    "safe_mechanical_lifting",
    "toxic_gas",
    "work_at_height",
]

DEFAULT_THRESHOLD = 0.50

TEXT_COLUMN_CANDIDATES = [
    "text_normalized",
    "narrative_sanitized",
    "narrative",
    "text",
]


# =============================================================================
# ENVIRONMENT
# =============================================================================

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# =============================================================================
# HELPERS
# =============================================================================


def print_section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def json_safe(value: Any) -> Any:
    """
    Convert numpy / torch / NaN values into JSON-safe Python values.
    """
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None

    if isinstance(value, torch.Tensor):
        return json_safe(value.detach().cpu().tolist())

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    return value


def find_text_column(df: pd.DataFrame) -> str:
    for column in TEXT_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "Could not find a usable text column. "
        f"Expected one of: {TEXT_COLUMN_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def parse_sif_label(value: Any) -> float:
    """
    Returns:
        1.0 -> positive
        0.0 -> negative
        NaN -> unknown

    'possible' is deliberately treated as unknown.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if text in {
        "yes",
        "sif",
        "sif_p",
        "positive",
        "1",
        "true",
    }:
        return 1.0

    if text in {
        "no",
        "non_sif",
        "non-sif",
        "negative",
        "0",
        "false",
    }:
        return 0.0

    # "possible" is intentionally unknown.
    if text in {
        "possible",
        "unknown",
        "uncertain",
        "",
        "nan",
        "none",
        "null",
    }:
        return np.nan

    return np.nan


def parse_iogp_label(value: Any) -> float:
    """
    Expected IOGP encoding:
        1  = positive
        0  = negative
       -1  = unknown

    Returns NaN for unknown.
    """
    if pd.isna(value):
        return np.nan

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = str(value).strip().lower()

        if text in {"1", "true", "yes", "positive"}:
            return 1.0

        if text in {"0", "false", "no", "negative"}:
            return 0.0

        return np.nan

    if numeric == 1:
        return 1.0

    if numeric == 0:
        return 0.0

    if numeric == -1:
        return np.nan

    return np.nan


def get_output_tensor(outputs: Any, preferred_names: list[str]) -> torch.Tensor:
    """
    Extract a tensor from the model output.

    Supports:
      - dictionary-like outputs
      - objects with attributes
    """
    if isinstance(outputs, dict):
        for name in preferred_names:
            if name in outputs:
                return outputs[name]

    for name in preferred_names:
        if hasattr(outputs, name):
            value = getattr(outputs, name)
            if isinstance(value, torch.Tensor):
                return value

    if isinstance(outputs, (tuple, list)):
        tensors = [x for x in outputs if isinstance(x, torch.Tensor)]

        if len(tensors) >= 2:
            # Model convention:
            #   outputs[0] = SIF
            #   outputs[1] = IOGP
            if "sif" in preferred_names[0]:
                return tensors[0]

            return tensors[1]

    raise RuntimeError(
        "Could not find the requested model output. "
        f"Expected one of: {preferred_names}"
    )


def instantiate_model(model_class: Any) -> torch.nn.Module:
    """
    Instantiate MultiTaskDeberta while accommodating the constructor used
    by the repository's training implementation.

    The function inspects the constructor rather than hard-coding a single
    signature.
    """
    signature = inspect.signature(model_class)
    parameters = signature.parameters

    kwargs: dict[str, Any] = {}

    possible_values = {
        "model_name": MODEL_NAME,
        "model_name_or_path": MODEL_NAME,
        "pretrained_model_name": MODEL_NAME,
        "backbone_name": MODEL_NAME,
        "num_iogp_rules": len(IOGP_RULES),
        "num_rules": len(IOGP_RULES),
        "n_rules": len(IOGP_RULES),
        "max_length": MAX_LENGTH,
        "lora_enabled": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": [
            "query_proj",
            "key_proj",
            "value_proj",
        ],
    }

    for parameter_name, parameter in parameters.items():
        if parameter_name in possible_values:
            kwargs[parameter_name] = possible_values[parameter_name]

    print("Model constructor parameters detected:")
    for key, value in kwargs.items():
        print(f"  {key}: {value}")

    try:
        model = model_class(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "Could not instantiate MultiTaskDeberta using the repository "
            "constructor.\n\n"
            f"Detected constructor:\n{signature}\n\n"
            f"Arguments attempted:\n{kwargs}\n\n"
            f"Original error:\n{exc}"
        ) from exc

    return model


def load_model() -> torch.nn.Module:
    """
    Load the repository model architecture and the trained checkpoint.
    """
    print_section("LOADING MODEL")

    # The repository's training script owns the actual model architecture.
    modeling_dir = Path(__file__).resolve().parent

    if str(modeling_dir) not in sys.path:
        sys.path.insert(0, str(modeling_dir))

    try:
        from train_multitask_deberta import MultiTaskDeberta
    except Exception as exc:
        raise RuntimeError(
            "Could not import MultiTaskDeberta from "
            "train_multitask_deberta.py.\n"
            f"Error: {exc}"
        ) from exc

    model = instantiate_model(MultiTaskDeberta)

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    print(f"Checkpoint: {CHECKPOINT_FILE}")

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a dictionary.")

    required_key = "model_state_dict"

    if required_key not in checkpoint:
        raise RuntimeError(
            f"Checkpoint does not contain '{required_key}'. "
            f"Available keys: {list(checkpoint.keys())}"
        )

    checkpoint_model_name = checkpoint.get("model_name")

    if checkpoint_model_name is not None:
        print(f"Checkpoint model name: {checkpoint_model_name}")

        if checkpoint_model_name != MODEL_NAME:
            raise RuntimeError(
                "Checkpoint backbone mismatch:\n"
                f"Expected: {MODEL_NAME}\n"
                f"Found:    {checkpoint_model_name}"
            )

    checkpoint_rules = checkpoint.get("rules")

    if checkpoint_rules is not None:
        if list(checkpoint_rules) != IOGP_RULES:
            raise RuntimeError(
                "Checkpoint IOGP rule order does not match the required order.\n"
                f"Expected: {IOGP_RULES}\n"
                f"Found:    {checkpoint_rules}"
            )

        print("Checkpoint IOGP rule order: PASS")

    checkpoint_max_length = checkpoint.get("max_length")

    if checkpoint_max_length is not None:
        print(f"Checkpoint max length: {checkpoint_max_length}")

        if int(checkpoint_max_length) != MAX_LENGTH:
            raise RuntimeError(
                "Checkpoint max sequence length mismatch:\n"
                f"Expected: {MAX_LENGTH}\n"
                f"Found:    {checkpoint_max_length}"
            )

    state_dict = checkpoint[required_key]

    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(f"Missing keys: {len(missing_keys)}")
    print(f"Unexpected keys: {len(unexpected_keys)}")

    if missing_keys:
        print("\nMissing keys:")
        for key in missing_keys:
            print(f"  {key}")

        raise RuntimeError(
            "Checkpoint loading produced missing model keys."
        )

    if unexpected_keys:
        print("\nUnexpected checkpoint keys:")
        for key in unexpected_keys:
            print(f"  {key}")

        raise RuntimeError(
            "Checkpoint loading produced unexpected model keys."
        )

    model.eval()
    model.to(DEVICE)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return model


def run_model_inference(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run inference without training.

    Handles the FP16 encoder / FP32 task-head dtype issue that occurred
    during the previous checkpoint test.

    Returns:
        sif_logits:  [N]
        iogp_logits: [N, 9]
    """
    all_sif: list[np.ndarray] = []
    all_iogp: list[np.ndarray] = []

    total = len(texts)

    start_time = time.perf_counter()

    for start in range(0, total, BATCH_SIZE):
        batch_texts = texts[start : start + BATCH_SIZE]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        encoded = {
            key: value.to(DEVICE)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            try:
                outputs = model(**encoded)

            except RuntimeError as exc:
                error_text = str(exc)

                if (
                    "mat1 and mat2 must have the same dtype"
                    not in error_text
                ):
                    raise

                sif_head = getattr(model, "sif_head", None)
                iogp_head = getattr(model, "iogp_head", None)

                if sif_head is None or iogp_head is None:
                    raise RuntimeError(
                        "Model produced a dtype mismatch, but sif_head "
                        "or iogp_head could not be found."
                    ) from exc

                sif_original_dtype = next(
                    sif_head.parameters()
                ).dtype

                iogp_original_dtype = next(
                    iogp_head.parameters()
                ).dtype

                # Match the FP16 encoder representation.
                sif_head.half()
                iogp_head.half()

                try:
                    outputs = model(**encoded)

                finally:
                    if sif_original_dtype == torch.float32:
                        sif_head.float()

                    if iogp_original_dtype == torch.float32:
                        iogp_head.float()

        sif_logits = get_output_tensor(
            outputs,
            ["sif_logits", "sif_logit"],
        )

        iogp_logits = get_output_tensor(
            outputs,
            ["iogp_logits", "iogp_logit", "rule_logits"],
        )

        if not isinstance(sif_logits, torch.Tensor):
            raise RuntimeError("SIF output is not a tensor.")

        if not isinstance(iogp_logits, torch.Tensor):
            raise RuntimeError("IOGP output is not a tensor.")

        sif_logits = sif_logits.detach().float().cpu()
        iogp_logits = iogp_logits.detach().float().cpu()

        if sif_logits.ndim == 2 and sif_logits.shape[1] == 1:
            sif_logits = sif_logits[:, 0]

        if sif_logits.ndim != 1:
            raise RuntimeError(
                f"Expected SIF logits shape [batch] or [batch,1], "
                f"got {tuple(sif_logits.shape)}"
            )

        if iogp_logits.ndim != 2:
            raise RuntimeError(
                f"Expected IOGP logits shape [batch,9], "
                f"got {tuple(iogp_logits.shape)}"
            )

        if iogp_logits.shape[1] != len(IOGP_RULES):
            raise RuntimeError(
                f"Expected {len(IOGP_RULES)} IOGP logits, "
                f"got {iogp_logits.shape[1]}"
            )

        if not torch.isfinite(sif_logits).all():
            raise RuntimeError("SIF logits contain non-finite values.")

        if not torch.isfinite(iogp_logits).all():
            raise RuntimeError("IOGP logits contain non-finite values.")

        all_sif.append(sif_logits.numpy())
        all_iogp.append(iogp_logits.numpy())

        processed = min(start + BATCH_SIZE, total)

        if start == 0 or processed == total or (start // BATCH_SIZE + 1) % 10 == 0:
            percentage = 100.0 * processed / total

            print(
                f"  {processed:5d}/{total:5d} "
                f"({percentage:6.2f}%)"
            )

    elapsed = time.perf_counter() - start_time

    sif_array = np.concatenate(all_sif, axis=0)
    iogp_array = np.concatenate(all_iogp, axis=0)

    print(f"\nInference time: {elapsed:.3f} seconds")
    print(
        f"Records/second: "
        f"{len(texts) / elapsed:.2f}"
    )

    return sif_array, iogp_array


def sigmoid(values: np.ndarray) -> np.ndarray:
    """
    Numerically stable sigmoid.
    """
    values = np.asarray(values, dtype=np.float64)

    result = np.empty_like(values)

    positive = values >= 0

    result[positive] = 1.0 / (
        1.0 + np.exp(-values[positive])
    )

    exp_values = np.exp(values[~positive])

    result[~positive] = exp_values / (
        1.0 + exp_values
    )

    return result


def threshold_grid() -> np.ndarray:
    """
    Thresholds from 0.00 through 1.00 inclusive.
    """
    count = int(round(1.0 / THRESHOLD_STEP))

    return np.array(
        [
            round(i * THRESHOLD_STEP, 6)
            for i in range(count + 1)
        ],
        dtype=np.float64,
    )


def calculate_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    accuracy = (
        float(np.mean(predictions == y_true))
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(
            fp / (fp + tn)
        ) if (fp + tn) > 0 else None,
        "positive_predictions": int(
            predictions.sum()
        ),
    }


def select_sif_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Select the highest threshold whose recall remains >= 0.95.

    Why highest?
    A higher threshold generally reduces false positives while maintaining
    the required recall. This is preferable to simply selecting the lowest
    threshold that achieves recall >= 0.95.
    """
    results = []

    for threshold in threshold_grid():
        metrics = calculate_binary_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        results.append(metrics)

    eligible = [
        item
        for item in results
        if item["recall"] >= SIF_TARGET_RECALL
    ]

    if not eligible:
        raise RuntimeError(
            "No SIF threshold achieved the required "
            f"recall >= {SIF_TARGET_RECALL:.2f} "
            "on the calibration set."
        )

    # Highest threshold while preserving recall requirement.
    selected = max(
        eligible,
        key=lambda item: item["threshold"],
    )

    return selected, results


def calculate_auc_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    unique = np.unique(y_true)

    if len(unique) == 2:
        result["pr_auc"] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )

        result["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None

    return result


def calibrate_iogp_rule(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """
    Find the threshold with the best F1 when both positive and negative
    calibration labels exist.

    If the calibration labels contain only one class, threshold selection
    is not statistically meaningful, so we retain 0.50 and explicitly mark
    the rule as insufficiently calibrated.
    """
    positive_count = int(np.sum(y_true == 1))
    negative_count = int(np.sum(y_true == 0))

    auc_metrics = calculate_auc_metrics(
        y_true,
        probabilities,
    )

    if positive_count == 0 or negative_count == 0:
        default_metrics = calculate_binary_metrics(
            y_true,
            probabilities,
            DEFAULT_THRESHOLD,
        )

        return {
            "status": "insufficient_class_coverage",
            "valid": int(len(y_true)),
            "positives": positive_count,
            "negatives": negative_count,
            "prevalence": float(
                positive_count / len(y_true)
            ),
            "selected_threshold": DEFAULT_THRESHOLD,
            "threshold_selection_method": (
                "default_0.50_due_to_single_class"
            ),
            "selected_metrics": default_metrics,
            **auc_metrics,
        }

    threshold_results = []

    for threshold in threshold_grid():
        metrics = calculate_binary_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        threshold_results.append(metrics)

    # Maximize F1.
    #
    # Tie-break:
    #   1. higher recall
    #   2. higher threshold
    #
    # This avoids unnecessarily low thresholds when F1 is tied.
    selected = max(
        threshold_results,
        key=lambda item: (
            item["f1"],
            item["recall"],
            item["threshold"],
        ),
    )

    return {
        "status": "calibrated",
        "valid": int(len(y_true)),
        "positives": positive_count,
        "negatives": negative_count,
        "prevalence": float(
            positive_count / len(y_true)
        ),
        "selected_threshold": float(
            selected["threshold"]
        ),
        "threshold_selection_method": (
            "maximum_f1"
        ),
        "selected_metrics": selected,
        **auc_metrics,
    }


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    print_section("RISKFORGE CALIBRATION-SET THRESHOLD SELECTION")

    print("IMPORTANT:")
    print("  No training will be performed.")
    print("  No model weights will be modified.")
    print("  Only the calibration split will be used.")
    print("  The held-out test set will NOT be read.")

    print()
    print(f"Calibration file:")
    print(f"  {CALIBRATION_FILE}")

    print()
    print(f"Checkpoint:")
    print(f"  {CHECKPOINT_FILE}")

    print()
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"SIF target recall: >= {SIF_TARGET_RECALL}")
    print(f"Threshold step: {THRESHOLD_STEP}")
    print(f"Device: {DEVICE}")

    # -------------------------------------------------------------------------
    # Validate files
    # -------------------------------------------------------------------------

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found:\n"
            f"{CALIBRATION_FILE}"
        )

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load calibration dataset
    # -------------------------------------------------------------------------

    print_section("LOADING CALIBRATION DATASET")

    df = pd.read_csv(CALIBRATION_FILE)

    print(f"Calibration records: {len(df):,}")

    if len(df) == 0:
        raise RuntimeError(
            "Calibration dataset is empty."
        )

    # Incident ID validation.
    if "incident_id" in df.columns:
        duplicate_ids = int(
            df["incident_id"].duplicated().sum()
        )

        print(
            f"Duplicate incident IDs: "
            f"{duplicate_ids}"
        )

        if duplicate_ids:
            raise RuntimeError(
                "Calibration dataset contains duplicate "
                "incident IDs."
            )

    # -------------------------------------------------------------------------
    # Find text
    # -------------------------------------------------------------------------

    text_column = find_text_column(df)

    print(f"Text column: {text_column}")

    texts = (
        df[text_column]
        .fillna("")
        .astype(str)
        .tolist()
    )

    empty_text_count = sum(
        not text.strip()
        for text in texts
    )

    print(
        f"Empty text records: "
        f"{empty_text_count}"
    )

    if empty_text_count:
        raise RuntimeError(
            "Calibration data contains empty narratives."
        )

    # -------------------------------------------------------------------------
    # Parse labels
    # -------------------------------------------------------------------------

    print_section("VALIDATING CALIBRATION LABELS")

    if "sif_potential" not in df.columns:
        raise RuntimeError(
            "Calibration file does not contain "
            "'sif_potential'."
        )

    sif_true = np.array(
        [
            parse_sif_label(value)
            for value in df["sif_potential"]
        ],
        dtype=np.float64,
    )

    sif_valid_mask = np.isfinite(sif_true)

    print(
        f"SIF valid labels: "
        f"{int(sif_valid_mask.sum()):,}"
    )

    print(
        f"SIF unknown labels: "
        f"{int((~sif_valid_mask).sum()):,}"
    )

    if int(sif_valid_mask.sum()) < 2:
        raise RuntimeError(
            "Not enough valid SIF calibration labels."
        )

    sif_valid_values = sif_true[sif_valid_mask]

    sif_positive_count = int(
        np.sum(sif_valid_values == 1)
    )

    sif_negative_count = int(
        np.sum(sif_valid_values == 0)
    )

    print(
        f"SIF positives: "
        f"{sif_positive_count:,}"
    )

    print(
        f"SIF negatives: "
        f"{sif_negative_count:,}"
    )

    if sif_positive_count == 0 or sif_negative_count == 0:
        raise RuntimeError(
            "SIF calibration set must contain both "
            "positive and negative labels."
        )

    # IOGP labels.
    iogp_true = np.full(
        (len(df), len(IOGP_RULES)),
        np.nan,
        dtype=np.float64,
    )

    missing_rule_columns = []

    for index, rule in enumerate(IOGP_RULES):
        if rule not in df.columns:
            missing_rule_columns.append(rule)
            continue

        iogp_true[:, index] = np.array(
            [
                parse_iogp_label(value)
                for value in df[rule]
            ],
            dtype=np.float64,
        )

    if missing_rule_columns:
        raise RuntimeError(
            "Missing IOGP columns:\n"
            + "\n".join(
                f"  {rule}"
                for rule in missing_rule_columns
            )
        )

    total_iogp_valid = int(
        np.isfinite(iogp_true).sum()
    )

    total_iogp_unknown = int(
        np.isnan(iogp_true).sum()
    )

    print(
        f"IOGP valid cells: "
        f"{total_iogp_valid:,}"
    )

    print(
        f"IOGP unknown cells: "
        f"{total_iogp_unknown:,}"
    )

    # -------------------------------------------------------------------------
    # Load tokenizer
    # -------------------------------------------------------------------------

    print_section("LOADING TOKENIZER")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    print(
        f"Tokenizer loaded: {MODEL_NAME}"
    )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    model = load_model()

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    print_section("RUNNING CALIBRATION INFERENCE")

    print(
        f"Evaluating {len(texts):,} calibration records..."
    )

    sif_logits, iogp_logits = run_model_inference(
        model,
        tokenizer,
        texts,
    )

    if len(sif_logits) != len(df):
        raise RuntimeError(
            "SIF prediction count does not match "
            "calibration dataset."
        )

    if iogp_logits.shape != (
        len(df),
        len(IOGP_RULES),
    ):
        raise RuntimeError(
            "IOGP prediction shape mismatch: "
            f"{iogp_logits.shape}"
        )

    sif_probabilities = sigmoid(
        sif_logits
    )

    iogp_probabilities = sigmoid(
        iogp_logits
    )

    # -------------------------------------------------------------------------
    # SIF calibration
    # -------------------------------------------------------------------------

    print_section("SIF-P CALIBRATION")

    sif_calibration_true = sif_true[
        sif_valid_mask
    ]

    sif_calibration_probabilities = (
        sif_probabilities[sif_valid_mask]
    )

    sif_auc = calculate_auc_metrics(
        sif_calibration_true.astype(np.int64),
        sif_calibration_probabilities,
    )

    selected_sif, sif_threshold_results = (
        select_sif_threshold(
            sif_calibration_true.astype(np.int64),
            sif_calibration_probabilities,
        )
    )

    print(
        f"Selected SIF threshold: "
        f"{selected_sif['threshold']:.3f}"
    )

    print(
        f"Recall:    "
        f"{selected_sif['recall']:.6f}"
    )

    print(
        f"Precision: "
        f"{selected_sif['precision']:.6f}"
    )

    print(
        f"F1:        "
        f"{selected_sif['f1']:.6f}"
    )

    print(
        f"FPR:       "
        f"{selected_sif['fpr']}"
    )

    print(
        f"TP: {selected_sif['tp']}  "
        f"FP: {selected_sif['fp']}  "
        f"TN: {selected_sif['tn']}  "
        f"FN: {selected_sif['fn']}"
    )

    print(
        f"PR-AUC:    "
        f"{sif_auc['pr_auc']}"
    )

    print(
        f"ROC-AUC:   "
        f"{sif_auc['roc_auc']}"
    )

    # -------------------------------------------------------------------------
    # IOGP calibration
    # -------------------------------------------------------------------------

    print_section("IOGP THRESHOLD CALIBRATION")

    iogp_results: dict[str, Any] = {}

    selected_iogp_thresholds: dict[str, float] = {}

    for index, rule in enumerate(IOGP_RULES):
        valid_mask = np.isfinite(
            iogp_true[:, index]
        )

        y_true = iogp_true[
            valid_mask,
            index,
        ].astype(np.int64)

        probabilities = iogp_probabilities[
            valid_mask,
            index,
        ]

        result = calibrate_iogp_rule(
            y_true,
            probabilities,
        )

        iogp_results[rule] = result

        selected_iogp_thresholds[rule] = float(
            result["selected_threshold"]
        )

        selected_metrics = result[
            "selected_metrics"
        ]

        print()
        print(rule)

        print(
            f"  Status:       "
            f"{result['status']}"
        )

        print(
            f"  Valid:        "
            f"{result['valid']:,}"
        )

        print(
            f"  Positives:    "
            f"{result['positives']:,}"
        )

        print(
            f"  Negatives:    "
            f"{result['negatives']:,}"
        )

        print(
            f"  Prevalence:   "
            f"{result['prevalence']:.6f}"
        )

        print(
            f"  Threshold:    "
            f"{result['selected_threshold']:.3f}"
        )

        print(
            f"  Precision:    "
            f"{selected_metrics['precision']:.6f}"
        )

        print(
            f"  Recall:       "
            f"{selected_metrics['recall']:.6f}"
        )

        print(
            f"  F1:           "
            f"{selected_metrics['f1']:.6f}"
        )

        print(
            f"  PR-AUC:       "
            f"{result['pr_auc']}"
        )

        print(
            f"  ROC-AUC:      "
            f"{result['roc_auc']}"
        )

    # -------------------------------------------------------------------------
    # Create prediction dataframe
    # -------------------------------------------------------------------------

    print_section("SAVING CALIBRATION PREDICTIONS")

    prediction_df = pd.DataFrame()

    if "incident_id" in df.columns:
        prediction_df["incident_id"] = (
            df["incident_id"].astype(str)
        )

    prediction_df["sif_probability"] = (
        sif_probabilities
    )

    prediction_df["sif_logit"] = sif_logits

    prediction_df["sif_valid_label"] = (
        sif_true
    )

    prediction_df["sif_prediction_at_selected_threshold"] = (
        sif_probabilities
        >= selected_sif["threshold"]
    ).astype(int)

    for index, rule in enumerate(IOGP_RULES):
        prediction_df[
            f"{rule}_probability"
        ] = iogp_probabilities[:, index]

        prediction_df[
            f"{rule}_logit"
        ] = iogp_logits[:, index]

        prediction_df[
            f"{rule}_valid_label"
        ] = iogp_true[:, index]

        prediction_df[
            f"{rule}_prediction_at_selected_threshold"
        ] = (
            iogp_probabilities[:, index]
            >= selected_iogp_thresholds[rule]
        ).astype(int)

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(
        f"Predictions saved to:\n"
        f"{PREDICTIONS_FILE}"
    )

    # -------------------------------------------------------------------------
    # Save threshold selection
    # -------------------------------------------------------------------------

    selected_thresholds = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "checkpoint": "best.pt",
        "calibration_file": "iogp_calibration.csv",
        "max_length": MAX_LENGTH,
        "sif": {
            "target_recall": SIF_TARGET_RECALL,
            "selected_threshold": float(
                selected_sif["threshold"]
            ),
            "selection_method": (
                "highest_threshold_with_recall_at_least_0.95"
            ),
            "calibration_metrics": selected_sif,
        },
        "iogp": {
            "rules": IOGP_RULES,
            "thresholds": selected_iogp_thresholds,
            "selection_method": (
                "maximum_f1_when_both_classes_available"
            ),
            "details": iogp_results,
        },
    }

    with open(
        THRESHOLDS_FILE,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_safe(selected_thresholds),
            handle,
            indent=2,
        )

    print(
        f"Selected thresholds saved to:\n"
        f"{THRESHOLDS_FILE}"
    )

    # -------------------------------------------------------------------------
    # Save detailed metrics
    # -------------------------------------------------------------------------

    detailed_metrics = {
        "schema_version": 1,
        "model_name": MODEL_NAME,
        "checkpoint": str(CHECKPOINT_FILE),
        "calibration_file": str(CALIBRATION_FILE),
        "records": len(df),
        "batch_size": BATCH_SIZE,
        "max_length": MAX_LENGTH,
        "device": str(DEVICE),
        "sif": {
            "valid": int(sif_valid_mask.sum()),
            "unknown": int((~sif_valid_mask).sum()),
            "positives": sif_positive_count,
            "negatives": sif_negative_count,
            "selected": selected_sif,
            "pr_auc": sif_auc["pr_auc"],
            "roc_auc": sif_auc["roc_auc"],
            "threshold_grid": sif_threshold_results,
        },
        "iogp": iogp_results,
    }

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            json_safe(detailed_metrics),
            handle,
            indent=2,
        )

    print(
        f"Detailed metrics saved to:\n"
        f"{METRICS_FILE}"
    )

    # -------------------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------------------

    print_section("CALIBRATION COMPLETE")

    print("SIF-P")
    print(
        f"  Selected threshold: "
        f"{selected_sif['threshold']:.3f}"
    )

    print(
        f"  Recall:    "
        f"{selected_sif['recall']:.4f}"
    )

    print(
        f"  Precision: "
        f"{selected_sif['precision']:.4f}"
    )

    print(
        f"  F1:        "
        f"{selected_sif['f1']:.4f}"
    )

    print(
        f"  FPR:       "
        f"{selected_sif['fpr']}"
    )

    print()
    print("IOGP thresholds:")

    for rule in IOGP_RULES:
        result = iogp_results[rule]

        print(
            f"  {rule:32s} "
            f"{result['selected_threshold']:.3f}"
        )

    print()
    print("Output files:")
    print(f"  {PREDICTIONS_FILE}")
    print(f"  {METRICS_FILE}")
    print(f"  {THRESHOLDS_FILE}")

    print()
    print(
        "IMPORTANT: These thresholds are calibration results only. "
        "They have NOT been evaluated against the held-out test set."
    )


if __name__ == "__main__":
    main()
