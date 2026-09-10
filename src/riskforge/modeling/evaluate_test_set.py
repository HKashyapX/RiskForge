"""
RiskForge held-out test-set evaluation.

Evaluates the already-trained best.pt checkpoint on:

    C:\\data\\processed\\iogp_test.csv

No training is performed.

The script reports:

SIF:
    - Recall
    - Precision
    - F1
    - PR-AUC
    - ROC-AUC
    - Accuracy
    - Confusion matrix
    - Brier score
    - ECE
    - Positive/negative counts

IOGP:
    - Per-rule precision
    - Per-rule recall
    - Per-rule F1
    - Per-rule PR-AUC
    - Per-rule prevalence
    - Macro F1
    - Micro F1
    - Label support

Additional:
    - Multi-label exact match where all 9 labels are known
    - Average inference time
    - Peak GPU memory
    - Raw predictions saved to CSV
    - Complete metrics saved to JSON

IMPORTANT:
    The test set is never used to select thresholds.
    The default classification threshold is 0.50.
    Threshold selection should be performed using the calibration set.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
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

DATA_DIR = modeling_data_dir()

TEST_FILE = DATA_DIR / "iogp_test.csv"

CHECKPOINT_DIR = (
    DATA_DIR
    / "deberta_multitask_checkpoints"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best.pt"
)

OUTPUT_DIR = (
    CHECKPOINT_DIR
    / "test_evaluation"
)

PREDICTIONS_FILE = (
    OUTPUT_DIR
    / "test_predictions.csv"
)

METRICS_FILE = (
    OUTPUT_DIR
    / "test_metrics_detailed.json"
)

MODEL_NAME = (
    "microsoft/deberta-v3-base"
)

MAX_LENGTH = 128

BATCH_SIZE = 16

THRESHOLD = 0.50

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


# =============================================================================
# DISPLAY
# =============================================================================

def header(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# SAFE METRICS
# =============================================================================

def safe_roc_auc(
    y_true,
    y_score,
):

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(
        roc_auc_score(
            y_true,
            y_score,
        )
    )


def safe_pr_auc(
    y_true,
    y_score,
):

    if len(np.unique(y_true)) < 2:
        return float("nan")

    return float(
        average_precision_score(
            y_true,
            y_score,
        )
    )


def safe_brier(
    y_true,
    y_score,
):

    if len(y_true) == 0:
        return float("nan")

    return float(
        brier_score_loss(
            y_true,
            y_score,
        )
    )


def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return float("nan")

    return float(
        numerator / denominator
    )


# =============================================================================
# EXPECTED CALIBRATION ERROR
# =============================================================================

def expected_calibration_error(
    y_true,
    y_score,
    n_bins: int = 10,
):

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    y_score = np.asarray(
        y_score,
        dtype=np.float64,
    )

    if len(y_true) == 0:
        return float("nan")

    ece = 0.0

    bin_edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    for index in range(n_bins):

        lower = bin_edges[index]
        upper = bin_edges[index + 1]

        if index == n_bins - 1:

            mask = (
                (y_score >= lower)
                & (y_score <= upper)
            )

        else:

            mask = (
                (y_score >= lower)
                & (y_score < upper)
            )

        if not mask.any():
            continue

        confidence = np.mean(
            y_score[mask]
        )

        accuracy = np.mean(
            y_true[mask]
        )

        fraction = (
            np.sum(mask)
            / len(y_true)
        )

        ece += (
            fraction
            * abs(
                confidence
                - accuracy
            )
        )

    return float(ece)


# =============================================================================
# MODEL IMPORT
# =============================================================================

def import_training_model():

    modeling_dir = (
        Path(__file__).resolve().parent
    )

    if str(modeling_dir) not in sys.path:

        sys.path.insert(
            0,
            str(modeling_dir),
        )

    from train_multitask_deberta import (
        MultiTaskDeberta,
    )

    return MultiTaskDeberta


# =============================================================================
# CHECKPOINT
# =============================================================================

def load_checkpoint():

    if not BEST_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{BEST_CHECKPOINT}"
        )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location="cpu",
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    print(
        f"Checkpoint: "
        f"{BEST_CHECKPOINT}"
    )

    print(
        "Checkpoint keys:"
    )

    for key in checkpoint.keys():

        print(
            f"  {key}"
        )

    return checkpoint


def extract_state_dict(
    checkpoint,
):

    if (
        "model_state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    elif (
        "state_dict"
        in checkpoint
    ):

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    else:

        raise RuntimeError(
            "No model_state_dict found "
            "in checkpoint."
        )

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned[key] = value

    return cleaned


# =============================================================================
# MODEL
# =============================================================================

def build_model(
    device,
):

    header(
        "LOADING MODEL"
    )

    MultiTaskDeberta = (
        import_training_model()
    )

    model = MultiTaskDeberta(
        MODEL_NAME,
        len(IOGP_RULES),
    )

    checkpoint = (
        load_checkpoint()
    )

    # -------------------------------------------------------------------------
    # Metadata validation.
    # -------------------------------------------------------------------------

    checkpoint_model_name = (
        checkpoint.get(
            "model_name"
        )
    )

    if (
        checkpoint_model_name
        is not None
    ):

        print(
            f"Checkpoint model name: "
            f"{checkpoint_model_name}"
        )

        if (
            checkpoint_model_name
            != MODEL_NAME
        ):

            raise RuntimeError(
                "Checkpoint model name "
                "does not match."
            )

    checkpoint_rules = (
        checkpoint.get(
            "rules"
        )
    )

    if checkpoint_rules is not None:

        if list(
            checkpoint_rules
        ) != IOGP_RULES:

            raise RuntimeError(
                "Checkpoint IOGP rule order "
                "does not match."
            )

        print(
            "Checkpoint IOGP rule order: PASS"
        )

    checkpoint_max_length = (
        checkpoint.get(
            "max_length"
        )
    )

    if checkpoint_max_length is not None:

        print(
            f"Checkpoint max length: "
            f"{checkpoint_max_length}"
        )

    # -------------------------------------------------------------------------
    # Load model state.
    # -------------------------------------------------------------------------

    state_dict = (
        extract_state_dict(
            checkpoint
        )
    )

    result = model.load_state_dict(
        state_dict,
        strict=False,
    )

    print(
        f"Missing keys: "
        f"{len(result.missing_keys)}"
    )

    print(
        f"Unexpected keys: "
        f"{len(result.unexpected_keys)}"
    )

    if result.missing_keys:

        for key in (
            result.missing_keys[:20]
        ):

            print(
                f"  MISSING: {key}"
            )

        raise RuntimeError(
            "Checkpoint has missing keys."
        )

    if result.unexpected_keys:

        for key in (
            result.unexpected_keys[:20]
        ):

            print(
                f"  UNEXPECTED: {key}"
            )

        raise RuntimeError(
            "Checkpoint has unexpected keys."
        )

    model.to(device)
    model.eval()

    return model


# =============================================================================
# SIF TARGETS
# =============================================================================

def parse_sif_targets(
    dataframe,
):

    if "sif_potential" not in (
        dataframe.columns
    ):

        raise ValueError(
            "Test file does not contain "
            "'sif_potential'."
        )

    values = (
        dataframe[
            "sif_potential"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.strip()
    )

    targets = np.full(
        len(values),
        -1,
        dtype=np.int8,
    )

    positive_values = {
        "yes",
        "sif",
        "sif_p",
    }

    negative_values = {
        "no",
        "non_sif",
        "non-sif",
    }

    positive_mask = (
        values.isin(
            positive_values
        )
    )

    negative_mask = (
        values.isin(
            negative_values
        )
    )

    targets[
        positive_mask.to_numpy()
    ] = 1

    targets[
        negative_mask.to_numpy()
    ] = 0

    return targets


# =============================================================================
# TEXT
# =============================================================================

def get_texts(
    dataframe,
):

    if "text_normalized" not in (
        dataframe.columns
    ):

        raise ValueError(
            "Test file does not contain "
            "'text_normalized'."
        )

    texts = (
        dataframe[
            "text_normalized"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    empty_count = sum(
        not text.strip()
        for text in texts
    )

    if empty_count:

        raise ValueError(
            f"Test set contains "
            f"{empty_count} empty narratives."
        )

    return texts


# =============================================================================
# IOGP TARGETS
# =============================================================================

def get_iogp_targets(
    dataframe,
):

    missing = [
        rule
        for rule in IOGP_RULES
        if rule not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            "Missing IOGP target columns:\n"
            + "\n".join(missing)
        )

    targets = (
        dataframe[
            IOGP_RULES
        ]
        .astype(np.int8)
        .to_numpy()
    )

    # Validate target values.
    invalid = ~np.isin(
        targets,
        [-1, 0, 1],
    )

    if invalid.any():

        raise ValueError(
            "IOGP targets contain values "
            "other than -1, 0, or 1."
        )

    return targets


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize_batch(
    tokenizer,
    texts,
    device,
):

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    return {
        key: value.to(device)
        for key, value in encoded.items()
    }


# =============================================================================
# INFERENCE
# =============================================================================

def infer_batch(
    model,
    tokenizer,
    texts,
    device,
):

    inputs = tokenize_batch(
        tokenizer,
        texts,
        device,
    )

    with torch.inference_mode():

        try:

            outputs = model(
                **inputs
            )

        except RuntimeError as exc:

            message = str(exc)

            if (
                "mat1 and mat2 must have "
                "the same dtype"
                not in message
            ):

                raise

            # -----------------------------------------------------------------
            # FP16 encoder / FP32 task-head compatibility.
            # -----------------------------------------------------------------

            sif_head = getattr(
                model,
                "sif_head",
                None,
            )

            iogp_head = getattr(
                model,
                "iogp_head",
                None,
            )

            if (
                sif_head is None
                or iogp_head is None
            ):

                raise RuntimeError(
                    "Could not find task heads "
                    "for dtype compatibility fix."
                )

            sif_dtype = next(
                sif_head.parameters()
            ).dtype

            iogp_dtype = next(
                iogp_head.parameters()
            ).dtype

            sif_head.half()
            iogp_head.half()

            try:

                outputs = model(
                    **inputs
                )

            finally:

                if (
                    sif_dtype
                    == torch.float32
                ):

                    sif_head.float()

                elif (
                    sif_dtype
                    == torch.float64
                ):

                    sif_head.double()

                if (
                    iogp_dtype
                    == torch.float32
                ):

                    iogp_head.float()

                elif (
                    iogp_dtype
                    == torch.float64
                ):

                    iogp_head.double()

    if not isinstance(
        outputs,
        dict,
    ):

        raise RuntimeError(
            "Model output is not a dictionary."
        )

    if (
        "sif_logits"
        not in outputs
    ):

        raise RuntimeError(
            "Missing sif_logits."
        )

    if (
        "iogp_logits"
        not in outputs
    ):

        raise RuntimeError(
            "Missing iogp_logits."
        )

    sif_logits = (
        outputs[
            "sif_logits"
        ]
    )

    iogp_logits = (
        outputs[
            "iogp_logits"
        ]
    )

    if not torch.isfinite(
        sif_logits
    ).all():

        raise RuntimeError(
            "SIF logits contain NaN/Inf."
        )

    if not torch.isfinite(
        iogp_logits
    ).all():

        raise RuntimeError(
            "IOGP logits contain NaN/Inf."
        )

    return (
        sif_logits
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1),

        iogp_logits
        .detach()
        .float()
        .cpu()
        .numpy(),
    )


# =============================================================================
# FULL TEST-SET INFERENCE
# =============================================================================

def run_test_inference(
    model,
    tokenizer,
    dataframe,
    texts,
    device,
):

    total_rows = len(
        dataframe
    )

    all_sif_logits = []
    all_iogp_logits = []

    total_time = 0.0

    print()
    print(
        f"Evaluating "
        f"{total_rows:,} test records..."
    )

    number_of_batches = math.ceil(
        total_rows
        / BATCH_SIZE
    )

    for batch_index in range(
        number_of_batches
    ):

        start_index = (
            batch_index
            * BATCH_SIZE
        )

        end_index = min(
            start_index
            + BATCH_SIZE,
            total_rows,
        )

        batch_texts = texts[
            start_index:end_index
        ]

        if device.type == "cuda":

            torch.cuda.synchronize()

        start_time = (
            time.perf_counter()
        )

        sif_logits, iogp_logits = (
            infer_batch(
                model,
                tokenizer,
                batch_texts,
                device,
            )
        )

        if device.type == "cuda":

            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start_time
        )

        total_time += elapsed

        all_sif_logits.append(
            sif_logits
        )

        all_iogp_logits.append(
            iogp_logits
        )

        if (
            batch_index == 0
            or (
                batch_index + 1
            ) % 10 == 0
            or (
                batch_index + 1
                == number_of_batches
            )
        ):

            completed = end_index

            percent = (
                completed
                / total_rows
                * 100.0
            )

            print(
                f"  Batch "
                f"{batch_index + 1:4d}/"
                f"{number_of_batches:4d} "
                f"| "
                f"{completed:5d}/"
                f"{total_rows:5d} "
                f"| "
                f"{percent:6.2f}%"
            )

    sif_logits = np.concatenate(
        all_sif_logits
    )

    iogp_logits = np.concatenate(
        all_iogp_logits
    )

    return (
        sif_logits,
        iogp_logits,
        total_time,
    )


# =============================================================================
# SIF EVALUATION
# =============================================================================

def evaluate_sif(
    sif_targets,
    sif_logits,
):

    header(
        "SIF-P TEST RESULTS"
    )

    valid = (
        sif_targets >= 0
    )

    y_true = (
        sif_targets[valid]
        .astype(int)
    )

    logits = (
        sif_logits[valid]
    )

    y_score = (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    logits,
                    -50,
                    50,
                )
            )
        )
    )

    y_pred = (
        y_score
        >= THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()
    )

    metrics = {}

    metrics[
        "sif_test_records_total"
    ] = int(
        len(sif_targets)
    )

    metrics[
        "sif_test_records_valid"
    ] = int(
        len(y_true)
    )

    metrics[
        "sif_positive_count"
    ] = int(
        np.sum(y_true == 1)
    )

    metrics[
        "sif_negative_count"
    ] = int(
        np.sum(y_true == 0)
    )

    metrics[
        "sif_threshold"
    ] = THRESHOLD

    metrics[
        "sif_accuracy"
    ] = float(
        accuracy_score(
            y_true,
            y_pred,
        )
    )

    metrics[
        "sif_recall"
    ] = float(
        recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    metrics[
        "sif_precision"
    ] = float(
        precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    metrics[
        "sif_f1"
    ] = float(
        f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )
    )

    metrics[
        "sif_pr_auc"
    ] = safe_pr_auc(
        y_true,
        y_score,
    )

    metrics[
        "sif_roc_auc"
    ] = safe_roc_auc(
        y_true,
        y_score,
    )

    metrics[
        "sif_brier_score"
    ] = safe_brier(
        y_true,
        y_score,
    )

    metrics[
        "sif_ece"
    ] = expected_calibration_error(
        y_true,
        y_score,
    )

    metrics[
        "sif_true_negative"
    ] = int(tn)

    metrics[
        "sif_false_positive"
    ] = int(fp)

    metrics[
        "sif_false_negative"
    ] = int(fn)

    metrics[
        "sif_true_positive"
    ] = int(tp)

    metrics[
        "sif_false_positive_rate"
    ] = safe_divide(
        fp,
        fp + tn,
    )

    metrics[
        "sif_negative_rate"
    ] = float(
        np.mean(
            y_true == 0
        )
    )

    metrics[
        "sif_positive_rate"
    ] = float(
        np.mean(
            y_true == 1
        )
    )

    print(
        f"Valid records:       "
        f"{len(y_true):,}"
    )

    print(
        f"Positive records:    "
        f"{np.sum(y_true == 1):,}"
    )

    print(
        f"Negative records:    "
        f"{np.sum(y_true == 0):,}"
    )

    print()
    print(
        f"Accuracy:             "
        f"{metrics['sif_accuracy']:.6f}"
    )

    print(
        f"Recall:               "
        f"{metrics['sif_recall']:.6f}"
    )

    print(
        f"Precision:            "
        f"{metrics['sif_precision']:.6f}"
    )

    print(
        f"F1:                   "
        f"{metrics['sif_f1']:.6f}"
    )

    print(
        f"PR-AUC:               "
        f"{metrics['sif_pr_auc']:.6f}"
    )

    print(
        f"ROC-AUC:              "
        f"{metrics['sif_roc_auc']:.6f}"
    )

    print(
        f"Brier score:          "
        f"{metrics['sif_brier_score']:.6f}"
    )

    print(
        f"ECE:                  "
        f"{metrics['sif_ece']:.6f}"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        f"  TN: {tn:,}"
    )

    print(
        f"  FP: {fp:,}"
    )

    print(
        f"  FN: {fn:,}"
    )

    print(
        f"  TP: {tp:,}"
    )

    return (
        metrics,
        y_score,
        y_pred,
    )


# =============================================================================
# IOGP EVALUATION
# =============================================================================

def evaluate_iogp(
    iogp_targets,
    iogp_logits,
):

    header(
        "IOGP TEST RESULTS"
    )

    all_true = []
    all_pred = []

    rule_metrics = {}

    for index, rule in enumerate(
        IOGP_RULES
    ):

        target_column = (
            iogp_targets[
                :,
                index,
            ]
        )

        valid = (
            target_column >= 0
        )

        y_true = (
            target_column[valid]
            .astype(int)
        )

        logits = (
            iogp_logits[
                valid,
                index,
            ]
        )

        if len(y_true) == 0:

            print()
            print(
                f"{rule}: "
                "NO VALID LABELS"
            )

            continue

        y_score = (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        logits,
                        -50,
                        50,
                    )
                )
            )
        )

        y_pred = (
            y_score
            >= THRESHOLD
        ).astype(int)

        precision = (
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        recall = (
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        f1 = (
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        pr_auc = safe_pr_auc(
            y_true,
            y_score,
        )

        support_positive = int(
            np.sum(
                y_true == 1
            )
        )

        support_negative = int(
            np.sum(
                y_true == 0
            )
        )

        prevalence = float(
            np.mean(
                y_true == 1
            )
        )

        rule_metrics[rule] = {
            "valid_records": int(
                len(y_true)
            ),
            "positive_count":
                support_positive,
            "negative_count":
                support_negative,
            "prevalence":
                prevalence,
            "precision":
                float(precision),
            "recall":
                float(recall),
            "f1":
                float(f1),
            "pr_auc":
                pr_auc,
        }

        all_true.append(
            y_true
        )

        all_pred.append(
            y_pred
        )

        print()
        print(
            f"{rule}"
        )

        print(
            f"  Valid:       "
            f"{len(y_true):,}"
        )

        print(
            f"  Positives:   "
            f"{support_positive:,}"
        )

        print(
            f"  Precision:   "
            f"{precision:.6f}"
        )

        print(
            f"  Recall:      "
            f"{recall:.6f}"
        )

        print(
            f"  F1:          "
            f"{f1:.6f}"
        )

        if np.isfinite(
            pr_auc
        ):

            print(
                f"  PR-AUC:      "
                f"{pr_auc:.6f}"
            )

        else:

            print(
                "  PR-AUC:      NaN"
            )

        print(
            f"  Prevalence:  "
            f"{prevalence:.6f}"
        )

    # -------------------------------------------------------------------------
    # Macro F1.
    # -------------------------------------------------------------------------

    valid_f1 = [
        values["f1"]
        for values in (
            rule_metrics.values()
        )
        if np.isfinite(
            values["f1"]
        )
    ]

    if valid_f1:

        macro_f1 = float(
            np.mean(valid_f1)
        )

    else:

        macro_f1 = float("nan")

    # -------------------------------------------------------------------------
    # Micro F1.
    # -------------------------------------------------------------------------

    if all_true:

        concatenated_true = (
            np.concatenate(
                all_true
            )
        )

        concatenated_pred = (
            np.concatenate(
                all_pred
            )
        )

        micro_f1 = float(
            f1_score(
                concatenated_true,
                concatenated_pred,
                zero_division=0,
            )
        )

    else:

        micro_f1 = float("nan")

    print()
    print(
        "-" * 80
    )

    print(
        f"IOGP Macro F1: "
        f"{macro_f1:.6f}"
    )

    print(
        f"IOGP Micro F1: "
        f"{micro_f1:.6f}"
    )

    return (
        rule_metrics,
        macro_f1,
        micro_f1,
    )


# =============================================================================
# MULTI-LABEL EXACT MATCH
# =============================================================================

def evaluate_exact_match(
    iogp_targets,
    iogp_logits,
):

    header(
        "IOGP EXACT-MATCH ANALYSIS"
    )

    # Only evaluate records for which all
    # nine labels are known.
    fully_known = np.all(
        iogp_targets >= 0,
        axis=1,
    )

    count = int(
        np.sum(fully_known)
    )

    if count == 0:

        print(
            "No records have all 9 "
            "IOGP labels known."
        )

        return {
            "fully_known_records": 0,
            "exact_match_accuracy":
                float("nan"),
        }

    y_true = (
        iogp_targets[
            fully_known
        ]
        .astype(int)
    )

    logits = (
        iogp_logits[
            fully_known
        ]
    )

    probabilities = (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    logits,
                    -50,
                    50,
                )
            )
        )
    )

    y_pred = (
        probabilities
        >= THRESHOLD
    ).astype(int)

    exact_match = np.all(
        y_true == y_pred,
        axis=1,
    )

    accuracy = float(
        np.mean(
            exact_match
        )
    )

    print(
        f"Fully-known records: "
        f"{count:,}"
    )

    print(
        f"Exact-match accuracy: "
        f"{accuracy:.6f}"
    )

    return {
        "fully_known_records":
            count,
        "exact_match_accuracy":
            accuracy,
    }


# =============================================================================
# SAVE PREDICTIONS
# =============================================================================

def save_predictions(
    dataframe,
    sif_logits,
    sif_targets,
    iogp_logits,
    iogp_targets,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = pd.DataFrame()

    if "incident_id" in (
        dataframe.columns
    ):

        result[
            "incident_id"
        ] = dataframe[
            "incident_id"
        ].values

    result[
        "sif_target"
    ] = sif_targets

    result[
        "sif_probability"
    ] = (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    sif_logits,
                    -50,
                    50,
                )
            )
        )
    )

    result[
        "sif_prediction"
    ] = (
        result[
            "sif_probability"
        ]
        >= THRESHOLD
    ).astype(int)

    for index, rule in enumerate(
        IOGP_RULES
    ):

        result[
            f"{rule}_target"
        ] = iogp_targets[
            :,
            index,
        ]

        probability = (
            1.0
            / (
                1.0
                + np.exp(
                    -np.clip(
                        iogp_logits[
                            :,
                            index,
                        ],
                        -50,
                        50,
                    )
                )
            )
        )

        result[
            f"{rule}_probability"
        ] = probability

        result[
            f"{rule}_prediction"
        ] = (
            probability
            >= THRESHOLD
        ).astype(int)

    result.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print()
    print(
        f"Predictions saved to:"
    )

    print(
        PREDICTIONS_FILE
    )


# =============================================================================
# JSON CLEANING
# =============================================================================

def make_json_safe(
    value,
):

    if isinstance(
        value,
        dict,
    ):

        return {
            str(key):
                make_json_safe(
                    item
                )
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        list,
    ):

        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        tuple,
    ):

        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.integer,
    ):

        return int(value)

    if isinstance(
        value,
        np.floating,
    ):

        if np.isnan(value):
            return None

        if np.isinf(value):

            if value > 0:
                return "Infinity"

            return "-Infinity"

        return float(value)

    if isinstance(
        value,
        float,
    ):

        if math.isnan(value):
            return None

        if math.isinf(value):

            if value > 0:
                return "Infinity"

            return "-Infinity"

        return value

    return value


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print(
        "RISKFORGE HELD-OUT TEST-SET EVALUATION"
    )
    print("=" * 80)

    print()
    print(
        f"Test file:"
    )

    print(
        TEST_FILE
    )

    print()
    print(
        f"Checkpoint:"
    )

    print(
        BEST_CHECKPOINT
    )

    print()
    print(
        f"Batch size: "
        f"{BATCH_SIZE}"
    )

    print(
        f"Max sequence length: "
        f"{MAX_LENGTH}"
    )

    print(
        f"Classification threshold: "
        f"{THRESHOLD}"
    )

    # -------------------------------------------------------------------------
    # Device.
    # -------------------------------------------------------------------------

    if torch.cuda.is_available():

        device = torch.device(
            "cuda"
        )

        print()
        print(
            f"Device: "
            f"{device}"
        )

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    else:

        device = torch.device(
            "cpu"
        )

        print()
        print(
            "CUDA unavailable. "
            "Using CPU."
        )

    # -------------------------------------------------------------------------
    # Test dataset.
    # -------------------------------------------------------------------------

    header(
        "LOADING TEST DATASET"
    )

    if not TEST_FILE.exists():

        raise FileNotFoundError(
            "Test dataset not found:\n"
            f"{TEST_FILE}"
        )

    dataframe = pd.read_csv(
        TEST_FILE,
        low_memory=False,
    )

    print(
        f"Test records: "
        f"{len(dataframe):,}"
    )

    if len(dataframe) == 0:

        raise RuntimeError(
            "Test dataset is empty."
        )

    # -------------------------------------------------------------------------
    # ID validation.
    # -------------------------------------------------------------------------

    if "incident_id" in (
        dataframe.columns
    ):

        duplicate_ids = (
            dataframe[
                "incident_id"
            ]
            .duplicated()
            .sum()
        )

        print(
            f"Duplicate incident IDs: "
            f"{duplicate_ids}"
        )

        if duplicate_ids:

            raise RuntimeError(
                "Duplicate incident IDs "
                "found in test set."
            )

    # -------------------------------------------------------------------------
    # Targets.
    # -------------------------------------------------------------------------

    sif_targets = (
        parse_sif_targets(
            dataframe
        )
    )

    iogp_targets = (
        get_iogp_targets(
            dataframe
        )
    )

    texts = get_texts(
        dataframe
    )

    print(
        f"SIF valid labels: "
        f"{np.sum(sif_targets >= 0):,}"
    )

    print(
        f"SIF unknown labels: "
        f"{np.sum(sif_targets < 0):,}"
    )

    print(
        f"IOGP valid cells: "
        f"{np.sum(iogp_targets >= 0):,}"
    )

    print(
        f"IOGP unknown cells: "
        f"{np.sum(iogp_targets < 0):,}"
    )

    # -------------------------------------------------------------------------
    # Tokenizer.
    # -------------------------------------------------------------------------

    header(
        "LOADING TOKENIZER"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
    )

    # -------------------------------------------------------------------------
    # Model.
    # -------------------------------------------------------------------------

    model = build_model(
        device
    )

    # -------------------------------------------------------------------------
    # Inference.
    # -------------------------------------------------------------------------

    header(
        "RUNNING HELD-OUT TEST INFERENCE"
    )

    start_total = (
        time.perf_counter()
    )

    (
        sif_logits,
        iogp_logits,
        inference_time,
    ) = run_test_inference(
        model,
        tokenizer,
        dataframe,
        texts,
        device,
    )

    total_elapsed = (
        time.perf_counter()
        - start_total
    )

    print()
    print(
        f"Inference time: "
        f"{inference_time:.3f} seconds"
    )

    print(
        f"Total evaluation time: "
        f"{total_elapsed:.3f} seconds"
    )

    print(
        f"Records/second: "
        f"{len(dataframe) / inference_time:.2f}"
    )

    print(
        f"Average ms/record: "
        f"{inference_time / len(dataframe) * 1000:.3f}"
    )

    if device.type == "cuda":

        peak_memory = (
            torch.cuda.max_memory_allocated()
            / (1024 ** 3)
        )

        print(
            f"Peak GPU memory: "
            f"{peak_memory:.3f} GB"
        )

    else:

        peak_memory = None

    # -------------------------------------------------------------------------
    # SIF.
    # -------------------------------------------------------------------------

    (
        sif_metrics,
        sif_probabilities,
        sif_predictions,
    ) = evaluate_sif(
        sif_targets,
        sif_logits,
    )

    # -------------------------------------------------------------------------
    # IOGP.
    # -------------------------------------------------------------------------

    (
        iogp_rule_metrics,
        iogp_macro_f1,
        iogp_micro_f1,
    ) = evaluate_iogp(
        iogp_targets,
        iogp_logits,
    )

    # -------------------------------------------------------------------------
    # Exact match.
    # -------------------------------------------------------------------------

    exact_match_metrics = (
        evaluate_exact_match(
            iogp_targets,
            iogp_logits,
        )
    )

    # -------------------------------------------------------------------------
    # Save predictions.
    # -------------------------------------------------------------------------

    save_predictions(
        dataframe,
        sif_logits,
        sif_targets,
        iogp_logits,
        iogp_targets,
    )

    # -------------------------------------------------------------------------
    # Build complete metrics object.
    # -------------------------------------------------------------------------

    metrics = {
        "evaluation": {
            "model_name":
                MODEL_NAME,
            "checkpoint":
                str(BEST_CHECKPOINT),
            "test_file":
                str(TEST_FILE),
            "test_records":
                int(len(dataframe)),
            "max_length":
                MAX_LENGTH,
            "batch_size":
                BATCH_SIZE,
            "threshold":
                THRESHOLD,
        },

        "iogp_rule_order":
            IOGP_RULES,

        "sif": sif_metrics,

        "iogp": {
            "macro_f1":
                iogp_macro_f1,
            "micro_f1":
                iogp_micro_f1,
            "rules":
                iogp_rule_metrics,
        },

        "iogp_exact_match":
            exact_match_metrics,

        "performance": {
            "inference_seconds":
                inference_time,
            "total_evaluation_seconds":
                total_elapsed,
            "records_per_second":
                len(dataframe)
                / inference_time,
            "average_ms_per_record":
                inference_time
                / len(dataframe)
                * 1000,
            "peak_gpu_memory_gb":
                peak_memory,
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        METRICS_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            make_json_safe(
                metrics
            ),
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final summary.
    # -------------------------------------------------------------------------

    header(
        "FINAL TEST-SET SUMMARY"
    )

    print(
        f"SIF Accuracy:        "
        f"{sif_metrics['sif_accuracy']:.4f}"
    )

    print(
        f"SIF Recall:          "
        f"{sif_metrics['sif_recall']:.4f}"
    )

    print(
        f"SIF Precision:       "
        f"{sif_metrics['sif_precision']:.4f}"
    )

    print(
        f"SIF F1:              "
        f"{sif_metrics['sif_f1']:.4f}"
    )

    print(
        f"SIF PR-AUC:          "
        f"{sif_metrics['sif_pr_auc']:.4f}"
    )

    print(
        f"SIF ROC-AUC:         "
        f"{sif_metrics['sif_roc_auc']:.4f}"
    )

    print()

    print(
        f"IOGP Macro F1:       "
        f"{iogp_macro_f1:.4f}"
    )

    print(
        f"IOGP Micro F1:       "
        f"{iogp_micro_f1:.4f}"
    )

    print()

    print(
        f"Inference time:      "
        f"{inference_time:.2f} s"
    )

    print(
        f"Average latency:     "
        f"{inference_time / len(dataframe) * 1000:.3f} ms/record"
    )

    if peak_memory is not None:

        print(
            f"Peak GPU memory:     "
            f"{peak_memory:.3f} GB"
        )

    print()
    print(
        f"Detailed metrics:"
    )

    print(
        METRICS_FILE
    )

    print()
    print(
        f"Predictions:"
    )

    print(
        PREDICTIONS_FILE
    )

    print()
    print("=" * 80)
    print(
        "HELD-OUT TEST EVALUATION COMPLETE"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
