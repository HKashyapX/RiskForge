from __future__ import annotations

import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Configure CUDA allocator before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path(r"C:\data\processed")

TRAIN_FILE = DATA_DIR / "iogp_train.csv"
CALIBRATION_FILE = DATA_DIR / "iogp_calibration.csv"
TEST_FILE = DATA_DIR / "iogp_test.csv"

OUTPUT_DIR = DATA_DIR / "deberta_multitask_checkpoints"

MODEL_NAME = "microsoft/deberta-v3-base"

MAX_LENGTH = 128

# RTX 4050 6 GB configuration.
#
# Physical batch = 1
# Accumulation = 32
# Effective batch = 32
#
# This configuration is designed for full DeBERTa-v3-base fine-tuning on a
# 6 GB RTX 4050 under Windows WDDM. Sequence length is 128 to keep attention
# activation memory within the available CUDA budget.
PHYSICAL_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 32

EPOCHS = 5

LEARNING_RATE = 2e-5

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.10

MAX_GRAD_NORM = 1.0

SEED = 42

NUM_WORKERS = 0

SIF_LOSS_WEIGHT = 1.0

IOGP_LOSS_WEIGHT = 0.5

SIF_GAMMA = 2.0

SIF_ALPHA = 0.75

# Enable gradient checkpointing to reduce activation memory.
GRADIENT_CHECKPOINTING = True

# LoRA configuration for full DeBERTa-v3-base on a 6 GB RTX 4050.
# The base encoder remains the required DeBERTa-v3-base architecture but is
# frozen; only LoRA adapters and the two task heads are trainable.
LORA_ENABLED = True
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "query_proj",
    "key_proj",
    "value_proj",
]


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# EXACT IOGP RULE ORDER
# =============================================================================

RULES = [
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

NUM_RULES = len(RULES)

UNKNOWN = -1


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int) -> None:

    random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# DATASET
# =============================================================================

class IncidentDataset(Dataset):

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer,
        max_length: int,
    ):

        self.df = dataframe.reset_index(drop=True)

        self.tokenizer = tokenizer

        self.max_length = max_length

        self.texts = (
            self.df["text_normalized"]
            .fillna("")
            .astype(str)
            .tolist()
        )

        self.sif_labels = (
            self._load_sif_labels()
        )

        self.iogp_labels = (
            self.df[RULES]
            .astype(np.int8)
            .to_numpy()
        )

        self.iogp_weights = (
            self.df[
                [
                    f"{rule}_weight"
                    for rule in RULES
                ]
            ]
            .astype(np.float32)
            .to_numpy()
        )

    def _load_sif_labels(self) -> np.ndarray:

        if "sif_potential" not in self.df.columns:

            raise ValueError(
                "Training data does not contain "
                "sif_potential."
            )

        values = (
            self.df["sif_potential"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )

        labels = np.full(
            len(values),
            UNKNOWN,
            dtype=np.int8,
        )

        labels[
            values.isin(
                [
                    "yes",
                    "sif",
                    "sif_p",
                ]
            ).to_numpy()
        ] = 1

        labels[
            values.isin(
                [
                    "no",
                    "non_sif",
                    "non-sif",
                ]
            ).to_numpy()
        ] = 0

        # "possible" is intentionally NOT converted to 1.
        #
        # It is uncertain and should remain unknown until we create the
        # proper Dawid-Skene posterior target.

        return labels

    def __len__(self) -> int:

        return len(self.texts)

    def __getitem__(self, index: int):

        text = self.texts[index]

        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        item = {
            "input_ids": (
                encoded["input_ids"]
                .squeeze(0)
            ),

            "attention_mask": (
                encoded["attention_mask"]
                .squeeze(0)
            ),

            "sif_label": torch.tensor(
                self.sif_labels[index],
                dtype=torch.long,
            ),

            "iogp_labels": torch.tensor(
                self.iogp_labels[index],
                dtype=torch.float32,
            ),

            "iogp_weights": torch.tensor(
                self.iogp_weights[index],
                dtype=torch.float32,
            ),
        }

        if "token_type_ids" in encoded:

            item["token_type_ids"] = (
                encoded["token_type_ids"]
                .squeeze(0)
            )

        return item


# =============================================================================
# MODEL
# =============================================================================

class MultiTaskDeberta(nn.Module):

    def __init__(
        self,
        model_name: str,
        num_rules: int,
    ):

        super().__init__()

        self.encoder = AutoModel.from_pretrained(
            model_name,
            torch_dtype=(
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            ),
        )

        if GRADIENT_CHECKPOINTING:

            self.encoder.gradient_checkpointing_enable()

            if hasattr(
                self.encoder.config,
                "use_cache",
            ):

                self.encoder.config.use_cache = False

        if LORA_ENABLED:
            lora_config = LoraConfig(
                r=LORA_R,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                target_modules=LORA_TARGET_MODULES,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.encoder = get_peft_model(
                self.encoder,
                lora_config,
            )
            self.encoder.print_trainable_parameters()

        hidden_size = (
            self.encoder.config.hidden_size
        )

        self.sif_head = nn.Linear(
            hidden_size,
            1,
        )

        self.iogp_head = nn.Linear(
            hidden_size,
            num_rules,
        )


    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ):

        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:

            kwargs["token_type_ids"] = (
                token_type_ids
            )

        outputs = self.encoder(
            **kwargs
        )

        pooled = (
            outputs.last_hidden_state[:, 0]
        )

        sif_logits = self.sif_head(
            pooled
        )

        iogp_logits = self.iogp_head(
            pooled
        )

        return {
            "sif_logits": sif_logits,
            "iogp_logits": iogp_logits,
        }


# =============================================================================
# ASYMMETRIC FOCAL LOSS
# =============================================================================

class AsymmetricFocalLoss(nn.Module):

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
    ):

        super().__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:

        logits = logits.float().view(-1)

        targets = targets.float().view(-1)

        valid = targets >= 0

        if not valid.any():

            return logits.sum() * 0.0

        logits = logits[valid]

        targets = targets[valid]

        probabilities = torch.sigmoid(
            logits
        )

        pt = (
            probabilities * targets
            + (
                1.0 - probabilities
            ) * (
                1.0 - targets
            )
        )

        alpha_factor = (
            self.alpha * targets
            + (
                1.0 - self.alpha
            ) * (
                1.0 - targets
            )
        )

        focal_factor = (
            1.0 - pt
        ).pow(self.gamma)

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none",
        )

        return (
            alpha_factor
            * focal_factor
            * bce
        ).mean()


# =============================================================================
# MASKED WEIGHTED BCE
# =============================================================================

def masked_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:

    logits = logits.float()

    targets = targets.float()

    weights = weights.float()

    valid = targets >= 0

    if not valid.any():

        return logits.sum() * 0.0

    safe_targets = torch.where(
        valid,
        targets,
        torch.zeros_like(targets),
    )

    raw_loss = (
        nn.functional
        .binary_cross_entropy_with_logits(
            logits,
            safe_targets,
            reduction="none",
        )
    )

    effective_weights = (
        weights
        * valid.float()
    )

    denominator = (
        effective_weights.sum()
        .clamp_min(1.0)
    )

    return (
        raw_loss
        * effective_weights
    ).sum() / denominator


# =============================================================================
# SAFE METRICS
# =============================================================================

def safe_average_precision(
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


# =============================================================================
# EVALUATION
# =============================================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    all_sif_logits = []
    all_sif_targets = []

    all_iogp_logits = []
    all_iogp_targets = []

    total_loss = 0.0
    batches = 0

    sif_loss_fn = AsymmetricFocalLoss(
        gamma=SIF_GAMMA,
        alpha=SIF_ALPHA,
    )

    for batch in loader:

        input_ids = batch[
            "input_ids"
        ].to(
            device,
            non_blocking=True,
        )

        attention_mask = batch[
            "attention_mask"
        ].to(
            device,
            non_blocking=True,
        )

        token_type_ids = batch.get(
            "token_type_ids"
        )

        if token_type_ids is not None:

            token_type_ids = (
                token_type_ids.to(
                    device,
                    non_blocking=True,
                )
            )

        sif_targets = batch[
            "sif_label"
        ].to(device)

        iogp_targets = batch[
            "iogp_labels"
        ].to(device)

        iogp_weights = batch[
            "iogp_weights"
        ].to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

        sif_loss = sif_loss_fn(
            outputs["sif_logits"],
            sif_targets,
        )

        iogp_loss = masked_weighted_bce(
            outputs["iogp_logits"],
            iogp_targets,
            iogp_weights,
        )

        loss = (
            SIF_LOSS_WEIGHT * sif_loss
            + IOGP_LOSS_WEIGHT * iogp_loss
        )

        total_loss += float(
            loss.item()
        )

        batches += 1

        all_sif_logits.append(
            outputs["sif_logits"]
            .float()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        all_sif_targets.append(
            sif_targets
            .cpu()
            .numpy()
            .reshape(-1)
        )

        all_iogp_logits.append(
            outputs["iogp_logits"]
            .float()
            .cpu()
            .numpy()
        )

        all_iogp_targets.append(
            iogp_targets
            .cpu()
            .numpy()
        )

    sif_logits = np.concatenate(
        all_sif_logits
    )

    sif_targets = np.concatenate(
        all_sif_targets
    )

    iogp_logits = np.concatenate(
        all_iogp_logits
    )

    iogp_targets = np.concatenate(
        all_iogp_targets
    )

    metrics = {}

    # -------------------------------------------------------------------------
    # SIF
    # -------------------------------------------------------------------------

    sif_mask = (
        sif_targets >= 0
    )

    if sif_mask.any():

        y_true = (
            sif_targets[sif_mask]
            .astype(int)
        )

        logits = sif_logits[
            sif_mask
        ]

        y_score = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    logits,
                    -50,
                    50,
                )
            )
        )

        y_pred = (
            y_score >= 0.5
        ).astype(int)

        metrics["sif_recall"] = float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        metrics["sif_precision"] = float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        metrics["sif_f1"] = float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        metrics["sif_pr_auc"] = (
            safe_average_precision(
                y_true,
                y_score,
            )
        )

        metrics["sif_roc_auc"] = (
            safe_roc_auc(
                y_true,
                y_score,
            )
        )

    # -------------------------------------------------------------------------
    # IOGP
    # -------------------------------------------------------------------------

    f1_values = []

    for index, rule in enumerate(RULES):

        mask = (
            iogp_targets[:, index]
            >= 0
        )

        if not mask.any():
            continue

        y_true = (
            iogp_targets[
                mask,
                index,
            ]
            .astype(int)
        )

        logits = (
            iogp_logits[
                mask,
                index,
            ]
        )

        y_score = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    logits,
                    -50,
                    50,
                )
            )
        )

        y_pred = (
            y_score >= 0.5
        ).astype(int)

        rule_f1 = float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        metrics[
            f"{rule}_precision"
        ] = float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        metrics[
            f"{rule}_recall"
        ] = float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        )

        metrics[
            f"{rule}_f1"
        ] = rule_f1

        metrics[
            f"{rule}_pr_auc"
        ] = safe_average_precision(
            y_true,
            y_score,
        )

        f1_values.append(
            rule_f1
        )

    if f1_values:

        metrics[
            "iogp_macro_f1"
        ] = float(
            np.mean(f1_values)
        )

    metrics["loss"] = (
        total_loss
        / max(batches, 1)
    )

    return metrics


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    best_metric,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "scheduler_state_dict":
                scheduler.state_dict(),

            "scaler_state_dict":
                scaler.state_dict(),

            "epoch":
                epoch,

            "global_step":
                global_step,

            "best_metric":
                best_metric,

            "model_name":
                MODEL_NAME,

            "max_length":
                MAX_LENGTH,

            "rules":
                RULES,

            "seed":
                SEED,
            "lora_enabled":
                LORA_ENABLED,
            "lora_r":
                LORA_R,
            "lora_alpha":
                LORA_ALPHA,
            "lora_dropout":
                LORA_DROPOUT,
            "lora_target_modules":
                LORA_TARGET_MODULES,
        },
        path,
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("RISKFORGE MULTITASK DEBERTA TRAINING")
    print("=" * 80)

    seed_everything(SEED)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA: "
            f"{torch.version.cuda}"
        )

        properties = (
            torch.cuda.get_device_properties(0)
        )

        total_memory = (
            properties.total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU memory: "
            f"{total_memory:.2f} GB"
        )

        print(
            f"Physical batch size: "
            f"{PHYSICAL_BATCH_SIZE}"
        )

        print(
            f"Gradient accumulation: "
            f"{GRADIENT_ACCUMULATION_STEPS}"
        )

        print(
            f"Effective batch size: "
            f"{PHYSICAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}"
        )

        print(
            f"Gradient checkpointing: "
            f"{GRADIENT_CHECKPOINTING}"
        )

    # -------------------------------------------------------------------------
    # DATA
    # -------------------------------------------------------------------------

    print("\nLoading datasets...")

    train_df = pd.read_csv(
        TRAIN_FILE,
        low_memory=False,
    )

    calibration_df = pd.read_csv(
        CALIBRATION_FILE,
        low_memory=False,
    )

    test_df = pd.read_csv(
        TEST_FILE,
        low_memory=False,
    )

    print(
        f"Train:       {len(train_df):,}"
    )

    print(
        f"Calibration: {len(calibration_df):,}"
    )

    print(
        f"Test:        {len(test_df):,}"
    )

    # -------------------------------------------------------------------------
    # ID CHECK
    # -------------------------------------------------------------------------

    train_ids = set(
        train_df["incident_id"]
    )

    calibration_ids = set(
        calibration_df["incident_id"]
    )

    test_ids = set(
        test_df["incident_id"]
    )

    if train_ids & calibration_ids:
        raise RuntimeError(
            "Train/calibration leakage."
        )

    if train_ids & test_ids:
        raise RuntimeError(
            "Train/test leakage."
        )

    if calibration_ids & test_ids:
        raise RuntimeError(
            "Calibration/test leakage."
        )

    print(
        "ID leakage check: PASSED"
    )

    # -------------------------------------------------------------------------
    # TOKENIZER
    # -------------------------------------------------------------------------

    print()
    print(
        f"Loading tokenizer: "
        f"{MODEL_NAME}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
    )

    # -------------------------------------------------------------------------
    # DATASETS
    # -------------------------------------------------------------------------

    train_dataset = IncidentDataset(
        train_df,
        tokenizer,
        MAX_LENGTH,
    )

    calibration_dataset = IncidentDataset(
        calibration_df,
        tokenizer,
        MAX_LENGTH,
    )

    test_dataset = IncidentDataset(
        test_df,
        tokenizer,
        MAX_LENGTH,
    )

    # -------------------------------------------------------------------------
    # LOADERS
    # -------------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    calibration_loader = DataLoader(
        calibration_dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=PHYSICAL_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------------------------------------------------------
    # MODEL
    # -------------------------------------------------------------------------

    print()
    print(
        f"Loading model: "
        f"{MODEL_NAME}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = MultiTaskDeberta(
        MODEL_NAME,
        NUM_RULES,
    )

    # Keep trainable LoRA adapters and task heads in FP32 so GradScaler can
    # safely unscale their gradients. The frozen DeBERTa backbone remains FP16.
    if LORA_ENABLED:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()

    print(
        "Moving model to GPU..."
    )

    model.to(
        DEVICE,
        non_blocking=True,
    )

    print(
        "Model loaded successfully."
    )

    if torch.cuda.is_available():

        allocated = (
            torch.cuda.memory_allocated()
            / (1024 ** 3)
        )

        reserved = (
            torch.cuda.memory_reserved()
            / (1024 ** 3)
        )

        print(
            f"GPU allocated after model load: "
            f"{allocated:.2f} GB"
        )

        print(
            f"GPU reserved after model load: "
            f"{reserved:.2f} GB"
        )

    # -------------------------------------------------------------------------
    # OPTIMIZER
    # -------------------------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    updates_per_epoch = math.ceil(
        len(train_loader)
        / GRADIENT_ACCUMULATION_STEPS
    )

    total_steps = (
        updates_per_epoch
        * EPOCHS
    )

    warmup_steps = int(
        total_steps
        * WARMUP_RATIO
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    )

    # -------------------------------------------------------------------------
    # AMP
    # -------------------------------------------------------------------------

    use_amp = (
        DEVICE.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    # -------------------------------------------------------------------------
    # LOSS
    # -------------------------------------------------------------------------

    sif_loss_fn = AsymmetricFocalLoss(
        gamma=SIF_GAMMA,
        alpha=SIF_ALPHA,
    )

    # -------------------------------------------------------------------------
    # TRAINING
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    best_metric = -float("inf")
    global_step = 0

    history = []

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0.0
        epoch_sif_loss = 0.0
        epoch_iogp_loss = 0.0

        optimizer.zero_grad(
            set_to_none=True
        )

        epoch_start = time.time()

        progress = tqdm(
            train_loader,
            desc=(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS}"
            ),
        )

        for batch_index, batch in enumerate(
            progress
        ):

            input_ids = batch[
                "input_ids"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            attention_mask = batch[
                "attention_mask"
            ].to(
                DEVICE,
                non_blocking=True,
            )

            token_type_ids = batch.get(
                "token_type_ids"
            )

            if token_type_ids is not None:

                token_type_ids = (
                    token_type_ids.to(
                        DEVICE,
                        non_blocking=True,
                    )
                )

            sif_targets = batch[
                "sif_label"
            ].to(DEVICE)

            iogp_targets = batch[
                "iogp_labels"
            ].to(DEVICE)

            iogp_weights = batch[
                "iogp_weights"
            ].to(DEVICE)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=use_amp,
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )

                sif_loss = sif_loss_fn(
                    outputs["sif_logits"],
                    sif_targets,
                )

                iogp_loss = (
                    masked_weighted_bce(
                        outputs["iogp_logits"],
                        iogp_targets,
                        iogp_weights,
                    )
                )

                loss = (
                    SIF_LOSS_WEIGHT
                    * sif_loss
                    + IOGP_LOSS_WEIGHT
                    * iogp_loss
                )

                loss_for_backward = (
                    loss
                    / GRADIENT_ACCUMULATION_STEPS
                )

            scaler.scale(
                loss_for_backward
            ).backward()

            epoch_loss += float(
                loss.detach().item()
            )

            epoch_sif_loss += float(
                sif_loss.detach().item()
            )

            epoch_iogp_loss += float(
                iogp_loss.detach().item()
            )

            if (
                (
                    batch_index + 1
                )
                % GRADIENT_ACCUMULATION_STEPS
                == 0
                or
                (
                    batch_index + 1
                )
                == len(train_loader)
            ):

                scaler.unscale_(
                    optimizer
                )

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    MAX_GRAD_NORM,
                )

                scaler.step(
                    optimizer
                )

                scaler.update()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

            progress.set_postfix(
                loss=(
                    epoch_loss
                    / (batch_index + 1)
                ),
                lr=(
                    scheduler.get_last_lr()[0]
                ),
            )

        # ---------------------------------------------------------------------
        # CALIBRATION EVALUATION
        # ---------------------------------------------------------------------

        batches = max(
            len(train_loader),
            1,
        )

        train_loss = (
            epoch_loss / batches
        )

        train_sif_loss = (
            epoch_sif_loss / batches
        )

        train_iogp_loss = (
            epoch_iogp_loss / batches
        )

        calibration_metrics = evaluate(
            model,
            calibration_loader,
            DEVICE,
        )

        elapsed = (
            time.time()
            - epoch_start
        )

        print()
        print("-" * 80)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"SIF loss: "
            f"{train_sif_loss:.6f}"
        )

        print(
            f"IOGP loss: "
            f"{train_iogp_loss:.6f}"
        )

        print(
            f"Calibration loss: "
            f"{calibration_metrics['loss']:.6f}"
        )

        for key, value in (
            calibration_metrics.items()
        ):

            if key == "loss":
                continue

            if isinstance(value, float):

                print(
                    f"{key}: "
                    f"{value:.6f}"
                )

        print(
            f"Epoch time: "
            f"{elapsed / 60:.2f} min"
        )

        # ---------------------------------------------------------------------
        # CHECKPOINT
        # ---------------------------------------------------------------------

        metric = calibration_metrics.get(
            "iogp_macro_f1",
            -float("inf"),
        )

        if (
            math.isfinite(metric)
            and metric > best_metric
        ):

            best_metric = metric

            save_checkpoint(
                OUTPUT_DIR / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_metric,
            )

            print(
                "Saved BEST checkpoint."
            )

        save_checkpoint(
            OUTPUT_DIR / "latest.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            global_step,
            best_metric,
        )

        history.append(
            {
                "epoch":
                    epoch + 1,

                "train_loss":
                    train_loss,

                "train_sif_loss":
                    train_sif_loss,

                "train_iogp_loss":
                    train_iogp_loss,

                **{
                    f"calibration_{key}":
                        value
                    for key, value
                    in calibration_metrics.items()
                },
            }
        )

        with open(
            OUTPUT_DIR / "history.json",
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                history,
                handle,
                indent=2,
            )

        if torch.cuda.is_available():

            peak_memory = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 3)
            )

            print(
                f"Peak GPU memory: "
                f"{peak_memory:.2f} GB"
            )

            torch.cuda.reset_peak_memory_stats()

    # -------------------------------------------------------------------------
    # FINAL TEST
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    best_checkpoint = (
        OUTPUT_DIR / "best.pt"
    )

    if best_checkpoint.exists():

        checkpoint = torch.load(
            best_checkpoint,
            map_location="cpu",
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model.to(DEVICE)

    test_metrics = evaluate(
        model,
        test_loader,
        DEVICE,
    )

    for key, value in test_metrics.items():

        if isinstance(value, float):

            print(
                f"{key}: "
                f"{value:.6f}"
            )

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            test_metrics,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # CONFIG
    # -------------------------------------------------------------------------

    config = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "physical_batch_size":
            PHYSICAL_BATCH_SIZE,
        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,
        "effective_batch_size":
            (
                PHYSICAL_BATCH_SIZE
                * GRADIENT_ACCUMULATION_STEPS
            ),
        "epochs": EPOCHS,
        "learning_rate":
            LEARNING_RATE,
        "weight_decay":
            WEIGHT_DECAY,
        "warmup_ratio":
            WARMUP_RATIO,
        "max_grad_norm":
            MAX_GRAD_NORM,
        "seed":
            SEED,
        "gradient_checkpointing":
            GRADIENT_CHECKPOINTING,
        "lora_enabled":
            LORA_ENABLED,
        "lora_r":
            LORA_R,
        "lora_alpha":
            LORA_ALPHA,
        "lora_dropout":
            LORA_DROPOUT,
        "lora_target_modules":
            LORA_TARGET_MODULES,
        "sif_loss_weight":
            SIF_LOSS_WEIGHT,
        "iogp_loss_weight":
            IOGP_LOSS_WEIGHT,
        "sif_gamma":
            SIF_GAMMA,
        "sif_alpha":
            SIF_ALPHA,
        "rules":
            RULES,
    }

    with open(
        OUTPUT_DIR / "training_config.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            config,
            handle,
            indent=2,
        )

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Checkpoint directory:\n"
        f"  {OUTPUT_DIR}"
    )

    print(
        f"\nBest checkpoint:\n"
        f"  {OUTPUT_DIR / 'best.pt'}"
    )

    print(
        f"\nTest metrics:\n"
        f"  {OUTPUT_DIR / 'test_metrics.json'}"
    )


if __name__ == "__main__":
    main()