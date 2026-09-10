"""
RiskForge - Numerical PyTorch vs ONNX Parity Validation

IMPORTANT:
- Does NOT train.
- Does NOT modify checkpoint weights.
- Does NOT export ONNX.
- Loads the exact MultiTaskDeberta implementation used for training.
- Merges the trained LoRA adapters.
- Converts the inference model to FP32.
- Runs PyTorch and ONNX Runtime on identical inputs.
- Compares raw SIF and IOGP logits numerically.

ONNX inputs:
    input_ids
    attention_mask
    token_type_ids

ONNX outputs:
    sif_logits  [batch, 1]
    iogp_logits [batch, 9]
"""

from __future__ import annotations

import json
import sys
from typing import Dict, List, Tuple

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from riskforge.modeling.config import modeling_data_dir


# =============================================================================
# PATHS
# =============================================================================

DATA_ROOT = modeling_data_dir()

CHECKPOINT_PATH = (
    DATA_ROOT
    / "deberta_multitask_checkpoints"
    / "best.pt"
)

ONNX_PATH = (
    DATA_ROOT
    / "deberta_multitask.onnx"
)

MANIFEST_PATH = (
    DATA_ROOT
    / "deberta_multitask.onnx.manifest.json"
)


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_NAME = "microsoft/deberta-v3-base"

MAX_LENGTH = 128

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


# =============================================================================
# NUMERICAL TOLERANCES
# =============================================================================

SIF_ATOL = 1e-4
SIF_RTOL = 1e-4

IOGP_ATOL = 1e-4
IOGP_RTOL = 1e-4


# =============================================================================
# OUTPUT HELPERS
# =============================================================================


def print_header(title: str) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def fail(message: str) -> None:

    print()
    print("PARITY STATUS: FAIL")
    print(f"Reason: {message}")

    raise SystemExit(1)


# =============================================================================
# LOAD PROJECT MODEL
# =============================================================================


def load_project_model(
    device: torch.device,
):
    """
    Load the exact MultiTaskDeberta implementation from the training script.

    The class is defined in:

        riskforge.modeling.train_multitask_deberta

    This avoids recreating a potentially different architecture.
    """

    try:

        from riskforge.modeling.train_multitask_deberta import (
            MultiTaskDeberta,
        )

    except Exception as exc:

        fail(
            "Could not import MultiTaskDeberta from "
            "riskforge.modeling.train_multitask_deberta.\n"
            f"Original error: {exc}"
        )

    print("Model class: MultiTaskDeberta")
    print(
        "Source: riskforge.modeling.train_multitask_deberta"
    )

    # -------------------------------------------------------------------------
    # Load checkpoint metadata first.
    # -------------------------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
    )

    if "model_state_dict" not in checkpoint:

        fail(
            "Checkpoint does not contain model_state_dict."
        )

    print()
    print("Checkpoint metadata:")

    print(
        f"  model_name: "
        f"{checkpoint.get('model_name')}"
    )

    print(
        f"  max_length: "
        f"{checkpoint.get('max_length')}"
    )

    print(
        f"  lora_enabled: "
        f"{checkpoint.get('lora_enabled')}"
    )

    print(
        f"  lora_r: "
        f"{checkpoint.get('lora_r')}"
    )

    print(
        f"  lora_alpha: "
        f"{checkpoint.get('lora_alpha')}"
    )

    print(
        f"  lora_dropout: "
        f"{checkpoint.get('lora_dropout')}"
    )

    print(
        f"  lora_target_modules: "
        f"{checkpoint.get('lora_target_modules')}"
    )

    # -------------------------------------------------------------------------
    # Validate checkpoint metadata.
    # -------------------------------------------------------------------------

    checkpoint_model_name = checkpoint.get(
        "model_name"
    )

    if (
        checkpoint_model_name is not None
        and checkpoint_model_name != MODEL_NAME
    ):

        fail(
            "Checkpoint model name mismatch.\n"
            f"Checkpoint: {checkpoint_model_name}\n"
            f"Expected:   {MODEL_NAME}"
        )

    checkpoint_max_length = checkpoint.get(
        "max_length"
    )

    if (
        checkpoint_max_length is not None
        and int(checkpoint_max_length) != MAX_LENGTH
    ):

        fail(
            "Checkpoint max_length mismatch.\n"
            f"Checkpoint: {checkpoint_max_length}\n"
            f"Expected:   {MAX_LENGTH}"
        )

    checkpoint_rules = checkpoint.get(
        "rules"
    )

    if checkpoint_rules is not None:

        if list(checkpoint_rules) != RULES:

            fail(
                "Checkpoint IOGP rule order mismatch.\n"
                f"Checkpoint: {checkpoint_rules}\n"
                f"Expected:   {RULES}"
            )

        print(
            "Checkpoint IOGP rule order: PASS"
        )

    # -------------------------------------------------------------------------
    # Construct exact training architecture.
    #
    # The training script reads these values from its own module constants.
    # We intentionally instantiate the same class rather than inventing a
    # second model implementation.
    # -------------------------------------------------------------------------

    model = MultiTaskDeberta(
        model_name=MODEL_NAME,
        num_rules=len(RULES),
    )

    # -------------------------------------------------------------------------
    # Load trained checkpoint.
    # -------------------------------------------------------------------------

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=False,
        )
    )

    if missing_keys:

        fail(
            "Missing checkpoint keys:\n"
            + "\n".join(
                f"  {key}"
                for key in missing_keys
            )
        )

    if unexpected_keys:

        fail(
            "Unexpected checkpoint keys:\n"
            + "\n".join(
                f"  {key}"
                for key in unexpected_keys
            )
        )

    print(
        "Checkpoint state_dict loading: PASS"
    )

    # -------------------------------------------------------------------------
    # CPU + FP32.
    #
    # The ONNX model is intended for CPUExecutionProvider.
    # Running PyTorch on CPU removes GPU-vs-CPU kernel differences from the
    # parity comparison.
    # -------------------------------------------------------------------------

    model = model.to(device)

    model.float()

    model.eval()

    print(
        f"PyTorch device: {device}"
    )

    print(
        "PyTorch inference dtype: FP32"
    )

    # -------------------------------------------------------------------------
    # Merge LoRA.
    # -------------------------------------------------------------------------

    encoder = getattr(
        model,
        "encoder",
        None,
    )

    if encoder is None:

        fail(
            "Model does not contain the expected encoder."
        )

    if hasattr(
        encoder,
        "merge_and_unload",
    ):

        print()
        print(
            "Found PEFT model at model.encoder"
        )

        model.encoder = (
            encoder.merge_and_unload()
        )

        print(
            "LoRA adapters merged successfully."
        )

    else:

        print()
        print(
            "WARNING: model.encoder does not expose "
            "merge_and_unload()."
        )

        print(
            "Continuing without an additional LoRA merge."
        )

    # Ensure merged inference model remains FP32.

    model.float()

    model.eval()

    return model


# =============================================================================
# TOKENIZATION
# =============================================================================


def tokenize(
    tokenizer,
    texts: List[str],
    max_length: int,
) -> Dict[str, torch.Tensor]:

    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    required_inputs = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]

    missing = [
        name
        for name in required_inputs
        if name not in encoded
    ]

    if missing:

        fail(
            "Tokenizer did not produce required inputs: "
            + ", ".join(missing)
        )

    return {
        "input_ids": encoded[
            "input_ids"
        ].to(
            dtype=torch.int64
        ),

        "attention_mask": encoded[
            "attention_mask"
        ].to(
            dtype=torch.int64
        ),

        "token_type_ids": encoded[
            "token_type_ids"
        ].to(
            dtype=torch.int64
        ),
    }


# =============================================================================
# PYTORCH INFERENCE
# =============================================================================


@torch.no_grad()
def run_pytorch(
    model,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"],
    )

    if not isinstance(outputs, dict):

        fail(
            "Expected PyTorch model output to be a dict, "
            f"got {type(outputs)}."
        )

    if "sif_logits" not in outputs:

        fail(
            "PyTorch output does not contain sif_logits."
        )

    if "iogp_logits" not in outputs:

        fail(
            "PyTorch output does not contain iogp_logits."
        )

    sif_logits = outputs[
        "sif_logits"
    ]

    iogp_logits = outputs[
        "iogp_logits"
    ]

    sif = (
        sif_logits
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    iogp = (
        iogp_logits
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    return sif, iogp


# =============================================================================
# ONNX RUNTIME
# =============================================================================


def load_onnx_session() -> ort.InferenceSession:

    session = ort.InferenceSession(
        str(ONNX_PATH),
        providers=[
            "CPUExecutionProvider"
        ],
    )

    providers = session.get_providers()

    print(
        f"Providers: {providers}"
    )

    if providers != [
        "CPUExecutionProvider"
    ]:

        fail(
            "Expected CPUExecutionProvider only.\n"
            f"Actual providers: {providers}"
        )

    return session


def validate_runtime_interface(
    session: ort.InferenceSession,
) -> None:

    actual_inputs = [
        item.name
        for item in session.get_inputs()
    ]

    actual_outputs = [
        item.name
        for item in session.get_outputs()
    ]

    expected_inputs = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]

    expected_outputs = [
        "sif_logits",
        "iogp_logits",
    ]

    print()
    print(
        "Runtime inputs:"
    )

    for item in session.get_inputs():

        print(
            f"  {item.name}: "
            f"shape={item.shape}, "
            f"type={item.type}"
        )

    print()
    print(
        "Runtime outputs:"
    )

    for item in session.get_outputs():

        print(
            f"  {item.name}: "
            f"shape={item.shape}, "
            f"type={item.type}"
        )

    if actual_inputs != expected_inputs:

        fail(
            "ONNX input order mismatch.\n"
            f"Actual:   {actual_inputs}\n"
            f"Expected: {expected_inputs}"
        )

    if actual_outputs != expected_outputs:

        fail(
            "ONNX output order mismatch.\n"
            f"Actual:   {actual_outputs}\n"
            f"Expected: {expected_outputs}"
        )

    print()
    print(
        "ONNX input order: PASS"
    )

    print(
        "ONNX output order: PASS"
    )


def run_onnx(
    session: ort.InferenceSession,
    inputs: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:

    ort_inputs = {
        "input_ids": (
            inputs["input_ids"]
            .cpu()
            .numpy()
            .astype(np.int64)
        ),

        "attention_mask": (
            inputs["attention_mask"]
            .cpu()
            .numpy()
            .astype(np.int64)
        ),

        "token_type_ids": (
            inputs["token_type_ids"]
            .cpu()
            .numpy()
            .astype(np.int64)
        ),
    }

    outputs = session.run(
        [
            "sif_logits",
            "iogp_logits",
        ],
        ort_inputs,
    )

    if len(outputs) != 2:

        fail(
            f"Expected 2 ONNX outputs, got {len(outputs)}."
        )

    sif = np.asarray(
        outputs[0],
        dtype=np.float32,
    )

    iogp = np.asarray(
        outputs[1],
        dtype=np.float32,
    )

    return sif, iogp


# =============================================================================
# NUMERICAL COMPARISON
# =============================================================================


def compare_arrays(
    label: str,
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
    atol: float,
    rtol: float,
) -> Dict[str, float | bool]:

    if pytorch_output.shape != onnx_output.shape:

        print(
            f"  {label}:"
        )

        print(
            f"    PyTorch shape: {pytorch_output.shape}"
        )

        print(
            f"    ONNX shape:    {onnx_output.shape}"
        )

        print(
            "    status:        FAIL"
        )

        return {
            "pass": False,
            "shape_match": False,
            "max_abs_diff": float("inf"),
            "mean_abs_diff": float("inf"),
            "max_rel_diff": float("inf"),
        }

    pytorch64 = (
        pytorch_output
        .astype(np.float64)
    )

    onnx64 = (
        onnx_output
        .astype(np.float64)
    )

    absolute_difference = np.abs(
        pytorch64 - onnx64
    )

    max_abs_diff = float(
        np.max(
            absolute_difference
        )
    )

    mean_abs_diff = float(
        np.mean(
            absolute_difference
        )
    )

    denominator = np.maximum(
        np.maximum(
            np.abs(pytorch64),
            np.abs(onnx64),
        ),
        1e-12,
    )

    relative_difference = (
        absolute_difference
        / denominator
    )

    max_rel_diff = float(
        np.max(
            relative_difference
        )
    )

    passed = bool(
        np.allclose(
            pytorch_output,
            onnx_output,
            atol=atol,
            rtol=rtol,
        )
    )

    print()
    print(
        f"  {label}:"
    )

    print(
        f"    shape:         "
        f"{pytorch_output.shape}"
    )

    print(
        f"    max abs diff:  "
        f"{max_abs_diff:.10f}"
    )

    print(
        f"    mean abs diff: "
        f"{mean_abs_diff:.10f}"
    )

    print(
        f"    max rel diff:  "
        f"{max_rel_diff:.10f}"
    )

    print(
        f"    tolerance:     "
        f"atol={atol}, rtol={rtol}"
    )

    print(
        f"    status:        "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return {
        "pass": passed,
        "shape_match": True,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
    }


# =============================================================================
# TEST CASES
# =============================================================================
#
# These intentionally cover:
#
# - short sequence
# - medium sequence
# - batch size 2
# - batch size 4
# - batch size 8
# - long sequence
#
# The tokenizer determines the actual padded sequence length for each case.
#


TEST_CASES = [

    (
        "single_short",

        [
            "Worker entered the area and failed "
            "to follow the required safety procedure."
        ],

        16,
    ),

    (
        "single_medium",

        [
            "During maintenance work, the crew isolated "
            "the equipment before starting the task, "
            "but a hazardous condition was identified "
            "during the operation."
        ],

        64,
    ),

    (
        "batch_2",

        [
            "Worker entered a confined space without "
            "following the required entry procedure.",

            "A vehicle was being operated during "
            "field activity.",
        ],

        64,
    ),

    (
        "batch_4",

        [
            "The worker was exposed to a line of fire hazard.",

            "Energy isolation requirements were reviewed "
            "before maintenance.",

            "Hot work was performed near operating equipment.",

            "The task involved working at height.",
        ],

        96,
    ),

    (
        "batch_8",

        [
            "A safety barrier was bypassed during the task.",

            "The work involved a confined space.",

            "A vehicle was being driven at the worksite.",

            "Equipment energy isolation was required.",

            "Hot work controls were applied.",

            "The worker entered a line of fire.",

            "Mechanical lifting was performed safely.",

            "A toxic gas hazard was identified.",
        ],

        128,
    ),

    (
        "batch_4_long",

        [
            (
                "During a planned maintenance operation, workers "
                "were required to isolate equipment, verify the "
                "energy state, establish appropriate controls, "
                "maintain a safe position relative to moving "
                "equipment, and complete the work according "
                "to the approved procedure."
            ),

            (
                "A field vehicle was used to transport personnel "
                "and equipment while the crew followed the "
                "applicable driving requirements and site controls."
            ),

            (
                "Personnel performing hot work established "
                "the required controls and monitored the "
                "surrounding area for hazards."
            ),

            (
                "The crew performed work at height while using "
                "the required fall-protection and access controls."
            ),
        ],

        128,
    ),
]


# =============================================================================
# MANIFEST VALIDATION
# =============================================================================


def validate_manifest() -> None:

    if not MANIFEST_PATH.is_file():

        print(
            "Manifest: not found "
            "(skipping manifest consistency check)"
        )

        return

    print(
        f"Manifest: {MANIFEST_PATH}"
    )

    try:

        manifest = json.loads(
            MANIFEST_PATH.read_text(
                encoding="utf-8"
            )
        )

    except Exception as exc:

        fail(
            f"Could not parse ONNX manifest: {exc}"
        )

    manifest_rules = manifest.get(
        "iogp_rules"
    )

    if manifest_rules is None:

        manifest_rules = manifest.get(
            "rules"
        )

    if manifest_rules is not None:

        if list(manifest_rules) != RULES:

            fail(
                "Manifest IOGP rule order mismatch.\n"
                f"Manifest: {manifest_rules}\n"
                f"Expected: {RULES}"
            )

        print(
            "Manifest IOGP rule order: PASS"
        )

    else:

        print(
            "Manifest IOGP rule order: "
            "NOT CHECKED (field not present)"
        )

    manifest_max_length = manifest.get(
        "max_seq_length"
    )

    if manifest_max_length is None:

        manifest_max_length = manifest.get(
            "max_length"
        )

    if manifest_max_length is not None:

        if int(manifest_max_length) != MAX_LENGTH:

            fail(
                "Manifest max sequence length mismatch.\n"
                f"Manifest: {manifest_max_length}\n"
                f"Expected: {MAX_LENGTH}"
            )

        print(
            "Manifest max sequence length: PASS"
        )

    else:

        print(
            "Manifest max sequence length: "
            "NOT CHECKED (field not present)"
        )


# =============================================================================
# MAIN
# =============================================================================


def main() -> int:

    print_header(
        "RISKFORGE NUMERICAL PYTORCH vs ONNX PARITY"
    )

    print()
    print(
        "NO TRAINING WILL BE PERFORMED."
    )

    print(
        "NO MODEL WEIGHTS WILL BE MODIFIED."
    )

    print(
        "NO ONNX EXPORT WILL BE PERFORMED."
    )

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------

    print_header(
        "ENVIRONMENT"
    )

    print(
        f"PyTorch: {torch.__version__}"
    )

    print(
        f"CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    # -------------------------------------------------------------------------
    # File validation
    # -------------------------------------------------------------------------

    print_header(
        "CHECKING MODEL FILES"
    )

    if not CHECKPOINT_PATH.is_file():

        fail(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    if not ONNX_PATH.is_file():

        fail(
            f"ONNX model not found: {ONNX_PATH}"
        )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    print(
        f"ONNX:      {ONNX_PATH}"
    )

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------

    print_header(
        "LOADING TOKENIZER"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    print(
        f"Tokenizer loaded: {MODEL_NAME}"
    )

    # -------------------------------------------------------------------------
    # PyTorch model
    # -------------------------------------------------------------------------

    print_header(
        "LOADING PYTORCH MODEL"
    )

    # CPU is deliberate because ONNX Runtime is being tested on CPU.
    device = torch.device(
        "cpu"
    )

    model = load_project_model(
        device
    )

    print(
        "PyTorch model loading: PASS"
    )

    # -------------------------------------------------------------------------
    # ONNX Runtime
    # -------------------------------------------------------------------------

    print_header(
        "LOADING ONNX RUNTIME"
    )

    session = load_onnx_session()

    validate_runtime_interface(
        session
    )

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------

    print_header(
        "VALIDATING MANIFEST"
    )

    validate_manifest()

    # -------------------------------------------------------------------------
    # Numerical parity
    # -------------------------------------------------------------------------

    print_header(
        "RUNNING NUMERICAL PARITY TESTS"
    )

    results = []

    overall_pass = True

    for (
        test_name,
        texts,
        max_length,
    ) in TEST_CASES:

        print()
        print("-" * 80)

        print(
            f"TEST CASE: {test_name}"
        )

        print("-" * 80)

        print(
            f"Requested batch size: "
            f"{len(texts)}"
        )

        print(
            f"Requested max length: "
            f"{max_length}"
        )

        # ---------------------------------------------------------------------
        # Tokenization
        # ---------------------------------------------------------------------

        inputs = tokenize(
            tokenizer=tokenizer,
            texts=texts,
            max_length=max_length,
        )

        actual_batch = int(
            inputs[
                "input_ids"
            ].shape[0]
        )

        actual_sequence = int(
            inputs[
                "input_ids"
            ].shape[1]
        )

        print(
            f"Actual batch size:    "
            f"{actual_batch}"
        )

        print(
            f"Actual sequence:      "
            f"{actual_sequence}"
        )

        # ---------------------------------------------------------------------
        # PyTorch
        # ---------------------------------------------------------------------

        pytorch_sif, pytorch_iogp = (
            run_pytorch(
                model,
                inputs,
            )
        )

        # ---------------------------------------------------------------------
        # ONNX
        # ---------------------------------------------------------------------

        onnx_sif, onnx_iogp = (
            run_onnx(
                session,
                inputs,
            )
        )

        # ---------------------------------------------------------------------
        # Shape checks
        # ---------------------------------------------------------------------

        expected_sif_shape = (
            actual_batch,
            1,
        )

        expected_iogp_shape = (
            actual_batch,
            9,
        )

        if pytorch_sif.shape != expected_sif_shape:

            fail(
                f"{test_name}: PyTorch SIF shape "
                f"{pytorch_sif.shape}, expected "
                f"{expected_sif_shape}"
            )

        if onnx_sif.shape != expected_sif_shape:

            fail(
                f"{test_name}: ONNX SIF shape "
                f"{onnx_sif.shape}, expected "
                f"{expected_sif_shape}"
            )

        if pytorch_iogp.shape != expected_iogp_shape:

            fail(
                f"{test_name}: PyTorch IOGP shape "
                f"{pytorch_iogp.shape}, expected "
                f"{expected_iogp_shape}"
            )

        if onnx_iogp.shape != expected_iogp_shape:

            fail(
                f"{test_name}: ONNX IOGP shape "
                f"{onnx_iogp.shape}, expected "
                f"{expected_iogp_shape}"
            )

        print()
        print(
            "Output shapes: PASS"
        )

        # ---------------------------------------------------------------------
        # Numerical comparison
        # ---------------------------------------------------------------------

        print()
        print(
            "Comparing raw logits:"
        )

        sif_result = compare_arrays(
            label="SIF logits",
            pytorch_output=pytorch_sif,
            onnx_output=onnx_sif,
            atol=SIF_ATOL,
            rtol=SIF_RTOL,
        )

        iogp_result = compare_arrays(
            label="IOGP logits",
            pytorch_output=pytorch_iogp,
            onnx_output=onnx_iogp,
            atol=IOGP_ATOL,
            rtol=IOGP_RTOL,
        )

        case_pass = bool(
            sif_result["pass"]
            and iogp_result["pass"]
        )

        if not case_pass:

            overall_pass = False

        results.append(
            {
                "test": test_name,

                "requested_batch": len(
                    texts
                ),

                "actual_batch": actual_batch,

                "requested_max_length": max_length,

                "actual_sequence": actual_sequence,

                "sif": sif_result,

                "iogp": iogp_result,

                "pass": case_pass,
            }
        )

        print()

        print(
            f"{test_name}: "
            f"{'PASS' if case_pass else 'FAIL'}"
        )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print_header(
        "PARITY SUMMARY"
    )

    print(
        f"{'Test case':<20}"
        f"{'SIF max abs':>16}"
        f"{'IOGP max abs':>16}"
        f"{'Status':>12}"
    )

    print(
        "-" * 66
    )

    for result in results:

        print(
            f"{result['test']:<20}"
            f"{result['sif']['max_abs_diff']:>16.10f}"
            f"{result['iogp']['max_abs_diff']:>16.10f}"
            f"{'PASS' if result['pass'] else 'FAIL':>12}"
        )

    print()

    max_sif_abs = max(
        float(
            result[
                "sif"
            ][
                "max_abs_diff"
            ]
        )
        for result in results
    )

    max_iogp_abs = max(
        float(
            result[
                "iogp"
            ][
                "max_abs_diff"
            ]
        )
        for result in results
    )

    mean_sif_abs = float(
        np.mean(
            [
                float(
                    result[
                        "sif"
                    ][
                        "mean_abs_diff"
                    ]
                )
                for result in results
            ]
        )
    )

    mean_iogp_abs = float(
        np.mean(
            [
                float(
                    result[
                        "iogp"
                    ][
                        "mean_abs_diff"
                    ]
                )
                for result in results
            ]
        )
    )

    print(
        f"Maximum SIF absolute difference:  "
        f"{max_sif_abs:.10f}"
    )

    print(
        f"Maximum IOGP absolute difference: "
        f"{max_iogp_abs:.10f}"
    )

    print(
        f"Mean SIF absolute difference:     "
        f"{mean_sif_abs:.10f}"
    )

    print(
        f"Mean IOGP absolute difference:    "
        f"{mean_iogp_abs:.10f}"
    )

    # -------------------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------------------

    print_header(
        "FINAL RESULT"
    )

    if overall_pass:

        print(
            "PyTorch vs ONNX numerical parity: PASS"
        )

        print()

        print(
            "All test cases passed within "
            "the configured tolerances."
        )

        print()

        print(
            f"SIF tolerance:  "
            f"atol={SIF_ATOL}, rtol={SIF_RTOL}"
        )

        print(
            f"IOGP tolerance: "
            f"atol={IOGP_ATOL}, rtol={IOGP_RTOL}"
        )

        print()

        print(
            "PARITY STATUS: PASS"
        )

        return 0

    print(
        "PyTorch vs ONNX numerical parity: FAIL"
    )

    print()

    print(
        "At least one test case exceeded "
        "the configured tolerances."
    )

    print()

    print(
        f"SIF tolerance:  "
        f"atol={SIF_ATOL}, rtol={SIF_RTOL}"
    )

    print(
        f"IOGP tolerance: "
        f"atol={IOGP_ATOL}, rtol={IOGP_RTOL}"
    )

    print()

    print(
        "PARITY STATUS: FAIL"
    )

    return 1


if __name__ == "__main__":

    sys.exit(
        main()
    )
