"""
RiskForge trained-model validation suite.

Loads the existing best.pt checkpoint and validates it without retraining.

Tests:
1. IOGP label ordering
2. Checkpoint integrity
3. Model output shapes
4. Real-text inference
5. NaN / Inf detection
6. Dynamic batch sizes
7. Deterministic inference
8. CPU inference

This script does NOT train, modify, or save the model checkpoint.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from riskforge.modeling.config import modeling_data_dir


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "microsoft/deberta-v3-base"

CHECKPOINT_DIR = modeling_data_dir() / "deberta_multitask_checkpoints"

BEST_CHECKPOINT = CHECKPOINT_DIR / "best.pt"

MAX_LENGTH = 128

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
# TEST CASES
# =============================================================================

TEST_CASES = [
    (
        "driving",
        (
            "A company vehicle was travelling between two facilities when "
            "the driver lost control of the vehicle and struck a roadside "
            "barrier."
        ),
    ),
    (
        "work_at_height",
        (
            "An employee working from a scaffold at approximately 20 feet "
            "fell while performing maintenance work."
        ),
    ),
    (
        "confined_space",
        (
            "A worker entered a confined tank to perform inspection and "
            "became unconscious inside the vessel."
        ),
    ),
    (
        "hot_work",
        (
            "Workers were using an oxy acetylene torch to cut a metal "
            "structure during maintenance activities."
        ),
    ),
    (
        "toxic_gas",
        (
            "A worker was exposed to hydrogen sulfide gas while working "
            "near a process vessel and required medical treatment."
        ),
    ),
    (
        "line_of_fire",
        (
            "An employee was struck by a moving suspended load while "
            "working near a crane lifting operation."
        ),
    ),
    (
        "energy_isolation",
        (
            "A technician began maintenance on equipment without isolating "
            "the electrical energy source."
        ),
    ),
    (
        "safe_mechanical_lifting",
        (
            "A crane was used to lift a heavy metal plate while workers "
            "remained clear of the suspended load."
        ),
    ),
    (
        "bypassing_safety_controls",
        (
            "An operator bypassed an interlock and defeated a machine "
            "safety control in order to continue production."
        ),
    ),
]


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def print_pass() -> None:
    print("PASS")


# =============================================================================
# MODEL IMPORT
# =============================================================================

def import_training_model():
    """
    Import the exact MultiTaskDeberta implementation used for training.
    """

    modeling_dir = Path(__file__).resolve().parent

    if str(modeling_dir) not in sys.path:
        sys.path.insert(0, str(modeling_dir))

    from train_multitask_deberta import MultiTaskDeberta

    return MultiTaskDeberta


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint():

    print(
        f"Checkpoint:\n{BEST_CHECKPOINT}"
    )

    if not BEST_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{BEST_CHECKPOINT}"
        )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location="cpu",
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    print(
        f"Checkpoint loaded: {type(checkpoint).__name__}"
    )

    print(
        "Checkpoint keys:",
        ", ".join(
            str(key)
            for key in checkpoint.keys()
        ),
    )

    return checkpoint


def extract_state_dict(checkpoint):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint[
            "state_dict"
        ]

    elif (
        "model" in checkpoint
        and isinstance(
            checkpoint["model"],
            dict,
        )
    ):

        state_dict = checkpoint[
            "model"
        ]

    else:

        raise RuntimeError(
            "Could not find model_state_dict "
            "in checkpoint."
        )

    if not isinstance(
        state_dict,
        dict,
    ):
        raise RuntimeError(
            "Model state_dict is not a dictionary."
        )

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned_state_dict[key] = value

    return cleaned_state_dict


# =============================================================================
# MODEL CONSTRUCTION
# =============================================================================

def build_model(device):

    print_header("LOADING MODEL")

    MultiTaskDeberta = import_training_model()

    model = MultiTaskDeberta(
        MODEL_NAME,
        len(IOGP_RULES),
    )

    checkpoint = load_checkpoint()

    # -------------------------------------------------------------------------
    # Verify checkpoint metadata.
    # -------------------------------------------------------------------------

    checkpoint_model_name = checkpoint.get(
        "model_name"
    )

    if checkpoint_model_name is not None:

        print(
            f"Checkpoint model name: "
            f"{checkpoint_model_name}"
        )

        if checkpoint_model_name != MODEL_NAME:

            raise RuntimeError(
                "Checkpoint model name does not match "
                f"{MODEL_NAME}."
            )

    checkpoint_rules = checkpoint.get(
        "rules"
    )

    if checkpoint_rules is not None:

        if list(checkpoint_rules) != IOGP_RULES:

            raise RuntimeError(
                "Checkpoint IOGP rule order does not "
                "match the required order."
            )

        print(
            "Checkpoint IOGP rule order: PASSED"
        )

    checkpoint_max_length = checkpoint.get(
        "max_length"
    )

    if checkpoint_max_length is not None:

        print(
            f"Checkpoint max length: "
            f"{checkpoint_max_length}"
        )

        if int(checkpoint_max_length) != MAX_LENGTH:

            print(
                "WARNING: test MAX_LENGTH differs "
                "from checkpoint max_length."
            )

    # -------------------------------------------------------------------------
    # Verify LoRA metadata.
    # -------------------------------------------------------------------------

    checkpoint_lora_enabled = checkpoint.get(
        "lora_enabled"
    )

    if checkpoint_lora_enabled is not None:
        print(
            f"Checkpoint LoRA enabled: "
            f"{checkpoint_lora_enabled}"
        )

    checkpoint_lora_r = checkpoint.get(
        "lora_r"
    )

    if checkpoint_lora_r is not None:
        print(
            f"Checkpoint LoRA r: "
            f"{checkpoint_lora_r}"
        )

    checkpoint_lora_alpha = checkpoint.get(
        "lora_alpha"
    )

    if checkpoint_lora_alpha is not None:
        print(
            f"Checkpoint LoRA alpha: "
            f"{checkpoint_lora_alpha}"
        )

    checkpoint_lora_dropout = checkpoint.get(
        "lora_dropout"
    )

    if checkpoint_lora_dropout is not None:
        print(
            f"Checkpoint LoRA dropout: "
            f"{checkpoint_lora_dropout}"
        )

    checkpoint_lora_targets = checkpoint.get(
        "lora_target_modules"
    )

    if checkpoint_lora_targets is not None:
        print(
            "Checkpoint LoRA target modules: "
            f"{checkpoint_lora_targets}"
        )

    # -------------------------------------------------------------------------
    # Load model weights.
    # -------------------------------------------------------------------------

    state_dict = extract_state_dict(
        checkpoint
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

        print()
        print("Missing keys:")

        for key in result.missing_keys[:20]:
            print(
                f"  {key}"
            )

        raise RuntimeError(
            "Checkpoint is missing model parameters."
        )

    if result.unexpected_keys:

        print()
        print("Unexpected keys:")

        for key in result.unexpected_keys[:20]:
            print(
                f"  {key}"
            )

        raise RuntimeError(
            "Checkpoint contains unexpected model parameters."
        )

    model.to(device)
    model.eval()

    return model


# =============================================================================
# TOKENIZATION
# =============================================================================

def tokenize(
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

def run_inference(
    model,
    tokenizer,
    texts,
    device,
):
    """
    Run inference while handling the FP16 encoder / FP32 task-head
    boundary used by the trained LoRA model.

    The checkpoint is never modified.
    """

    inputs = tokenize(
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

            dtype_error = (
                "mat1 and mat2 must have the same dtype"
                in message
            )

            if not dtype_error:
                raise

            # -----------------------------------------------------------------
            # The trained configuration can have:
            #
            # DeBERTa encoder = FP16
            # LoRA adapters    = FP32
            # SIF head         = FP32
            # IOGP head        = FP32
            #
            # Temporarily convert the task heads to FP16 for inference.
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

            if sif_head is None:

                raise RuntimeError(
                    "Model does not contain sif_head."
                )

            if iogp_head is None:

                raise RuntimeError(
                    "Model does not contain iogp_head."
                )

            sif_original_dtype = next(
                sif_head.parameters()
            ).dtype

            iogp_original_dtype = next(
                iogp_head.parameters()
            ).dtype

            sif_head.half()
            iogp_head.half()

            try:

                outputs = model(
                    **inputs
                )

            finally:

                # Restore original task-head dtypes.

                if (
                    sif_original_dtype
                    == torch.float32
                ):
                    sif_head.float()

                elif (
                    sif_original_dtype
                    == torch.float64
                ):
                    sif_head.double()

                if (
                    iogp_original_dtype
                    == torch.float32
                ):
                    iogp_head.float()

                elif (
                    iogp_original_dtype
                    == torch.float64
                ):
                    iogp_head.double()

    # -------------------------------------------------------------------------
    # Validate output structure.
    # -------------------------------------------------------------------------

    if not isinstance(
        outputs,
        dict,
    ):

        raise RuntimeError(
            "Model output is not a dictionary."
        )

    if "sif_logits" not in outputs:

        raise RuntimeError(
            "Missing model output: sif_logits"
        )

    if "iogp_logits" not in outputs:

        raise RuntimeError(
            "Missing model output: iogp_logits"
        )

    sif_logits = outputs[
        "sif_logits"
    ]

    iogp_logits = outputs[
        "iogp_logits"
    ]

    if not isinstance(
        sif_logits,
        torch.Tensor,
    ):

        raise RuntimeError(
            "sif_logits is not a torch.Tensor."
        )

    if not isinstance(
        iogp_logits,
        torch.Tensor,
    ):

        raise RuntimeError(
            "iogp_logits is not a torch.Tensor."
        )

    # -------------------------------------------------------------------------
    # Numerical validity.
    # -------------------------------------------------------------------------

    if not torch.isfinite(
        sif_logits
    ).all():

        raise RuntimeError(
            "sif_logits contains NaN or Inf."
        )

    if not torch.isfinite(
        iogp_logits
    ).all():

        raise RuntimeError(
            "iogp_logits contains NaN or Inf."
        )

    # -------------------------------------------------------------------------
    # Return CPU FP32 tensors.
    # -------------------------------------------------------------------------

    return (
        sif_logits.detach().float().cpu(),
        iogp_logits.detach().float().cpu(),
    )


# =============================================================================
# PYTEST FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def tokenizer():
    """
    Load the DeBERTa tokenizer once for the pytest module.
    """
    print_header("LOADING TOKENIZER")

    return AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )


@pytest.fixture(scope="module")
def device():
    """
    Use CUDA when available; otherwise CPU.
    """
    selected_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(
        f"Test device: {selected_device}"
    )

    return selected_device


@pytest.fixture(scope="module")
def model(device):
    """
    Build and load the existing trained checkpoint once for pytest.

    No training is performed and the checkpoint is never modified.
    """
    return build_model(device)


# =============================================================================
# TEST 1 — ARCHITECTURE / OUTPUT SHAPES
# =============================================================================

def test_architecture(
    model,
    tokenizer,
    device,
):

    print_header(
        "TEST 1 — ARCHITECTURE / OUTPUT SHAPES"
    )

    texts = [
        "Worker was injured during maintenance.",
        "Vehicle struck another vehicle during transport.",
    ]

    sif, iogp = run_inference(
        model,
        tokenizer,
        texts,
        device,
    )

    print(
        f"SIF shape:  {tuple(sif.shape)}"
    )

    print(
        f"IOGP shape: {tuple(iogp.shape)}"
    )

    if sif.shape != (2, 1):

        raise AssertionError(
            "Expected SIF shape (2, 1), "
            f"got {tuple(sif.shape)}"
        )

    if iogp.shape != (2, 9):

        raise AssertionError(
            "Expected IOGP shape (2, 9), "
            f"got {tuple(iogp.shape)}"
        )

    print_pass()


# =============================================================================
# TEST 2 — REAL TEXT INFERENCE
# =============================================================================

def test_real_inference(
    model,
    tokenizer,
    device,
):

    print_header(
        "TEST 2 — REAL TEXT INFERENCE"
    )

    texts = [
        text
        for _, text in TEST_CASES
    ]

    sif, iogp = run_inference(
        model,
        tokenizer,
        texts,
        device,
    )

    sif_probability = torch.sigmoid(
        sif[:, 0]
    )

    iogp_probability = torch.sigmoid(
        iogp
    )

    for index, (
        expected_rule,
        text,
    ) in enumerate(TEST_CASES):

        scores = iogp_probability[
            index
        ]

        ranked = torch.argsort(
            scores,
            descending=True,
        )

        print()
        print(
            f"Expected concept: "
            f"{expected_rule}"
        )

        print(
            f"SIF probability: "
            f"{float(sif_probability[index]):.4f}"
        )

        print(
            "Top IOGP predictions:"
        )

        for rule_index in ranked[:3]:

            rule_index = int(
                rule_index
            )

            print(
                f"  "
                f"{IOGP_RULES[rule_index]:32s} "
                f"{float(scores[rule_index]):.4f}"
            )

    print()
    print_pass()


# =============================================================================
# TEST 3 — BATCH SIZE
# =============================================================================

def test_batch_sizes(
    model,
    tokenizer,
    device,
):

    print_header(
        "TEST 3 — BATCH SIZE TEST"
    )

    base_text = (
        "An employee was performing maintenance "
        "work when an incident occurred."
    )

    reference_sif = None
    reference_iogp = None

    for batch_size in (
        1,
        2,
        4,
        8,
        16,
    ):

        texts = [
            base_text
            for _ in range(batch_size)
        ]

        if device.type == "cuda":

            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        start = time.perf_counter()

        sif, iogp = run_inference(
            model,
            tokenizer,
            texts,
            device,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        print(
            f"Batch {batch_size:2d}: "
            f"SIF {tuple(sif.shape)}, "
            f"IOGP {tuple(iogp.shape)}, "
            f"{elapsed * 1000:.2f} ms"
        )

        # ---------------------------------------------------------------------
        # Validate dynamic batch dimensions.
        # ---------------------------------------------------------------------

        expected_sif_shape = (
            batch_size,
            1,
        )

        expected_iogp_shape = (
            batch_size,
            9,
        )

        if sif.shape != expected_sif_shape:

            raise AssertionError(
                "Incorrect SIF dynamic batch shape: "
                f"expected {expected_sif_shape}, "
                f"got {tuple(sif.shape)}"
            )

        if iogp.shape != expected_iogp_shape:

            raise AssertionError(
                "Incorrect IOGP dynamic batch shape: "
                f"expected {expected_iogp_shape}, "
                f"got {tuple(iogp.shape)}"
            )

        # ---------------------------------------------------------------------
        # Numerical consistency.
        #
        # The same text should produce essentially the same result regardless
        # of batch size. FP16 GPU kernels can introduce small numerical
        # differences depending on execution shape.
        #
        # We therefore validate reasonable numerical agreement rather than
        # requiring bit-for-bit equality.
        # ---------------------------------------------------------------------

        if batch_size == 1:

            reference_sif = sif[0].clone()

            reference_iogp = (
                iogp[0].clone()
            )

        else:

            sif_difference = torch.max(
                torch.abs(
                    reference_sif - sif[0]
                )
            ).item()

            iogp_difference = torch.max(
                torch.abs(
                    reference_iogp - iogp[0]
                )
            ).item()

            print(
                f"  Max SIF difference:  "
                f"{sif_difference:.8f}"
            )

            print(
                f"  Max IOGP difference: "
                f"{iogp_difference:.8f}"
            )

            if not torch.allclose(
                reference_sif,
                sif[0],
                rtol=5e-3,
                atol=5e-3,
            ):

                raise AssertionError(
                    "SIF output changed excessively "
                    f"when batch size changed to {batch_size}. "
                    f"Maximum difference: "
                    f"{sif_difference:.8f}"
                )

            if not torch.allclose(
                reference_iogp,
                iogp[0],
                rtol=5e-3,
                atol=5e-3,
            ):

                raise AssertionError(
                    "IOGP output changed excessively "
                    f"when batch size changed to {batch_size}. "
                    f"Maximum difference: "
                    f"{iogp_difference:.8f}"
                )

    print_pass()


# =============================================================================
# TEST 4 — DETERMINISM
# =============================================================================

def test_determinism(
    model,
    tokenizer,
    device,
):

    print_header(
        "TEST 4 — DETERMINISTIC INFERENCE"
    )

    text = (
        "A worker entered a confined space during "
        "maintenance and was exposed to hazardous "
        "conditions."
    )

    sif1, iogp1 = run_inference(
        model,
        tokenizer,
        [text],
        device,
    )

    sif2, iogp2 = run_inference(
        model,
        tokenizer,
        [text],
        device,
    )

    sif_difference = torch.max(
        torch.abs(
            sif1 - sif2
        )
    ).item()

    iogp_difference = torch.max(
        torch.abs(
            iogp1 - iogp2
        )
    ).item()

    print(
        f"Maximum SIF difference:  "
        f"{sif_difference:.10f}"
    )

    print(
        f"Maximum IOGP difference: "
        f"{iogp_difference:.10f}"
    )

    if not torch.allclose(
        sif1,
        sif2,
        rtol=1e-5,
        atol=1e-5,
    ):

        raise AssertionError(
            "SIF inference is not deterministic."
        )

    if not torch.allclose(
        iogp1,
        iogp2,
        rtol=1e-5,
        atol=1e-5,
    ):

        raise AssertionError(
            "IOGP inference is not deterministic."
        )

    print_pass()


# =============================================================================
# TEST 5 — CPU INFERENCE
# =============================================================================

def test_cpu(
    tokenizer,
):

    print_header(
        "TEST 5 — CPU INFERENCE"
    )

    cpu = torch.device(
        "cpu"
    )

    model = build_model(
        cpu
    )

    sif, iogp = run_inference(
        model,
        tokenizer,
        [
            (
                "A worker was struck by "
                "a moving object during maintenance."
            )
        ],
        cpu,
    )

    print(
        f"CPU SIF shape:  "
        f"{tuple(sif.shape)}"
    )

    print(
        f"CPU IOGP shape: "
        f"{tuple(iogp.shape)}"
    )

    if sif.shape != (1, 1):

        raise AssertionError(
            "CPU SIF shape is incorrect."
        )

    if iogp.shape != (1, 9):

        raise AssertionError(
            "CPU IOGP shape is incorrect."
        )

    if not torch.isfinite(
        sif
    ).all():

        raise AssertionError(
            "CPU SIF contains NaN or Inf."
        )

    if not torch.isfinite(
        iogp
    ).all():

        raise AssertionError(
            "CPU IOGP contains NaN or Inf."
        )

    print_pass()


# =============================================================================
# TEST 6 — IOGP LABEL ORDER
# =============================================================================

def test_label_order():

    print_header(
        "TEST 6 — IOGP LABEL ORDER"
    )

    print(
        json.dumps(
            {
                "iogp_rule_order": IOGP_RULES
            },
            indent=2,
        )
    )

    if len(IOGP_RULES) != 9:

        raise AssertionError(
            "IOGP rule count is not 9."
        )

    if len(
        set(IOGP_RULES)
    ) != 9:

        raise AssertionError(
            "Duplicate IOGP rule detected."
        )

    print_pass()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("RISKFORGE TRAINED MODEL VALIDATION")
    print("=" * 80)

    print(
        f"Checkpoint:\n"
        f"{BEST_CHECKPOINT}"
    )

    print(
        f"Model:\n"
        f"{MODEL_NAME}"
    )

    print(
        f"PyTorch:\n"
        f"{torch.__version__}"
    )

    print(
        f"CUDA available:\n"
        f"{torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU:\n"
            f"{torch.cuda.get_device_name(0)}"
        )

    # -------------------------------------------------------------------------
    # IOGP label order.
    # -------------------------------------------------------------------------

    test_label_order()

    # -------------------------------------------------------------------------
    # Load tokenizer completely offline.
    # -------------------------------------------------------------------------

    print_header(
        "LOADING TOKENIZER"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        local_files_only=True,
    )

    # -------------------------------------------------------------------------
    # GPU validation.
    # -------------------------------------------------------------------------

    if torch.cuda.is_available():

        gpu = torch.device(
            "cuda"
        )

        model = build_model(
            gpu
        )

        print_header(
            "GPU MODEL READY"
        )

        print(
            f"Allocated GPU memory: "
            f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        )

        test_architecture(
            model,
            tokenizer,
            gpu,
        )

        test_real_inference(
            model,
            tokenizer,
            gpu,
        )

        test_batch_sizes(
            model,
            tokenizer,
            gpu,
        )

        test_determinism(
            model,
            tokenizer,
            gpu,
        )

        # ---------------------------------------------------------------------
        # Release GPU memory before CPU test.
        # ---------------------------------------------------------------------

        del model

        torch.cuda.empty_cache()

    else:

        print(
            "\nCUDA unavailable — "
            "skipping GPU tests."
        )

    # -------------------------------------------------------------------------
    # CPU validation.
    # -------------------------------------------------------------------------

    test_cpu(
        tokenizer
    )

    # -------------------------------------------------------------------------
    # Complete.
    # -------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("ALL MODEL VALIDATION TESTS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()
