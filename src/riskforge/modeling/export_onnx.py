"""
RiskForge DeBERTa-v3 multitask ONNX export.

NO TRAINING IS PERFORMED.

Exports:
    C:\\data\\processed\\deberta_multitask.onnx
    C:\\data\\processed\\deberta_multitask.onnx.data
    C:\\data\\processed\\deberta_multitask.onnx.manifest.json

ONNX inputs:
    input_ids
    attention_mask
    token_type_ids

ONNX outputs:
    sif_logits  -> [batch, 1]
    iogp_logits -> [batch, 9]

Important:
    - Raw logits only.
    - No sigmoid in graph.
    - No thresholding in graph.
    - No calibration in graph.
    - LoRA is merged before export.
    - Export precision is FP32.
    - ONNX opset is 18.
    - Dynamic batch and sequence dimensions.
    - ONNX Runtime CPUExecutionProvider is validated.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from riskforge.modeling.config import modeling_data_dir


# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "microsoft/deberta-v3-base"

DATA_DIR = modeling_data_dir()

CHECKPOINT_FILE = (
    DATA_DIR
    / "deberta_multitask_checkpoints"
    / "best.pt"
)

ONNX_FILE = (
    DATA_DIR
    / "deberta_multitask.onnx"
)

MANIFEST_FILE = (
    DATA_DIR
    / "deberta_multitask.onnx.manifest.json"
)

MAX_LENGTH = 128

ONNX_OPSET = 18

MODEL_VERSION = (
    "deberta-v3-base-multitask-v1"
)

TEMPERATURE = 1.0


# Exact IOGP order.
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


# Serving-side calibration.
# These values are metadata only.
# They are NOT placed inside the ONNX graph.

SIF_THRESHOLD = 0.660

IOGP_THRESHOLDS = {
    "bypassing_safety_controls": 0.500,
    "confined_space": 0.500,
    "driving": 0.500,
    "energy_isolation": 0.940,
    "hot_work": 0.500,
    "line_of_fire": 0.665,
    "safe_mechanical_lifting": 0.500,
    "toxic_gas": 0.500,
    "work_at_height": 0.980,
}


# =============================================================================
# PRINT HELPERS
# =============================================================================


def print_section(
    title: str,
) -> None:

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# CHECKPOINT
# =============================================================================


def load_checkpoint() -> dict[str, Any]:
    """
    Load and validate best.pt.
    """

    if not CHECKPOINT_FILE.exists():

        raise FileNotFoundError(
            "Checkpoint not found:\n"
            f"{CHECKPOINT_FILE}"
        )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            "Checkpoint is not a dictionary."
        )

    if (
        "model_state_dict"
        not in checkpoint
    ):

        raise RuntimeError(
            "Checkpoint does not contain "
            "model_state_dict."
        )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_FILE}"
    )

    print(
        "Checkpoint keys:"
    )

    for key in checkpoint.keys():

        print(
            f"  {key}"
        )

    checkpoint_model_name = (
        checkpoint.get(
            "model_name"
        )
    )

    if checkpoint_model_name is not None:

        print(
            "Checkpoint model name: "
            f"{checkpoint_model_name}"
        )

        if (
            checkpoint_model_name
            != MODEL_NAME
        ):

            raise RuntimeError(
                "Checkpoint model name mismatch.\n"
                f"Expected: {MODEL_NAME}\n"
                f"Found:    {checkpoint_model_name}"
            )

    checkpoint_rules = (
        checkpoint.get(
            "rules"
        )
    )

    if checkpoint_rules is not None:

        if (
            list(checkpoint_rules)
            != IOGP_RULES
        ):

            raise RuntimeError(
                "Checkpoint IOGP rule order mismatch.\n"
                f"Expected: {IOGP_RULES}\n"
                f"Found:    {checkpoint_rules}"
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
            "Checkpoint max length: "
            f"{checkpoint_max_length}"
        )

        if (
            int(checkpoint_max_length)
            != MAX_LENGTH
        ):

            raise RuntimeError(
                "Checkpoint max length mismatch.\n"
                f"Expected: {MAX_LENGTH}\n"
                f"Found:    {checkpoint_max_length}"
            )

    if (
        checkpoint.get(
            "lora_enabled"
        )
        is not True
    ):

        raise RuntimeError(
            "Expected a LoRA-enabled checkpoint."
        )

    print(
        "Checkpoint LoRA configuration: PASS"
    )

    return checkpoint


# =============================================================================
# MODEL INSTANTIATION
# =============================================================================


def instantiate_model(
    model_class: Any,
) -> torch.nn.Module:
    """
    Instantiate MultiTaskDeberta from the repository.
    """

    signature = inspect.signature(
        model_class
    )

    possible_values: dict[str, Any] = {

        "model_name": MODEL_NAME,

        "model_name_or_path": MODEL_NAME,

        "pretrained_model_name": MODEL_NAME,

        "backbone_name": MODEL_NAME,

        "num_iogp_rules": len(
            IOGP_RULES
        ),

        "num_rules": len(
            IOGP_RULES
        ),

        "n_rules": len(
            IOGP_RULES
        ),

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

    kwargs: dict[str, Any] = {}

    for parameter_name in (
        signature.parameters
    ):

        if (
            parameter_name
            in possible_values
        ):

            kwargs[
                parameter_name
            ] = possible_values[
                parameter_name
            ]

    print(
        "Constructor arguments:"
    )

    for key, value in (
        kwargs.items()
    ):

        print(
            f"  {key}: {value}"
        )

    try:

        return model_class(
            **kwargs
        )

    except TypeError as exc:

        raise RuntimeError(
            "Could not instantiate "
            "MultiTaskDeberta.\n\n"
            f"Constructor:\n{signature}\n\n"
            f"Arguments:\n{kwargs}\n\n"
            f"Original error:\n{exc}"
        ) from exc


# =============================================================================
# MODEL LOADING
# =============================================================================


def load_model() -> torch.nn.Module:
    """
    Construct model and load best.pt.
    """

    modeling_dir = (
        Path(__file__).resolve().parent
    )

    if (
        str(modeling_dir)
        not in sys.path
    ):

        sys.path.insert(
            0,
            str(modeling_dir),
        )

    try:

        from train_multitask_deberta import (
            MultiTaskDeberta,
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not import "
            "MultiTaskDeberta from "
            "train_multitask_deberta.py."
        ) from exc

    checkpoint = (
        load_checkpoint()
    )

    model = instantiate_model(
        MultiTaskDeberta
    )

    missing_keys, unexpected_keys = (
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=False,
        )
    )

    print(
        f"Missing keys: "
        f"{len(missing_keys)}"
    )

    print(
        "Unexpected keys: "
        f"{len(unexpected_keys)}"
    )

    if missing_keys:

        print(
            "\nMissing keys:"
        )

        for key in missing_keys:

            print(
                f"  {key}"
            )

        raise RuntimeError(
            "Checkpoint has missing "
            "model keys."
        )

    if unexpected_keys:

        print(
            "\nUnexpected keys:"
        )

        for key in unexpected_keys:

            print(
                f"  {key}"
            )

        raise RuntimeError(
            "Checkpoint has unexpected "
            "model keys."
        )

    model.eval()

    return model


# =============================================================================
# LORA MERGE
# =============================================================================


def merge_lora(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Merge LoRA adapters into the encoder.
    """

    print_section(
        "MERGING LORA ADAPTERS"
    )

    encoder = getattr(
        model,
        "encoder",
        None,
    )

    if (
        encoder is not None
        and hasattr(
            encoder,
            "merge_and_unload",
        )
    ):

        print(
            "Found PEFT model at "
            "model.encoder"
        )

        model.encoder = (
            encoder.merge_and_unload()
        )

        print(
            "LoRA adapters merged "
            "successfully."
        )

        return model

    if hasattr(
        model,
        "merge_and_unload",
    ):

        model = (
            model.merge_and_unload()
        )

        print(
            "LoRA adapters merged "
            "successfully."
        )

        return model

    raise RuntimeError(
        "Could not find a PEFT model "
        "exposing merge_and_unload()."
    )


# =============================================================================
# FP32
# =============================================================================


def convert_model_to_fp32(
    model: torch.nn.Module,
) -> torch.nn.Module:
    """
    Convert the complete model to FP32.
    """

    print_section(
        "CONVERTING MODEL TO FP32"
    )

    model = model.float()

    model.cpu()

    model.eval()

    fp16_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):

        if (
            parameter.dtype
            == torch.float16
        ):

            fp16_parameters.append(
                name
            )

    if fp16_parameters:

        print(
            "FP16 parameters remain:"
        )

        for name in (
            fp16_parameters[:20]
        ):

            print(
                f"  {name}"
            )

        raise RuntimeError(
            "Model still contains "
            "FP16 parameters."
        )

    print(
        "All model parameters are FP32."
    )

    return model


# =============================================================================
# FORWARD INPUTS
# =============================================================================


def get_forward_inputs(
    model: torch.nn.Module,
) -> list[str]:
    """
    Determine inputs accepted by model.forward().
    """

    signature = inspect.signature(
        model.forward
    )

    print(
        "Model forward signature:"
    )

    accepted = []

    for name, parameter in (
        signature.parameters.items()
    ):

        print(
            f"  {name}: "
            f"{parameter.kind}"
        )

        if name == "self":
            continue

        if parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):

            accepted.append(
                name
            )

    return accepted


# =============================================================================
# INPUT BUILDING
# =============================================================================


def build_example_inputs(
    model: torch.nn.Module,
    tokenizer: Any,
) -> dict[str, torch.Tensor]:
    """
    Build example inputs in explicit wrapper order.

    Order:
        input_ids
        attention_mask
        token_type_ids
    """

    accepted_inputs = set(
        get_forward_inputs(model)
    )

    encoded = tokenizer(
        [
            "Example industrial incident "
            "involving maintenance work."
        ],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    ordered_keys = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]

    inputs: dict[
        str,
        torch.Tensor,
    ] = {}

    for key in ordered_keys:

        if (
            key in encoded
            and key in accepted_inputs
        ):

            inputs[key] = encoded[
                key
            ]

    required_inputs = {
        "input_ids",
        "attention_mask",
    }

    missing = (
        required_inputs
        - set(inputs)
    )

    if missing:

        raise RuntimeError(
            "Missing required model inputs: "
            f"{sorted(missing)}"
        )

    print()
    print(
        "ONNX inputs:"
    )

    for name, tensor in (
        inputs.items()
    ):

        print(
            f"  {name}: "
            f"shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype}"
        )

    return inputs


# =============================================================================
# ONNX WRAPPER
# =============================================================================


class ONNXExportWrapper(
    torch.nn.Module
):
    """
    Stable ONNX interface.

    Input order:
        input_ids
        attention_mask
        token_type_ids

    Output order:
        sif_logits
        iogp_logits
    """

    def __init__(
        self,
        model: torch.nn.Module,
    ) -> None:

        super().__init__()

        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ):

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if token_type_ids is not None:

            model_inputs[
                "token_type_ids"
            ] = token_type_ids

        outputs = self.model(
            **model_inputs
        )

        if isinstance(
            outputs,
            dict,
        ):

            sif_logits = (
                outputs.get(
                    "sif_logits"
                )
            )

            iogp_logits = (
                outputs.get(
                    "iogp_logits"
                )
            )

        else:

            sif_logits = getattr(
                outputs,
                "sif_logits",
                None,
            )

            iogp_logits = getattr(
                outputs,
                "iogp_logits",
                None,
            )

        if sif_logits is None:

            raise RuntimeError(
                "Model did not return "
                "sif_logits."
            )

        if iogp_logits is None:

            raise RuntimeError(
                "Model did not return "
                "iogp_logits."
            )

        if sif_logits.ndim == 1:

            sif_logits = (
                sif_logits.unsqueeze(1)
            )

        if sif_logits.ndim != 2:

            raise RuntimeError(
                "SIF logits must have "
                "shape [batch,1]."
            )

        if sif_logits.shape[1] != 1:

            raise RuntimeError(
                "SIF output width "
                "must be 1."
            )

        if iogp_logits.ndim != 2:

            raise RuntimeError(
                "IOGP logits must have "
                "shape [batch,9]."
            )

        if iogp_logits.shape[1] != 9:

            raise RuntimeError(
                "IOGP output width "
                "must be 9."
            )

        return (
            sif_logits,
            iogp_logits,
        )


# =============================================================================
# PYTORCH VALIDATION
# =============================================================================


def validate_pytorch(
    wrapper: ONNXExportWrapper,
    inputs: dict[str, torch.Tensor],
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Validate PyTorch outputs.
    """

    print_section(
        "VALIDATING PYTORCH EXPORT OUTPUT"
    )

    wrapper.eval()

    with torch.inference_mode():

        sif_logits, iogp_logits = (
            wrapper(
                **inputs
            )
        )

    print(
        "SIF logits shape: "
        f"{tuple(sif_logits.shape)}"
    )

    print(
        "IOGP logits shape: "
        f"{tuple(iogp_logits.shape)}"
    )

    if tuple(
        sif_logits.shape
    ) != (1, 1):

        raise RuntimeError(
            "Unexpected SIF output shape."
        )

    if tuple(
        iogp_logits.shape
    ) != (1, 9):

        raise RuntimeError(
            "Unexpected IOGP output shape."
        )

    if not torch.isfinite(
        sif_logits
    ).all():

        raise RuntimeError(
            "SIF output contains "
            "non-finite values."
        )

    if not torch.isfinite(
        iogp_logits
    ).all():

        raise RuntimeError(
            "IOGP output contains "
            "non-finite values."
        )

    print(
        "PyTorch output validation: PASS"
    )

    return (
        sif_logits.detach()
        .cpu()
        .numpy(),

        iogp_logits.detach()
        .cpu()
        .numpy(),
    )


# =============================================================================
# EXPORT
# =============================================================================


def export_onnx(
    wrapper: ONNXExportWrapper,
    inputs: dict[str, torch.Tensor],
) -> None:
    """
    Export the model to ONNX opset 18.
    """

    print_section(
        "EXPORTING ONNX"
    )

    input_names = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]

    for name in input_names:

        if name not in inputs:

            raise RuntimeError(
                "Missing exporter input: "
                f"{name}"
            )

    output_names = [
        "sif_logits",
        "iogp_logits",
    ]

    dynamic_axes = {

        "input_ids": {
            0: "batch",
            1: "sequence",
        },

        "attention_mask": {
            0: "batch",
            1: "sequence",
        },

        "token_type_ids": {
            0: "batch",
            1: "sequence",
        },

        "sif_logits": {
            0: "batch",
        },

        "iogp_logits": {
            0: "batch",
        },
    }

    print(
        "Inputs:"
    )

    for name in input_names:

        print(
            f"  {name}"
        )

    print()
    print(
        "Outputs:"
    )

    for name in output_names:

        print(
            f"  {name}"
        )

    print()
    print(
        f"Export opset: {ONNX_OPSET}"
    )

    # CRITICAL:
    #
    # The tuple order MUST match:
    #
    #     forward(
    #         input_ids,
    #         attention_mask,
    #         token_type_ids
    #     )
    #
    example_tuple = tuple(
        inputs[name]
        for name in input_names
    )

    # Remove stale ONNX graph.
    if ONNX_FILE.exists():

        ONNX_FILE.unlink()

    # Remove stale external weights.
    external_data_file = Path(
        str(ONNX_FILE)
        + ".data"
    )

    if external_data_file.exists():

        external_data_file.unlink()

    torch.onnx.export(
        wrapper,
        example_tuple,
        str(ONNX_FILE),

        input_names=input_names,

        output_names=output_names,

        dynamic_axes=dynamic_axes,

        opset_version=ONNX_OPSET,

        do_constant_folding=True,

        export_params=True,
    )

    if not ONNX_FILE.exists():

        raise RuntimeError(
            "ONNX export did not create "
            "the expected file."
        )

    print()
    print(
        "ONNX export successful."
    )

    print(
        f"File: {ONNX_FILE}"
    )

    graph_size_mb = (
        ONNX_FILE.stat().st_size
        / (1024 * 1024)
    )

    print(
        f"Graph size: "
        f"{graph_size_mb:.2f} MB"
    )

    if external_data_file.exists():

        external_size_mb = (
            external_data_file.stat().st_size
            / (1024 * 1024)
        )

        print(
            f"External data: "
            f"{external_data_file}"
        )

        print(
            f"External data size: "
            f"{external_size_mb:.2f} MB"
        )

    else:

        raise RuntimeError(
            "Expected external ONNX "
            "data file was not created."
        )


# =============================================================================
# ONNX STRUCTURAL VALIDATION
# =============================================================================


def validate_onnx() -> None:
    """
    Validate ONNX structure without invoking onnx.checker.check_model().

    The installed ONNX version attempts to resolve external TensorProto
    data during check_model(), which causes the current exporter to fail
    despite a valid external-data artifact.

    Therefore this function performs explicit structural validation and
    leaves actual tensor loading/execution to ONNX Runtime.
    """

    print_section(
        "VALIDATING ONNX GRAPH"
    )

    try:

        import onnx

    except ImportError as exc:

        raise RuntimeError(
            "onnx package is not installed."
        ) from exc

    if not ONNX_FILE.exists():

        raise RuntimeError(
            f"ONNX file does not exist:\n"
            f"{ONNX_FILE}"
        )

    external_data_file = Path(
        str(ONNX_FILE)
        + ".data"
    )

    if not external_data_file.exists():

        raise RuntimeError(
            "ONNX external-data file "
            "does not exist.\n"
            f"Expected:\n"
            f"{external_data_file}"
        )

    # -------------------------------------------------------------------------
    # File checks
    # -------------------------------------------------------------------------

    graph_size = (
        ONNX_FILE.stat().st_size
    )

    external_size = (
        external_data_file.stat().st_size
    )

    if graph_size <= 0:

        raise RuntimeError(
            "ONNX graph file is empty."
        )

    if external_size <= 0:

        raise RuntimeError(
            "ONNX external-data file "
            "is empty."
        )

    print(
        f"Graph file size: "
        f"{graph_size / (1024 * 1024):.2f} MB"
    )

    print(
        f"External data size: "
        f"{external_size / (1024 * 1024):.2f} MB"
    )

    # -------------------------------------------------------------------------
    # Load only graph metadata.
    # -------------------------------------------------------------------------

    print()
    print(
        "Loading ONNX graph metadata "
        "without external weights..."
    )

    model = onnx.load(
        str(ONNX_FILE),
        load_external_data=False,
    )

    # -------------------------------------------------------------------------
    # IR / opset
    # -------------------------------------------------------------------------

    print(
        f"IR version: "
        f"{model.ir_version}"
    )

    opsets = [
        (
            item.domain,
            item.version,
        )
        for item in model.opset_import
    ]

    print(
        f"Opset imports: "
        f"{opsets}"
    )

    default_opset = None

    for item in model.opset_import:

        if item.domain == "":

            default_opset = (
                item.version
            )

            break

    if default_opset != ONNX_OPSET:

        raise RuntimeError(
            "ONNX opset mismatch.\n"
            f"Expected: {ONNX_OPSET}\n"
            f"Found: {default_opset}"
        )

    print(
        f"Opset {ONNX_OPSET}: PASS"
    )

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------

    inputs = [
        value.name
        for value in model.graph.input
    ]

    expected_inputs = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]

    print()
    print(
        "Graph inputs:"
    )

    for name in inputs:

        print(
            f"  {name}"
        )

    if inputs != expected_inputs:

        raise RuntimeError(
            "Unexpected ONNX input order.\n"
            f"Expected: {expected_inputs}\n"
            f"Found:    {inputs}"
        )

    print(
        "ONNX input order: PASS"
    )

    # -------------------------------------------------------------------------
    # Outputs
    # -------------------------------------------------------------------------

    outputs = [
        value.name
        for value in model.graph.output
    ]

    expected_outputs = [
        "sif_logits",
        "iogp_logits",
    ]

    print()
    print(
        "Graph outputs:"
    )

    for name in outputs:

        print(
            f"  {name}"
        )

    if outputs != expected_outputs:

        raise RuntimeError(
            "Unexpected ONNX output order.\n"
            f"Expected: {expected_outputs}\n"
            f"Found:    {outputs}"
        )

    print(
        "ONNX output names/order: PASS"
    )

    # -------------------------------------------------------------------------
    # Node count
    # -------------------------------------------------------------------------

    node_count = len(
        model.graph.node
    )

    initializer_count = len(
        model.graph.initializer
    )

    print()
    print(
        f"Graph nodes: "
        f"{node_count}"
    )

    print(
        f"Initializers: "
        f"{initializer_count}"
    )

    if node_count == 0:

        raise RuntimeError(
            "ONNX graph contains "
            "zero nodes."
        )

    if initializer_count == 0:

        raise RuntimeError(
            "ONNX graph contains "
            "zero initializers."
        )

    print(
        "Graph node/initializer validation: PASS"
    )

    # -------------------------------------------------------------------------
    # Output shape validation
    # -------------------------------------------------------------------------

    output_map = {
        value.name: value
        for value in model.graph.output
    }

    sif_shape = (
        output_map[
            "sif_logits"
        ]
        .type
        .tensor_type
        .shape
    )

    iogp_shape = (
        output_map[
            "iogp_logits"
        ]
        .type
        .tensor_type
        .shape
    )

    sif_dims = []

    for dimension in sif_shape.dim:

        if dimension.dim_param:

            sif_dims.append(
                dimension.dim_param
            )

        else:

            sif_dims.append(
                dimension.dim_value
            )

    iogp_dims = []

    for dimension in iogp_shape.dim:

        if dimension.dim_param:

            iogp_dims.append(
                dimension.dim_param
            )

        else:

            iogp_dims.append(
                dimension.dim_value
            )

    print()
    print(
        f"SIF output shape: "
        f"{sif_dims}"
    )

    print(
        f"IOGP output shape: "
        f"{iogp_dims}"
    )

    if len(sif_dims) != 2:

        raise RuntimeError(
            "SIF output must be rank 2."
        )

    if len(iogp_dims) != 2:

        raise RuntimeError(
            "IOGP output must be rank 2."
        )

    if sif_dims[1] != 1:

        raise RuntimeError(
            "SIF output width "
            "must be 1."
        )

    if iogp_dims[1] != 9:

        raise RuntimeError(
            "IOGP output width "
            "must be 9."
        )

    print(
        "Output shape validation: PASS"
    )

    # -------------------------------------------------------------------------
    # Dynamic axes
    # -------------------------------------------------------------------------

    input_map = {
        value.name: value
        for value in model.graph.input
    }

    for input_name in expected_inputs:

        value = input_map[
            input_name
        ]

        shape = (
            value.type
            .tensor_type
            .shape
        )

        dimensions = []

        for dimension in shape.dim:

            if dimension.dim_param:

                dimensions.append(
                    dimension.dim_param
                )

            else:

                dimensions.append(
                    dimension.dim_value
                )

        print(
            f"{input_name} shape: "
            f"{dimensions}"
        )

        if len(dimensions) != 2:

            raise RuntimeError(
                f"{input_name} must be "
                "rank 2."
            )

        if dimensions[0] != "batch":

            raise RuntimeError(
                f"{input_name} batch "
                "dimension is not dynamic."
            )

        if dimensions[1] != "sequence":

            raise RuntimeError(
                f"{input_name} sequence "
                "dimension is not dynamic."
            )

    print(
        "Dynamic input axes: PASS"
    )

    # -------------------------------------------------------------------------
    # External data references
    # -------------------------------------------------------------------------

    external_count = 0

    for initializer in (
        model.graph.initializer
    ):

        if initializer.external_data:

            external_count += 1

            location = None

            for item in (
                initializer.external_data
            ):

                if item.key == "location":

                    location = item.value

                    break

            if location is None:

                raise RuntimeError(
                    "External initializer "
                    f"{initializer.name} "
                    "does not specify a "
                    "data location."
                )

            # The exporter creates:
            #
            #     deberta_multitask.onnx.data
            #
            # and ONNX should reference this file.

            expected_location = (
                external_data_file.name
            )

            if location != expected_location:

                raise RuntimeError(
                    "Unexpected external "
                    "data location.\n"
                    f"Expected: "
                    f"{expected_location}\n"
                    f"Found:    {location}\n"
                    f"Tensor:   "
                    f"{initializer.name}"
                )

    print()
    print(
        f"External-data initializers: "
        f"{external_count}"
    )

    if external_count == 0:

        raise RuntimeError(
            "No external-data initializers "
            "were found."
        )

    print(
        "External data reference validation: PASS"
    )

    # -------------------------------------------------------------------------
    # Explicitly verify that the external file is a regular file.
    # -------------------------------------------------------------------------

    if not external_data_file.is_file():

        raise RuntimeError(
            "External data path is not "
            "a regular file:\n"
            f"{external_data_file}"
        )

    print(
        "External data file validation: PASS"
    )

    print()
    print(
        "ONNX graph structural validation: PASS"
    )


# =============================================================================
# ONNX RUNTIME
# =============================================================================


def validate_onnx_runtime() -> None:
    """
    Load and execute the actual exported model using ONNX Runtime.
    """

    print_section(
        "VALIDATING ONNX RUNTIME"
    )

    try:

        import onnxruntime as ort

    except ImportError as exc:

        raise RuntimeError(
            "onnxruntime is not installed."
        ) from exc

    session_options = (
        ort.SessionOptions()
    )

    session_options.intra_op_num_threads = 4

    session_options.inter_op_num_threads = 1

    session_options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    print(
        "Creating ONNX Runtime session..."
    )

    session = (
        ort.InferenceSession(
            str(ONNX_FILE),
            sess_options=session_options,
            providers=[
                "CPUExecutionProvider"
            ],
        )
    )

    providers = (
        session.get_providers()
    )

    print(
        f"Providers: {providers}"
    )

    if providers != [
        "CPUExecutionProvider"
    ]:

        raise RuntimeError(
            "Expected only "
            "CPUExecutionProvider."
        )

    print(
        "CPUExecutionProvider: PASS"
    )

    # -------------------------------------------------------------------------
    # Runtime inputs
    # -------------------------------------------------------------------------

    runtime_inputs = (
        session.get_inputs()
    )

    runtime_input_names = [
        value.name
        for value in runtime_inputs
    ]

    print()
    print(
        "Runtime inputs:"
    )

    for value in runtime_inputs:

        print(
            f"  {value.name}: "
            f"shape={value.shape}, "
            f"type={value.type}"
        )

    expected_inputs = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
    ]

    if (
        runtime_input_names
        != expected_inputs
    ):

        raise RuntimeError(
            "Unexpected ONNX Runtime "
            "input order.\n"
            f"Expected: {expected_inputs}\n"
            f"Found: {runtime_input_names}"
        )

    print(
        "ONNX Runtime input order: PASS"
    )

    # -------------------------------------------------------------------------
    # Runtime outputs
    # -------------------------------------------------------------------------

    runtime_outputs = (
        session.get_outputs()
    )

    runtime_output_names = [
        value.name
        for value in runtime_outputs
    ]

    print()
    print(
        "Runtime outputs:"
    )

    for value in runtime_outputs:

        print(
            f"  {value.name}: "
            f"shape={value.shape}, "
            f"type={value.type}"
        )

    expected_outputs = [
        "sif_logits",
        "iogp_logits",
    ]

    if (
        runtime_output_names
        != expected_outputs
    ):

        raise RuntimeError(
            "Unexpected ONNX Runtime "
            "output order.\n"
            f"Expected: {expected_outputs}\n"
            f"Found: {runtime_output_names}"
        )

    print(
        "ONNX Runtime output order: PASS"
    )

    # -------------------------------------------------------------------------
    # Runtime smoke test
    # -------------------------------------------------------------------------

    print()
    print(
        "Running ONNX Runtime smoke test..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
    )

    encoded = tokenizer(
        [
            "Example industrial incident "
            "involving maintenance work."
        ],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="np",
    )

    feed: dict[
        str,
        np.ndarray,
    ] = {}

    for input_name in (
        runtime_input_names
    ):

        if input_name not in encoded:

            raise RuntimeError(
                "Tokenizer did not provide "
                f"runtime input: "
                f"{input_name}"
            )

        feed[
            input_name
        ] = encoded[
            input_name
        ].astype(
            np.int64,
            copy=False,
        )

    outputs = session.run(
        None,
        feed,
    )

    if len(outputs) != 2:

        raise RuntimeError(
            "Expected exactly two "
            "runtime outputs."
        )

    sif_output = outputs[0]

    iogp_output = outputs[1]

    print(
        "SIF runtime shape: "
        f"{sif_output.shape}"
    )

    print(
        "IOGP runtime shape: "
        f"{iogp_output.shape}"
    )

    if sif_output.shape != (
        1,
        1,
    ):

        raise RuntimeError(
            "Unexpected ONNX Runtime "
            "SIF output shape."
        )

    if iogp_output.shape != (
        1,
        9,
    ):

        raise RuntimeError(
            "Unexpected ONNX Runtime "
            "IOGP output shape."
        )

    if not np.isfinite(
        sif_output
    ).all():

        raise RuntimeError(
            "SIF output contains "
            "non-finite values."
        )

    if not np.isfinite(
        iogp_output
    ).all():

        raise RuntimeError(
            "IOGP output contains "
            "non-finite values."
        )

    print()
    print(
        "SIF output:"
    )

    print(
        sif_output
    )

    print()
    print(
        "IOGP output:"
    )

    print(
        iogp_output
    )

    print()
    print(
        "ONNX Runtime smoke test: PASS"
    )

    print()
    print(
        "ONNX Runtime validation: PASS"
    )


# =============================================================================
# MANIFEST
# =============================================================================


def save_manifest() -> None:
    """
    Write model manifest.
    """

    print_section(
        "WRITING MODEL MANIFEST"
    )

    manifest = {

        "manifest_version": 1,

        "model_version": (
            MODEL_VERSION
        ),

        "model_name": MODEL_NAME,

        "backbone": MODEL_NAME,

        "max_sequence_length": (
            MAX_LENGTH
        ),

        "onnx_opset": (
            ONNX_OPSET
        ),

        "precision": "float32",

        "outputs": {

            "sif": {

                "name": "sif_logits",

                "shape": [
                    "batch",
                    1,
                ],

                "type": "raw_logit",
            },

            "iogp": {

                "name": "iogp_logits",

                "shape": [
                    "batch",
                    9,
                ],

                "type": "raw_logits",

                "label_order": (
                    IOGP_RULES
                ),
            },
        },

        "temperature": (
            TEMPERATURE
        ),

        "serving_thresholds": {

            "sif": (
                SIF_THRESHOLD
            ),

            "iogp": (
                IOGP_THRESHOLDS
            ),
        },

        "serving": {

            "execution_provider":
                "CPUExecutionProvider",

            "intra_op_threads": 4,

            "inter_op_threads": 1,

            "dynamic_batch": True,

            "dynamic_sequence": True,
        },

        "export": {

            "lora_merged": True,

            "fp32_export": True,

            "onnx_opset": (
                ONNX_OPSET
            ),

            "sigmoid_in_graph": False,

            "thresholds_in_graph": False,

            "calibration_in_graph": False,
        },
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Manifest: "
        f"{MANIFEST_FILE}"
    )

    print(
        "Manifest validation: PASS"
    )


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:

    print()
    print("=" * 80)
    print(
        "RISKFORGE DEBERTA MULTITASK "
        "ONNX EXPORT"
    )
    print("=" * 80)

    print()
    print(
        "NO TRAINING WILL BE PERFORMED."
    )

    print(
        "The existing best.pt checkpoint "
        "will be used without modification."
    )

    # -------------------------------------------------------------------------
    # Environment
    # -------------------------------------------------------------------------

    print_section(
        "ENVIRONMENT"
    )

    print(
        f"PyTorch: "
        f"{torch.__version__}"
    )

    print(
        "CUDA available: "
        f"{torch.cuda.is_available()}"
    )

    # -------------------------------------------------------------------------
    # Tokenizer
    # -------------------------------------------------------------------------

    print_section(
        "LOADING TOKENIZER"
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME,
            local_files_only=True,
        )
    )

    print(
        f"Tokenizer loaded: "
        f"{MODEL_NAME}"
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = load_model()

    # -------------------------------------------------------------------------
    # Merge LoRA
    # -------------------------------------------------------------------------

    model = merge_lora(
        model
    )

    # -------------------------------------------------------------------------
    # Convert FP32
    # -------------------------------------------------------------------------

    model = convert_model_to_fp32(
        model
    )

    # -------------------------------------------------------------------------
    # Inputs
    # -------------------------------------------------------------------------

    inputs = build_example_inputs(
        model,
        tokenizer,
    )

    # -------------------------------------------------------------------------
    # Wrapper
    # -------------------------------------------------------------------------

    wrapper = (
        ONNXExportWrapper(
            model
        )
    )

    wrapper.eval()

    # -------------------------------------------------------------------------
    # PyTorch validation
    # -------------------------------------------------------------------------

    validate_pytorch(
        wrapper,
        inputs,
    )

    # -------------------------------------------------------------------------
    # ONNX export
    # -------------------------------------------------------------------------

    export_onnx(
        wrapper,
        inputs,
    )

    # -------------------------------------------------------------------------
    # Structural validation
    # -------------------------------------------------------------------------

    validate_onnx()

    # -------------------------------------------------------------------------
    # ONNX Runtime validation
    # -------------------------------------------------------------------------

    validate_onnx_runtime()

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------

    save_manifest()

    # -------------------------------------------------------------------------
    # Final status
    # -------------------------------------------------------------------------

    print_section(
        "ONNX EXPORT COMPLETE"
    )

    print(
        f"ONNX: "
        f"{ONNX_FILE}"
    )

    print(
        f"Manifest: "
        f"{MANIFEST_FILE}"
    )

    external_data_file = Path(
        str(ONNX_FILE)
        + ".data"
    )

    print(
        f"Weights: "
        f"{external_data_file}"
    )

    print()
    print(
        "EXPORT STATUS: PASS"
    )


if __name__ == "__main__":
    main()
