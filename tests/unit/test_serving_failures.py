import json

import numpy as np
import pytest

from riskforge.core.contracts import LifeSavingRule
from riskforge.serving.artifact import ArtifactValidationError, ModelArtifactManifest
from riskforge.serving.engine import ONNXInferenceEngine
from riskforge.serving.exceptions import (
    ArtifactLoadingError,
    InferenceFailureError,
    InvalidOutputError,
    WarmupError,
)
from tests.unit.test_serving import _record


class _Node:
    def __init__(self, name: str) -> None:
        self.name = name


class _FailureSession:
    def __init__(self, *, outputs=None, error: Exception | None = None) -> None:
        self.outputs = outputs
        self.error = error

    def get_inputs(self):
        return [_Node("input_ids"), _Node("attention_mask")]

    def get_outputs(self):
        return [_Node("sif_logits"), _Node("iogp_logits")]

    def run(self, output_names, feed):
        if self.error is not None:
            raise self.error
        return self.outputs


def _manifest(checksum: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_sha256": checksum,
        "backbone": "test",
        "max_sequence_length": 8,
        "quantization": "NONE",
        "temperature": 1.0,
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["sif_logits", "iogp_logits"],
        "iogp_rule_order": [rule.value for rule in LifeSavingRule],
    }


def test_missing_artifacts_and_malformed_manifest_are_typed(tmp_path) -> None:
    with pytest.raises(ArtifactLoadingError, match="manifest"):
        ModelArtifactManifest.load(tmp_path / "missing.json")

    malformed = tmp_path / "manifest.json"
    malformed.write_text("{", encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="manifest"):
        ModelArtifactManifest.load(malformed)

    with pytest.raises(ArtifactLoadingError, match="model not found"):
        ONNXInferenceEngine(tmp_path / "missing.onnx")


def test_bad_checksum_is_artifact_loading_error(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(_manifest("0" * 64)), encoding="utf-8")

    with pytest.raises(ArtifactLoadingError, match="checksum"):
        ONNXInferenceEngine.from_artifact(model, manifest, warmup=False)


def test_session_failure_is_inference_failure() -> None:
    engine = ONNXInferenceEngine(
        "unused.onnx", session=_FailureSession(error=RuntimeError("injected"))
    )
    with pytest.raises(InferenceFailureError, match="injected"):
        engine.infer(_record(), np.ones(8), np.ones(8))


@pytest.mark.parametrize(
    "outputs, message",
    [
        ([np.zeros((1, 1))], "returned 1 outputs"),
        ([np.zeros((1, 1)), np.zeros((1, 8))], "IOGP output"),
    ],
)
def test_invalid_output_count_and_width_are_typed(outputs, message) -> None:
    engine = ONNXInferenceEngine("unused.onnx", session=_FailureSession(outputs=outputs))
    with pytest.raises(InvalidOutputError, match=message):
        engine.infer(_record(), np.ones(8), np.ones(8))


def test_warmup_failure_is_distinct_from_runtime_inference() -> None:
    engine = ONNXInferenceEngine(
        "unused.onnx", session=_FailureSession(error=RuntimeError("warmup injected"))
    )
    with pytest.raises(WarmupError, match="warmup injected"):
        engine.warmup()
