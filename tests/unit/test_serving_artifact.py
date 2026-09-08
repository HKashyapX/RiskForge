import hashlib
import json

import pytest

from riskforge.core.contracts import LifeSavingRule
from riskforge.serving.artifact import (
    ArtifactValidationError,
    ModelArtifactManifest,
    TemperatureScaler,
    validate_model_checksum,
)


def _manifest(model_sha256: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_sha256": model_sha256,
        "backbone": "microsoft/deberta-v3-base",
        "max_sequence_length": 256,
        "quantization": "INT8",
        "temperature": 1.5,
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["sif_logits", "iogp_logits"],
        "iogp_rule_order": [rule.value for rule in LifeSavingRule],
    }


def test_manifest_loads_and_validates_session(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest("a" * 64)), encoding="utf-8")

    manifest = ModelArtifactManifest.load(path)

    manifest.validate_session(
        ("input_ids", "attention_mask"), ("sif_logits", "iogp_logits")
    )
    with pytest.raises(ArtifactValidationError, match="input names"):
        manifest.validate_session(("tokens",), manifest.output_names)


def test_manifest_rejects_rule_reordering_and_temperature(tmp_path) -> None:
    payload = _manifest("b" * 64)
    payload["iogp_rule_order"] = list(reversed(payload["iogp_rule_order"]))
    payload["temperature"] = 0
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ArtifactValidationError):
        ModelArtifactManifest.load(path)


def test_checksum_validation(tmp_path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"riskforge")
    expected = hashlib.sha256(b"riskforge").hexdigest()

    validate_model_checksum(model, expected)
    with pytest.raises(ArtifactValidationError, match="checksum"):
        validate_model_checksum(model, "0" * 64)


def test_temperature_scaler_preserves_half_and_softens_confidence() -> None:
    scaler = TemperatureScaler(temperature=2.0)

    assert scaler(0.5) == pytest.approx(0.5)
    assert 0.5 < scaler(0.9) < 0.9
    assert 0.1 < scaler(0.1) < 0.5

