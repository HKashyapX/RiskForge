"""Validation and calibration support for deployable ONNX artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from riskforge.core.contracts import LifeSavingRule


class ArtifactValidationError(ValueError):
    """Raised when a model bundle is unsafe or incompatible with serving."""


@dataclass(frozen=True)
class ModelArtifactManifest:
    schema_version: int
    model_sha256: str
    backbone: str
    max_sequence_length: int
    quantization: str
    temperature: float
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    iogp_rule_order: tuple[str, ...]

    @classmethod
    def load(cls, path: str | Path) -> ModelArtifactManifest:
        manifest_path = Path(path)
        try:
            payload: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactValidationError(f"cannot read artifact manifest: {error}") from error
        if not isinstance(payload, dict):
            raise ArtifactValidationError("artifact manifest must be a JSON object")
        try:
            manifest = cls(
                schema_version=int(payload["schema_version"]),
                model_sha256=str(payload["model_sha256"]),
                backbone=str(payload["backbone"]),
                max_sequence_length=int(payload["max_sequence_length"]),
                quantization=str(payload["quantization"]),
                temperature=float(payload["temperature"]),
                input_names=tuple(str(value) for value in payload["input_names"]),
                output_names=tuple(str(value) for value in payload["output_names"]),
                iogp_rule_order=tuple(str(value) for value in payload["iogp_rule_order"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ArtifactValidationError(f"invalid artifact manifest field: {error}") from error
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ArtifactValidationError("unsupported artifact manifest schema version")
        if len(self.model_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.model_sha256.lower()
        ):
            raise ArtifactValidationError("model_sha256 must be a hexadecimal SHA-256 digest")
        if not self.backbone:
            raise ArtifactValidationError("backbone must not be empty")
        if self.max_sequence_length < 1:
            raise ArtifactValidationError("max_sequence_length must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ArtifactValidationError("temperature must be finite and positive")
        if len(set(self.input_names)) != len(self.input_names) or not self.input_names:
            raise ArtifactValidationError("input_names must be non-empty and unique")
        if len(set(self.output_names)) != len(self.output_names) or len(self.output_names) < 2:
            raise ArtifactValidationError("output_names must contain unique SIF and IOGP outputs")
        expected_rules = tuple(rule.value for rule in LifeSavingRule)
        if self.iogp_rule_order != expected_rules:
            raise ArtifactValidationError("IOGP rule order does not match the serving contract")

    def validate_session(
        self, input_names: tuple[str, ...], output_names: tuple[str, ...]
    ) -> None:
        if input_names != self.input_names:
            raise ArtifactValidationError("ONNX input names differ from the artifact manifest")
        if output_names != self.output_names:
            raise ArtifactValidationError("ONNX output names differ from the artifact manifest")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as artifact:
        for chunk in iter(lambda: artifact.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_model_checksum(model_path: str | Path, expected_sha256: str) -> None:
    actual = sha256_file(model_path)
    if not hmac.compare_digest(actual.lower(), expected_sha256.lower()):
        raise ArtifactValidationError("ONNX model checksum does not match the manifest")


class TemperatureScaler:
    def __init__(self, temperature: float, epsilon: float = 1e-7) -> None:
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ArtifactValidationError("temperature must be finite and positive")
        if not 0.0 < epsilon < 0.5:
            raise ValueError("epsilon must be between zero and 0.5")
        self.temperature = temperature
        self.epsilon = epsilon

    def __call__(self, probability: float) -> float:
        clipped = min(max(float(probability), self.epsilon), 1.0 - self.epsilon)
        logit = math.log(clipped / (1.0 - clipped))
        return 1.0 / (1.0 + math.exp(-logit / self.temperature))
