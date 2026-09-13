"""Exporter ↔ serving manifest contract tests.

The ONNX exporter (``riskforge.modeling.export_onnx.save_manifest``) and the
serving-side validator (``riskforge.serving.artifact.ModelArtifactManifest``)
must agree on one canonical schema.  These tests pin that agreement without
needing torch or a trained model: they statically inspect the exporter's
manifest construction and validate its shape against the serving schema.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from riskforge.core.contracts import LifeSavingRule
from riskforge.serving.artifact import ArtifactValidationError, ModelArtifactManifest

EXPORTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "riskforge"
    / "modeling"
    / "export_onnx.py"
)

CANONICAL_KEYS = {
    "schema_version",
    "model_version",
    "model_sha256",
    "backbone",
    "max_sequence_length",
    "quantization",
    "temperature",
    "input_names",
    "output_names",
    "iogp_rule_order",
}


def _manifest_dict_keys(source: str) -> set[str]:
    """Collect top-level string keys assigned in ``save_manifest``'s dict."""
    tree = ast.parse(source)
    keys: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "save_manifest":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Dict):
                    for key in sub.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            keys.add(key.value)
    return keys


class TestExporterServingSchemaAlignment:
    def test_exporter_emits_every_canonical_key(self) -> None:
        source = EXPORTER_PATH.read_text(encoding="utf-8")
        emitted = _manifest_dict_keys(source)
        missing = CANONICAL_KEYS - emitted
        assert not missing, f"exporter manifest is missing canonical keys: {sorted(missing)}"

    def test_exporter_no_longer_emits_legacy_schema_key(self) -> None:
        source = EXPORTER_PATH.read_text(encoding="utf-8")
        emitted = _manifest_dict_keys(source)
        assert "manifest_version" not in emitted, (
            "exporter still writes the legacy 'manifest_version' key that the "
            "serving validator rejects; use 'schema_version'"
        )

    def test_exporter_writes_checksum_matching_serving_format(self) -> None:
        source = EXPORTER_PATH.read_text(encoding="utf-8")
        assert "model_sha256" in source
        assert "hashlib.sha256" in source, (
            "exporter must embed a real SHA-256 digest of the exported ONNX file"
        )


class TestServingAcceptsExporterShape:
    def _exporter_shaped_manifest(self, model_sha256: str) -> dict[str, object]:
        """Mirror the exporter's canonical manifest exactly."""
        return {
            "schema_version": 1,
            "model_version": "deberta-v3-base-multitask-v1",
            "model_sha256": model_sha256,
            "backbone": "microsoft/deberta-v3-base",
            "max_sequence_length": 128,
            "quantization": "none",
            "temperature": 1.0,
            "input_names": ["input_ids", "attention_mask", "token_type_ids"],
            "output_names": ["sif_logits", "iogp_logits"],
            "iogp_rule_order": [rule.value for rule in LifeSavingRule],
        }

    def test_serving_validator_accepts_exporter_manifest(self, tmp_path: Path) -> None:
        manifest = self._exporter_shaped_manifest("c" * 64)
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = ModelArtifactManifest.load(path)
        assert loaded.model_version == "deberta-v3-base-multitask-v1"
        assert loaded.input_names == ("input_ids", "attention_mask", "token_type_ids")

    @pytest.mark.parametrize("missing_key", sorted(CANONICAL_KEYS))
    def test_serving_rejects_manifest_missing_any_canonical_key(
        self, tmp_path: Path, missing_key: str
    ) -> None:
        manifest = self._exporter_shaped_manifest("d" * 64)
        del manifest[missing_key]
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ArtifactValidationError):
            ModelArtifactManifest.load(path)

    def test_serving_rejects_legacy_exporter_manifest(self, tmp_path: Path) -> None:
        legacy = {
            "manifest_version": 1,
            "model_version": "deberta-v3-base-multitask-v1",
            "backbone": "microsoft/deberta-v3-base",
            "max_sequence_length": 128,
            "temperature": 1.0,
        }
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(legacy), encoding="utf-8")
        with pytest.raises(ArtifactValidationError):
            ModelArtifactManifest.load(path)
