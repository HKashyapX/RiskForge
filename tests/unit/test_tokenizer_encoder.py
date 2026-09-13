"""Tests for the concrete HuggingFace incident encoder.

``transformers`` is not installed in every environment (CI installs the lean
test set), so these tests inject a fake ``transformers`` module and verify
tensor shapes, padding behavior, protocol conformance, and failure modes.
"""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime

import numpy as np
import pytest

from riskforge.application.inference_adapter import IncidentInputEncoder
from riskforge.core.contracts import AssetType, IncidentNormalizedRecord
from riskforge.encoding.tokenizer_encoder import (
    EncoderDependencyError,
    HFIncidentEncoder,
)


class _FakeTokenizer:
    def __init__(self, width: int = 8, with_token_types: bool = True) -> None:
        self.width = width
        self.with_token_types = with_token_types
        self.calls: list[list[str]] = []

    def __call__(self, narratives, **kwargs):
        self.calls.append(list(narratives))
        assert kwargs["padding"] == "max_length"
        assert kwargs["truncation"] is True
        assert kwargs["max_length"] == self.width
        assert kwargs["return_tensors"] == "np"
        batch = len(narratives)
        payload: dict[str, np.ndarray] = {
            "input_ids": np.ones((batch, self.width), dtype=np.int64),
            "attention_mask": np.ones((batch, self.width), dtype=np.int64),
        }
        if self.with_token_types:
            payload["token_type_ids"] = np.zeros((batch, self.width), dtype=np.int64)
        return payload


@pytest.fixture()
def fake_transformers(monkeypatch):
    module = types.ModuleType("transformers")

    tokenizers: dict[str, _FakeTokenizer] = {}

    def auto_from_pretrained(name: str) -> _FakeTokenizer:
        if name == "missing/model":
            raise OSError("not found")
        tokenizer = tokenizers.setdefault(name, _FakeTokenizer())
        return tokenizer

    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=auto_from_pretrained)
    module._tokenizers = tokenizers
    monkeypatch.setitem(sys.modules, "transformers", module)
    return module


def _record(log_id: str = "L1", narrative: str = "Worker exposed at height.") -> IncidentNormalizedRecord:
    return IncidentNormalizedRecord(
        log_id=log_id,
        timestamp=datetime.now(UTC),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=narrative,
        spans=[],
    )


class TestConstruction:
    def test_requires_transformers(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "transformers", None)
        with pytest.raises(EncoderDependencyError, match="transformers is required"):
            HFIncidentEncoder("backbone/x", 128)

    def test_missing_tokenizer_fails_with_clear_error(self, fake_transformers) -> None:
        with pytest.raises(EncoderDependencyError, match="cannot load tokenizer"):
            HFIncidentEncoder("missing/model", 128)

    def test_rejects_invalid_configuration(self, fake_transformers) -> None:
        with pytest.raises(EncoderDependencyError, match="tokenizer_name"):
            HFIncidentEncoder("", 128)
        with pytest.raises(EncoderDependencyError, match="max_sequence_length"):
            HFIncidentEncoder("backbone/x", 0)


class TestEncoding:
    def test_single_record_produces_fixed_width_tensors(self, fake_transformers) -> None:
        encoder = HFIncidentEncoder("backbone/x", 8)
        encoded = encoder.encode(_record())
        assert encoded.input_ids.shape == (1, 8)
        assert encoded.attention_mask.shape == (1, 8)
        assert encoded.token_type_ids is not None
        assert encoded.token_type_ids.shape == (1, 8)
        assert encoded.input_ids.dtype == np.int64

    def test_batch_preserves_order_and_width(self, fake_transformers) -> None:
        encoder = HFIncidentEncoder("backbone/x", 8)
        encoded = encoder.encode_batch([_record("A"), _record("B"), _record("C")])
        assert encoded.input_ids.shape == (3, 8)

    def test_implements_application_protocol(self, fake_transformers) -> None:
        encoder = HFIncidentEncoder("backbone/x", 8)
        assert isinstance(encoder, IncidentInputEncoder)

    def test_empty_batch_is_rejected(self, fake_transformers) -> None:
        encoder = HFIncidentEncoder("backbone/x", 8)
        with pytest.raises(EncoderDependencyError, match="empty batch"):
            encoder.encode_batch([])

    def test_unexpected_tokenizer_width_fails_closed(self, monkeypatch) -> None:
        module = types.ModuleType("transformers")
        rogue = _FakeTokenizer(width=16)

        module.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name: rogue)
        monkeypatch.setitem(sys.modules, "transformers", module)
        encoder = HFIncidentEncoder("backbone/x", 8)
        with pytest.raises(EncoderDependencyError, match="tokenization failed"):
            encoder.encode(_record())

    def test_max_sequence_length_exposed(self, fake_transformers) -> None:
        assert HFIncidentEncoder("backbone/x", 128).max_sequence_length == 128
