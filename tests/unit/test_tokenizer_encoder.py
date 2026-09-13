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
def fake_transformers(monkeypatch, tmp_path):
    module = types.ModuleType("transformers")

    tokenizers: dict[str, _FakeTokenizer] = {}

    def auto_from_pretrained(name: str, local_files_only: bool = False) -> _FakeTokenizer:
        if name == "missing/model":
            raise OSError("not found")
        tokenizer = tokenizers.setdefault(name, _FakeTokenizer())
        return tokenizer

    module.AutoTokenizer = types.SimpleNamespace(from_pretrained=auto_from_pretrained)
    module._tokenizers = tokenizers
    monkeypatch.setitem(sys.modules, "transformers", module)

    # Every construction test needs a local tokenizer directory; create one
    # and hand back its path so tests mirror the air-gap policy.
    local = tmp_path / "tokenizer"
    local.mkdir(exist_ok=True)
    module._local_dir = str(local)
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
    def test_requires_transformers(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setitem(sys.modules, "transformers", None)
        local = tmp_path / "tokenizer"
        local.mkdir()
        with pytest.raises(EncoderDependencyError, match="transformers is required"):
            HFIncidentEncoder(str(local), 128)

    def test_missing_tokenizer_fails_with_clear_error(self, fake_transformers, tmp_path) -> None:
        # A directory that exists but whose load fails is a load error.
        missing_local = tmp_path / "empty-but-missing-load"
        missing_local.mkdir()

        def _raise(name, local_files_only=False):
            raise OSError("not found")

        fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=_raise)
        with pytest.raises(EncoderDependencyError, match="cannot load tokenizer"):
            HFIncidentEncoder(str(missing_local), 128)

    def test_rejects_remote_hub_identifiers(self, fake_transformers) -> None:
        # org/model hub ids imply a network fetch — refused outright.
        with pytest.raises(EncoderDependencyError, match="local"):
            HFIncidentEncoder("backbone/x", 128)

    def test_rejects_missing_local_directory(self, fake_transformers, tmp_path) -> None:
        with pytest.raises(EncoderDependencyError, match="local"):
            HFIncidentEncoder(str(tmp_path / "nope"), 128)

    def test_rejects_invalid_configuration(self, fake_transformers) -> None:
        with pytest.raises(EncoderDependencyError, match="tokenizer_name"):
            HFIncidentEncoder("", 128)
        with pytest.raises(EncoderDependencyError, match="max_sequence_length"):
            HFIncidentEncoder(fake_transformers._local_dir, 0)


class TestEncoding:
    def test_single_record_produces_fixed_width_tensors(self, fake_transformers) -> None:
        encoder = HFIncidentEncoder(fake_transformers._local_dir, 8)
        encoded = encoder.encode(_record())
        assert encoded.input_ids.shape == (1, 8)
        assert encoded.attention_mask.shape == (1, 8)
        assert encoded.token_type_ids is not None
        assert encoded.token_type_ids.shape == (1, 8)
        assert encoded.input_ids.dtype == np.int64

    def test_batch_preserves_order_and_width(self, fake_transformers) -> None:
        encoder = HFIncidentEncoder(fake_transformers._local_dir, 8)
        encoded = encoder.encode_batch([_record("A"), _record("B"), _record("C")])
        assert encoded.input_ids.shape == (3, 8)

    def test_implements_application_protocol(self, fake_transformers) -> None:
        encoder = HFIncidentEncoder(fake_transformers._local_dir, 8)
        assert isinstance(encoder, IncidentInputEncoder)

    def test_empty_batch_is_rejected(self, fake_transformers) -> None:
        encoder = HFIncidentEncoder(fake_transformers._local_dir, 8)
        with pytest.raises(EncoderDependencyError, match="empty batch"):
            encoder.encode_batch([])

    def test_unexpected_tokenizer_width_fails_closed(self, monkeypatch, tmp_path) -> None:
        module = types.ModuleType("transformers")
        rogue = _FakeTokenizer(width=16)

        module.AutoTokenizer = types.SimpleNamespace(
            from_pretrained=lambda name, local_files_only=False: rogue
        )
        monkeypatch.setitem(sys.modules, "transformers", module)
        local = tmp_path / "tokenizer"
        local.mkdir()
        encoder = HFIncidentEncoder(str(local), 8)
        with pytest.raises(EncoderDependencyError, match="tokenization failed"):
            encoder.encode(_record())

    def test_max_sequence_length_exposed(self, fake_transformers) -> None:
        assert HFIncidentEncoder(fake_transformers._local_dir, 128).max_sequence_length == 128
