"""Concrete HuggingFace-backed incident encoder.

Implements the application-facing ``IncidentInputEncoder`` protocol: every
normalized incident becomes right-padded int64 tensors sized to the
artifact's ``max_sequence_length`` so the ONNX session always receives a
fixed-width batch.  ``transformers`` is imported lazily with a precise
failure message for air-gapped hosts that never installed it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from riskforge.application.inference_adapter import EncodedIncident
from riskforge.core.contracts import IncidentNormalizedRecord


class EncoderDependencyError(RuntimeError):
    """Raised when the tokenizer stack is unavailable or misconfigured."""


class HFIncidentEncoder:
    """Tokenize incident narratives with a HuggingFace tokenizer.

    Parameters
    ----------
    tokenizer_name:
        HuggingFace model id or local path of the tokenizer to load.  Must
        match the artifact manifest's ``backbone`` — the production composer
        enforces that agreement at startup.
    max_sequence_length:
        Fixed output width; every batch is padded/truncated to this length.
    """

    def __init__(self, tokenizer_name: str, max_sequence_length: int) -> None:
        if not tokenizer_name:
            raise EncoderDependencyError("tokenizer_name must not be empty")
        if max_sequence_length < 1:
            raise EncoderDependencyError("max_sequence_length must be positive")
        self._max_sequence_length = max_sequence_length
        try:
            from transformers import AutoTokenizer
        except ImportError as error:  # pragma: no cover - depends on extras
            raise EncoderDependencyError(
                "transformers is required for model-backed encoding; install "
                "the serving extras or deploy a host with transformers present"
            ) from error
        try:
            self._tokenizer: Any = AutoTokenizer.from_pretrained(tokenizer_name)
        except Exception as error:
            raise EncoderDependencyError(
                f"cannot load tokenizer {tokenizer_name!r}: {error}"
            ) from error

    @property
    def max_sequence_length(self) -> int:
        return self._max_sequence_length

    def encode(self, record: IncidentNormalizedRecord) -> EncodedIncident:
        return self._encode_batch([record])

    def encode_batch(
        self, records: Sequence[IncidentNormalizedRecord]
    ) -> EncodedIncident:
        if not records:
            raise EncoderDependencyError("cannot encode an empty batch")
        return self._encode_batch(list(records))

    def _encode_batch(
        self, records: list[IncidentNormalizedRecord]
    ) -> EncodedIncident:
        narratives = [record.raw_narrative or "" for record in records]
        try:
            encoded = self._tokenizer(
                narratives,
                padding="max_length",
                truncation=True,
                max_length=self._max_sequence_length,
                return_tensors="np",
            )
        except Exception as error:
            raise EncoderDependencyError(
                f"tokenization failed: {error}"
            ) from error
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64)
        token_type_ids = encoded.get("token_type_ids")
        token_types = (
            None if token_type_ids is None else np.asarray(token_type_ids, dtype=np.int64)
        )
        if input_ids.ndim != 2 or input_ids.shape[1] != self._max_sequence_length:
            raise EncoderDependencyError(
                "tokenizer produced unexpected sequence width "
                f"{input_ids.shape}; expected {self._max_sequence_length}"
            )
        return EncodedIncident(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_types,
        )
