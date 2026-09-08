"""CPU-only ONNX Runtime execution with bounded threading and dynamic batching."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from riskforge.core.contracts import IncidentNormalizedRecord, ModelInferenceResult
from riskforge.serving.artifact import (
    ModelArtifactManifest,
    TemperatureScaler,
    validate_model_checksum,
)
from riskforge.serving.postprocessor import InferencePostprocessor


class ONNXInferenceEngine:
    @classmethod
    def from_artifact(
        cls,
        model_path: str | Path,
        manifest_path: str | Path,
        *,
        max_batch_size: int = 32,
        warmup: bool = True,
    ) -> ONNXInferenceEngine:
        manifest = ModelArtifactManifest.load(manifest_path)
        validate_model_checksum(model_path, manifest.model_sha256)
        engine = cls(
            model_path,
            postprocessor=InferencePostprocessor(
                calibrator=TemperatureScaler(manifest.temperature)
            ),
            max_batch_size=max_batch_size,
        )
        manifest.validate_session(engine.input_names, engine.output_names)
        if warmup:
            engine.warmup(manifest.max_sequence_length)
        return engine

    def __init__(
        self,
        model_path: str | Path,
        *,
        postprocessor: InferencePostprocessor | None = None,
        max_batch_size: int = 32,
        session: Any | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.model_path = Path(model_path)
        self.max_batch_size = max_batch_size
        self.postprocessor = postprocessor or InferencePostprocessor()
        if session is None:
            if not self.model_path.is_file():
                raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError("onnxruntime is required for model serving") from error
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(
                str(self.model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        self.session = session
        self.input_names = tuple(item.name for item in self.session.get_inputs())
        self.output_names = tuple(item.name for item in self.session.get_outputs())
        if len(self.output_names) < 2:
            raise ValueError("the ONNX model must expose SIF and IOGP outputs")

    def warmup(self, sequence_length: int = 8) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        sample = np.zeros((1, sequence_length), dtype=np.int64)
        feed = self._input_feed(sample, np.ones_like(sample), None)
        outputs = self.session.run(None, feed)
        self._split_outputs(outputs)

    def _input_feed(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        token_type_ids: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        ids = np.asarray(input_ids, dtype=np.int64)
        mask = np.asarray(attention_mask, dtype=np.int64)
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise ValueError("input_ids and attention_mask must be equally shaped rank-2 arrays")
        token_types = None if token_type_ids is None else np.asarray(token_type_ids, dtype=np.int64)
        if token_types is not None and token_types.shape != ids.shape:
            raise ValueError("token_type_ids must match input_ids")
        feed: dict[str, np.ndarray] = {}
        for name in self.input_names:
            lowered = name.lower()
            if "attention" in lowered and "mask" in lowered:
                feed[name] = mask
            elif "token_type" in lowered or "segment" in lowered:
                feed[name] = np.zeros_like(ids) if token_types is None else token_types
            elif "input" in lowered and "id" in lowered:
                feed[name] = ids
        missing = set(self.input_names) - set(feed)
        if missing:
            raise ValueError(f"unsupported ONNX inputs: {', '.join(sorted(missing))}")
        return feed

    def _split_outputs(self, outputs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        named = dict(zip(self.output_names, outputs, strict=True))
        sif_name = next((name for name in self.output_names if "sif" in name.lower()), None)
        rule_name = next(
            (
                name
                for name in self.output_names
                if "iogp" in name.lower() or "rule" in name.lower()
            ),
            None,
        )
        if sif_name is not None and rule_name is not None and sif_name != rule_name:
            return np.asarray(named[sif_name]), np.asarray(named[rule_name])
        return np.asarray(outputs[0]), np.asarray(outputs[1])

    def _run_batch(
        self,
        records: Sequence[IncidentNormalizedRecord],
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        token_type_ids: np.ndarray | None,
    ) -> list[ModelInferenceResult]:
        if not records:
            return []
        if len(records) != len(input_ids):
            raise ValueError("record and tensor batch sizes must match")
        feed = self._input_feed(input_ids, attention_mask, token_type_ids)
        started = perf_counter()
        outputs = self.session.run(None, feed)
        latency_ms = (perf_counter() - started) * 1000.0
        sif_logits, iogp_logits = self._split_outputs(outputs)
        return self.postprocessor.process_batch(
            records, sif_logits, iogp_logits, latency_ms=latency_ms
        )

    def infer_batch(
        self,
        records: Sequence[IncidentNormalizedRecord],
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        token_type_ids: np.ndarray | None = None,
    ) -> list[ModelInferenceResult]:
        ids = np.asarray(input_ids)
        mask = np.asarray(attention_mask)
        token_types = None if token_type_ids is None else np.asarray(token_type_ids)
        if len(records) != len(ids):
            raise ValueError("record and tensor batch sizes must match")
        results: list[ModelInferenceResult] = []
        for start in range(0, len(records), self.max_batch_size):
            stop = min(start + self.max_batch_size, len(records))
            results.extend(
                self._run_batch(
                    records[start:stop],
                    ids[start:stop],
                    mask[start:stop],
                    None if token_types is None else token_types[start:stop],
                )
            )
        return results

    def infer(
        self,
        record: IncidentNormalizedRecord,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        token_type_ids: np.ndarray | None = None,
    ) -> ModelInferenceResult:
        def batched(values: np.ndarray) -> np.ndarray:
            array = np.asarray(values)
            return array[None, :] if array.ndim == 1 else array

        return self.infer_batch(
            [record],
            batched(input_ids),
            batched(attention_mask),
            None if token_type_ids is None else batched(token_type_ids),
        )[0]

    predict = infer
    predict_batch = infer_batch


InferenceEngine = ONNXInferenceEngine
