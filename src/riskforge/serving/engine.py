"""CPU-only ONNX Runtime execution with bounded threading and dynamic batching."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
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
from riskforge.serving.exceptions import (
    ArtifactLoadingError,
    InferenceFailureError,
    InputCompatibilityError,
    InvalidOutputError,
    ServingError,
    WarmupError,
)
from riskforge.serving.postprocessor import InferencePostprocessor

logger = logging.getLogger("riskforge.serving.engine")

# Default inference timeout in seconds
_DEFAULT_INFERENCE_TIMEOUT_S = 30.0
# Number of consecutive failures before marking engine degraded
_DEGRADED_THRESHOLD = 5


class ONNXInferenceEngine:
    @classmethod
    def from_artifact(
        cls,
        model_path: str | Path,
        manifest_path: str | Path,
        *,
        max_batch_size: int = 32,
        warmup: bool = True,
        inference_timeout_s: float = _DEFAULT_INFERENCE_TIMEOUT_S,
    ) -> ONNXInferenceEngine:
        logger.info("loading model artifact", extra={"model_path": str(model_path)})
        try:
            manifest = ModelArtifactManifest.load(manifest_path)
            validate_model_checksum(model_path, manifest.model_sha256)
            engine = cls(
                model_path,
                postprocessor=InferencePostprocessor(
                    calibrator=TemperatureScaler(manifest.temperature)
                ),
                max_batch_size=max_batch_size,
                inference_timeout_s=inference_timeout_s,
            )
            manifest.validate_session(engine.input_names, engine.output_names)
        except ServingError:
            raise
        except Exception as error:
            raise ArtifactLoadingError(f"cannot load model artifact: {error}") from error
        if warmup:
            try:
                engine.warmup(manifest.max_sequence_length)
            except WarmupError:
                raise
            except Exception as error:
                raise WarmupError(f"model warm-up failed: {error}") from error
        logger.info(
            "model artifact loaded successfully",
            extra={
                "model_path": str(model_path),
                "temperature": manifest.temperature,
                "max_batch_size": max_batch_size,
                "input_names": list(engine.input_names),
                "output_names": list(engine.output_names),
            },
        )
        return engine

    def __init__(
        self,
        model_path: str | Path,
        *,
        postprocessor: InferencePostprocessor | None = None,
        max_batch_size: int = 32,
        session: Any | None = None,
        inference_timeout_s: float = _DEFAULT_INFERENCE_TIMEOUT_S,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.model_path = Path(model_path)
        self.max_batch_size = max_batch_size
        self.inference_timeout_s = inference_timeout_s
        self.postprocessor = postprocessor or InferencePostprocessor()
        self._consecutive_failures = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="onnx-infer")
        if session is None:
            if not self.model_path.is_file():
                raise ArtifactLoadingError(f"ONNX model not found: {self.model_path}")
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise ArtifactLoadingError("onnxruntime is required for model serving") from error
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            try:
                session = ort.InferenceSession(
                    str(self.model_path),
                    sess_options=options,
                    providers=["CPUExecutionProvider"],
                )
            except Exception as error:
                raise ArtifactLoadingError(f"cannot create ONNX Runtime session: {error}") from error
        self.session = session
        try:
            self.input_names = tuple(item.name for item in self.session.get_inputs())
            self.output_names = tuple(item.name for item in self.session.get_outputs())
        except Exception as error:
            raise ArtifactLoadingError(f"cannot inspect ONNX model interface: {error}") from error
        if len(self.output_names) < 2:
            raise InvalidOutputError("the ONNX model must expose SIF and IOGP outputs")

    @property
    def is_degraded(self) -> bool:
        """True when consecutive failures exceed the degradation threshold."""
        return self._consecutive_failures >= _DEGRADED_THRESHOLD

    def warmup(self, sequence_length: int = 8) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        sample = np.zeros((1, sequence_length), dtype=np.int64)
        feed = self._input_feed(sample, np.ones_like(sample), None)
        try:
            outputs = self.session.run(None, feed)
            self._split_outputs(outputs)
            logger.info("model warmup completed", extra={"sequence_length": sequence_length})
        except Exception as error:
            raise WarmupError(f"model warm-up failed: {error}") from error

    def _input_feed(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        token_type_ids: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        ids = np.asarray(input_ids, dtype=np.int64)
        mask = np.asarray(attention_mask, dtype=np.int64)
        if ids.ndim != 2 or mask.shape != ids.shape:
            raise InputCompatibilityError(
                "input_ids and attention_mask must be equally shaped rank-2 arrays"
            )
        token_types = None if token_type_ids is None else np.asarray(token_type_ids, dtype=np.int64)
        if token_types is not None and token_types.shape != ids.shape:
            raise InputCompatibilityError("token_type_ids must match input_ids")
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
            raise InputCompatibilityError(
                f"unsupported ONNX inputs: {', '.join(sorted(missing))}"
            )
        return feed

    def _split_outputs(self, outputs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        if len(outputs) != len(self.output_names):
            raise InvalidOutputError(
                f"runtime returned {len(outputs)} outputs; expected {len(self.output_names)}"
            )
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
            raise InputCompatibilityError("record and tensor batch sizes must match")
        feed = self._input_feed(input_ids, attention_mask, token_type_ids)
        started = perf_counter()
        try:
            # Run inference with timeout protection
            future = self._executor.submit(self.session.run, None, feed)
            try:
                outputs = future.result(timeout=self.inference_timeout_s)
            except FuturesTimeoutError as error:
                self._consecutive_failures += 1
                raise InferenceFailureError(
                    f"ONNX Runtime inference timed out after {self.inference_timeout_s}s"
                ) from error
        except InferenceFailureError:
            raise
        except Exception as error:
            self._consecutive_failures += 1
            raise InferenceFailureError(f"ONNX Runtime inference failed: {error}") from error
        latency_ms = (perf_counter() - started) * 1000.0
        self._consecutive_failures = 0  # reset on success
        logger.debug(
            "inference completed",
            extra={
                "batch_size": len(records),
                "latency_ms": round(latency_ms, 2),
            },
        )
        sif_logits, iogp_logits = self._split_outputs(outputs)
        try:
            return self.postprocessor.process_batch(
                records, sif_logits, iogp_logits, latency_ms=latency_ms
            )
        except (IndexError, TypeError, ValueError) as error:
            raise InvalidOutputError(f"invalid model output: {error}") from error

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
            raise InputCompatibilityError("record and tensor batch sizes must match")
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
