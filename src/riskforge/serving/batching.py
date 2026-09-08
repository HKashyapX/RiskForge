"""Bounded concurrent request coalescing for synchronous ONNX inference."""

from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Lock, Thread
from time import monotonic
from typing import Self

import numpy as np

from riskforge.core.contracts import IncidentNormalizedRecord, ModelInferenceResult
from riskforge.serving.engine import ONNXInferenceEngine
from riskforge.serving.exceptions import (
    BatcherClosedError,
    InputCompatibilityError,
    RequestQueueFullError,
    ServingError,
)


@dataclass(frozen=True)
class _Request:
    record: IncidentNormalizedRecord
    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: np.ndarray | None
    future: Future[ModelInferenceResult]

    @property
    def shape_key(self) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...] | None]:
        return (
            self.input_ids.shape,
            self.attention_mask.shape,
            None if self.token_type_ids is None else self.token_type_ids.shape,
        )


_STOP = object()


class ConcurrentRequestBatcher:
    """Coalesce concurrent single-record requests into bounded engine batches."""

    def __init__(
        self,
        engine: ONNXInferenceEngine,
        *,
        max_batch_size: int | None = None,
        max_queue_size: int = 256,
        max_queue_delay_ms: float = 5.0,
    ) -> None:
        resolved_batch_size = engine.max_batch_size if max_batch_size is None else max_batch_size
        if resolved_batch_size < 1 or resolved_batch_size > engine.max_batch_size:
            raise ValueError("max_batch_size must be between 1 and the engine maximum")
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        if not 0.0 <= max_queue_delay_ms <= 1000.0:
            raise ValueError("max_queue_delay_ms must be between 0 and 1000")

        self.engine = engine
        self.max_batch_size = resolved_batch_size
        self.max_queue_delay_seconds = max_queue_delay_ms / 1000.0
        self._queue: Queue[_Request | object] = Queue(maxsize=max_queue_size)
        self._state_lock = Lock()
        self._accepting = True
        self._worker = Thread(target=self._run, name="riskforge-inference-batcher", daemon=True)
        self._worker.start()

    @property
    def pending_count(self) -> int:
        """Return the approximate number of requests waiting in the bounded queue."""
        return self._queue.qsize()

    @property
    def is_shutdown(self) -> bool:
        with self._state_lock:
            return not self._accepting

    def submit(
        self,
        record: IncidentNormalizedRecord,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
        token_type_ids: np.ndarray | None = None,
    ) -> Future[ModelInferenceResult]:
        request = _Request(
            record=record,
            input_ids=self._single_tensor(input_ids, "input_ids"),
            attention_mask=self._single_tensor(attention_mask, "attention_mask"),
            token_type_ids=(
                None
                if token_type_ids is None
                else self._single_tensor(token_type_ids, "token_type_ids")
            ),
            future=Future(),
        )
        if request.input_ids.shape != request.attention_mask.shape:
            raise InputCompatibilityError("input_ids and attention_mask must have matching shapes")
        if request.token_type_ids is not None and request.token_type_ids.shape != request.input_ids.shape:
            raise InputCompatibilityError("token_type_ids must match input_ids")

        with self._state_lock:
            if not self._accepting:
                raise BatcherClosedError("request batcher is shutting down")
            try:
                self._queue.put_nowait(request)
            except Full as error:
                raise RequestQueueFullError("inference request queue is full") from error
        return request.future

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        """Stop accepting work and optionally cancel requests that have not started."""
        with self._state_lock:
            if not self._accepting:
                if wait:
                    self._worker.join()
                return
            self._accepting = False

        if cancel_pending:
            while True:
                try:
                    item = self._queue.get_nowait()
                except Empty:
                    break
                if isinstance(item, _Request):
                    item.future.cancel()
                self._queue.task_done()
        self._queue.put(_STOP)
        if wait:
            self._worker.join()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown(wait=True, cancel_pending=exc_type is not None)

    @staticmethod
    def _single_tensor(values: np.ndarray, name: str) -> np.ndarray:
        array = np.asarray(values)
        if array.ndim == 2 and array.shape[0] == 1:
            array = array[0]
        if array.ndim != 1:
            raise InputCompatibilityError(f"{name} must describe exactly one rank-1 record")
        return array.copy()

    def _run(self) -> None:
        deferred: _Request | object | None = None
        while True:
            item = deferred if deferred is not None else self._queue.get()
            deferred = None
            if item is _STOP:
                if deferred is None:
                    self._queue.task_done()
                return
            assert isinstance(item, _Request)
            batch = [item]
            deadline = monotonic() + self.max_queue_delay_seconds

            while len(batch) < self.max_batch_size:
                timeout = max(0.0, deadline - monotonic())
                if timeout == 0.0:
                    break
                try:
                    candidate = self._queue.get(timeout=timeout)
                except Empty:
                    break
                if candidate is _STOP:
                    deferred = candidate
                    break
                assert isinstance(candidate, _Request)
                if candidate.shape_key != batch[0].shape_key:
                    deferred = candidate
                    break
                batch.append(candidate)

            runnable = [request for request in batch if request.future.set_running_or_notify_cancel()]
            if runnable:
                self._execute(runnable)
            for _ in batch:
                self._queue.task_done()

    def _execute(self, requests: list[_Request]) -> None:
        try:
            token_types = None
            if requests[0].token_type_ids is not None:
                token_types = np.stack([request.token_type_ids for request in requests])
            results = self.engine.infer_batch(
                [request.record for request in requests],
                np.stack([request.input_ids for request in requests]),
                np.stack([request.attention_mask for request in requests]),
                token_types,
            )
            if len(results) != len(requests):
                raise RuntimeError("inference engine returned an unexpected result count")
        except (ServingError, RuntimeError, TypeError, ValueError) as error:
            for request in requests:
                request.future.set_exception(error)
            return
        for request, result in zip(requests, results, strict=True):
            request.future.set_result(result)
