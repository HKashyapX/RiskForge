from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock

import numpy as np
import pytest

from riskforge.serving.batching import ConcurrentRequestBatcher
from riskforge.serving.engine import ONNXInferenceEngine
from riskforge.serving.exceptions import BatcherClosedError, RequestQueueFullError
from tests.unit.test_serving import _Node, _record


class _RecordingSession:
    def __init__(self, *, block: bool = False) -> None:
        self.batch_sizes: list[int] = []
        self.started = Event()
        self.release = Event()
        self._block = block
        self._lock = Lock()

    def get_inputs(self):
        return [_Node("input_ids"), _Node("attention_mask")]

    def get_outputs(self):
        return [_Node("sif_logits"), _Node("iogp_logits")]

    def run(self, output_names, feed):
        batch_size = len(feed["input_ids"])
        with self._lock:
            self.batch_sizes.append(batch_size)
        self.started.set()
        if self._block:
            assert self.release.wait(timeout=2.0)
        return [np.zeros((batch_size, 1)), np.zeros((batch_size, 9))]


def _batcher(session, **kwargs) -> ConcurrentRequestBatcher:
    engine = ONNXInferenceEngine("unused.onnx", session=session, max_batch_size=8)
    return ConcurrentRequestBatcher(engine, max_queue_delay_ms=30.0, **kwargs)


def test_concurrent_requests_are_coalesced_with_deterministic_results() -> None:
    session = _RecordingSession()
    batcher = _batcher(session, max_batch_size=4)
    with ThreadPoolExecutor(max_workers=4) as callers:
        submissions = [
            callers.submit(batcher.submit, _record(), np.ones(8), np.ones(8))
            for _ in range(4)
        ]
        futures = [submission.result(timeout=1.0) for submission in submissions]

    results = [future.result(timeout=1.0) for future in futures]
    batcher.shutdown()

    assert session.batch_sizes == [4]
    assert [result.log_id for result in results] == ["SERVE_001"] * 4


def test_queue_capacity_is_bounded() -> None:
    session = _RecordingSession(block=True)
    batcher = _batcher(session, max_batch_size=1, max_queue_size=1)
    first = batcher.submit(_record(), np.ones(8), np.ones(8))
    assert session.started.wait(timeout=1.0)
    second = batcher.submit(_record(), np.ones(8), np.ones(8))

    with pytest.raises(RequestQueueFullError):
        batcher.submit(_record(), np.ones(8), np.ones(8))

    session.release.set()
    assert first.result(timeout=1.0)
    assert second.result(timeout=1.0)
    batcher.shutdown()


def test_cancelled_request_is_not_executed() -> None:
    session = _RecordingSession(block=True)
    batcher = _batcher(session, max_batch_size=1, max_queue_size=2)
    first = batcher.submit(_record(), np.ones(8), np.ones(8))
    assert session.started.wait(timeout=1.0)
    cancelled = batcher.submit(_record(), np.ones(8), np.ones(8))
    assert cancelled.cancel()

    session.release.set()
    assert first.result(timeout=1.0)
    batcher.shutdown()

    assert cancelled.cancelled()
    assert session.batch_sizes == [1]


def test_graceful_shutdown_drains_work_and_rejects_new_requests() -> None:
    session = _RecordingSession()
    batcher = _batcher(session, max_batch_size=2)
    future = batcher.submit(_record(), np.ones(8), np.ones(8))

    batcher.shutdown(wait=True)

    assert future.done() and not future.cancelled()
    with pytest.raises(BatcherClosedError):
        batcher.submit(_record(), np.ones(8), np.ones(8))
