from time import sleep

import pytest

from riskforge.serving.benchmark import (
    PeakRssMonitor,
    benchmark_concurrent_requests,
    benchmark_warm_batches,
    current_rss_bytes,
    latency_summary,
    percentile,
)
from tests.unit.test_serving import _Session


class _BenchmarkEngine:
    max_batch_size = 32

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def infer_batch(self, records, input_ids, attention_mask, token_type_ids=None):
        self.batch_sizes.append(len(records))
        sleep(0.0001)
        return list(records)


def test_percentiles_and_latency_summary_are_deterministic() -> None:
    samples = [4.0, 1.0, 3.0, 2.0]
    assert percentile(samples, 50.0) == 2.5
    assert percentile(samples, 95.0) == pytest.approx(3.85)
    summary = latency_summary(samples)
    assert summary.sample_count == 4
    assert summary.minimum_ms == 1.0
    assert summary.maximum_ms == 4.0


def test_warm_benchmark_preserves_requested_batch_order() -> None:
    engine = _BenchmarkEngine()
    results = benchmark_warm_batches(
        engine, batch_sizes=(1, 8, 16, 32), iterations=2, sequence_length=8
    )
    assert [result.batch_size for result in results] == [1, 8, 16, 32]
    assert all(result.iterations == 2 for result in results)
    assert all(result.latency.sample_count == 2 for result in results)
    assert all(result.peak_rss_bytes >= result.baseline_rss_bytes > 0 for result in results)
    assert engine.batch_sizes == [1, 1, 1, 8, 8, 8, 16, 16, 16, 32, 32, 32]


def test_concurrent_benchmark_reports_all_requests() -> None:
    engine = _BenchmarkEngine()
    report = benchmark_concurrent_requests(
        engine, concurrency=4, iterations=2, sequence_length=8, max_queue_delay_ms=10.0
    )
    assert report["request_count"] == 8
    assert report["latency"]["sample_count"] == 8
    assert report["memory"]["peak_rss_bytes"] > 0
    assert sum(engine.batch_sizes) == 8


@pytest.mark.parametrize("samples, value", [([], 50.0), ([1.0], -1.0), ([1.0], 101.0)])
def test_percentile_rejects_invalid_arguments(samples, value) -> None:
    with pytest.raises(ValueError):
        percentile(samples, value)


def test_benchmark_rejects_batch_above_engine_limit() -> None:
    engine = _Session()
    engine.max_batch_size = 8
    with pytest.raises(ValueError, match="exceeds"):
        benchmark_warm_batches(engine, batch_sizes=(16,), iterations=1, sequence_length=8)


def test_peak_rss_monitor_reports_total_process_memory() -> None:
    assert current_rss_bytes() > 0
    with PeakRssMonitor(poll_interval_seconds=0.001) as monitor:
        allocation = bytearray(1_000_000)
        assert allocation
    summary = monitor.summary()
    assert summary.baseline_rss_bytes > 0
    assert summary.peak_rss_bytes >= summary.baseline_rss_bytes
    assert summary.final_rss_bytes > 0


def test_peak_rss_monitor_requires_positive_interval_and_completed_context() -> None:
    with pytest.raises(ValueError, match="positive"):
        PeakRssMonitor(0.0)
    monitor = PeakRssMonitor()
    with pytest.raises(RuntimeError, match="after monitoring"):
        monitor.summary()
