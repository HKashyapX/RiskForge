"""Benchmark tests for SpanPreservingGazetteer performance.

CONTEXT.md requires <5ms per incident record.
"""
import statistics
import time

from datetime import datetime, timezone

from riskforge.core.contracts import AssetType, IncidentRawRecord
from riskforge.normalization.gazetteer import SpanPreservingGazetteer, clear_config_cache

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

NARRATIVES = {
    "short_50chars": "BOP checked during inspection.",
    "medium_200chars": (
        "During the night shift at Well-12, the blowout preventer was tested. "
        "BOP passed all pressure tests. H2S levels were within acceptable limits."
    ),
    "long_500chars": (
        "During the night shift at Well-12, the blowout preventer was tested and "
        "found to be in good condition. BOP passed all pressure tests. H2S levels "
        "were within acceptable limits. The swabbing unit was deployed to clear "
        "the wellbore. Kill line pressure was monitored throughout. LOTO procedures "
        "were followed for energy isolation. The christmas tree valves were inspected. "
        "Monkey board access was restricted."
    ),
    "no_matches": (
        "Routine inspection completed. All systems nominal. "
        "No anomalies detected in standard procedures."
    ),
    "all_matches": (
        "BOP tested. H2S checked. Kill line inspected. LOTO verified. "
        "PRV serviced. Swabbing unit deployed. Monkey board accessed. "
        "Christmas tree inspected."
    ),
}

REPS = 1000
WARMUP = 100
SLA_MS = 5.0


def _make_record(narrative: str) -> IncidentRawRecord:
    return IncidentRawRecord(
        log_id="BENCH",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=narrative,
    )


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def test_p99_latency_below_5ms():
    """p99 end-to-end latency across all narratives must stay under 5ms."""
    clear_config_cache()
    gaz = SpanPreservingGazetteer()

    # Warmup: ensure JIT / cache effects are settled
    for _ in range(WARMUP):
        for text in NARRATIVES.values():
            gaz.process(_make_record(text))

    all_times_ns: list[int] = []

    for name, text in NARRATIVES.items():
        times: list[int] = []
        for _ in range(REPS):
            record = _make_record(text)
            start = time.perf_counter_ns()
            gaz.process(record)
            end = time.perf_counter_ns()
            times.append(end - start)
        times.sort()
        p50 = statistics.median(times)
        p99 = times[int(len(times) * 0.99)]
        print(f"  {name:25s}: median={p50/1e6:.3f}ms  p99={p99/1e6:.3f}ms")
        all_times_ns.extend(times)

    all_times_ns.sort()
    overall_p99 = all_times_ns[int(len(all_times_ns) * 0.99)]
    print(f"\n  Overall p99: {overall_p99/1e6:.3f}ms  (SLA: <{SLA_MS}ms)")
    assert overall_p99 / 1e6 < SLA_MS, (
        f"p99 latency {overall_p99/1e6:.3f}ms exceeds {SLA_MS}ms SLA"
    )


# ---------------------------------------------------------------------------
# Cache correctness
# ---------------------------------------------------------------------------

def test_cache_reuses_compiled_rules():
    """Two gazetteer instances for the same file share compiled rules."""
    clear_config_cache()
    gaz1 = SpanPreservingGazetteer()
    gaz2 = SpanPreservingGazetteer()
    assert gaz1.compiled_rules is gaz2.compiled_rules


def test_cache_clear_and_rebuild():
    """After clearing cache, new instance compiles fresh rules."""
    clear_config_cache()
    gaz1 = SpanPreservingGazetteer()
    id1 = id(gaz1.compiled_rules)
    clear_config_cache()
    gaz2 = SpanPreservingGazetteer()
    id2 = id(gaz2.compiled_rules)
    assert id1 != id2, "Cache should have been rebuilt with a new list"
    # But contents are identical
    assert len(gaz1.compiled_rules) == len(gaz2.compiled_rules)


def test_cache_independence():
    """Clearing cache does not break existing gazetteer instances."""
    clear_config_cache()
    gaz = SpanPreservingGazetteer()
    rules_ref = gaz.compiled_rules
    clear_config_cache()
    # Existing instance keeps its reference and still works
    assert gaz.compiled_rules is rules_ref
    record = _make_record("The BOP was inspected.")
    result = gaz.process(record)
    assert len(result.spans) == 1
