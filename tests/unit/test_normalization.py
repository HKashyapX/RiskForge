from datetime import datetime, timezone
import pytest
from riskforge.core.contracts import AssetType, IncidentRawRecord
from riskforge.normalization.gazetteer import SpanPreservingGazetteer

@pytest.fixture
def normalizer():
    return SpanPreservingGazetteer()

def test_span_preservation_offsets(normalizer):
    narrative = "Observed leak at BOP line while on monkey board."
    rec = IncidentRawRecord(
        log_id="NORM_001",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=narrative
    )

    normalized = normalizer.process(rec)
    assert normalized.raw_narrative == narrative

    spans = normalized.spans
    assert len(spans) == 2

    # BOP span verification
    bop_span = next(s for s in spans if s.canonical_form == "blowout_preventer")
    assert bop_span.text == "BOP"
    assert narrative[bop_span.start_char:bop_span.end_char] == "BOP"
    assert bop_span.entity_type == "BARRIER"

    # Monkey board span verification
    mb_span = next(s for s in spans if s.canonical_form == "monkey_board")
    assert mb_span.text == "monkey board"
    assert narrative[mb_span.start_char:mb_span.end_char] == "monkey board"
    assert mb_span.entity_type == "LOCATION"

def test_overlap_resolution_longest_match(normalizer):
    narrative = "Inspection completed on blowout preventer assembly."
    rec = IncidentRawRecord(
        log_id="NORM_002",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=narrative
    )

    normalized = normalizer.process(rec)
    assert len(normalized.spans) == 1
    span = normalized.spans[0]
    assert span.canonical_form == "blowout_preventer"
    assert span.text == "blowout preventer"
