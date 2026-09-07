import re
from pathlib import Path
from typing import List, Optional, Tuple
import yaml

from riskforge.core.contracts import EntitySpan, IncidentNormalizedRecord, IncidentRawRecord
from riskforge.core.exceptions import NormalizationError

class SpanPreservingGazetteer:
    def __init__(self, config_path: Optional[Path] = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[3] / "config" / "gazetteer_rules.yaml"

        if not config_path.exists():
            raise NormalizationError(f"Gazetteer configuration not found at {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self.compiled_rules: List[Tuple[re.Pattern, str, str]] = []
        for entry in data.get("rules", []):
            canonical = entry["canonical"]
            entity_type = entry["entity_type"]
            for pattern_str in entry.get("patterns", []):
                tokens = pattern_str.strip().split()
                if not tokens:
                    continue
                escaped_tokens = [re.escape(token) for token in tokens]
                pattern_body = r"\s+".join(escaped_tokens)
                pattern = re.compile(rf"\b{pattern_body}\b", re.IGNORECASE)
                self.compiled_rules.append((pattern, canonical, entity_type))

    def process(self, record: IncidentRawRecord) -> IncidentNormalizedRecord:
        raw_text = record.raw_narrative
        matched_spans: List[EntitySpan] = []

        for pattern, canonical, entity_type in self.compiled_rules:
            for match in pattern.finditer(raw_text):
                start, end = match.span()
                matched_spans.append(
                    EntitySpan(
                        text=raw_text[start:end],
                        canonical_form=canonical,
                        start_char=start,
                        end_char=end,
                        entity_type=entity_type
                    )
                )

        matched_spans.sort(key=lambda s: (s.start_char, -(s.end_char - s.start_char)))
        non_overlapping_spans: List[EntitySpan] = []
        last_end = -1

        for span in matched_spans:
            if span.start_char >= last_end:
                non_overlapping_spans.append(span)
                last_end = span.end_char

        return IncidentNormalizedRecord(
            log_id=record.log_id,
            timestamp=record.timestamp,
            asset_id=record.asset_id,
            asset_type=record.asset_type,
            raw_narrative=raw_text,
            spans=non_overlapping_spans
        )
