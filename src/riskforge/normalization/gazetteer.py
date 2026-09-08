import hashlib
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple
import yaml

from riskforge.core.contracts import EntitySpan, IncidentNormalizedRecord, IncidentRawRecord
from riskforge.core.exceptions import NormalizationError

_CONFIG_CACHE: dict[str, list] = {}


def clear_config_cache() -> None:
    """Clear the compiled-rules cache. For testing only."""
    _CONFIG_CACHE.clear()


def _build_nfc_to_orig(text: str, nfc: str) -> List[int]:
    """Map each NFC character position to its corresponding original text position.

    When NFC merges base + combining characters (e.g. 'e' + U+0301 -> 'é'),
    the merged NFC character maps to the original base character position, and
    the combining character's position is skipped in the mapping.
    """
    mapping: List[int] = []
    nfc_len_so_far = 0
    for i in range(len(text)):
        new_nfc_len = len(unicodedata.normalize('NFC', text[: i + 1]))
        while nfc_len_so_far < new_nfc_len:
            mapping.append(i)
            nfc_len_so_far += 1
    return mapping


def _preprocess_with_mapping(text: str) -> Tuple[str, List[int]]:
    """Normalize Unicode (NFC) and whitespace while building original-index mapping.

    Returns (preprocessed_text, mapping) where
        mapping[preprocessed_idx] = original_idx
    so that span offsets can be mapped back to the original raw narrative.

    The original text is never mutated.
    """
    nfc = unicodedata.normalize('NFC', text)
    nfc_to_orig = _build_nfc_to_orig(text, nfc)

    result: List[str] = []
    mapping: List[int] = []
    prev_was_space = False
    for nfc_idx, ch in enumerate(nfc):
        if ch.isspace():
            if not prev_was_space:
                result.append(' ')
                mapping.append(nfc_to_orig[nfc_idx])
                prev_was_space = True
        else:
            result.append(ch)
            mapping.append(nfc_to_orig[nfc_idx])
            prev_was_space = False

    return ''.join(result), mapping

class SpanPreservingGazetteer:
    def __init__(self, config_path: Optional[Path] = None) -> None:
        if config_path is None:
            config_path = Path(__file__).resolve().parents[3] / "config" / "gazetteer_rules.yaml"

        if not config_path.exists():
            raise NormalizationError(f"Gazetteer configuration not found at {config_path}")

        cache_key = hashlib.md5(config_path.read_bytes()).hexdigest()

        if cache_key in _CONFIG_CACHE:
            self.compiled_rules = _CONFIG_CACHE[cache_key]
            return

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        compiled: List[Tuple[re.Pattern, str, str]] = []
        for entry in data.get("rules", []):
            canonical = entry["canonical"]
            entity_type = entry["entity_type"]
            for pattern_str in entry.get("patterns", []):
                tokens = pattern_str.strip().split()
                if not tokens:
                    continue
                escaped_tokens = [re.escape(token) for token in tokens]
                pattern_body = r"[\s\-]+".join(escaped_tokens)
                pattern = re.compile(
                    rf"(?<!\w){pattern_body}(?!\w)", re.IGNORECASE
                )
                compiled.append((pattern, canonical, entity_type))

        self.compiled_rules = compiled
        _CONFIG_CACHE[cache_key] = compiled

    def process(self, record: IncidentRawRecord) -> IncidentNormalizedRecord:
        raw_text = record.raw_narrative
        clean_text, mapping = _preprocess_with_mapping(raw_text)
        matched_spans: List[EntitySpan] = []

        for pattern, canonical, entity_type in self.compiled_rules:
            for match in pattern.finditer(clean_text):
                start, end = match.span()
                orig_start = mapping[start]
                orig_end = mapping[end - 1] + 1
                matched_spans.append(
                    EntitySpan(
                        text=raw_text[orig_start:orig_end],
                        canonical_form=canonical,
                        start_char=orig_start,
                        end_char=orig_end,
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
