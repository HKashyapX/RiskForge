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
            self.compiled_rules, self._first_token_index = _CONFIG_CACHE[cache_key]
            return

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Decompose every configured surface form into lowercase word tokens.
        # A pattern token may contain a literal hyphen (e.g. "x-tree"); hyphens
        # inside a pattern token must match literally, while hyphens BETWEEN
        # tokens act as flexible separators ("bop ram" matches "bop ram" and
        # "bop-ram"), exactly like the previous "[\s\-]+"-joined regexes with
        # word-boundary lookarounds.  Matching then walks the narrative
        # token-by-token via a first-token index — O(text tokens) rather than
        # O(patterns x characters) — which keeps latency inside the <5ms SLA
        # as the vocabulary grows.
        # Decompose every configured surface form into lowercase word tokens.
        # A pattern token may contain literal hyphens or dots (e.g. "x-tree",
        # "b.o.p"); those characters become flexible separators, matching the
        # previous "[\s\-]+"-joined regexes with word-boundary lookarounds
        # ("bop ram", "bop-ram", "bop - ram" were all equivalent before).
        # Matching then walks the narrative token-by-token via a first-token
        # index — O(text tokens) rather than O(patterns x characters) — which
        # keeps latency inside the <5ms SLA as the vocabulary grows.
        matchers: List[Tuple[Tuple[str, ...], str, str]] = []
        for entry in data.get("rules", []):
            canonical = entry["canonical"]
            entity_type = entry["entity_type"]
            for pattern_str in entry.get("patterns", []):
                subtokens: List[str] = []
                for tok in pattern_str.strip().split():
                    parts = tok.lower().replace(".", "-").split("-")
                    if any(part == "" for part in parts):
                        raise NormalizationError(
                            f"Invalid gazetteer pattern {pattern_str!r}: separator at token edge"
                        )
                    if any(not (ch.isalnum() or ch == "_") for ch in tok.replace("-", "").replace(".", "")):
                        raise NormalizationError(
                            f"Invalid gazetteer pattern {pattern_str!r}: unsupported characters"
                        )
                    subtokens.extend(parts)
                matchers.append((tuple(subtokens), canonical, entity_type))

        first_token_index: dict = {}
        for idx, (subtokens, _, _) in enumerate(matchers):
            first_token_index.setdefault(subtokens[0], []).append(idx)

        self.compiled_rules = matchers
        self._first_token_index = first_token_index
        _CONFIG_CACHE[cache_key] = (matchers, first_token_index)

    def process(self, record: IncidentRawRecord) -> IncidentNormalizedRecord:
        raw_text = record.raw_narrative
        clean_text, mapping = _preprocess_with_mapping(raw_text)
        matched_spans: List[EntitySpan] = []

        token_spans = [
            (m.start(), m.end(), m.group().lower())
            for m in re.finditer(r"\w+", clean_text)
        ]

        for i, (start, _tok_end, tok) in enumerate(token_spans):
            for matcher_idx in self._first_token_index.get(tok, ()):
                subtokens, canonical, entity_type = self.compiled_rules[matcher_idx]
                n_subs = len(subtokens)
                if i + n_subs > len(token_spans):
                    continue
                matched = True
                for k in range(1, n_subs):
                    if token_spans[i + k][2] != subtokens[k]:
                        matched = False
                        break
                    gap = clean_text[token_spans[i + k - 1][1]:token_spans[i + k][0]]
                    if any(ch not in " -." for ch in gap):
                        matched = False
                        break
                if not matched:
                    continue
                end = token_spans[i + n_subs - 1][1]
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
