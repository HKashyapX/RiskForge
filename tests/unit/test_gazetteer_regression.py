"""
Regression test suite for SpanPreservingGazetteer.

Purpose: Lock in ALL current behavior before any engine changes.
Every test represents a verified behavior of the current implementation.
If any test fails after a change, that change has broken existing behavior.

Convention: Each test uses (narrative, expected_spans) tuples where
expected_spans is a list of dicts with keys:
  canonical_form, text, start_char, end_char, entity_type
"""
from datetime import datetime, timezone
from typing import List, Dict
import pytest

from riskforge.core.contracts import AssetType, IncidentRawRecord
from riskforge.normalization.gazetteer import SpanPreservingGazetteer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def gazetteer():
    """Shared gazetteer instance loaded from the real config."""
    return SpanPreservingGazetteer()


def _make_record(narrative: str) -> IncidentRawRecord:
    """Helper to build an IncidentRawRecord with minimal boilerplate."""
    return IncidentRawRecord(
        log_id="REG_TEST",
        timestamp=datetime.now(timezone.utc),
        asset_id="RIG_01",
        asset_type=AssetType.DRILLING_RIG,
        raw_narrative=narrative,
    )


def _assert_spans(gazetteer, narrative: str, expected: List[Dict], record_id: str = "REG"):
    """Assert exact span list from gazetteer against expected list.

    Checks:
      1. Correct number of spans
      2. Each span matches (canonical_form, text, start_char, end_char, entity_type)
      3. Offsets are valid: raw_narrative[s:e] == span.text
      4. Spans are in order and non-overlapping
    """
    rec = _make_record(narrative)
    normalized = gazetteer.process(rec)

    actual_spans = normalized.spans
    assert len(actual_spans) == len(expected), (
        f"[{record_id}] Expected {len(expected)} spans, got {len(actual_spans)}. "
        f"Got: {[(s.canonical_form, s.text) for s in actual_spans]}"
    )

    last_end = -1
    for i, (actual, exp) in enumerate(zip(actual_spans, expected)):
        assert actual.canonical_form == exp["canonical_form"], (
            f"[{record_id}] span[{i}] canonical_form: expected '{exp['canonical_form']}', "
            f"got '{actual.canonical_form}'"
        )
        assert actual.text == exp["text"], (
            f"[{record_id}] span[{i}] text: expected '{exp['text']}', got '{actual.text}'"
        )
        assert actual.start_char == exp["start_char"], (
            f"[{record_id}] span[{i}] start_char: expected {exp['start_char']}, "
            f"got {actual.start_char}"
        )
        assert actual.end_char == exp["end_char"], (
            f"[{record_id}] span[{i}] end_char: expected {exp['end_char']}, "
            f"got {actual.end_char}"
        )
        assert actual.entity_type == exp["entity_type"], (
            f"[{record_id}] span[{i}] entity_type: expected '{exp['entity_type']}', "
            f"got '{actual.entity_type}'"
        )

        # Offset invariant: raw_narrative[s:e] == span.text
        assert narrative[actual.start_char:actual.end_char] == actual.text, (
            f"[{record_id}] span[{i}] Offset invariant violated: "
            f"raw_narrative[{actual.start_char}:{actual.end_char}] = "
            f"'{narrative[actual.start_char:actual.end_char]}' != span.text = '{actual.text}'"
        )

        # Non-overlapping
        assert actual.start_char >= last_end, (
            f"[{record_id}] span[{i}] overlaps with previous span "
            f"(start={actual.start_char} < last_end={last_end})"
        )
        last_end = actual.end_char


# ===========================================================================
# SECTION 1: GOLDEN MASTER TESTS (all 8 rules)
# ===========================================================================
# Each rule tested with multiple pattern variants.
# These lock in the exact behavior of every rule in gazetteer_rules.yaml.

class TestGoldenMasterBlowoutPreventer:
    """Rule: blowout_preventer (BARRIER)
    Patterns: bop, b.o.p, blowout preventer, blow out preventer
    """

    def test_bop_lowercase(self, gazetteer):
        _assert_spans(gazetteer, "The bop was inspected.", [
            {"canonical_form": "blowout_preventer", "text": "bop",
             "start_char": 4, "end_char": 7, "entity_type": "BARRIER"},
        ], "bp_lower")

    def test_bop_uppercase(self, gazetteer):
        _assert_spans(gazetteer, "Checked the BOP before drilling.", [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 12, "end_char": 15, "entity_type": "BARRIER"},
        ], "bp_upper")

    def test_bop_mixed_case(self, gazetteer):
        _assert_spans(gazetteer, "Inspected the Bop valve.", [
            {"canonical_form": "blowout_preventer", "text": "Bop",
             "start_char": 14, "end_char": 17, "entity_type": "BARRIER"},
        ], "bp_mixed")

    def test_b_o_p_dotted(self, gazetteer):
        _assert_spans(gazetteer, "Checked the b.o.p system.", [
            {"canonical_form": "blowout_preventer", "text": "b.o.p",
             "start_char": 12, "end_char": 17, "entity_type": "BARRIER"},
        ], "bp_dotted")

    def test_b_o_p_upper_dotted(self, gazetteer):
        _assert_spans(gazetteer, "Inspected B.O.P before shift.", [
            {"canonical_form": "blowout_preventer", "text": "B.O.P",
             "start_char": 10, "end_char": 15, "entity_type": "BARRIER"},
        ], "bp_upper_dotted")

    def test_blowout_preventer_two_words(self, gazetteer):
        _assert_spans(gazetteer, "The blowout preventer was tested.", [
            {"canonical_form": "blowout_preventer", "text": "blowout preventer",
             "start_char": 4, "end_char": 21, "entity_type": "BARRIER"},
        ], "bp_two_words")

    def test_blow_out_preventer_three_words(self, gazetteer):
        _assert_spans(gazetteer, "We tested the blow out preventer.", [
            {"canonical_form": "blowout_preventer", "text": "blow out preventer",
             "start_char": 14, "end_char": 32, "entity_type": "BARRIER"},
        ], "bp_three_words")


class TestGoldenMasterChristmasTree:
    """Rule: christmas_tree (ASSET)
    Patterns: xtree, x-tree, xmas tree, christmas tree
    """

    def test_xtree(self, gazetteer):
        _assert_spans(gazetteer, "Checked the xtree assembly.", [
            {"canonical_form": "christmas_tree", "text": "xtree",
             "start_char": 12, "end_char": 17, "entity_type": "ASSET"},
        ], "ct_xtree")

    def test_x_tree_hyphen(self, gazetteer):
        _assert_spans(gazetteer, "Inspected x-tree valve.", [
            {"canonical_form": "christmas_tree", "text": "x-tree",
             "start_char": 10, "end_char": 16, "entity_type": "ASSET"},
        ], "ct_x_tree")

    def test_xmas_tree(self, gazetteer):
        _assert_spans(gazetteer, "The xmas tree was replaced.", [
            {"canonical_form": "christmas_tree", "text": "xmas tree",
             "start_char": 4, "end_char": 13, "entity_type": "ASSET"},
        ], "ct_xmas_tree")

    def test_christmas_tree(self, gazetteer):
        _assert_spans(gazetteer, "Serviced the christmas tree today.", [
            {"canonical_form": "christmas_tree", "text": "christmas tree",
             "start_char": 13, "end_char": 27, "entity_type": "ASSET"},
        ], "ct_christmas_tree")


class TestGoldenMasterSwabUnit:
    """Rule: swab_unit (ACTIVITY)
    Patterns: swab unit, swabbing unit, swabbing
    """

    def test_swab_unit(self, gazetteer):
        _assert_spans(gazetteer, "The swab unit was deployed.", [
            {"canonical_form": "swab_unit", "text": "swab unit",
             "start_char": 4, "end_char": 13, "entity_type": "ACTIVITY"},
        ], "su_swab_unit")

    def test_swabbing_unit(self, gazetteer):
        _assert_spans(gazetteer, "Completed swabbing unit operation.", [
            {"canonical_form": "swab_unit", "text": "swabbing unit",
             "start_char": 10, "end_char": 23, "entity_type": "ACTIVITY"},
        ], "su_swabbing_unit")

    def test_swabbing_standalone(self, gazetteer):
        """swabbing alone matches (single-token pattern)."""
        _assert_spans(gazetteer, "Swabbing is complete.", [
            {"canonical_form": "swab_unit", "text": "Swabbing",
             "start_char": 0, "end_char": 8, "entity_type": "ACTIVITY"},
        ], "su_swabbing_standalone")


class TestGoldenMasterMonkeyBoard:
    """Rule: monkey_board (LOCATION)
    Patterns: monkey board, m-board, mboard
    """

    def test_monkey_board(self, gazetteer):
        _assert_spans(gazetteer, "Climbed to the monkey board.", [
            {"canonical_form": "monkey_board", "text": "monkey board",
             "start_char": 15, "end_char": 27, "entity_type": "LOCATION"},
        ], "mb_monkey_board")

    def test_m_board_hyphen(self, gazetteer):
        _assert_spans(gazetteer, "Working from m-board position.", [
            {"canonical_form": "monkey_board", "text": "m-board",
             "start_char": 13, "end_char": 20, "entity_type": "LOCATION"},
        ], "mb_m_board")

    def test_mboard_joined(self, gazetteer):
        _assert_spans(gazetteer, "Secured the mboard location.", [
            {"canonical_form": "monkey_board", "text": "mboard",
             "start_char": 12, "end_char": 18, "entity_type": "LOCATION"},
        ], "mb_mboard")


class TestGoldenMasterWellControlLine:
    """Rule: well_control_line (BARRIER)
    Patterns: kill line, choke line
    """

    def test_kill_line(self, gazetteer):
        _assert_spans(gazetteer, "Opened the kill line valve.", [
            {"canonical_form": "well_control_line", "text": "kill line",
             "start_char": 11, "end_char": 20, "entity_type": "BARRIER"},
        ], "wcl_kill_line")

    def test_choke_line(self, gazetteer):
        _assert_spans(gazetteer, "Adjusted choke line pressure.", [
            {"canonical_form": "well_control_line", "text": "choke line",
             "start_char": 9, "end_char": 19, "entity_type": "BARRIER"},
        ], "wcl_choke_line")


class TestGoldenMasterPressureReliefValve:
    """Rule: pressure_relief_valve (BARRIER)
    Patterns: prv, psv, safety valve, relief valve
    """

    def test_prv(self, gazetteer):
        _assert_spans(gazetteer, "The PRV was tested.", [
            {"canonical_form": "pressure_relief_valve", "text": "PRV",
             "start_char": 4, "end_char": 7, "entity_type": "BARRIER"},
        ], "prv_prv")

    def test_psv(self, gazetteer):
        _assert_spans(gazetteer, "Inspected PSV before startup.", [
            {"canonical_form": "pressure_relief_valve", "text": "PSV",
             "start_char": 10, "end_char": 13, "entity_type": "BARRIER"},
        ], "prv_psv")

    def test_safety_valve(self, gazetteer):
        _assert_spans(gazetteer, "Replaced the safety valve today.", [
            {"canonical_form": "pressure_relief_valve", "text": "safety valve",
             "start_char": 13, "end_char": 25, "entity_type": "BARRIER"},
        ], "prv_safety_valve")

    def test_relief_valve(self, gazetteer):
        _assert_spans(gazetteer, "Checked relief valve settings.", [
            {"canonical_form": "pressure_relief_valve", "text": "relief valve",
             "start_char": 8, "end_char": 20, "entity_type": "BARRIER"},
        ], "prv_relief_valve")


class TestGoldenMasterEnergyIsolationLoto:
    """Rule: energy_isolation_loto (BARRIER)
    Patterns: loto, lock out tag out, lockout tagout
    """

    def test_loto(self, gazetteer):
        _assert_spans(gazetteer, "Applied LOTO before maintenance.", [
            {"canonical_form": "energy_isolation_loto", "text": "LOTO",
             "start_char": 8, "end_char": 12, "entity_type": "BARRIER"},
        ], "eil_loto")

    def test_lock_out_tag_out(self, gazetteer):
        _assert_spans(gazetteer, "Completed lock out tag out procedure.", [
            {"canonical_form": "energy_isolation_loto", "text": "lock out tag out",
             "start_char": 10, "end_char": 26, "entity_type": "BARRIER"},
        ], "eil_lock_out_tag_out")

    def test_lockout_tagout_joined(self, gazetteer):
        _assert_spans(gazetteer, "Verified lockout tagout compliance.", [
            {"canonical_form": "energy_isolation_loto", "text": "lockout tagout",
             "start_char": 9, "end_char": 23, "entity_type": "BARRIER"},
        ], "eil_lockout_tagout")


class TestGoldenMasterHydrogenSulfide:
    """Rule: hydrogen_sulfide (HAZARD)
    Patterns: h2s, sour gas, hydrogen sulfide
    """

    def test_h2s(self, gazetteer):
        _assert_spans(gazetteer, "Detected H2S at location.", [
            {"canonical_form": "hydrogen_sulfide", "text": "H2S",
             "start_char": 9, "end_char": 12, "entity_type": "HAZARD"},
        ], "hs_h2s")

    def test_sour_gas(self, gazetteer):
        _assert_spans(gazetteer, "Alert for sour gas detected.", [
            {"canonical_form": "hydrogen_sulfide", "text": "sour gas",
             "start_char": 10, "end_char": 18, "entity_type": "HAZARD"},
        ], "hs_sour_gas")

    def test_hydrogen_sulfide(self, gazetteer):
        _assert_spans(gazetteer, "Measured hydrogen sulfide levels.", [
            {"canonical_form": "hydrogen_sulfide", "text": "hydrogen sulfide",
             "start_char": 9, "end_char": 25, "entity_type": "HAZARD"},
        ], "hs_hydrogen_sulfide")


# ===========================================================================
# SECTION 2: FALSE-POSITIVE BOUNDARY TESTS (negative tests)
# ===========================================================================
# These verify the engine does NOT match in cases where it should not.

class TestFalsePositiveBoundaries:
    """Verify the engine does NOT produce false positives."""

    def test_no_match_joined_tokens(self, gazetteer):
        """blowoutpreventer (no space) should NOT match blowout preventer."""
        _assert_spans(gazetteer, "Inspected blowoutpreventer today.", [], "fp_joined")

    def test_no_match_prefix(self, gazetteer):
        """xbop should NOT match bop."""
        _assert_spans(gazetteer, "Found xbop in the system.", [], "fp_prefix")

    def test_no_match_suffix(self, gazetteer):
        """bopx should NOT match bop."""
        _assert_spans(gazetteer, "Found bopx in the line.", [], "fp_suffix")

    def test_no_match_word_containing_bop(self, gazetteer):
        """bopping should NOT match bop (b is word char after p)."""
        _assert_spans(gazetteer, "They were bopping along.", [], "fp_bopping")

    def test_no_match_word_containing_h2s(self, gazetteer):
        """h2so4 should NOT match h2s."""
        _assert_spans(gazetteer, "Found H2SO4 in the lab.", [], "fp_h2s_formula")

    def test_no_match_joined_swabbingunit(self, gazetteer):
        """swabbingunit (no space) should NOT match swabbing unit."""
        _assert_spans(gazetteer, "The swabbingunit was deployed.", [], "fp_swab_joined")

    def test_no_match_xmas_tree_joined(self, gazetteer):
        """xmastree (no space) should NOT match xmas tree."""
        _assert_spans(gazetteer, "Replaced xmastree today.", [], "fp_xmas_joined")

    def test_no_match_mboard_prefix(self, gazetteer):
        """amboard should NOT match mboard (a is word char before m)."""
        _assert_spans(gazetteer, "They used the amboard panel.", [], "fp_mboard_prefix")

    def test_no_match_mboard_suffix(self, gazetteer):
        """mboards should NOT match mboard (s is word char after d)."""
        _assert_spans(gazetteer, "Checked the mboards in storage.", [], "fp_mboard_suffix")

    def test_no_match_loto_suffix(self, gazetteer):
        """lotol should NOT match loto (l is word char after o)."""
        _assert_spans(gazetteer, "Applied lotol protocol.", [], "fp_loto_suffix")

    def test_no_match_lockouttagout_joined(self, gazetteer):
        """lockouttagoutmission (too long) should NOT match lockout tagout."""
        _assert_spans(gazetteer, "Did lockouttagoutmission today.", [], "fp_lockout_long")

    def test_no_match_prv_prefix(self, gazetteer):
        """sprv should NOT match prv (s is word char before p)."""
        _assert_spans(gazetteer, "Found sprv in the valve list.", [], "fp_prv_prefix")

    def test_no_match_psv_suffix(self, gazetteer):
        """psvs should NOT match psv (s is word char after v)."""
        _assert_spans(gazetteer, "Checked psvs in the system.", [], "fp_psv_suffix")


# ===========================================================================
# SECTION 3: OVERLAP RESOLUTION TESTS
# ===========================================================================
# Verify greedy longest-match-first behavior.

class TestOverlapResolution:

    def test_longer_match_wins_blowout_preventer(self, gazetteer):
        """'blowout preventer' (17 chars) beats 'bop' (3 chars) at same position."""
        _assert_spans(gazetteer, "Inspected blowout preventer assembly.", [
            {"canonical_form": "blowout_preventer", "text": "blowout preventer",
             "start_char": 10, "end_char": 27, "entity_type": "BARRIER"},
        ], "ov_bop_vs_blowout")

    def test_longer_match_wins_swabbing(self, gazetteer):
        """'swabbing unit' (13 chars) beats 'swabbing' (8 chars)."""
        _assert_spans(gazetteer, "Deployed swabbing unit operation.", [
            {"canonical_form": "swab_unit", "text": "swabbing unit",
             "start_char": 9, "end_char": 22, "entity_type": "ACTIVITY"},
        ], "ov_swabbing")

    def test_non_overlapping_coexist(self, gazetteer):
        """Non-overlapping spans from different rules coexist."""
        _assert_spans(gazetteer,
            "Checked BOP and monkey board before kill line test.",
            [
                {"canonical_form": "blowout_preventer", "text": "BOP",
                 "start_char": 8, "end_char": 11, "entity_type": "BARRIER"},
                {"canonical_form": "monkey_board", "text": "monkey board",
                 "start_char": 16, "end_char": 28, "entity_type": "LOCATION"},
                {"canonical_form": "well_control_line", "text": "kill line",
                 "start_char": 36, "end_char": 45, "entity_type": "BARRIER"},
            ], "ov_coexist")

    def test_overlapping_shorter_earlier_loses(self, gazetteer):
        """When 'bop' and 'blowout preventer' overlap, longest wins."""
        narrative = "The blowout preventer BOP was checked."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "blowout preventer",
             "start_char": 4, "end_char": 21, "entity_type": "BARRIER"},
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 22, "end_char": 25, "entity_type": "BARRIER"},
        ], "ov_shorter_earlier")

    def test_sequential_spans_non_overlapping(self, gazetteer):
        """Multiple spans that don't overlap coexist in order."""
        narrative = "Replaced PRV and tested H2S alarm."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "pressure_relief_valve", "text": "PRV",
             "start_char": 9, "end_char": 12, "entity_type": "BARRIER"},
            {"canonical_form": "hydrogen_sulfide", "text": "H2S",
             "start_char": 24, "end_char": 27, "entity_type": "HAZARD"},
        ], "ov_sequential")


# ===========================================================================
# SECTION 4: OFFSET PRESERVATION TESTS
# ===========================================================================
# Verify exact character offset invariant in edge cases.

class TestOffsetPreservation:

    def test_offsets_with_leading_whitespace(self, gazetteer):
        """Offsets are correct even with leading whitespace."""
        narrative = "   The BOP was checked."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 7, "end_char": 10, "entity_type": "BARRIER"},
        ], "off_leading_ws")

    def test_offsets_with_punctuation_before(self, gazetteer):
        """Offsets are correct when token follows punctuation."""
        narrative = "The BOP. was checked."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 4, "end_char": 7, "entity_type": "BARRIER"},
        ], "off_punct_before")

    def test_offsets_with_punctuation_after(self, gazetteer):
        """Offsets are correct when token precedes punctuation."""
        narrative = "Checked the BOP, before shift."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 12, "end_char": 15, "entity_type": "BARRIER"},
        ], "off_punct_after")

    def test_offsets_with_parentheses(self, gazetteer):
        """Offsets are correct with parenthesized tokens."""
        narrative = "Inspected (BOP) during round."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 11, "end_char": 14, "entity_type": "BARRIER"},
        ], "off_parens")

    def test_offsets_with_apostrophe(self, gazetteer):
        """BOP's: apostrophe is non-word char, \\b fires, matches BOP."""
        narrative = "The BOP's seal was intact."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 4, "end_char": 7, "entity_type": "BARRIER"},
        ], "off_apostrophe")

    def test_offsets_with_tab(self, gazetteer):
        """Offsets correct with embedded tabs."""
        narrative = "The\tBOP\twas\tchecked."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 4, "end_char": 7, "entity_type": "BARRIER"},
        ], "off_tab")

    def test_offsets_with_newline(self, gazetteer):
        """Offsets correct with embedded newlines."""
        narrative = "The\nBOP\nwas checked."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 4, "end_char": 7, "entity_type": "BARRIER"},
        ], "off_newline")

    def test_offsets_multiple_spaces_between_tokens(self, gazetteer):
        """Multiple spaces between multi-word tokens still match."""
        narrative = "Checked the blowout  preventer."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "blowout  preventer",
             "start_char": 12, "end_char": 30, "entity_type": "BARRIER"},
        ], "off_multi_space")

    def test_offsets_end_of_string(self, gazetteer):
        """Span at end of string has correct end_char == len(narrative)."""
        narrative = "Checked the BOP"
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 12, "end_char": 15, "entity_type": "BARRIER"},
        ], "off_end_of_string")
        assert len(narrative) == 15

    def test_offsets_start_of_string(self, gazetteer):
        """Span at start of string has start_char == 0."""
        narrative = "BOP was checked."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 0, "end_char": 3, "entity_type": "BARRIER"},
        ], "off_start_of_string")

    def test_offsets_unicode_before_match(self, gazetteer):
        """Unicode characters before match don't corrupt offsets."""
        narrative = "Café BOP inspection."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 5, "end_char": 8, "entity_type": "BARRIER"},
        ], "off_unicode_before")

    def test_offsets_emoji_in_narrative(self, gazetteer):
        """Emoji characters don't corrupt offsets (emoji is multi-byte but single codepoint)."""
        narrative = "⚠️ BOP alarm triggered."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 3, "end_char": 6, "entity_type": "BARRIER"},
        ], "off_emoji")


# ===========================================================================
# SECTION 5: EMPTY / NO-MATCH / ALL-MATCH EDGE CASES
# ===========================================================================

class TestEdgeCases:

    def test_empty_narrative(self, gazetteer):
        """Empty narrative produces zero spans."""
        _assert_spans(gazetteer, "", [], "edge_empty")

    def test_no_matches(self, gazetteer):
        """Narrative with no domain entities produces zero spans."""
        _assert_spans(gazetteer, "Routine inspection completed.", [], "edge_no_match")

    def test_single_character_narrative(self, gazetteer):
        """Single character narrative produces zero spans."""
        _assert_spans(gazetteer, "X", [], "edge_single_char")

    def test_all_tokens_match(self, gazetteer):
        """Narrative that is entirely a matched token."""
        _assert_spans(gazetteer, "BOP", [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 0, "end_char": 3, "entity_type": "BARRIER"},
        ], "edge_all_match")

    def test_multiple_occurrences_same_rule(self, gazetteer):
        """Same token appears twice - both should match if non-overlapping."""
        narrative = "BOP was checked, then BOP again."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 0, "end_char": 3, "entity_type": "BARRIER"},
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 22, "end_char": 25, "entity_type": "BARRIER"},
        ], "edge_two_occurrences")


# ===========================================================================
# SECTION 6: RECORD FIELDS PRESERVED
# ===========================================================================

class TestRecordPreservation:

    def test_record_fields_preserved(self, gazetteer):
        """All record fields (except raw_narrative) are preserved in output."""
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        rec = IncidentRawRecord(
            log_id="PRESERVE_001",
            timestamp=ts,
            asset_id="RIG_42",
            asset_type=AssetType.WORKOVER_RIG,
            raw_narrative="Checked the BOP.",
        )
        normalized = gazetteer.process(rec)
        assert normalized.log_id == "PRESERVE_001"
        assert normalized.timestamp == ts
        assert normalized.asset_id == "RIG_42"
        assert normalized.asset_type == AssetType.WORKOVER_RIG
        assert normalized.raw_narrative == "Checked the BOP."

    def test_raw_narrative_not_mutated(self, gazetteer):
        """The original raw_narrative string is never mutated."""
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        narrative = "Checked the BOP before shift."
        rec = IncidentRawRecord(
            log_id="PRESERVE_002",
            timestamp=ts,
            asset_id="RIG_01",
            asset_type=AssetType.DRILLING_RIG,
            raw_narrative=narrative,
        )
        original_narrative = narrative[:]
        _ = gazetteer.process(rec)
        assert narrative == original_narrative, "raw_narrative was mutated!"


# ===========================================================================
# SECTION 7: COMPLEX COMPOUND NARRATIVES
# ===========================================================================
# Realistic narratives combining multiple entities.

class TestCompoundNarratives:

    def test_multi_entity_narrative(self, gazetteer):
        """Multiple entities from different rules in one narrative."""
        narrative = "Operator on monkey board reported H2S leak near BOP. PRV activated."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "monkey_board", "text": "monkey board",
             "start_char": 12, "end_char": 24, "entity_type": "LOCATION"},
            {"canonical_form": "hydrogen_sulfide", "text": "H2S",
             "start_char": 34, "end_char": 37, "entity_type": "HAZARD"},
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 48, "end_char": 51, "entity_type": "BARRIER"},
            {"canonical_form": "pressure_relief_valve", "text": "PRV",
             "start_char": 53, "end_char": 56, "entity_type": "BARRIER"},
        ], "compound_multi")

    def test_long_narrative(self, gazetteer):
        """Long narrative with entities scattered throughout."""
        narrative = (
            "During the night shift, the derrickman observed fluid returns from "
            "the kill line. The blowout preventer was tested at 0800 hours. "
            "Maintenance crew applied LOTO before starting repair work on the "
            "xmas tree assembly. H2S levels were monitored throughout the operation."
        )
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "well_control_line", "text": "kill line",
             "start_char": 71, "end_char": 80, "entity_type": "BARRIER"},
            {"canonical_form": "blowout_preventer", "text": "blowout preventer",
             "start_char": 86, "end_char": 103, "entity_type": "BARRIER"},
            {"canonical_form": "energy_isolation_loto", "text": "LOTO",
             "start_char": 155, "end_char": 159, "entity_type": "BARRIER"},
            {"canonical_form": "christmas_tree", "text": "xmas tree",
             "start_char": 195, "end_char": 204, "entity_type": "ASSET"},
            {"canonical_form": "hydrogen_sulfide", "text": "H2S",
             "start_char": 215, "end_char": 218, "entity_type": "HAZARD"},
        ], "compound_long")


# ===========================================================================
# SECTION 8: UNICODE / WHITESPACE PREPROCESSING TESTS
# ===========================================================================
# Verify the preprocessing layer handles Unicode NFC normalization and
# whitespace collapsing while preserving exact original-text offsets.

class TestUnicodePreprocessing:
    """NFC normalization and whitespace collapsing with offset preservation."""

    def test_nfd_combining_accent_to_nfc(self, gazetteer):
        """NFD 'cafe\\u0301' (e + combining acute) normalizes to NFC 'é'.
        BOP offset must point into the original (NFD) text."""
        narrative = "cafe\u0301 BOP inspection."
        # Original positions: c=0 a=1 f=2 e=3 \u0301=4 ' '=5 B=6 O=7 P=8
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 6, "end_char": 9, "entity_type": "BARRIER"},
        ], "uni_nfd_combining")

    def test_nfc_precomposed_unchanged(self, gazetteer):
        """Precomposed 'café' (NFC) produces same offsets as before."""
        narrative = "café BOP inspection."
        # Original positions: c=0 a=1 f=2 é=3 ' '=4 B=5 O=6 P=7
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 5, "end_char": 8, "entity_type": "BARRIER"},
        ], "uni_nfc_precomposed")

    def test_double_tab_collapse(self, gazetteer):
        """Two consecutive tabs collapse to one space; offsets correct."""
        narrative = "The\t\tBOP\t\twas checked."
        # Original: T=0 h=1 e=2 \t=3 \t=4 B=5 O=6 P=7
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 5, "end_char": 8, "entity_type": "BARRIER"},
        ], "uni_double_tab")

    def test_crlf_collapse(self, gazetteer):
        """\\r\\n collapses to single space; offsets point to original \\r."""
        narrative = "The\r\nBOP\r\nwas checked."
        # Original: T=0 h=1 e=2 \r=3 \n=4 B=5 O=6 P=7
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 5, "end_char": 8, "entity_type": "BARRIER"},
        ], "uni_crlf")

    def test_mixed_whitespace_collapse(self, gazetteer):
        """Space+tab+newline collapse to single space; offsets correct."""
        narrative = "The \t\n BOP was checked."
        # Original: T=0 h=1 e=2 ' '=3 \t=4 \n=5 ' '=6 B=7 O=8 P=9
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "blowout_preventer", "text": "BOP",
             "start_char": 7, "end_char": 10, "entity_type": "BARRIER"},
        ], "uni_mixed_ws")

    def test_nfd_before_multiword_match(self, gazetteer):
        """NFD combining accent before a multi-word match; offset invariant holds."""
        narrative = "caf\u0065\u0301 blowout  preventer."
        # NFD: 'e' + combining acute. After NFC: 'café blowout preventer.'
        # 'blowout' starts at original index 6 in NFD text
        rec = _make_record(narrative)
        normalized = gazetteer.process(rec)
        assert len(normalized.spans) == 1
        span = normalized.spans[0]
        # Offset invariant: raw_narrative[start:end] == span.text
        assert narrative[span.start_char:span.end_char] == span.text
        assert span.canonical_form == "blowout_preventer"


# ===========================================================================
# SECTION 9: HYPHEN SEPARATOR TESTS
# ===========================================================================
# Verify that multi-token patterns match both space-separated and
# hyphen-separated variants (e.g. "kill line" and "kill-line").
# This covers the oil & gas domain convention of hyphenating compound terms.

class TestHyphenSeparator:
    """Multi-token patterns match hyphen-separated variants."""

    def test_kill_line_hyphenated(self, gazetteer):
        """'kill-line' matches the kill line rule."""
        _assert_spans(gazetteer, "Opened the kill-line valve.", [
            {"canonical_form": "well_control_line", "text": "kill-line",
             "start_char": 11, "end_char": 20, "entity_type": "BARRIER"},
        ], "hy_kill_line")

    def test_blow_out_preventer_hyphenated(self, gazetteer):
        """'blow-out preventer' matches the blowout preventer rule."""
        _assert_spans(gazetteer, "Inspected blow-out preventer assembly.", [
            {"canonical_form": "blowout_preventer", "text": "blow-out preventer",
             "start_char": 10, "end_char": 28, "entity_type": "BARRIER"},
        ], "hy_blow_out")

    def test_blow_out_preventer_double_hyphen(self, gazetteer):
        """'blow-out-preventer' (all hyphens) matches."""
        _assert_spans(gazetteer, "Checked blow-out-preventer status.", [
            {"canonical_form": "blowout_preventer", "text": "blow-out-preventer",
             "start_char": 8, "end_char": 26, "entity_type": "BARRIER"},
        ], "hy_blow_out_double")

    def test_lock_out_tag_out_hyphenated(self, gazetteer):
        """'lock-out-tag-out' matches the LOTO rule."""
        _assert_spans(gazetteer, "Applied lock-out-tag-out procedure.", [
            {"canonical_form": "energy_isolation_loto", "text": "lock-out-tag-out",
             "start_char": 8, "end_char": 24, "entity_type": "BARRIER"},
        ], "hy_loto_hyphenated")

    def test_lockout_tagout_hyphenated(self, gazetteer):
        """'lockout-tagout' (partial hyphen) matches."""
        _assert_spans(gazetteer, "Verified lockout-tagout compliance.", [
            {"canonical_form": "energy_isolation_loto", "text": "lockout-tagout",
             "start_char": 9, "end_char": 23, "entity_type": "BARRIER"},
        ], "hy_lockout_tagout")

    def test_lock_out_mixed_separator(self, gazetteer):
        """'lock-out tag out' (mixed hyphen/space) matches."""
        _assert_spans(gazetteer, "Applied lock-out tag out procedure.", [
            {"canonical_form": "energy_isolation_loto", "text": "lock-out tag out",
             "start_char": 8, "end_char": 24, "entity_type": "BARRIER"},
        ], "hy_loto_mixed")

    def test_choke_line_hyphenated(self, gazetteer):
        """'choke-line' matches the choke line rule."""
        _assert_spans(gazetteer, "Adjusted choke-line pressure.", [
            {"canonical_form": "well_control_line", "text": "choke-line",
             "start_char": 9, "end_char": 19, "entity_type": "BARRIER"},
        ], "hy_choke_line")

    def test_space_variants_still_work(self, gazetteer):
        """Original space-separated patterns still match."""
        _assert_spans(gazetteer, "Opened the kill line valve.", [
            {"canonical_form": "well_control_line", "text": "kill line",
             "start_char": 11, "end_char": 20, "entity_type": "BARRIER"},
        ], "hy_space_still_works")

    def test_no_false_positives_hyphen_joined(self, gazetteer):
        """'killline' (no separator) does NOT match."""
        _assert_spans(gazetteer, "Checked killline status.", [], "hy_no_fp_joined")

    def test_no_false_positives_prefix_suffix(self, gazetteer):
        """'xkill-linex' does NOT match kill line."""
        _assert_spans(gazetteer, "Found xkill-linex in system.", [], "hy_no_fp_affix")

    def test_hyphen_offsets_correct(self, gazetteer):
        """Offsets are correct for hyphenated matches."""
        narrative = "Opened the kill-line valve."
        _assert_spans(gazetteer, narrative, [
            {"canonical_form": "well_control_line", "text": "kill-line",
             "start_char": 11, "end_char": 20, "entity_type": "BARRIER"},
        ], "hy_offsets")
        assert narrative[11:20] == "kill-line"
