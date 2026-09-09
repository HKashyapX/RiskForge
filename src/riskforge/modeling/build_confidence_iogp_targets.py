from __future__ import annotations

from pathlib import Path
import re

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = Path(r"C:\data\processed")

CLEAN_FILE = DATA_DIR / "clean_incidents.parquet"
AUDIT_FILE = DATA_DIR / "iogp_manual_audit_completed.csv"

OUTPUT_FILE = DATA_DIR / "iogp_confidence_targets.csv"


# ============================================================================
# EXACT IOGP LIFE-SAVING RULE ORDER
# ============================================================================

RULES = [
    "bypassing_safety_controls",
    "confined_space",
    "driving",
    "energy_isolation",
    "hot_work",
    "line_of_fire",
    "safe_mechanical_lifting",
    "toxic_gas",
    "work_at_height",
]


# ============================================================================
# LABEL STATES
#
#  1 = positive
#  0 = audited negative
# -1 = unknown / abstain
# ============================================================================

POSITIVE = 1
NEGATIVE = 0
UNKNOWN = -1


# ============================================================================
# CONFIDENCE
#
# 3 = very high confidence
# 2 = high confidence
# 1 = weak / not used for training
# 0 = abstain
# ============================================================================


def norm(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).lower()


def has(text: str, patterns: list[str]) -> bool:
    return any(
        re.search(pattern, text, re.I)
        for pattern in patterns
    )


def score(
    text: str,
    patterns: list[tuple[int, str]],
) -> int:
    """
    Return highest matching confidence.
    """

    best = 0

    for confidence, pattern in patterns:
        if re.search(pattern, text, re.I):
            best = max(best, confidence)

    return best


# ============================================================================
# RULE 1 - BYPASSING SAFETY CONTROLS
# ============================================================================

def confidence_bypassing(text: str) -> int:

    if not text:
        return 0

    patterns = [
        (3, r"\bbypassed\b.{0,80}\binterlock\b"),
        (3, r"\bdefeated\b.{0,80}\binterlock\b"),
        (3, r"\bdisabled\b.{0,80}\binterlock\b"),
        (3, r"\bdefeated\b.{0,80}\bsafety device\b"),
        (3, r"\bdisabled\b.{0,80}\bsafety device\b"),

        (2, r"\bbypass(?:ed|ing)?\b.{0,80}\bsafety control\b"),
        (2, r"\bremoved\b.{0,60}\bsafety guard\b"),
        (2, r"\bremoved\b.{0,60}\bmachine guard\b"),
    ]

    return score(text, patterns)


# ============================================================================
# RULE 2 - CONFINED SPACE
# ============================================================================

def confidence_confined(text: str) -> int:

    if not text:
        return 0

    explicit = [
        (3, r"\bpermit[- ]required confined space\b"),
        (3, r"\bconfined-space entry\b"),
        (3, r"\bconfined space entry\b"),
        (3, r"\bconfined space\b"),

        (2, r"\bconfined-space\b"),
    ]

    return score(text, explicit)


# ============================================================================
# RULE 3 - DRIVING
# ============================================================================

def confidence_driving(text: str) -> int:

    if not text:
        return 0

    # Pedestrian struck by vehicle = not automatically Driving.
    pedestrian = [
        r"\bwas struck by a vehicle\b",
        r"\bwas struck by a truck\b",
        r"\bwas hit by a vehicle\b",
        r"\bwas hit by a truck\b",
        r"\bbacked over\b.{0,100}\bworker\b",
        r"\bbacked over\b.{0,100}\bemployee\b",
    ]

    if has(text, pedestrian):
        return 0

    patterns = [
        (3, r"\bwas driving\b"),
        (3, r"\bwere driving\b"),
        (3, r"\bwhile driving\b"),
        (3, r"\bdriving the vehicle\b"),
        (3, r"\bdriving the truck\b"),

        (2, r"\boperating a truck\b"),
        (2, r"\boperating the truck\b"),
        (2, r"\boperating a forklift\b"),
        (2, r"\boperating the forklift\b"),
        (2, r"\boperating a grader\b"),
        (2, r"\boperating a scraper\b"),
        (2, r"\boperating a roller\b"),
        (2, r"\boperating a paver\b"),
        (2, r"\boperating a tractor\b"),
        (2, r"\boperating a dump truck\b"),
        (2, r"\boperating a haul truck\b"),
        (2, r"\boperating a utility vehicle\b"),
    ]

    return score(text, patterns)


# ============================================================================
# RULE 4 - ENERGY ISOLATION
# ============================================================================

def confidence_energy(text: str) -> int:

    if not text:
        return 0

    patterns = [
        (3, r"\bfailed to de[- ]energize\b"),
        (3, r"\bnot de[- ]energized\b"),
        (3, r"\bnot deenergized\b"),
        (3, r"\bfailed to isolate\b"),
        (3, r"\bwithout isolating\b"),
        (3, r"\bwithout isolation\b"),
        (3, r"\bunexpectedly energized\b"),
        (3, r"\bunintentionally energized\b"),
        (3, r"\blive wire\b"),
        (3, r"\blive conductor\b"),
        (3, r"\belectrocuted\b"),

        (2, r"\benergized\b.{0,80}\bcontact\b"),
        (2, r"\bcontacted\b.{0,80}\benergized\b"),
        (2, r"\btouched\b.{0,80}\benergized\b"),
        (2, r"\bcontact with energized\b"),
        (2, r"\blive electrical\b"),
        (2, r"\belectrical shock\b"),
        (2, r"\belectric shock\b"),
    ]

    return score(text, patterns)


# ============================================================================
# RULE 5 - HOT WORK
# ============================================================================

def confidence_hot_work(text: str) -> int:

    if not text:
        return 0

    patterns = [
        (3, r"\bwas welding\b"),
        (3, r"\bwere welding\b"),
        (3, r"\bwhile welding\b"),
        (3, r"\busing an acetylene torch\b"),
        (3, r"\busing a cutting torch\b"),
        (3, r"\busing an oxygen[- ]acetylene\b"),
        (3, r"\bcutting with a torch\b"),
        (3, r"\btorch cutting\b"),
        (3, r"\bflame cutting\b"),
        (3, r"\bplasma cutting\b"),
        (3, r"\bthermal cutting\b"),

        (2, r"\bwelder\b"),
        (2, r"\bbrazing\b"),
        (2, r"\bsoldering\b"),
        (2, r"\bhot work\b"),
    ]

    # Don't classify a weld failure by itself as hot work.
    if has(
        text,
        [
            r"\bweld failed\b",
            r"\bweld broke\b",
            r"\bweld failure\b",
            r"\bwelded joint failed\b",
        ],
    ):
        active = [
            r"\bwas welding\b",
            r"\bwere welding\b",
            r"\bwhile welding\b",
            r"\busing\b.{0,60}\btorch\b",
            r"\busing\b.{0,60}\bacetylene\b",
        ]

        if not has(text, active):
            return 0

    return score(text, patterns)


# ============================================================================
# RULE 6 - LINE OF FIRE
#
# This is intentionally much stricter.
# ============================================================================

def confidence_line_of_fire(text: str) -> int:

    if not text:
        return 0

    # ------------------------------------------------------------------------
    # Confidence 3:
    # explicit mechanism + physical exposure
    # ------------------------------------------------------------------------

    very_high = [
        r"\bcaught between\b",
        r"\btrapped between\b",
        r"\bpinned between\b",
        r"\bcrushed between\b",

        r"\bcaught in\b.{0,100}\bmachine\b",
        r"\bcaught in\b.{0,100}\bequipment\b",

        r"\btrapped under\b.{0,100}\b(?:machine|equipment|vehicle|truck|load|beam|pipe|debris)\b",
        r"\bpinned under\b.{0,100}\b(?:machine|equipment|vehicle|truck|load|beam|pipe|debris)\b",

        r"\bbacked over\b",
        r"\brun over\b",

        r"\bfalling load\b.{0,100}\b(?:struck|hit|injur|worker|employee)\b",
        r"\bfalling object\b.{0,100}\b(?:struck|hit|injur|worker|employee)\b",

        r"\bobject fell\b.{0,100}\b(?:worker|employee|person)\b",
        r"\bload fell\b.{0,100}\b(?:worker|employee|person)\b",
        r"\bbeam fell\b.{0,100}\b(?:worker|employee|person)\b",
        r"\bpipe fell\b.{0,100}\b(?:worker|employee|person)\b",

        r"\bboom\b.{0,120}\b(?:trapped|pinned|crushed)\b",
        r"\bbucket\b.{0,120}\b(?:trapped|pinned|crushed)\b",
    ]

    if has(text, very_high):
        return 3

    # ------------------------------------------------------------------------
    # Vehicle strike
    # ------------------------------------------------------------------------

    vehicle = [
        r"\bstruck by a vehicle\b",
        r"\bstruck by the vehicle\b",
        r"\bstruck by a truck\b",
        r"\bstruck by the truck\b",
        r"\bhit by a vehicle\b",
        r"\bhit by a truck\b",
    ]

    if has(text, vehicle):
        return 3

    # ------------------------------------------------------------------------
    # Specific object/equipment strike
    # ------------------------------------------------------------------------

    specific = [
        r"\bstruck by\b.{0,100}\bboom\b",
        r"\bstruck by\b.{0,100}\bbucket\b",
        r"\bstruck by\b.{0,100}\bpipe\b",
        r"\bstruck by\b.{0,100}\bbeam\b",
        r"\bstruck by\b.{0,100}\bload\b",
        r"\bstruck by\b.{0,100}\bobject\b",
        r"\bstruck by\b.{0,100}\bequipment\b",
        r"\bstruck by\b.{0,100}\bmachine\b",
        r"\bstruck by\b.{0,100}\btool\b",
        r"\bstruck by\b.{0,100}\bdisc\b",
        r"\bstruck by\b.{0,100}\brock\b",
        r"\bstruck by\b.{0,100}\bdebris\b",

        r"\bhit by\b.{0,100}\bboom\b",
        r"\bhit by\b.{0,100}\bbucket\b",
        r"\bhit by\b.{0,100}\bpipe\b",
        r"\bhit by\b.{0,100}\bbeam\b",
        r"\bhit by\b.{0,100}\bload\b",
        r"\bhit by\b.{0,100}\bobject\b",
        r"\bhit by\b.{0,100}\bequipment\b",
        r"\bhit by\b.{0,100}\bmachine\b",
        r"\bhit by\b.{0,100}\btool\b",
        r"\bhit by\b.{0,100}\bdisc\b",
        r"\bhit by\b.{0,100}\brock\b",
        r"\bhit by\b.{0,100}\bdebris\b",
    ]

    if has(text, specific):
        return 3

    # ------------------------------------------------------------------------
    # Pinch/crush only when the object/mechanism is explicitly identified.
    # ------------------------------------------------------------------------

    pinch = [
        r"\bpinched\b.{0,100}\bfinger\b.{0,100}\b(?:machine|equipment|object|door|pipe|metal)\b",
        r"\bpinched\b.{0,100}\bhand\b.{0,100}\b(?:machine|equipment|object|door|pipe|metal)\b",
        r"\bpinched\b.{0,100}\bfoot\b.{0,100}\b(?:machine|equipment|object|door|pipe|metal)\b",
        r"\bcrushed\b.{0,100}\bfinger\b.{0,100}\b(?:machine|equipment|object|door|pipe|metal)\b",
        r"\bcrushed\b.{0,100}\bhand\b.{0,100}\b(?:machine|equipment|object|door|pipe|metal)\b",
        r"\bcrushed\b.{0,100}\bleg\b.{0,100}\b(?:machine|equipment|object|door|pipe|metal)\b",
    ]

    if has(text, pinch):
        return 3

    return 0


# ============================================================================
# RULE 7 - SAFE MECHANICAL LIFTING
# ============================================================================

def confidence_lifting(text: str) -> int:

    if not text:
        return 0

    personnel = [
        r"\blifted an employee\b",
        r"\blifted a worker\b",
        r"\blifted a person\b",
        r"\blifted personnel\b",
        r"\bhoisted an employee\b",
        r"\bhoisted a worker\b",
        r"\bhoisted a person\b",
        r"\bhoisted personnel\b",
        r"\bemployee\b.{0,60}\bwas lifted\b",
        r"\bemployee\b.{0,60}\bwas hoisted\b",
        r"\bworker\b.{0,60}\bwas lifted\b",
        r"\bworker\b.{0,60}\bwas hoisted\b",
        r"\bpersonnel lifting\b",
        r"\blifting personnel\b",
    ]

    if has(text, personnel):
        return 0

    equipment = [
        r"\bcrane\b",
        r"\bhoist\b",
        r"\bwinch\b",
        r"\bpulley\b",
        r"\bchain hoist\b",
        r"\bcome[- ]along\b",
    ]

    action = [
        r"\blift(?:ed|ing)?\b",
        r"\bhoist(?:ed|ing)?\b",
        r"\braised\b",
        r"\blowered\b",
        r"\brigging\b",
    ]

    load = [
        r"\bload\b",
        r"\bbeam\b",
        r"\bpipe\b",
        r"\bcompressor\b",
        r"\bplate\b",
        r"\bequipment\b",
        r"\bmaterial\b",
        r"\bsteel\b",
        r"\bskid\b",
        r"\bshore jack\b",
        r"\bmachinery\b",
    ]

    if (
        has(text, equipment)
        and has(text, action)
        and has(text, load)
    ):
        return 3

    explicit = [
        r"\bcrane lift\b",
        r"\bcrane lifted\b",
        r"\blifting a load\b",
        r"\blifting equipment\b",
        r"\blifting a beam\b",
        r"\blifting a compressor\b",
        r"\blifting pipe\b",
        r"\blifting steel\b",
        r"\bhoisting a load\b",
    ]

    if has(text, explicit):
        return 3

    return 0


# ============================================================================
# RULE 8 - TOXIC GAS
# ============================================================================

def confidence_toxic_gas(text: str) -> int:

    if not text:
        return 0

    patterns = [
        (3, r"\bh2s\b"),
        (3, r"\bhydrogen sulfide\b"),
        (3, r"\bammonia\b"),
        (3, r"\btoxic gas\b"),
        (3, r"\bpoisonous gas\b"),
        (3, r"\bgas exposure\b"),
        (3, r"\bgas inhalation\b"),
        (3, r"\bcarbon monoxide\b"),
        (3, r"\bco exposure\b"),
        (3, r"\btoxic fumes\b"),
    ]

    return score(text, patterns)


# ============================================================================
# RULE 9 - WORK AT HEIGHT
# ============================================================================

def confidence_height(text: str) -> int:

    if not text:
        return 0

    fall = [
        r"\bfell\b",
        r"\bfall\b",
        r"\bfalling\b",
        r"\bejected\b",
        r"\bslipped\b",
        r"\bplunged\b",
    ]

    if not has(text, fall):
        return 0

    explicit_height = [
        r"\bladder\b",
        r"\broof\b",
        r"\bscaffold\b",
        r"\bscaffolding\b",
        r"\baerial lift\b",
        r"\bboom lift\b",
        r"\bman lift\b",
        r"\bmanlift\b",
        r"\bscissor lift\b",
        r"\bwork platform\b",
        r"\belevated platform\b",
        r"\btower\b",
        r"\bcatwalk\b",
        r"\bfloor opening\b",
        r"\binspection pit\b",
        r"\bsteel structure\b",
    ]

    if has(text, explicit_height):
        return 3

    distance = [
        r"\b\d+\s*(?:feet|ft|foot)\b.{0,80}\babove\b",
        r"\b\d+\s*(?:feet|ft|foot)\b.{0,80}\bhigh\b",
        r"\bfrom\b.{0,40}\b\d+\s*(?:feet|ft|foot)\b",
    ]

    if has(text, distance):
        return 2

    return 0


CONFIDENCE_FUNCTIONS = {
    "bypassing_safety_controls": confidence_bypassing,
    "confined_space": confidence_confined,
    "driving": confidence_driving,
    "energy_isolation": confidence_energy,
    "hot_work": confidence_hot_work,
    "line_of_fire": confidence_line_of_fire,
    "safe_mechanical_lifting": confidence_lifting,
    "toxic_gas": confidence_toxic_gas,
    "work_at_height": confidence_height,
}


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 78)
    print("CONFIDENCE-SCORED IOGP TARGET BUILDER")
    print("=" * 78)

    # ------------------------------------------------------------------------
    # LOAD CLEAN CORPUS
    # ------------------------------------------------------------------------

    print(f"\nReading clean corpus:\n  {CLEAN_FILE}")

    if not CLEAN_FILE.exists():
        raise FileNotFoundError(CLEAN_FILE)

    clean = pd.read_parquet(CLEAN_FILE)

    print(f"Clean incidents: {len(clean):,}")

    required = {
        "incident_id",
        "narrative_sanitized",
        "title_normalized",
        "text_normalized",
    }

    missing = required - set(clean.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    clean["incident_id"] = (
        clean["incident_id"]
        .astype(str)
        .str.strip()
    )

    if clean["incident_id"].duplicated().any():
        raise ValueError(
            "Duplicate incident IDs found."
        )

    clean["_text"] = (
        clean["narrative_sanitized"]
        .fillna("")
        .astype(str)
        + " "
        + clean["title_normalized"]
        .fillna("")
        .astype(str)
    ).map(norm)

    # ------------------------------------------------------------------------
    # GENERATE CONFIDENCE SCORES
    # ------------------------------------------------------------------------

    print("\nGenerating confidence scores...")

    for rule in RULES:

        print(f"  {rule}")

        clean[f"{rule}_confidence"] = [
            CONFIDENCE_FUNCTIONS[rule](value)
            for value in clean["_text"]
        ]

        clean[rule] = pd.array(
            [
                POSITIVE if confidence > 0 else UNKNOWN
                for confidence in clean[
                    f"{rule}_confidence"
                ]
            ],
            dtype="Int8",
        )

    # ------------------------------------------------------------------------
    # LOAD MANUAL AUDIT
    # ------------------------------------------------------------------------

    print(f"\nReading manual audit:\n  {AUDIT_FILE}")

    if not AUDIT_FILE.exists():
        raise FileNotFoundError(AUDIT_FILE)

    audit = pd.read_csv(AUDIT_FILE)

    required_audit = {
        "incident_id",
        "predicted_rule",
        "decision",
    }

    missing = required_audit - set(audit.columns)

    if missing:
        raise ValueError(
            f"Missing audit columns: {sorted(missing)}"
        )

    audit["incident_id"] = (
        audit["incident_id"]
        .astype(str)
        .str.strip()
    )

    audit["predicted_rule"] = (
        audit["predicted_rule"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    audit["decision"] = (
        audit["decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    valid_decisions = {
        "KEEP",
        "REMOVE",
        "REVIEW",
    }

    invalid_decisions = sorted(
        set(audit["decision"]) - valid_decisions
    )

    if invalid_decisions:
        raise ValueError(
            f"Invalid decisions: {invalid_decisions}"
        )

    invalid_rules = sorted(
        set(audit["predicted_rule"]) - set(RULES)
    )

    if invalid_rules:
        raise ValueError(
            f"Invalid rules: {invalid_rules}"
        )

    clean_ids = set(clean["incident_id"])

    missing_audit = sorted(
        set(audit["incident_id"]) - clean_ids
    )

    if missing_audit:
        raise ValueError(
            f"{len(missing_audit)} audit IDs are missing "
            "from clean corpus."
        )

    audited_ids = set(
        audit["incident_id"]
    )

    print(f"Audit rows: {len(audit):,}")
    print(
        f"Unique audited incidents: "
        f"{len(audited_ids):,}"
    )

    # ------------------------------------------------------------------------
    # AUDIT LOOKUP
    # ------------------------------------------------------------------------

    audit_lookup: dict[tuple[str, str], str] = {}

    for row in audit.itertuples(index=False):

        key = (
            str(row.incident_id),
            str(row.predicted_rule),
        )

        decision = str(row.decision).upper()

        if key in audit_lookup:

            previous = audit_lookup[key]

            if previous == "REVIEW":
                continue

            if decision == "REVIEW":
                audit_lookup[key] = "REVIEW"
                continue

            if previous != decision:
                raise ValueError(
                    f"Conflicting decisions for {key}: "
                    f"{previous} vs {decision}"
                )

        else:
            audit_lookup[key] = decision

    # ------------------------------------------------------------------------
    # APPLY AUDIT
    #
    # For audited incidents, every rule must be explicitly represented.
    # Missing rule entries become UNKNOWN.
    # ------------------------------------------------------------------------

    print("\nApplying authoritative manual audit...")

    for idx in clean.index:

        incident_id = clean.at[
            idx,
            "incident_id",
        ]

        if incident_id not in audited_ids:
            continue

        for rule in RULES:

            key = (
                incident_id,
                rule,
            )

            if key not in audit_lookup:

                clean.at[
                    idx,
                    rule,
                ] = UNKNOWN

                clean.at[
                    idx,
                    f"{rule}_confidence",
                ] = 0

                continue

            decision = audit_lookup[key]

            if decision == "KEEP":

                clean.at[
                    idx,
                    rule,
                ] = POSITIVE

                clean.at[
                    idx,
                    f"{rule}_confidence",
                ] = 3

            elif decision == "REMOVE":

                clean.at[
                    idx,
                    rule,
                ] = NEGATIVE

                clean.at[
                    idx,
                    f"{rule}_confidence",
                ] = 3

            elif decision == "REVIEW":

                clean.at[
                    idx,
                    rule,
                ] = UNKNOWN

                clean.at[
                    idx,
                    f"{rule}_confidence",
                ] = 0

    # ------------------------------------------------------------------------
    # PROVENANCE
    # ------------------------------------------------------------------------

    clean["label_source"] = "weak"
    clean["label_status"] = "abstain"

    clean.loc[
        clean["incident_id"].isin(audited_ids),
        "label_source",
    ] = "manual_audit"

    clean.loc[
        clean["incident_id"].isin(audited_ids),
        "label_status",
    ] = "audited"

    # ------------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("CONFIDENCE TARGET RESULTS")
    print("=" * 78)

    print(
        f"Total incidents:   {len(clean):,}"
    )

    print(
        f"Audited incidents: {len(audited_ids):,}"
    )

    print("\nRule distribution:")

    for rule in RULES:

        positive = int(
            (clean[rule] == POSITIVE).sum()
        )

        negative = int(
            (clean[rule] == NEGATIVE).sum()
        )

        unknown = int(
            (clean[rule] == UNKNOWN).sum()
        )

        confidence_3 = int(
            (clean[f"{rule}_confidence"] == 3).sum()
        )

        confidence_2 = int(
            (clean[f"{rule}_confidence"] == 2).sum()
        )

        rate = (
            positive / len(clean) * 100
        )

        print(
            f"{rule:30s} "
            f"positive={positive:6d} "
            f"negative={negative:5d} "
            f"unknown={unknown:6d} "
            f"conf3={confidence_3:6d} "
            f"conf2={confidence_2:6d} "
            f"rate={rate:6.2f}%"
        )

    # ------------------------------------------------------------------------
    # WEAK POSITIVE DISTRIBUTION
    # ------------------------------------------------------------------------

    weak_mask = clean[
        "label_source"
    ].eq("weak")

    print("\nWeak positive confidence:")

    for rule in RULES:

        conf3 = int(
            (
                weak_mask
                & clean[f"{rule}_confidence"].eq(3)
            ).sum()
        )

        conf2 = int(
            (
                weak_mask
                & clean[f"{rule}_confidence"].eq(2)
            ).sum()
        )

        conf1 = int(
            (
                weak_mask
                & clean[f"{rule}_confidence"].eq(1)
            ).sum()
        )

        print(
            f"{rule:30s} "
            f"conf3={conf3:6d} "
            f"conf2={conf2:6d} "
            f"conf1={conf1:6d}"
        )

    # ------------------------------------------------------------------------
    # MULTI-LABEL DISTRIBUTION
    # ------------------------------------------------------------------------

    positive_counts = (
        clean[RULES]
        .eq(POSITIVE)
        .sum(axis=1)
    )

    print("\nMulti-label positive distribution:")

    print(
        f"zero positive:        "
        f"{int((positive_counts == 0).sum()):,}"
    )

    print(
        f"exactly one positive: "
        f"{int((positive_counts == 1).sum()):,}"
    )

    print(
        f"two+ positives:       "
        f"{int((positive_counts >= 2).sum()):,}"
    )

    print(
        f"maximum positives:    "
        f"{int(positive_counts.max()):,}"
    )

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    base = [
        "incident_id",
        "title_normalized",
        "narrative_sanitized",
        "text_normalized",
    ]

    optional = [
        "event_date",
        "employer",
        "state",
        "naics",
        "hazard_class",
        "actual_severity",
        "actual_severity_confidence",
        "sif_potential",
        "sif_confidence",
        "pipeline_version",
    ]

    columns = base.copy()

    for column in optional:

        if column in clean.columns:
            columns.append(column)

    columns.extend(RULES)

    columns.extend(
        f"{rule}_confidence"
        for rule in RULES
    )

    columns.extend(
        [
            "label_source",
            "label_status",
        ]
    )

    final = clean[columns].copy()

    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    if len(final) != 122285:
        raise RuntimeError(
            f"Expected 122,285 rows, got {len(final)}"
        )

    if final["incident_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate incident IDs in output."
        )

    for rule in RULES:

        values = set(
            final[rule]
            .dropna()
            .astype(int)
            .unique()
        )

        if not values.issubset(
            {
                POSITIVE,
                NEGATIVE,
                UNKNOWN,
            }
        ):
            raise RuntimeError(
                f"Invalid values in {rule}: {values}"
            )

        confidence_values = set(
            final[f"{rule}_confidence"]
            .dropna()
            .astype(int)
            .unique()
        )

        if not confidence_values.issubset(
            {0, 1, 2, 3}
        ):
            raise RuntimeError(
                f"Invalid confidence values in {rule}: "
                f"{confidence_values}"
            )

    print()
    print(
        f"Output written to:\n  {OUTPUT_FILE}"
    )

    print("\nValidation passed.")
    print("=" * 78)


if __name__ == "__main__":
    main()