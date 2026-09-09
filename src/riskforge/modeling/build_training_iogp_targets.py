from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

DATA_DIR = Path(r"C:\data\processed")

CLEAN_FILE = DATA_DIR / "clean_incidents.parquet"
CONFIDENCE_FILE = DATA_DIR / "iogp_confidence_targets.csv"
AUDIT_FILE = DATA_DIR / "iogp_manual_audit_completed.csv"

OUTPUT_FILE = DATA_DIR / "iogp_training_targets.csv"


# ============================================================================
# EXACT IOGP ORDER
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


POSITIVE = 1
NEGATIVE = 0
UNKNOWN = -1


# ============================================================================
# WEAK-LABEL POLICY
#
# Rules other than Line of Fire:
#   confidence 3 -> positive
#   confidence 2 -> positive
#
# Line of Fire:
#   ONLY manually audited KEEP -> positive
#   manually audited REMOVE -> negative
#   everything else -> unknown
#
# We intentionally do NOT train on the 39k automatically detected
# Line-of-Fire examples.
# ============================================================================

WEAK_CONFIDENCE_RULES = [
    "bypassing_safety_controls",
    "confined_space",
    "driving",
    "energy_isolation",
    "hot_work",
    "safe_mechanical_lifting",
    "toxic_gas",
    "work_at_height",
]


def main() -> None:

    print("=" * 80)
    print("FINAL IOGP TRAINING TARGET BUILDER")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # LOAD CONFIDENCE DATA
    # ------------------------------------------------------------------------

    print(f"\nReading confidence targets:")
    print(f"  {CONFIDENCE_FILE}")

    if not CONFIDENCE_FILE.exists():
        raise FileNotFoundError(CONFIDENCE_FILE)

    df = pd.read_csv(CONFIDENCE_FILE)

    print(f"Rows: {len(df):,}")

    required = {
        "incident_id",
        *RULES,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df["incident_id"] = (
        df["incident_id"]
        .astype(str)
        .str.strip()
    )

    if df["incident_id"].duplicated().any():
        raise ValueError(
            "Duplicate incident IDs found."
        )

    # ------------------------------------------------------------------------
    # INITIALIZE ALL RULES AS UNKNOWN
    # ------------------------------------------------------------------------

    for rule in RULES:
        df[rule] = UNKNOWN

    # ------------------------------------------------------------------------
    # APPLY HIGH-CONFIDENCE WEAK LABELS
    # ------------------------------------------------------------------------

    print("\nApplying high-confidence weak labels...")

    for rule in WEAK_CONFIDENCE_RULES:

        confidence_column = f"{rule}_confidence"

        if confidence_column not in df.columns:
            raise ValueError(
                f"Missing confidence column: "
                f"{confidence_column}"
            )

        confidence = pd.to_numeric(
            df[confidence_column],
            errors="coerce",
        ).fillna(0)

        df.loc[
            confidence >= 2,
            rule,
        ] = POSITIVE

        print(
            f"{rule:30s} "
            f"weak positives="
            f"{int((confidence >= 2).sum()):6d}"
        )

    # ------------------------------------------------------------------------
    # LINE OF FIRE IS SPECIAL
    # ------------------------------------------------------------------------

    df["line_of_fire"] = UNKNOWN

    print(
        "\nLine of Fire weak labels: DISABLED"
    )

    print(
        "Line of Fire will use manual audit only."
    )

    # ------------------------------------------------------------------------
    # LOAD MANUAL AUDIT
    # ------------------------------------------------------------------------

    print(f"\nReading manual audit:")
    print(f"  {AUDIT_FILE}")

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
            f"Invalid audit decisions: "
            f"{invalid_decisions}"
        )

    invalid_rules = sorted(
        set(audit["predicted_rule"]) - set(RULES)
    )

    if invalid_rules:
        raise ValueError(
            f"Invalid audit rules: {invalid_rules}"
        )

    # ------------------------------------------------------------------------
    # VALIDATE AUDIT IDS
    # ------------------------------------------------------------------------

    clean_ids = set(df["incident_id"])

    missing_ids = sorted(
        set(audit["incident_id"]) - clean_ids
    )

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids)} audit IDs are not "
            f"present in confidence dataset."
        )

    # ------------------------------------------------------------------------
    # BUILD AUDIT LOOKUP
    # ------------------------------------------------------------------------

    audit_lookup = {}

    for row in audit.itertuples(index=False):

        incident_id = str(row.incident_id)
        rule = str(row.predicted_rule)
        decision = str(row.decision).upper()

        key = (incident_id, rule)

        if key in audit_lookup:

            previous = audit_lookup[key]

            if previous != decision:

                raise ValueError(
                    f"Conflicting audit decisions for "
                    f"{incident_id} / {rule}: "
                    f"{previous} vs {decision}"
                )

        else:
            audit_lookup[key] = decision

    audited_ids = set(
        audit["incident_id"]
    )

    print(
        f"Audit rows: {len(audit):,}"
    )

    print(
        f"Audited incidents: {len(audited_ids):,}"
    )

    # ------------------------------------------------------------------------
    # APPLY MANUAL AUDIT
    #
    # IMPORTANT:
    # For every audited incident, every rule must be explicitly represented.
    #
    # KEEP   -> 1
    # REMOVE -> 0
    # REVIEW -> -1
    # Missing rule -> -1
    # ------------------------------------------------------------------------

    print(
        "\nApplying manual audit as authoritative labels..."
    )

    for idx in df.index:

        incident_id = df.at[
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

                df.at[
                    idx,
                    rule,
                ] = UNKNOWN

                continue

            decision = audit_lookup[key]

            if decision == "KEEP":

                df.at[
                    idx,
                    rule,
                ] = POSITIVE

            elif decision == "REMOVE":

                df.at[
                    idx,
                    rule,
                ] = NEGATIVE

            elif decision == "REVIEW":

                df.at[
                    idx,
                    rule,
                ] = UNKNOWN

    # ------------------------------------------------------------------------
    # LABEL SOURCE
    # ------------------------------------------------------------------------

    df["label_source"] = "weak"

    df.loc[
        df["incident_id"].isin(audited_ids),
        "label_source",
    ] = "manual_audit"

    # ------------------------------------------------------------------------
    # SAMPLE WEIGHT
    #
    # These weights will later be used by the training pipeline.
    #
    # Manual audit:
    #   1.00
    #
    # Weak confidence 3:
    #   0.75
    #
    # Weak confidence 2:
    #   0.40
    #
    # Unknown:
    #   0.00
    # ------------------------------------------------------------------------

    for rule in RULES:

        df[f"{rule}_weight"] = 0.0

    # Weak labels
    for rule in WEAK_CONFIDENCE_RULES:

        confidence_column = (
            f"{rule}_confidence"
        )

        confidence = pd.to_numeric(
            df[confidence_column],
            errors="coerce",
        ).fillna(0)

        df.loc[
            confidence == 3,
            f"{rule}_weight",
        ] = 0.75

        df.loc[
            confidence == 2,
            f"{rule}_weight",
        ] = 0.40

    # Line of Fire has no weak-label weight.
    df["line_of_fire_weight"] = 0.0

    # ------------------------------------------------------------------------
    # MANUAL AUDIT ALWAYS OVERRIDES SAMPLE WEIGHT
    # ------------------------------------------------------------------------

    for idx in df.index:

        incident_id = df.at[
            idx,
            "incident_id",
        ]

        if incident_id not in audited_ids:
            continue

        for rule in RULES:

            decision = audit_lookup.get(
                (incident_id, rule)
            )

            if decision in {
                "KEEP",
                "REMOVE",
            }:

                df.at[
                    idx,
                    f"{rule}_weight",
                ] = 1.00

            else:

                df.at[
                    idx,
                    f"{rule}_weight",
                ] = 0.00

    # ------------------------------------------------------------------------
    # TRAINING ELIGIBILITY
    # ------------------------------------------------------------------------

    # An incident can participate in training if at least one rule has a
    # known target.

    known_mask = (
        df[RULES]
        .ne(UNKNOWN)
        .any(axis=1)
    )

    df["training_eligible"] = known_mask

    # ------------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL TRAINING TARGET STATISTICS")
    print("=" * 80)

    print(
        f"Total incidents: "
        f"{len(df):,}"
    )

    print(
        f"Training-eligible incidents: "
        f"{int(known_mask.sum()):,}"
    )

    print(
        f"Fully unknown incidents: "
        f"{int((~known_mask).sum()):,}"
    )

    print("\nRule statistics:")

    for rule in RULES:

        positive = int(
            (df[rule] == POSITIVE).sum()
        )

        negative = int(
            (df[rule] == NEGATIVE).sum()
        )

        unknown = int(
            (df[rule] == UNKNOWN).sum()
        )

        weighted = int(
            (df[f"{rule}_weight"] > 0).sum()
        )

        print(
            f"{rule:30s} "
            f"positive={positive:6d} "
            f"negative={negative:6d} "
            f"unknown={unknown:6d} "
            f"weighted={weighted:6d}"
        )

    # ------------------------------------------------------------------------
    # MULTI-LABEL DISTRIBUTION
    # ------------------------------------------------------------------------

    positive_count = (
        df[RULES]
        .eq(POSITIVE)
        .sum(axis=1)
    )

    print("\nMulti-label distribution:")

    print(
        f"zero positives:        "
        f"{int((positive_count == 0).sum()):,}"
    )

    print(
        f"exactly one positive: "
        f"{int((positive_count == 1).sum()):,}"
    )

    print(
        f"two+ positives:       "
        f"{int((positive_count >= 2).sum()):,}"
    )

    print(
        f"maximum positives:    "
        f"{int(positive_count.max()):,}"
    )

    # ------------------------------------------------------------------------
    # SAVE ONLY TRAINING-RELEVANT COLUMNS
    # ------------------------------------------------------------------------

    base_columns = [
        "incident_id",
    ]

    for column in [
        "title_normalized",
        "narrative_sanitized",
        "text_normalized",
        "event_date",
        "hazard_class",
        "actual_severity",
        "sif_potential",
    ]:

        if column in df.columns:
            base_columns.append(column)

    columns = base_columns.copy()

    columns.extend(RULES)

    columns.extend(
        f"{rule}_weight"
        for rule in RULES
    )

    columns.extend(
        [
            "label_source",
            "training_eligible",
        ]
    )

    final = df[columns].copy()

    final.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # FINAL VALIDATION
    # ------------------------------------------------------------------------

    if len(final) != len(df):
        raise RuntimeError(
            "Output row count changed."
        )

    if final["incident_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate incident IDs found."
        )

    for rule in RULES:

        values = set(
            final[rule]
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
                f"Invalid labels for {rule}: "
                f"{values}"
            )

        weights = set(
            final[f"{rule}_weight"]
            .astype(float)
            .unique()
        )

        if not weights.issubset(
            {
                0.0,
                0.40,
                0.75,
                1.00,
            }
        ):
            raise RuntimeError(
                f"Invalid weights for {rule}: "
                f"{weights}"
            )

    # ------------------------------------------------------------------------
    # IMPORTANT SANITY CHECK
    # ------------------------------------------------------------------------

    # There must be NO automatically generated Line-of-Fire positives.
    # Every positive must correspond to a manual KEEP.

    line_positive_ids = set(
        final.loc[
            final["line_of_fire"] == POSITIVE,
            "incident_id",
        ]
    )

    audited_line_keep_ids = set(
        audit.loc[
            (audit["predicted_rule"] == "line_of_fire")
            & (audit["decision"] == "KEEP"),
            "incident_id",
        ]
        .astype(str)
    )

    if line_positive_ids != audited_line_keep_ids:

        extra = sorted(
            line_positive_ids - audited_line_keep_ids
        )

        missing = sorted(
            audited_line_keep_ids - line_positive_ids
        )

        raise RuntimeError(
            "Line-of-Fire audit integrity check failed.\n"
            f"Extra positives: {extra[:10]}\n"
            f"Missing positives: {missing[:10]}"
        )

    print()
    print(
        "Line-of-Fire integrity check: PASSED"
    )

    print(
        f"\nOutput written to:\n"
        f"  {OUTPUT_FILE}"
    )

    print()
    print("=" * 80)
    print("FINAL TARGET BUILD VALIDATION PASSED")
    print("=" * 80)


if __name__ == "__main__":
    main()