from __future__ import annotations

import numpy as np
import pandas as pd

from riskforge.modeling.config import modeling_data_dir


# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = modeling_data_dir()

INPUT_FILE = DATA_DIR / "iogp_training_targets.csv"

TRAIN_FILE = DATA_DIR / "iogp_train.csv"
CALIBRATION_FILE = DATA_DIR / "iogp_calibration.csv"
TEST_FILE = DATA_DIR / "iogp_test.csv"

SEED = 42

TRAIN_FRACTION = 0.80
CALIBRATION_FRACTION = 0.10
TEST_FRACTION = 0.10


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


UNKNOWN = -1


# ============================================================================
# HELPERS
# ============================================================================

def label_signature(row: pd.Series) -> str:
    """
    Create a deterministic signature from the known positive labels.

    Unknown labels are ignored because they do not represent a negative.
    """

    positives = [
        rule
        for rule in RULES
        if int(row[rule]) == 1
    ]

    if not positives:
        return "NONE"

    return "+".join(positives)


def check_ids(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:

    train_ids = set(train["incident_id"])
    calibration_ids = set(calibration["incident_id"])
    test_ids = set(test["incident_id"])

    train_cal = train_ids & calibration_ids
    train_test = train_ids & test_ids
    cal_test = calibration_ids & test_ids

    if train_cal or train_test or cal_test:
        raise RuntimeError(
            "DATA LEAKAGE DETECTED\n"
            f"train/calibration overlap: {len(train_cal)}\n"
            f"train/test overlap: {len(train_test)}\n"
            f"calibration/test overlap: {len(cal_test)}"
        )


def print_distribution(
    name: str,
    df: pd.DataFrame,
) -> None:

    print()
    print(f"{name}: {len(df):,} incidents")

    for rule in RULES:

        positive = int(
            (df[rule] == 1).sum()
        )

        negative = int(
            (df[rule] == 0).sum()
        )

        unknown = int(
            (df[rule] == UNKNOWN).sum()
        )

        print(
            f"  {rule:30s} "
            f"positive={positive:6d} "
            f"negative={negative:6d} "
            f"unknown={unknown:6d}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:

    print("=" * 80)
    print("IOGP DATASET SPLITTER")
    print("=" * 80)

    # ------------------------------------------------------------------------
    # VALIDATE FRACTIONS
    # ------------------------------------------------------------------------

    total_fraction = (
        TRAIN_FRACTION
        + CALIBRATION_FRACTION
        + TEST_FRACTION
    )

    if not np.isclose(total_fraction, 1.0):
        raise ValueError(
            "Train/calibration/test fractions must sum to 1.0"
        )

    # ------------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------------

    print()
    print("Reading:")
    print(f"  {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    df = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    # ------------------------------------------------------------------------
    # REQUIRED COLUMNS
    # ------------------------------------------------------------------------

    required = {
        "incident_id",
        *RULES,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing)}"
        )

    # ------------------------------------------------------------------------
    # ID VALIDATION
    # ------------------------------------------------------------------------

    df["incident_id"] = (
        df["incident_id"]
        .astype(str)
        .str.strip()
    )

    if df["incident_id"].duplicated().any():

        duplicates = int(
            df["incident_id"].duplicated().sum()
        )

        raise ValueError(
            f"Duplicate incident IDs: {duplicates}"
        )

    # ------------------------------------------------------------------------
    # LABEL VALIDATION
    # ------------------------------------------------------------------------

    for rule in RULES:

        df[rule] = pd.to_numeric(
            df[rule],
            errors="raise",
        ).astype("int8")

        invalid = ~df[rule].isin(
            [-1, 0, 1]
        )

        if invalid.any():

            values = sorted(
                df.loc[invalid, rule]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Invalid labels in {rule}: {values}"
            )

    # ------------------------------------------------------------------------
    # TRAINING ELIGIBILITY
    # ------------------------------------------------------------------------

    known_mask = (
        df[RULES]
        .ne(UNKNOWN)
        .any(axis=1)
    )

    eligible = df.loc[
        known_mask
    ].copy()

    print(
        f"Training-eligible incidents: "
        f"{len(eligible):,}"
    )

    # ------------------------------------------------------------------------
    # LABEL SIGNATURE
    # ------------------------------------------------------------------------

    eligible["_signature"] = eligible.apply(
        label_signature,
        axis=1,
    )

    print()
    print("Label signatures:")

    signature_counts = (
        eligible["_signature"]
        .value_counts()
    )

    for signature, count in signature_counts.items():

        print(
            f"  {signature:55s} "
            f"{int(count):6d}"
        )

    # ------------------------------------------------------------------------
    # STRATIFIED RANDOM SPLIT
    #
    # We stratify by the complete positive-label signature so that the
    # approximate multi-label structure is preserved.
    # ------------------------------------------------------------------------

    rng = np.random.default_rng(SEED)

    train_parts = []
    calibration_parts = []
    test_parts = []

    for signature, group in eligible.groupby(
        "_signature",
        sort=True,
    ):

        # IMPORTANT:
        # copy=True ensures the NumPy array is writable.
        indices = group.index.to_numpy(copy=True)

        rng.shuffle(indices)

        n = len(indices)

        n_train = int(
            round(n * TRAIN_FRACTION)
        )

        n_calibration = int(
            round(n * CALIBRATION_FRACTION)
        )

        # ------------------------------------------------------------
        # Ensure test receives at least one sample when possible.
        # ------------------------------------------------------------

        if n >= 3:

            n_train = max(
                1,
                min(
                    n_train,
                    n - 2,
                ),
            )

            n_calibration = max(
                1,
                min(
                    n_calibration,
                    n - n_train - 1,
                ),
            )

        elif n == 2:

            n_train = 1
            n_calibration = 0

        else:

            n_train = 1
            n_calibration = 0

        train_idx = indices[
            :n_train
        ]

        calibration_idx = indices[
            n_train:
            n_train + n_calibration
        ]

        test_idx = indices[
            n_train + n_calibration:
        ]

        train_parts.append(
            eligible.loc[train_idx]
        )

        if len(calibration_idx):

            calibration_parts.append(
                eligible.loc[calibration_idx]
            )

        if len(test_idx):

            test_parts.append(
                eligible.loc[test_idx]
            )

    # ------------------------------------------------------------------------
    # COMBINE
    # ------------------------------------------------------------------------

    train = pd.concat(
        train_parts,
        ignore_index=True,
    )

    calibration = (
        pd.concat(
            calibration_parts,
            ignore_index=True,
        )
        if calibration_parts
        else pd.DataFrame(
            columns=eligible.columns
        )
    )

    test = (
        pd.concat(
            test_parts,
            ignore_index=True,
        )
        if test_parts
        else pd.DataFrame(
            columns=eligible.columns
        )
    )

    # ------------------------------------------------------------------------
    # SHUFFLE EACH SPLIT
    # ------------------------------------------------------------------------

    train = train.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    calibration = calibration.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    test = test.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    # ------------------------------------------------------------------------
    # REMOVE INTERNAL COLUMN
    # ------------------------------------------------------------------------

    for split in (
        train,
        calibration,
        test,
    ):

        if "_signature" in split.columns:

            split.drop(
                columns=["_signature"],
                inplace=True,
            )

    # ------------------------------------------------------------------------
    # LEAKAGE CHECK
    # ------------------------------------------------------------------------

    check_ids(
        train,
        calibration,
        test,
    )

    # ------------------------------------------------------------------------
    # CHECK TOTALS
    # ------------------------------------------------------------------------

    if (
        len(train)
        + len(calibration)
        + len(test)
        != len(eligible)
    ):

        raise RuntimeError(
            "Split counts do not add up."
        )

    # ------------------------------------------------------------------------
    # CHECK MANUAL AUDIT DISTRIBUTION
    # ------------------------------------------------------------------------

    if "label_source" in eligible.columns:

        for name, split in [
            ("TRAIN", train),
            ("CALIBRATION", calibration),
            ("TEST", test),
        ]:

            audited = int(
                (
                    split["label_source"]
                    == "manual_audit"
                ).sum()
            )

            print(
                f"{name} manual-audit incidents: "
                f"{audited:,}"
            )

    # ------------------------------------------------------------------------
    # PRINT SPLIT SIZES
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SPLIT SIZES")
    print("=" * 80)

    total = len(eligible)

    print(
        f"TRAIN:       {len(train):7,} "
        f"({len(train) / total * 100:6.2f}%)"
    )

    print(
        f"CALIBRATION: {len(calibration):7,} "
        f"({len(calibration) / total * 100:6.2f}%)"
    )

    print(
        f"TEST:        {len(test):7,} "
        f"({len(test) / total * 100:6.2f}%)"
    )

    print(
        f"TOTAL:       {total:7,}"
    )

    # ------------------------------------------------------------------------
    # RULE DISTRIBUTIONS
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("RULE DISTRIBUTIONS")
    print("=" * 80)

    print_distribution(
        "TRAIN",
        train,
    )

    print_distribution(
        "CALIBRATION",
        calibration,
    )

    print_distribution(
        "TEST",
        test,
    )

    # ------------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------------

    train.to_csv(
        TRAIN_FILE,
        index=False,
    )

    calibration.to_csv(
        CALIBRATION_FILE,
        index=False,
    )

    test.to_csv(
        TEST_FILE,
        index=False,
    )

    # ------------------------------------------------------------------------
    # RELOAD CHECK
    # ------------------------------------------------------------------------

    train_check = pd.read_csv(
        TRAIN_FILE,
        low_memory=False,
    )

    calibration_check = pd.read_csv(
        CALIBRATION_FILE,
        low_memory=False,
    )

    test_check = pd.read_csv(
        TEST_FILE,
        low_memory=False,
    )

    check_ids(
        train_check,
        calibration_check,
        test_check,
    )

    # ------------------------------------------------------------------------
    # FINAL OUTPUT
    # ------------------------------------------------------------------------

    print()
    print("=" * 80)
    print("SPLIT VALIDATION PASSED")
    print("=" * 80)

    print()
    print("Train:")
    print(f"  {TRAIN_FILE}")

    print()
    print("Calibration:")
    print(f"  {CALIBRATION_FILE}")

    print()
    print("Test:")
    print(f"  {TEST_FILE}")


if __name__ == "__main__":
    main()
