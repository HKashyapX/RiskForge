import pandas as pd
from pathlib import Path

INPUT = Path(r"C:\data\processed\iogp_pilot_review.csv")
OUTPUT = Path(r"C:\data\processed\iogp_manual_audit.csv")

print(f"Reading: {INPUT}")

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT}")

df = pd.read_csv(INPUT)

required = [
    "incident_id",
    "predicted_rule",
    "title",
    "narrative",
]

missing = [column for column in required if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Fields to be completed during manual review.
df["decision"] = ""
df["review_reason"] = ""
df["reviewer"] = ""
df["reviewed"] = False

# Keep only the columns needed for the audit.
df = df[
    [
        "incident_id",
        "predicted_rule",
        "title",
        "narrative",
        "decision",
        "review_reason",
        "reviewer",
        "reviewed",
    ]
]

# Stable ordering makes the audit easier to review.
df = (
    df.sort_values(
        ["predicted_rule", "incident_id"],
        kind="stable",
    )
    .reset_index(drop=True)
)

df.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig",
)

print()
print("=" * 70)
print("IOGP MANUAL AUDIT FILE CREATED")
print("=" * 70)
print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")
print(f"Rows  : {len(df):,}")
print()

print("Rows by rule:")
print(
    df.groupby("predicted_rule")
    .size()
    .sort_index()
    .to_string()
)

print()
print("Decision values:")
print("  KEEP   = clearly represents the predicted IOGP rule")
print("  REMOVE = clearly does not represent the predicted IOGP rule")
print("  REVIEW = ambiguous; needs domain judgment")
print()
print("Do not change:")
print("  incident_id")
print("  predicted_rule")
print("  title")
print("  narrative")
print()
print("The audit columns are:")
print("  decision")
print("  review_reason")
print("  reviewer")
print("  reviewed")
