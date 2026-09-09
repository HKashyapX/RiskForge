# OSHA Sanitization and Labeling Audit

## Purpose

This directory contains the audit trail for the OSHA incident-data sanitization, normalization, labeling, deduplication, and dataset-generation pipeline.

The audit files are designed to make the processed dataset reproducible and reviewable without exposing sensitive source information.

## Source datasets

The pipeline used three source datasets:

1. `January2015toNovember2025.*`

   * Primary OSHA incident corpus.
   * Contains incident metadata, injury outcomes, event/source classifications, and incident narratives.

2. `osha.xlsx`

   * Historical OSHA narrative/enrichment dataset.
   * Used for additional narrative, keyword, and event context.

3. `tagged1000.xlsx`

   * 1,000 tagged reference incidents.
   * Used as the gold/reference dataset for hazard taxonomy mapping and supervised labeling.

All 1,000 tagged reference IDs were matched to records in `osha.xlsx`.

## Processing stages

The pipeline performs the following major transformations:

### 1. Schema detection

Source-specific column names are mapped to a canonical incident representation.

The pipeline does not assume that the three source files use identical column names or structures.

### 2. Record validation

Invalid, empty, or unusable records are removed.

Records that cannot be parsed safely are excluded rather than silently converted into potentially incorrect data.

### 3. Duplicate removal

Duplicate detection is performed in two stages:

* Exact duplicate detection.
* Conservative near-duplicate detection.

Final duplicate-removal counts:

* Exact duplicates removed: 26
* Near duplicates removed: 3
* Total removed: 29

Final clean corpus:

* 122,285 unique incident records

### 4. Cross-source enrichment

Matching incident IDs are used to enrich the main OSHA corpus with information from `osha.xlsx` and the `tagged1000.xlsx` reference set.

Source provenance remains available so that inferred information can be distinguished from source facts.

### 5. PII sanitization

Obvious personally identifying or unnecessarily identifying information is removed or generalized.

Examples include:

* Person names
* Telephone numbers
* Email addresses
* Personal identifiers
* Medical record identifiers
* Exact street addresses
* Building/unit identifiers
* Equipment serial numbers
* Vehicle/license identifiers
* Organization names where unnecessarily identifying

Safety-relevant information is intentionally preserved.

Examples of retained information include:

* Worker role
* Job activity
* Equipment type
* Material involved
* Hazard
* Exposure
* Worksite type
* Sequence of events
* Injury mechanism
* Barrier/control information

### 6. Text normalization

Text is normalized for downstream modeling while preserving the original narrative separately.

The intent is to improve consistency without destroying important incident details.

### 7. Terminology normalization

Common injury, hazard, equipment, and activity terminology is standardized.

For example, different expressions describing the same equipment or hazard can be mapped to a common canonical term.

Source wording should not be treated as changed OSHA fact; normalized terminology is a derived feature.

### 8. Severity labeling

Actual incident outcome is derived separately from SIF potential.

Examples of actual outcome information include:

* Fatality
* Hospitalization
* Amputation
* Loss of eye
* Other injury outcome

Actual severity must not be used as a direct substitute for SIF potential.

### 9. Hazard taxonomy mapping

Hazard labels from `tagged1000.xlsx`, particularly the `Tagged2` taxonomy, are mapped into canonical hazard classes.

Source OSHA event descriptions remain separate from the normalized hazard label.

### 10. Weak-label generation

Where gold/reference labels are unavailable, conservative rules are used to generate weak labels for the broader OSHA corpus.

Possible derived labels include:

* Hazard class
* SIF potential
* SIF mechanism
* Activity class
* Barrier/control failure

Weak labels must be treated as inferred information rather than confirmed source facts.

### 11. Label provenance

Every label should retain information about its source or confidence.

Recommended distinction:

* `gold` — supplied by the tagged reference dataset
* `weak` — automatically inferred or mapped
* `unknown`/`unlabeled` — insufficient evidence

### 12. Dataset splitting

The labeled reference records are divided into:

* Training: 799
* Validation: 100
* Test: 101
* Combined evaluation: 201

Evaluation records should remain isolated from training data and should not be duplicated through exact or near-identical narratives.

## Final processed datasets

The processed directory contains:

```text
processed/
├── clean_incidents.parquet
├── clean_incidents.jsonl
├── sif_training.jsonl
├── sif_validation.jsonl
├── sif_test.jsonl
├── sif_eval.jsonl
└── label_report.csv
```

### `clean_incidents.parquet`

This is the master analytical dataset.

It is intended for:

* Data analysis
* SQL/DuckDB
* Pandas
* Polars
* Spark
* Machine-learning preprocessing

### `clean_incidents.jsonl`

This is the JSONL representation of the cleaned incident corpus.

Each line contains one incident record.

### `sif_training.jsonl`

Training records for SIF/hazard modeling.

### `sif_validation.jsonl`

Validation records used for model selection and tuning.

### `sif_test.jsonl`

Held-out test records for final model evaluation.

### `sif_eval.jsonl`

Combined evaluation set containing validation and test records.

### `label_report.csv`

Summary of label distributions and labeling outcomes.

## Audit files

The audit directory contains:

```text
audit/
├── transformation_log.jsonl
├── duplicate_log.csv
├── pii_redaction_log.csv
├── validation_report.json
└── README.md
```

### `transformation_log.jsonl`

Records the major processing stages in order, including:

* Schema mapping
* Validation
* Deduplication
* Cross-source enrichment
* PII sanitization
* Text normalization
* Taxonomy mapping
* Severity labeling
* Weak-label generation
* Dataset splitting
* Parquet export
* Final validation

### `duplicate_log.csv`

Documents exact and near-duplicate removal.

Current totals:

```text
Exact duplicates:       26
Near duplicates:         3
Total removed:          29
```

### `pii_redaction_log.csv`

Documents the classes of PII/generalized identifiers handled by the sanitizer.

The log records the sanitization rule without reproducing sensitive source information.

### `validation_report.json`

Contains the final machine-readable validation summary, including:

* Dataset counts
* Split counts
* Duplicate counts
* Validation checks
* Output-file status
* Data-governance principles

## Canonical record principles

The final representation should maintain a strict distinction between three types of information.

### Source facts

Information directly present in the OSHA source.

Examples:

```text
Incident date
OSHA event
Nature of injury
Body part
Hospitalization
Amputation
Loss of eye
Narrative
```

### Normalized features

Information standardized from source facts.

Examples:

```text
Normalized equipment
Normalized activity
Normalized hazard terminology
Normalized worksite type
```

### Inferred labels

Information produced by rules, mappings, or modeling.

Examples:

```text
Hazard class
SIF potential
SIF mechanism
Barrier failure
Activity class
```

An inferred label must never be presented as though it were an original OSHA field.

## SIF modeling principle

Actual severity and SIF potential are separate concepts.

For example, an incident can have:

```text
Actual severity: nonfatal
SIF potential: high
```

Therefore:

```text
fatality ≠ automatic SIF label
hospitalization ≠ automatic SIF label
amputation ≠ automatic SIF label
```

SIF potential should be based on the hazard mechanism and potential for serious/fatal consequence, not merely the observed injury outcome.

## Reproducibility

The main processing script is:

```text
scripts/sanitize_and_label.py
```

The recommended execution sequence is:

```text
raw source files
        ↓
sanitize_and_label.py
        ↓
clean_incidents.jsonl
        ↓
clean_incidents.parquet
        ↓
training / validation / test JSONL
```

The transformation log and validation report should be regenerated whenever the pipeline logic changes.

## Final validation requirements

Before using the dataset for production training, verify:

1. Every JSONL line parses as valid JSON.
2. Parquet row count matches the intended cleaned corpus.
3. Incident IDs are unique where required.
4. Train/validation/test leakage has been checked.
5. No obvious PII remains in sanitized text.
6. Gold labels and weak labels remain distinguishable.
7. Actual severity and SIF potential remain separate.
8. Canonical hazard labels belong to the approved taxonomy.
9. Source facts are not overwritten by inferred labels.
10. The transformation and validation logs are stored with the dataset version.

## Dataset versioning

When the sanitization rules, taxonomy, weak-label rules, or source files change, create a new dataset version.

Recommended format:

```text
osha_sif_v1.0
osha_sif_v1.1
osha_sif_v2.0
```

Do not silently overwrite a previously released training dataset without updating the audit records.

## Current dataset status

```text
Source schema mapping:       completed
Record validation:           completed
Duplicate removal:           completed
PII sanitization:            completed
Text normalization:          completed
Terminology normalization:   completed
Severity labeling:           completed
Hazard mapping:              completed
Weak labeling:               completed
Dataset splitting:           completed
JSONL generation:            completed
Parquet export:              completed
Validation:                  completed
```
