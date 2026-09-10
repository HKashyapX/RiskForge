from __future__ import annotations

import json

from riskforge.modeling.dataset_contract import main, validate_sif_dataset


def _record(incident_id: str, narrative: str = "private narrative") -> dict[str, object]:
    return {
        "id": incident_id,
        "input": {"narrative": narrative},
        "labels": {
            "sif_potential": "yes",
            "sif_confidence": "high",
            "sif_label_source": "weak_supervision",
        },
        "provenance": {
            "pipeline_version": "test-v1",
            "source_datasets": ["synthetic-test"],
        },
    }


def test_valid_dataset_passes_without_exposing_values(tmp_path, capsys) -> None:
    path = tmp_path / "valid.jsonl"
    secret = "PRIVATE NARRATIVE VALUE"
    path.write_text(json.dumps(_record("sensitive-id", secret)) + "\n", encoding="utf-8")

    assert main([str(path)]) == 0

    output = capsys.readouterr().out
    assert output.startswith("PASS")
    assert secret not in output
    assert "sensitive-id" not in output


def test_missing_nested_fields_and_duplicates_fail(tmp_path) -> None:
    path = tmp_path / "invalid.jsonl"
    records = [
        _record("duplicate"),
        _record("duplicate"),
        {"id": "missing-fields"},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    report = validate_sif_dataset(path)

    assert not report.valid
    assert report.records == 3
    assert report.invalid_records == 2
    assert report.duplicate_ids == 1
    assert report.cross_split_duplicates == 0
    assert report.issue_counts["duplicate:id"] == 1
    assert report.issue_counts["missing:input.narrative"] == 1


def test_cli_detects_cross_split_leakage_without_exposing_id(tmp_path, capsys) -> None:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text(json.dumps(_record("leaked-private-id")) + "\n", encoding="utf-8")
    test.write_text(json.dumps(_record("leaked-private-id")) + "\n", encoding="utf-8")

    assert main([str(train), str(test)]) == 1

    output = capsys.readouterr().out
    assert "leakage:id: 1" in output
    assert "leaked-private-id" not in output


def test_malformed_json_is_counted_without_echoing_content(tmp_path, capsys) -> None:
    path = tmp_path / "malformed.jsonl"
    path.write_text('{"narrative":"SECRET"\n', encoding="utf-8")

    assert main([str(path)]) == 1

    output = capsys.readouterr().out
    assert "invalid:json: 1" in output
    assert "SECRET" not in output
