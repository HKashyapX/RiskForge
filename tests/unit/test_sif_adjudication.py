from __future__ import annotations

import csv
import json
import os

import pytest

from riskforge.modeling.sif_adjudication import create_adjudication_queue, main


def _record(incident_id: str, label: str, narrative: str) -> dict[str, object]:
    return {
        "id": incident_id,
        "input": {"narrative": narrative},
        "labels": {"sif_potential": label},
    }


def test_queue_contains_only_possible_records_and_hides_raw_ids(tmp_path) -> None:
    source = tmp_path / "validation.jsonl"
    output = tmp_path / "adjudication.csv"
    records = [
        _record("private-id-1", "yes", "not exported"),
        _record("private-id-2", "possible", "=formula-like narrative"),
    ]
    source.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    assert create_adjudication_queue([source], output) == 1

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["current_label"] == "possible"
    assert rows[0]["reviewer_label"] == ""
    assert rows[0]["narrative_for_review"].startswith("'=")
    assert "private-id-2" not in output.read_text(encoding="utf-8")
    if os.name == "posix":
        assert output.stat().st_mode & 0o777 == 0o600


def test_queue_refuses_to_overwrite_existing_output(tmp_path) -> None:
    source = tmp_path / "test.jsonl"
    output = tmp_path / "adjudication.csv"
    source.write_text(
        json.dumps(_record("private-id", "possible", "review me")) + "\n",
        encoding="utf-8",
    )
    output.write_text("existing", encoding="utf-8")

    with pytest.raises(FileExistsError):
        create_adjudication_queue([source], output)

    assert output.read_text(encoding="utf-8") == "existing"


def test_cli_reports_only_aggregate_metadata(tmp_path, capsys) -> None:
    source = tmp_path / "test.jsonl"
    output = tmp_path / "adjudication.csv"
    secret_id = "SECRET-ID"
    secret_narrative = "SECRET NARRATIVE"
    source.write_text(
        json.dumps(_record(secret_id, "possible", secret_narrative)) + "\n",
        encoding="utf-8",
    )

    assert main([str(source), "--output", str(output)]) == 0

    report = capsys.readouterr().out
    assert "rows=1" in report
    assert secret_id not in report
    assert secret_narrative not in report
