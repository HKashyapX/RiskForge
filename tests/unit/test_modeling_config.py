from __future__ import annotations

import json
from pathlib import Path

import pytest

from riskforge.modeling.config import MODELING_DATA_DIR_ENV, modeling_data_dir
from riskforge.modeling.dataset_inspection import inspect_dataset, main


def test_modeling_data_dir_defaults_to_project_relative_directory(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    assert modeling_data_dir(environment={}) == (tmp_path / "data" / "processed").resolve()


def test_modeling_data_dir_honors_environment(tmp_path) -> None:
    configured = tmp_path / "private-data"

    assert modeling_data_dir(environment={MODELING_DATA_DIR_ENV: str(configured)}) == configured


def test_jsonl_inspection_never_returns_record_values(tmp_path) -> None:
    path = tmp_path / "incidents.jsonl"
    secret_narrative = "worker identity and private incident narrative"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "incident_id": "one",
                        "input": {"narrative": secret_narrative},
                        "labels": {"sif": 1},
                    }
                ),
                json.dumps(
                    {
                        "incident_id": "two",
                        "input": {"narrative": "another private value"},
                        "labels": {"sif": None},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    report = inspect_dataset(path)

    assert report.records == 2
    assert report.fields == ("incident_id", "input", "labels")
    assert report.nested_schema["input.narrative"] == "str"
    assert report.nested_schema["labels.sif"] == "int|null"
    assert secret_narrative not in repr(report)


def test_cli_output_contains_schema_but_not_values(tmp_path, capsys) -> None:
    path = tmp_path / "incidents.jsonl"
    path.write_text('{"incident_id":"private-id","score":0.5}\n', encoding="utf-8")

    assert main([str(path)]) == 0

    output = capsys.readouterr().out
    assert "incident_id: str" in output
    assert "score: float" in output
    assert "private-id" not in output


def test_inspection_refuses_symlinks(tmp_path) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("{}\n", encoding="utf-8")
    link = tmp_path / "link.jsonl"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(ValueError, match="symbolic link"):
        inspect_dataset(link)
