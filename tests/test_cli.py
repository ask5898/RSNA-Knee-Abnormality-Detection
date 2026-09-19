# -*- coding: utf-8 -*-
"""The command line: the paths a user hits before they trust anything."""
from __future__ import annotations

import pytest

from rsna_knee.cli import main
from rsna_knee.config import resolve_data_path


def test_check_runs_and_reports(data_dir, capsys):
    assert main(["--data-path", data_dir, "--workers", "2", "check"]) == 0
    out = capsys.readouterr().out
    assert "sagittal" in out
    assert "series" in out


def test_labels_writes_csv_without_touching_dicoms(data_dir, tmp_path, capsys):
    out_path = str(tmp_path / "labels.csv")
    assert main(["--data-path", data_dir, "labels", "--out", out_path]) == 0
    import pandas as pd
    df = pd.read_csv(out_path)
    assert len(df) == 5
    assert df["medial_meniscus"].max() == 0.95
    assert "need review" in capsys.readouterr().out


def test_build_writes_the_index(data_dir, tmp_path):
    out_path = str(tmp_path / "index.parquet")
    assert main(["--data-path", data_dir, "--workers", "2", "build", "--out", out_path]) == 0
    from rsna_knee import KneeDatasetBuilder
    assert len(KneeDatasetBuilder.load(out_path).table) == 12


def test_subcommand_is_required(data_dir):
    with pytest.raises(SystemExit):
        main(["--data-path", data_dir])


# -- config ----------------------------------------------------------------
def test_explicit_path_wins(data_dir, monkeypatch):
    monkeypatch.setenv("RSNA_DATA_PATH", "/nowhere")
    assert resolve_data_path(data_dir) == data_dir


def test_env_var_is_used(data_dir, monkeypatch):
    monkeypatch.setenv("RSNA_DATA_PATH", data_dir)
    assert resolve_data_path() == data_dir


def test_missing_path_names_the_candidates(monkeypatch, tmp_path):
    monkeypatch.delenv("RSNA_DATA_PATH", raising=False)
    monkeypatch.chdir(tmp_path)              # no local data/ directory here
    with pytest.raises(FileNotFoundError, match="kaggle"):
        resolve_data_path()
