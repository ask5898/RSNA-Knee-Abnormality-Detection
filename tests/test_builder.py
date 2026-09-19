# -*- coding: utf-8 -*-
"""The join: labels broadcast to series, failures recorded, table cached."""
from __future__ import annotations

import os

import pandas as pd
import pytest

from rsna_knee import KneeDatasetBuilder, build_knee_dataset


# -- shape of the result ---------------------------------------------------
def test_one_row_per_scannable_series(built):
    df = built.table
    # 12 real series; the missing folder is dropped, the corrupt file is not a
    # whole series.
    assert len(df) == 12
    assert df["StudyInstanceUID"].nunique() == 5
    assert not df["SeriesInstanceUID"].duplicated().any()


def test_metadata_and_labels_are_on_the_same_row(built):
    row = built.table.set_index("SeriesInstanceUID").loc["1.2.3.000.0"]
    assert row["plane"] == "sagittal"          # from the pixels
    assert row["medial_meniscus"] == 0.95      # from the report
    assert row["language"] == "en"             # from train.csv


def test_labels_broadcast_to_every_series_of_a_study(built):
    per_study = built.table.groupby("StudyInstanceUID")["medial_meniscus"].nunique()
    assert (per_study == 1).all()


def test_group_column_is_the_study(built):
    """Split on this. A row-level split puts one report on both sides."""
    assert (built.table["group"] == built.table["StudyInstanceUID"]).all()


def test_n_series_in_study_counts_scanned_series(built):
    counts = built.table.set_index("SeriesInstanceUID")["n_series_in_study"]
    assert counts["1.2.3.000.0"] == 2          # the missing one is not counted
    assert counts["1.2.3.001.0"] == 3


def test_certainty_columns_accompany_the_labels(built):
    row = built.table.set_index("StudyInstanceUID").loc["1.2.3.000"].iloc[0]
    assert row["ACL_certainty"] == "negated"
    assert row["medial_meniscus_certainty"] == "definite"


def test_needs_review_survives_the_join(built):
    flagged = built.table[built.table["needs_review"]]["StudyInstanceUID"].unique()
    assert list(flagged) == ["1.2.3.003"]      # the unsupported-script report


# -- failures are data -----------------------------------------------------
def test_problem_series_are_recorded(built):
    ids = {e["SeriesInstanceUID"] for e in built.errors}
    assert "1.2.3.000.missing" in ids          # folder absent
    assert "1.2.3.002.0" in ids                # one corrupt file


def test_empty_series_dropped_by_default(built):
    assert "1.2.3.000.missing" not in set(built.table["SeriesInstanceUID"])


def test_empty_series_can_be_kept(data_dir):
    b = KneeDatasetBuilder(data_dir, workers=2, verbose=False, drop_empty_series=False)
    df = b.build()
    assert (df["n_slices"] == 0).sum() == 1


def test_study_with_no_images_stays_in_the_study_table(variant):
    """A label with no pixels behind it must not become a training row."""
    def add_orphan(df):
        return pd.concat([df, pd.DataFrame([{"StudyInstanceUID": "9.9.9",
                                             "report": "ACL tear.",
                                             "language": "en"}])])
    b = KneeDatasetBuilder(variant(add_orphan), workers=2, verbose=False)
    assert "9.9.9" not in set(b.build()["StudyInstanceUID"])
    assert "9.9.9" in set(b.study_table()["StudyInstanceUID"])


# -- report column resolution ----------------------------------------------
def test_sections_are_concatenated(variant):
    def split_sections(df):
        df["findings"] = df.pop("report")
        df["impression"] = "Correlate clinically."
    b = KneeDatasetBuilder(variant(split_sections), workers=2, verbose=False)
    df = b.build()
    assert b.resolve_report_columns(pd.read_csv(os.path.join(b.data_path, "train.csv"))) \
        == ["findings", "impression"]
    assert df.set_index("StudyInstanceUID").loc["1.2.3.000"].iloc[0]["medial_meniscus"] == 0.95


def test_unknown_column_name_falls_back_and_warns(variant):
    path = variant(lambda df: df.rename(columns={"report": "rad_free_text"}))
    b = KneeDatasetBuilder(path, workers=2, verbose=False)
    with pytest.warns(UserWarning, match="No known report column"):
        df = b.build()
    assert df["medial_meniscus"].max() == 0.95


def test_no_text_column_raises_with_the_column_list(variant):
    b = KneeDatasetBuilder(variant(lambda df: df.drop(columns=["report"])),
                           workers=2, verbose=False)
    with pytest.raises(KeyError, match="No report column found"):
        b.build()


def test_explicit_wrong_column_raises(data_dir):
    b = KneeDatasetBuilder(data_dir, report_col="nope", verbose=False)
    with pytest.raises(KeyError, match="not in"):
        b.build()


def test_explicit_column_is_used(data_dir):
    b = KneeDatasetBuilder(data_dir, report_col="report", workers=2, verbose=False)
    assert b.build()["medial_meniscus"].max() == 0.95


def test_missing_data_path_names_what_it_tried(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.csv"):
        KneeDatasetBuilder(str(tmp_path), verbose=False).build()


# -- options ---------------------------------------------------------------
def test_label_prefix_and_no_certainty(data_dir):
    b = KneeDatasetBuilder(data_dir, label_prefix="y_", include_certainty=False,
                           workers=2, verbose=False)
    df = b.build()
    assert "y_ACL" in df.columns and "ACL" not in df.columns
    assert not [c for c in df.columns if c.endswith("_certainty")]
    assert b.records[0].labels["ACL"] == 0.02       # records still key on the label


def test_limit_studies(data_dir):
    b = KneeDatasetBuilder(data_dir, workers=2, verbose=False)
    assert b.build(limit_studies=2)["StudyInstanceUID"].nunique() == 2


def test_labelling_runs_once(builder, monkeypatch):
    """The scan is slow; re-labelling on every accessor is just waste."""
    calls = []
    original = builder.labeler.extract
    monkeypatch.setattr(builder.labeler, "extract",
                        lambda t: (calls.append(t), original(t))[1])
    builder.build()
    builder.study_table()
    builder.review_queue()
    assert len(calls) == 5          # one per report, not fifteen


# -- records and dataset ---------------------------------------------------
def test_records_carry_paths_labels_and_report(built):
    rec = next(r for r in built.records if r.series_uid == "1.2.3.000.0")
    assert len(rec.paths) == 5
    assert rec.labels["medial_meniscus"] == 0.95
    assert "Medial meniscus" in rec.report
    assert rec.meta["weighting"] == "PD"


def test_dataset_round_trips_to_pixels(built):
    ds = built.dataset(normalize="zscore")
    item = ds[[r.series_uid for r in ds.records].index("1.2.3.000.0")]
    assert item["volume"].shape == (5, 32, 32)
    assert len(item["labels"]) == 12


def test_review_queue_lists_the_unreadable_report(built):
    q = built.review_queue()
    assert list(q["StudyInstanceUID"]) == ["1.2.3.003"]
    assert list(q["report_script"]) == ["han"]


def test_study_table_aggregates_series(built):
    st = built.study_table().set_index("StudyInstanceUID")
    assert st.loc["1.2.3.001", "n_series"] == 3
    assert st.loc["1.2.3.001", "planes"] == ["axial", "coronal", "sagittal"]


# -- cache -----------------------------------------------------------------
@pytest.mark.parametrize("name", ["index.parquet", "index.csv"])
def test_cache_round_trip(built, tmp_path, name):
    path = built.save(str(tmp_path / name))
    reloaded = KneeDatasetBuilder.load(path)
    assert reloaded.table.shape == built.table.shape
    assert reloaded.records[0].volume().shape == built.records[0].volume().shape


def test_save_falls_back_to_csv_without_a_parquet_engine(built, tmp_path, monkeypatch):
    import pandas as pd

    def no_engine(self, *a, **k):
        raise ImportError("no pyarrow")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", no_engine)
    with pytest.warns(UserWarning, match="no parquet engine"):
        path = built.save(str(tmp_path / "index.parquet"))
    assert path.endswith(".csv") and os.path.exists(path)


# -- convenience -----------------------------------------------------------
def test_build_knee_dataset_one_liner(data_dir):
    df, ds = build_knee_dataset(data_dir, workers=2, verbose=False)
    assert len(df) == len(ds) == 12


def test_repr_says_whether_it_is_built(data_dir):
    b = KneeDatasetBuilder(data_dir, verbose=False)
    assert "unbuilt" in repr(b)
    b.build()
    assert "12 series" in repr(b)
