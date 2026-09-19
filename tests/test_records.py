# -*- coding: utf-8 -*-
"""Reading pixels: order, rescale, normalisation, and the ragged case."""
from __future__ import annotations

import os

import numpy as np
import pydicom
import pytest

from rsna_knee import KneeDataset, SeriesRecord, scan_series

from conftest import write_series


@pytest.fixture
def record(data_dir):
    row = scan_series("1.2.3.000", "1.2.3.000.0",
                      os.path.join(data_dir, "train_series", "1.2.3.000", "1.2.3.000.0"))
    return SeriesRecord("1.2.3.000", "1.2.3.000.0", row["paths"], row,
                        {"ACL": 0.02, "medial_meniscus": 0.95})


def test_volume_shape_and_order(record):
    """Slice i has value 100(i+1), so the stack states its own order."""
    vol = record.volume()
    assert vol.shape == (5, 32, 32)
    assert [float(s.mean()) for s in vol] == [100.0, 200.0, 300.0, 400.0, 500.0]


def test_len_is_slice_count(record):
    assert len(record) == 5 == record.meta["n_slices"]


def test_rescale_is_applied(tmp_path):
    d = tmp_path / "rescaled"
    write_series(str(d), "S", "S.0", 2, "SAG PD", "sagittal")
    for f in os.listdir(d):
        ds = pydicom.dcmread(str(d / f))
        ds.RescaleSlope, ds.RescaleIntercept = 2.0, -50.0
        ds.save_as(str(d / f), enforce_file_format=True)
    row = scan_series("S", "S.0", str(d))
    rec = SeriesRecord("S", "S.0", row["paths"], row, {})
    assert float(rec.volume()[0].mean()) == 100.0 * 2 - 50
    assert float(rec.volume(apply_rescale=False)[0].mean()) == 100.0


def test_monochrome1_is_inverted(tmp_path):
    """MONOCHROME1 stores bright-is-low; left alone it trains on inverted images."""
    d = tmp_path / "mono1"
    write_series(str(d), "S", "S.0", 2, "SAG PD", "sagittal")
    for f in os.listdir(d):
        ds = pydicom.dcmread(str(d / f))
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.save_as(str(d / f), enforce_file_format=True)
    row = scan_series("S", "S.0", str(d))
    vol = SeriesRecord("S", "S.0", row["paths"], row, {}).volume()
    assert float(vol[0].max()) == 0.0        # a flat slice inverts to zero


@pytest.mark.parametrize("mode,check", [
    ("minmax", lambda v: (v.min(), v.max()) == (0.0, 1.0)),
    ("zscore", lambda v: abs(float(v.mean())) < 1e-5 and abs(float(v.std()) - 1) < 1e-5),
    (None, lambda v: v.max() == 500.0),
])
def test_normalize(record, mode, check):
    assert check(record.volume(normalize=mode))


def test_normalize_is_per_volume_not_per_slice(record):
    """Per-slice scaling destroys the between-slice intensity relationship."""
    vol = record.volume(normalize="minmax")
    assert len({float(s.mean()) for s in vol}) == 5


def test_unknown_normalize_raises(record):
    with pytest.raises(ValueError, match="unknown normalize"):
        record.volume(normalize="standardise")


def test_ragged_series_raises_with_its_shapes(data_dir):
    row = scan_series("1.2.3.001", "1.2.3.001.1",
                      os.path.join(data_dir, "train_series", "1.2.3.001", "1.2.3.001.1"))
    rec = SeriesRecord("1.2.3.001", "1.2.3.001.1", row["paths"], row, {})
    with pytest.raises(ValueError, match="ragged"):
        rec.volume()


def test_empty_series_raises(record):
    with pytest.raises(ValueError, match="no files"):
        SeriesRecord("s", "x", (), {}, {}).volume()


def test_label_vector_follows_column_order(record):
    v = record.label_vector(["medial_meniscus", "ACL"])
    assert v.dtype == np.float32                 # what a model wants to be handed
    assert v.tolist() == pytest.approx([0.95, 0.02])


def test_label_vector_is_nan_for_missing_label(record):
    assert np.isnan(record.label_vector(["effusion"])[0])


def test_dicoms_are_reread_in_order(record):
    numbers = [int(d.InstanceNumber) for d in record.dicoms(metadata_only=True)]
    assert numbers == sorted(numbers)


# -- KneeDataset -----------------------------------------------------------
def test_dataset_item(record):
    ds = KneeDataset([record], ["ACL", "medial_meniscus"])
    item = ds[0]
    assert item["volume"].shape == (5, 32, 32)
    assert item["labels"].tolist() == pytest.approx([0.02, 0.95])
    assert item["study_uid"] == "1.2.3.000"


def test_dataset_can_skip_pixels(record):
    """Metadata-only iteration must not touch the disk."""
    assert KneeDataset([record], ["ACL"], load_pixels=False)[0]["volume"] is None


def test_dataset_transform_applies(record):
    ds = KneeDataset([record], ["ACL"], transform=lambda s: s["volume"].shape)
    assert ds[0] == (5, 32, 32)


def test_filter_returns_a_view(record):
    ds = KneeDataset([record], ["ACL"])
    assert len(ds.filter(lambda r: r.plane == "sagittal")) == 1
    assert len(ds.filter(lambda r: r.plane == "axial")) == 0
    assert len(ds) == 1                      # the original is untouched


def test_record_exposes_plane_and_weighting(record):
    assert (record.plane, record.weighting) == ("sagittal", "PD")
