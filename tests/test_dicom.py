# -*- coding: utf-8 -*-
"""Header reading, slice ordering, and the failures a real corpus contains."""
from __future__ import annotations

import os

import numpy as np
import pydicom
import pytest

from rsna_knee import DICOMExtractor, infer_weighting, plane_of, scan_series
from rsna_knee.dicom import age_years, as_float, slice_normal

from conftest import write_series


# -- small helpers ---------------------------------------------------------
@pytest.mark.parametrize("value,expected", [
    ("034Y", 34.0), ("018M", 1.5), ("052W", 1.0), ("055", 55.0), (None, None), ("", None),
])
def test_age_years(value, expected):
    assert age_years(value) == pytest.approx(expected) if expected else age_years(value) is None


def test_as_float_handles_multivalue():
    from pydicom.multival import MultiValue
    from pydicom.valuerep import DSfloat
    assert as_float(MultiValue(DSfloat, ["40.5", "80.2"])) == 40.5
    assert as_float("abc") is None
    assert as_float(None, 1.0) == 1.0


@pytest.mark.parametrize("normal,expected", [
    ([0, 0, 1], "axial"), ([1, 0, 0], "sagittal"), ([0, 1, 0], "coronal"),
    ([0.6, 0.6, 0.5], "oblique"),      # knee series are often prescribed oblique
    (None, "unknown"),
])
def test_plane_of(normal, expected):
    assert plane_of(np.array(normal, float) if normal else None) == expected


@pytest.mark.parametrize("description,expected", [
    ("SAG PD FS", "PD"), ("COR T1", "T1"), ("AX T2 FS", "T2"),
    ("STIR COR", "STIR"),            # checked before T2, or fat-sat swallows it
    ("T2* MERGE", "T2*"), ("3 PLANE LOC", "localizer"), ("whatever", "unknown"),
])
def test_infer_weighting(description, expected):
    assert infer_weighting(description) == expected


# -- scanning --------------------------------------------------------------
def series_path(data_dir, study, series):
    return os.path.join(data_dir, "train_series", study, series)


def test_scan_reads_geometry_and_protocol(data_dir):
    row = scan_series("1.2.3.000", "1.2.3.000.0", series_path(data_dir, "1.2.3.000", "1.2.3.000.0"))
    assert row["n_slices"] == 5
    assert row["plane"] == "sagittal"
    assert row["weighting"] == "PD"
    assert row["fat_saturated"] is True
    assert row["slice_gap_median"] == pytest.approx(4.0)
    assert row["irregular_spacing"] is False
    assert row["fov_row_mm"] == pytest.approx(32 * 0.35)
    assert row["patient_age"] == 47.0
    assert row["laterality"] == "L"
    assert not row["scan_error"]


def test_slices_come_back_in_anatomical_order(data_dir):
    """Filenames run backwards on purpose; geometry must win."""
    from rsna_knee.dicom import slice_position

    row = scan_series("1.2.3.000", "1.2.3.000.0", series_path(data_dir, "1.2.3.000", "1.2.3.000.0"))
    sets = [pydicom.dcmread(p) for p in row["paths"]]
    normal = slice_normal(sets[0])
    positions = [slice_position(d, normal) for d in sets]
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)        # the geometric sort ran
    assert [os.path.basename(p) for p in row["paths"]][0] == "0005.dcm"


def test_missing_folder_is_a_row_not_an_exception(data_dir):
    row = scan_series("1.2.3.000", "gone", series_path(data_dir, "1.2.3.000", "gone"))
    assert row["missing_files"] is True
    assert row["n_slices"] == 0
    assert "no .dcm files" in row["scan_error"]


def test_corrupt_file_does_not_lose_the_series(data_dir):
    """One bad file among five good ones costs you the one, not the series."""
    row = scan_series("1.2.3.002", "1.2.3.002.0", series_path(data_dir, "1.2.3.002", "1.2.3.002.0"))
    assert row["n_slices"] == 5
    assert row["n_unreadable"] == 1
    assert row["scan_error"]


def test_series_without_geometry_falls_back_to_instance_number(tmp_path):
    """Localizers carry no position tags; the existing extractor raises on these."""
    d = tmp_path / "nogeo"
    write_series(str(d), "S", "S.0", 4, "3 PLANE LOC", "axial", geometry=False)
    row = scan_series("S", "S.0", str(d))
    assert row["n_slices"] == 4
    assert row["plane"] == "unknown"
    assert row["weighting"] == "localizer"
    numbers = [int(pydicom.dcmread(p).InstanceNumber) for p in row["paths"]]
    assert numbers == sorted(numbers)


def test_irregular_spacing_is_flagged(tmp_path):
    """A stack that is not evenly spaced cannot be treated as a regular volume."""
    d = tmp_path / "gappy"
    write_series(str(d), "S", "S.0", 4, "SAG PD", "sagittal")
    from conftest import plane_normal

    victim = sorted(os.listdir(d))[0]
    ds = pydicom.dcmread(str(d / victim))
    ds.ImagePositionPatient = list(plane_normal("sagittal") * 40.0)   # far from the rest
    ds.save_as(str(d / victim), enforce_file_format=True)
    assert scan_series("S", "S.0", str(d))["irregular_spacing"] is True


# -- the original extractor ------------------------------------------------
def test_extractor_lists_series_per_study(data_dir):
    mapping = DICOMExtractor(data_dir)._getSeriesInstanceUID("train_series.csv")
    assert mapping["1.2.3.001"] == ["1.2.3.001.0", "1.2.3.001.1", "1.2.3.001.2"]


def test_extractor_lists_studies(data_dir):
    assert len(DICOMExtractor(data_dir)._getStudyInstanceUID("train.csv")) == 5


def test_slice_normal_is_none_without_orientation(tmp_path):
    d = tmp_path / "nogeo2"
    write_series(str(d), "S", "S.0", 1, "LOC", "axial", geometry=False)
    ds = pydicom.dcmread(str(next(d.iterdir())))
    assert slice_normal(ds) is None
