# -*- coding: utf-8 -*-
"""A synthetic competition tree: real .dcm files, knee-ish headers, known answers.

Real DICOM files rather than mocks, because almost every bug this package can
have lives in the gap between what pydicom returns and what the code assumes --
a mock would agree with the assumption and prove nothing.

The fixture deliberately includes the awkward cases:

    study 0   clean, two series
    study 1   one series with a ragged final slice
    study 2   one series with a corrupt file among good ones
    study 3   a report in an unsupported script
    study 4   clean
    plus      a series listed in the csv whose folder does not exist

Slices are written with filenames in REVERSE anatomical order, so any test that
passes by accident of filesystem ordering fails here.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

# (language, report) -> the labels each one should produce are asserted in
# test_labeling.py; keep the two in step.
REPORTS = [
    ("en", "Findings: No ACL tear. Medial meniscus posterior horn tear. "
           "Moderate joint effusion.\nImpression: Medial meniscal tear."),
    ("ru", "Медиальный мениск: разрыв заднего рога. Передняя крестообразная "
           "связка интактна. Перелом не выявлен."),
    ("de", "Kein Erguss. Innenmeniskus unauffallig. Verdacht auf Bakerzyste."),
    ("ja", "前十字靭帯断裂を認める。"),          # script the vocabulary cannot read
    ("en", "Possible lateral meniscus tear. Patellofemoral chondromalacia "
           "patellae. No fracture."),
]

PLANE_IOP = {
    "sagittal": [0, 1, 0, 0, 0, -1],
    "coronal": [1, 0, 0, 0, 0, -1],
    "axial": [1, 0, 0, 0, 1, 0],
}

SERIES_PLAN = [("SAG PD FS", "sagittal", 5), ("COR T1", "coronal", 4), ("AX T2 FS", "axial", 3)]


def plane_normal(plane):
    """The direction slices actually step in for a given orientation.

    Worth computing rather than assuming: a sagittal series steps along x, not
    z, and a fixture that walks z regardless leaves every position projecting
    to the same value -- which makes the geometric sort fall through to
    InstanceNumber and quietly stop being tested.
    """
    iop = np.array(PLANE_IOP[plane], float)
    return np.cross(iop[:3], iop[3:])


def write_series(directory, study, series, n_slices, description, plane,
                 ragged=False, corrupt=False, geometry=True):
    """Write one series. Slice i sits 4i mm along the slice normal, value 100(i+1)."""
    os.makedirs(directory, exist_ok=True)
    for i in range(n_slices):
        meta = Dataset()
        meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"   # MR Image Storage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset(None, Dataset(), file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = meta.MediaStorageSOPClassUID
        ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
        ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
        ds.Modality, ds.BodyPartExamined, ds.ImageLaterality = "MR", "KNEE", "L"
        ds.SeriesDescription, ds.SeriesNumber = description, 3
        ds.Manufacturer, ds.ManufacturerModelName = "SIEMENS", "MAGNETOM Aera"
        ds.MagneticFieldStrength, ds.MRAcquisitionType = 1.5, "2D"
        ds.RepetitionTime, ds.EchoTime = 3500.0, 85.0
        ds.FlipAngle, ds.EchoTrainLength = 150.0, 11
        ds.ScanningSequence, ds.SequenceVariant = "SE", "SK"
        ds.PatientID, ds.PatientSex, ds.PatientAge = "P" + study[-3:], "M", "047Y"
        if geometry:
            ds.ImageOrientationPatient = PLANE_IOP[plane]
            ds.ImagePositionPatient = list(plane_normal(plane) * float(i) * 4.0)
        ds.InstanceNumber = i + 1
        ds.PixelSpacing = [0.35, 0.35]
        ds.SliceThickness, ds.SpacingBetweenSlices = 3.0, 4.0
        ds.PhotometricInterpretation, ds.SamplesPerPixel = "MONOCHROME2", 1
        ds.BitsAllocated = ds.BitsStored = 16
        ds.HighBit, ds.PixelRepresentation = 15, 0
        ds.RescaleSlope, ds.RescaleIntercept = 1.0, 0.0

        size = 24 if (ragged and i == n_slices - 1) else 32
        ds.Rows = ds.Columns = size
        ds.PixelData = np.full((size, size), 100 * (i + 1), np.uint16).tobytes()

        # reverse-numbered filenames: sorting by name gives the wrong order
        ds.save_as(os.path.join(directory, f"{n_slices - i:04d}.dcm"),
                   enforce_file_format=True)

    if corrupt:
        with open(os.path.join(directory, "9999.dcm"), "wb") as fh:
            fh.write(b"not a dicom at all")


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory) -> str:
    """Path to a complete synthetic competition tree. Built once per session."""
    root = str(tmp_path_factory.mktemp("rsna"))
    studies, series_rows = [], []

    for k, (language, text) in enumerate(REPORTS):
        study = f"1.2.3.{k:03d}"
        studies.append({"StudyInstanceUID": study, "report": text, "language": language})
        for j, (description, plane, n) in enumerate(SERIES_PLAN[: 2 + (k % 2)]):
            series = f"{study}.{j}"
            series_rows.append({"StudyInstanceUID": study, "SeriesInstanceUID": series})
            write_series(
                os.path.join(root, "train_series", study, series),
                study, series, n, description, plane,
                ragged=(k == 1 and j == 1),
                corrupt=(k == 2 and j == 0),
            )

    series_rows.append({"StudyInstanceUID": "1.2.3.000",
                        "SeriesInstanceUID": "1.2.3.000.missing"})

    pd.DataFrame(studies).to_csv(os.path.join(root, "train.csv"), index=False)
    pd.DataFrame(series_rows).to_csv(os.path.join(root, "train_series.csv"), index=False)
    return root


@pytest.fixture
def variant(data_dir, tmp_path):
    """A copy of the tree with train.csv rewritten by `mutate`."""
    import shutil

    def make(mutate):
        """`mutate` may edit the frame in place or return a new one."""
        target = tmp_path / "variant"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(data_dir, target)
        df = pd.read_csv(target / "train.csv")
        returned = mutate(df)
        if returned is not None:            # `or df` is ambiguous on a DataFrame
            df = returned
        df.to_csv(target / "train.csv", index=False)
        return str(target)

    return make


@pytest.fixture
def builder(data_dir):
    from rsna_knee import KneeDatasetBuilder
    return KneeDatasetBuilder(data_dir, workers=4, verbose=False)


@pytest.fixture
def built(builder):
    builder.build()
    return builder
