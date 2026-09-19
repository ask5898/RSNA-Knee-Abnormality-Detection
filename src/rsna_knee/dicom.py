# -*- coding: utf-8 -*-
"""Finding the pixels, and reading everything about them except the pixels.

`DICOMExtractor` walks the competition layout. `scan_series` turns one series
into a flat metadata row and the slice paths in anatomical order, reading
headers only -- which is what makes a corpus-wide scan affordable.
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from glob import glob
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pydicom

__all__ = [
    "DICOMExtractor", "scan_series", "sort_dicom_paths",
    "slice_normal", "slice_position", "plane_of", "infer_weighting",
    "first_tag", "as_float", "age_years",
]


# --------------------------------------------------------------------------
# Tag access
# --------------------------------------------------------------------------

def first_tag(ds, *names, default=None):
    """First present, non-empty attribute out of `names`.

    Vendors disagree on which tag carries a concept -- laterality is
    `ImageLaterality` on one scanner and `Laterality` on the next -- so every
    read goes through a list of candidates instead of one attribute access.
    """
    for n in names:
        v = getattr(ds, n, None)
        if v is not None and v != "":
            return v
    return default


def as_float(v, default=None):
    """DICOM numbers arrive as DSfloat, IS, str, or a 1-element list."""
    if v is None:
        return default
    if isinstance(v, (list, tuple, pydicom.multival.MultiValue)):
        v = v[0] if len(v) else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def age_years(value) -> float | None:
    """'034Y' -> 34.0, '018M' -> 1.5. Returns None on anything else."""
    if not value:
        return None
    m = re.fullmatch(r"(\d{1,3})\s*([DWMY])?", str(value).strip(), re.I)
    if not m:
        return as_float(value)
    n = float(m.group(1))
    return n / {"D": 365.25, "W": 52.0, "M": 12.0, "Y": 1.0}[(m.group(2) or "Y").upper()]


def slice_normal(ds) -> np.ndarray | None:
    """Unit vector out of the image plane, from ImageOrientationPatient."""
    iop = first_tag(ds, "ImageOrientationPatient")
    if iop is None or len(iop) != 6:
        return None
    iop = np.asarray(iop, dtype=float)
    return np.cross(iop[:3], iop[3:])


def slice_position(ds, normal: np.ndarray | None) -> float | None:
    """Where this slice sits along the stack axis, in mm."""
    ipp = first_tag(ds, "ImagePositionPatient")
    if ipp is None or normal is None or len(ipp) != 3:
        return None
    return float(np.dot(np.asarray(ipp, dtype=float), normal))


def plane_of(normal: np.ndarray | None) -> str:
    """'axial' | 'sagittal' | 'coronal' | 'oblique' | 'unknown'.

    The dominant axis of the slice normal names the plane, but knee MRI is
    routinely prescribed oblique to the ligaments, so a normal that is not
    clearly dominated by one axis (< 0.75, i.e. more than ~41 deg off) is
    reported as oblique rather than silently rounded to the nearest plane.
    """
    if normal is None or not np.any(normal):
        return "unknown"
    n = np.abs(normal / np.linalg.norm(normal))
    axis = int(np.argmax(n))
    if n[axis] < 0.75:
        return "oblique"
    return ("sagittal", "coronal", "axial")[axis]


# Knee MRI protocols are named, not tagged: the weighting lives in
# SeriesDescription as free text. Order matters -- STIR and fat-sat qualifiers
# are checked before the plain weightings so "T2 FS" does not stop at "t2".
_WEIGHTING_RULES: list[tuple[str, str]] = [
    (r"\bstir\b", "STIR"),
    (r"\bt2\s*[\*x]|\bt2star|\bmerge\b|\bmedic\b", "T2*"),
    (r"\bpd\b|\bproton|\bdp\b", "PD"),
    (r"\bt1\b", "T1"),
    (r"\bt2\b", "T2"),
    (r"\bflair\b", "FLAIR"),
    (r"\bdwi\b|\bdiffus", "DWI"),
    (r"\b3d\b|\bmprage|\bvibe\b|\bfiesta|\btrufi|\bdess\b|\bfspgr", "3D"),
    (r"\bloc\b|\bscout|\blocali|\bsurvey", "localizer"),
]
_FATSAT = re.compile(r"\bfs\b|\bfat\s*sat|\bspair|\bspir\b|\bsat\b|\bstir\b|\bfatsat", re.I)


def infer_weighting(description: str | None, ds=None) -> str:
    """Best guess at sequence weighting from the series name, then the tags.

    A guess, and labelled as one: the series description is free text a
    technologist typed. Use it to filter candidate sequences, not as a feature
    you would stake a prediction on.
    """
    text = (description or "").lower()
    for pattern, name in _WEIGHTING_RULES:
        if re.search(pattern, text):
            return name
    if ds is not None:                      # fall back to acquisition timings
        tr, te = as_float(first_tag(ds, "RepetitionTime")), as_float(first_tag(ds, "EchoTime"))
        if tr is not None and te is not None:
            if tr < 900 and te < 30:
                return "T1"
            if tr >= 2000 and te >= 60:
                return "T2"
            if tr >= 2000 and te < 40:
                return "PD"
    return "unknown"


def sort_dicom_paths(datasets: Sequence, paths: Sequence[str]) -> list[int]:
    """Indices that put a series in through-plane anatomical order.

    Geometry first (position projected on the slice normal), InstanceNumber
    second, filename last. The geometric sort is the only one that is correct
    when a series was acquired interleaved or reconstructed out of order, but
    localizers and some derived series carry no position tags at all, hence
    the ladder. Sorting by filename alone silently shuffles slice order on any
    scanner whose exports are not zero-padded.
    """
    order = list(range(len(datasets)))
    normal = next((n for n in (slice_normal(d) for d in datasets) if n is not None), None)
    positions = [slice_position(d, normal) for d in datasets]
    if all(p is not None for p in positions) and len(set(positions)) > 1:
        return sorted(order, key=lambda i: positions[i])

    instance = [as_float(first_tag(datasets[i], "InstanceNumber")) for i in order]
    if all(v is not None for v in instance) and len(set(instance)) > 1:
        return sorted(order, key=lambda i: instance[i])

    return sorted(order, key=lambda i: paths[i])


def scan_series(study_uid: str, series_uid: str, series_dir: str) -> dict:
    """Read every header in one series; return one flat metadata row.

    Never raises: a series that cannot be read comes back as a row with
    `scan_error` set, because a corpus-wide scan that dies on file 40,000 of
    50,000 has cost you the other 49,999.
    """
    row: dict[str, Any] = {
        "StudyInstanceUID": study_uid,
        "SeriesInstanceUID": series_uid,
        "series_dir": series_dir,
        "n_slices": 0,
        "missing_files": True,
        "scan_error": "",
        "paths": (),
    }

    paths = sorted(glob(os.path.join(series_dir, "*.dcm")))
    if not paths:
        row["scan_error"] = "no .dcm files"
        return row

    datasets, kept, failed = [], [], 0
    for p in paths:
        try:
            datasets.append(pydicom.dcmread(p, stop_before_pixels=True))
            kept.append(p)
        except Exception as exc:                        # unreadable single file
            failed += 1
            row["scan_error"] = f"{type(exc).__name__}: {exc}"
    if not datasets:
        return row

    try:
        order = sort_dicom_paths(datasets, kept)
    except Exception as exc:                            # bad geometry tags
        order = list(range(len(datasets)))
        row["scan_error"] = f"unsorted ({type(exc).__name__}: {exc})"

    datasets = [datasets[i] for i in order]
    ordered_paths = [kept[i] for i in order]
    head = datasets[0]

    normal = slice_normal(head)
    positions = [slice_position(d, normal) for d in datasets]
    known = [p for p in positions if p is not None]
    gaps = np.diff(known) if len(known) > 1 else np.array([])

    spacing = first_tag(head, "PixelSpacing", default=[None, None])
    rows_, cols_ = as_float(first_tag(head, "Rows")), as_float(first_tag(head, "Columns"))
    row_mm, col_mm = as_float(spacing[0]), as_float(spacing[1] if len(spacing) > 1 else None)
    desc = first_tag(head, "SeriesDescription", "ProtocolName", default="")

    row.update({
        "paths": tuple(ordered_paths),
        "n_slices": len(datasets),
        "n_unreadable": failed,
        "missing_files": False,

        # geometry
        "rows": int(rows_) if rows_ else None,
        "cols": int(cols_) if cols_ else None,
        "pixel_spacing_row": row_mm,
        "pixel_spacing_col": col_mm,
        "slice_thickness": as_float(first_tag(head, "SliceThickness")),
        "spacing_between_slices": as_float(first_tag(head, "SpacingBetweenSlices")),
        "slice_gap_median": float(np.median(np.abs(gaps))) if gaps.size else None,
        # A series whose slice spacing is not constant cannot be resampled as a
        # regular volume; flagging it beats discovering it in the model.
        "irregular_spacing": bool(gaps.size and np.ptp(np.abs(gaps)) > 0.51),
        "plane": plane_of(normal),
        "fov_row_mm": (rows_ * row_mm) if (rows_ and row_mm) else None,
        "fov_col_mm": (cols_ * col_mm) if (cols_ and col_mm) else None,
        "extent_mm": float(max(known) - min(known)) if len(known) > 1 else None,

        # protocol
        "modality": first_tag(head, "Modality", default=""),
        "series_description": str(desc),
        "series_number": as_float(first_tag(head, "SeriesNumber")),
        "weighting": infer_weighting(str(desc), head),
        "fat_saturated": bool(_FATSAT.search(str(desc) or "")),
        "mr_acquisition_type": first_tag(head, "MRAcquisitionType", default=""),
        "scanning_sequence": str(first_tag(head, "ScanningSequence", default="")),
        "sequence_variant": str(first_tag(head, "SequenceVariant", default="")),
        "repetition_time": as_float(first_tag(head, "RepetitionTime")),
        "echo_time": as_float(first_tag(head, "EchoTime")),
        "inversion_time": as_float(first_tag(head, "InversionTime")),
        "flip_angle": as_float(first_tag(head, "FlipAngle")),
        "echo_train_length": as_float(first_tag(head, "EchoTrainLength")),
        "magnetic_field_strength": as_float(first_tag(head, "MagneticFieldStrength")),
        "body_part": str(first_tag(head, "BodyPartExamined", default="")),
        "laterality": str(first_tag(head, "ImageLaterality", "Laterality", default="")),

        # scanner
        "manufacturer": str(first_tag(head, "Manufacturer", default="")),
        "model": str(first_tag(head, "ManufacturerModelName", default="")),

        # patient
        "patient_id": str(first_tag(head, "PatientID", default="")),
        "patient_sex": str(first_tag(head, "PatientSex", default="")),
        "patient_age": age_years(first_tag(head, "PatientAge")),

        # pixel pipeline
        "photometric": str(first_tag(head, "PhotometricInterpretation", default="")),
        "bits_stored": as_float(first_tag(head, "BitsStored")),
        "rescale_slope": as_float(first_tag(head, "RescaleSlope"), 1.0),
        "rescale_intercept": as_float(first_tag(head, "RescaleIntercept"), 0.0),
        "window_center": as_float(first_tag(head, "WindowCenter")),
        "window_width": as_float(first_tag(head, "WindowWidth")),
    })
    return row


# --------------------------------------------------------------------------
# Walking the competition layout
# --------------------------------------------------------------------------

class DICOMExtractor :
    def __init__(self, data_path) :
        self.data_path = data_path

    def _getStudyInstanceUID(self, file = 'train.csv') -> list[str] :
        df = pd.read_csv(os.path.join(self.data_path, file))
        return df['StudyInstanceUID'].to_list()

    def _getSeriesInstanceUID(self, file = 'train_series.csv') -> dict:
        df = pd.read_csv(os.path.join(self.data_path, file))
        seriesInstanceUID = defaultdict(list)
        for study, series in zip(df['StudyInstanceUID'], df['SeriesInstanceUID']) :
            seriesInstanceUID[study].append(series)

        return seriesInstanceUID

    
    def getDICOM(self, metadata_only = False, file = 'train_series'):
        for study, series in self._getSeriesInstanceUID().items():
            for ser in series:
                series_dir = os.path.join(self.data_path, file, study, ser)
                paths = sorted(glob(os.path.join(series_dir, "*.dcm")))
                
                if not paths:
                    continue
    
                datasets = []
                for p in paths:
                    try:
                        datasets.append(pydicom.dcmread(p, stop_before_pixels=metadata_only))
                    except Exception as e:
                        print(f"skipping {p}: {type(e).__name__}: {e}")
    
                if not datasets:
                    continue
    
                iop = np.array(datasets[0].ImageOrientationPatient, float)
                normal = np.cross(iop[:3], iop[3:])
                datasets.sort(key=lambda d: float(np.dot(np.array(d.ImagePositionPatient, float), normal)))
                yield (study, ser), datasets
