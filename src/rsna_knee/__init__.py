# -*- coding: utf-8 -*-
"""RSNA knee abnormality detection: DICOM series joined to report-derived labels.

    from rsna_knee import KneeDatasetBuilder

    builder = KneeDatasetBuilder("/kaggle/input/.../rsna-knee-abnormality-detection")
    df = builder.build()        # one row per series: metadata + labels
    ds = builder.dataset()      # indexable; volumes read on demand

Three layers, each usable on its own:

    rsna_knee.dicom      where the pixels are, and every header except the pixels
    rsna_knee.labeling   free-text reports -> 12 soft labels, in ~12 languages
    rsna_knee.builder    the join, and the cached table it produces
"""
from __future__ import annotations

from .builder import KneeDatasetBuilder, build_knee_dataset
from .config import resolve_data_path
from .dicom import DICOMExtractor, infer_weighting, plane_of, scan_series, sort_dicom_paths
from .labeling import (
    LABELS, Certainty, ClinicalNoteLabeler, LabelResult, Mention,
    NegationDetector, TextNormalizer, Vocabulary, build_default_vocabulary,
)
from .records import KneeDataset, SeriesRecord

__version__ = "0.1.0"

__all__ = [
    # dataset
    "KneeDatasetBuilder", "build_knee_dataset", "KneeDataset", "SeriesRecord",
    # dicom
    "DICOMExtractor", "scan_series", "sort_dicom_paths", "plane_of", "infer_weighting",
    # labeling
    "LABELS", "Certainty", "Mention", "LabelResult", "Vocabulary",
    "build_default_vocabulary", "TextNormalizer", "NegationDetector",
    "ClinicalNoteLabeler",
    # config
    "resolve_data_path",
    "__version__",
]
