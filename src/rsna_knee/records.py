# -*- coding: utf-8 -*-
"""One series, and an indexable view over many.

Both hold paths, not pixels: building thousands of records costs nothing and
`SeriesRecord.volume()` is where the IO happens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import pydicom

from .dicom import as_float, first_tag

__all__ = ["SeriesRecord", "KneeDataset"]


@dataclass
class SeriesRecord:
    """One series: its ordered file paths, its metadata, its study's labels.

    Holds paths, not pixels. Constructing thousands of these costs nothing;
    `volume()` is where the IO happens.
    """

    study_uid: str
    series_uid: str
    paths: tuple[str, ...]
    meta: dict = field(default_factory=dict)
    labels: dict[str, float] = field(default_factory=dict)
    report: str = ""

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def plane(self) -> str:
        return self.meta.get("plane", "unknown")

    @property
    def weighting(self) -> str:
        return self.meta.get("weighting", "unknown")

    def dicoms(self, metadata_only: bool = False) -> Iterator[pydicom.Dataset]:
        """Re-read the datasets, in anatomical order."""
        for p in self.paths:
            yield pydicom.dcmread(p, stop_before_pixels=metadata_only)

    def volume(self, dtype=np.float32, apply_rescale: bool = True,
               normalize: str | None = None) -> np.ndarray:
        """(n_slices, H, W) array, slices in through-plane order.

        normalize:
            None      raw stored values (after rescale)
            'minmax'  per-volume to [0, 1]
            'zscore'  per-volume zero mean, unit variance

        Per-volume, not per-slice, and per-volume, not per-dataset: MR
        intensities have no physical units and vary with coil, scanner and
        sequence, so a global normalisation constant is meaningless, while a
        per-slice one destroys the intensity relationship *between* slices
        that makes an effusion or a marrow oedema visible as it comes and goes
        through the stack.
        """
        if not self.paths:
            raise ValueError(f"series {self.series_uid} has no files")

        slices = []
        for p in self.paths:
            ds = pydicom.dcmread(p)
            arr = ds.pixel_array.astype(np.float32)
            if apply_rescale:
                arr = arr * as_float(first_tag(ds, "RescaleSlope"), 1.0) + as_float(first_tag(ds, "RescaleIntercept"), 0.0)
            if str(first_tag(ds, "PhotometricInterpretation", default="")) == "MONOCHROME1":
                arr = arr.max() - arr          # MONOCHROME1 stores inverted
            slices.append(arr)

        shapes = {s.shape for s in slices}
        if len(shapes) > 1:
            raise ValueError(
                f"series {self.series_uid} has ragged slices {sorted(shapes)}; "
                "resample or drop it before stacking"
            )

        vol = np.stack(slices).astype(dtype)
        if normalize == "minmax":
            lo, hi = float(vol.min()), float(vol.max())
            vol = (vol - lo) / (hi - lo) if hi > lo else np.zeros_like(vol)
        elif normalize == "zscore":
            sd = float(vol.std())
            vol = (vol - float(vol.mean())) / sd if sd > 0 else np.zeros_like(vol)
        elif normalize is not None:
            raise ValueError(f"unknown normalize={normalize!r}")
        return vol

    def label_vector(self, labels: Sequence[str]) -> np.ndarray:
        """Labels in a fixed column order -- the model's target vector."""
        return np.array([self.labels.get(k, np.nan) for k in labels], dtype=np.float32)


class KneeDataset:
    """Indexable view over the records. Works as a `torch.utils.data.Dataset`.

    Deliberately not a torch subclass: torch is not imported here, so the same
    object works in a notebook, in a sklearn loop, or inside a DataLoader.
    """

    def __init__(self, records: Sequence[SeriesRecord], labels: Sequence[str],
                 transform: Callable[[dict], Any] | None = None,
                 load_pixels: bool = True, **volume_kwargs):
        self.records = list(records)
        self.labels = list(labels)
        self.transform = transform
        self.load_pixels = load_pixels
        self.volume_kwargs = volume_kwargs

    def __len__(self) -> int:
        return len(self.records)

    def __repr__(self) -> str:
        return f"KneeDataset({len(self)} series, {len(self.labels)} labels)"

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        sample = {
            "volume": r.volume(**self.volume_kwargs) if self.load_pixels else None,
            "labels": r.label_vector(self.labels),
            "meta": r.meta,
            "study_uid": r.study_uid,
            "series_uid": r.series_uid,
        }
        return self.transform(sample) if self.transform else sample

    def filter(self, predicate: Callable[[SeriesRecord], bool]) -> "KneeDataset":
        """A new view over the records that pass. Cheap -- nothing is copied."""
        return KneeDataset([r for r in self.records if predicate(r)], self.labels,
                           self.transform, self.load_pixels, **self.volume_kwargs)
