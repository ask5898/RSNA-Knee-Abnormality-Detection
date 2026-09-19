# -*- coding: utf-8 -*-
"""Join the pixel side and the text side into one dataset.

`DICOMExtractor` knows where the images are. `ClinicalNoteLabeler` knows what
the report says. Nothing so far knows that a study has both. This module is
that missing piece:

    builder = KneeDatasetBuilder(DATA_PATH)
    df = builder.build()                  # one row per series: meta + labels
    ds = builder.dataset()                # indexable; volumes read on demand
    vol, y, meta = ds[0]["volume"], ds[0]["labels"], ds[0]["meta"]

DESIGN
------
Three ideas carry the whole thing:

1.  THE ROW IS A SERIES, THE LABEL IS A STUDY.
    Reports are written per study; pixels live per series; a knee study is
    typically 4-8 sequences. So labels are broadcast down to every series of
    their study and the study id is kept on every row, which is what you group
    by when you split train/val. Splitting on series leaks a study's report
    across the split and flatters your validation score. `build()` also emits
    `n_series_in_study` so a per-study aggregation is one groupby away.

2.  METADATA IS EAGER, PIXELS ARE LAZY.
    One header per slice is cheap and gives you everything you filter on
    (plane, weighting, slice count, spacing). Pixels are gigabytes. So the
    scan reads headers only (`stop_before_pixels=True`) and stores the sorted
    file paths; `SeriesRecord.volume()` reads pixels when you actually ask.
    The table is therefore small enough to cache to disk and re-load in
    seconds, which is the difference between iterating on your dataset and
    waiting on it.

3.  A FAILED SERIES IS DATA, NOT AN EXCEPTION.
    Unreadable files, missing folders, absent geometry tags, ragged slice
    shapes: on a real DICOM corpus all of these happen, and a scan that dies
    on the first one is useless. Every failure is recorded on the row
    (`scan_error`, `missing_files`) and surfaced in `builder.errors`, so a bad
    series is something you can count and inspect rather than something that
    ends the run.
"""
from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import pandas as pd

from .dicom import DICOMExtractor, scan_series
from .labeling import LABELS, ClinicalNoteLabeler
from .records import KneeDataset, SeriesRecord

__all__ = ["KneeDatasetBuilder", "build_knee_dataset"]


class KneeDatasetBuilder:
    """Reports -> labels, folders -> metadata, both -> one table.

        builder = KneeDatasetBuilder(DATA_PATH)
        df = builder.build()                    # scan + label + join
        builder.save("knee_index.parquet")      # cache; skip the scan next time
        ds = builder.dataset(normalize="zscore")

    The scan is the expensive step (one header read per slice), so it runs
    once, in a thread pool, and everything downstream works off the table.
    """

    # Candidate report columns, in the order they should be concatenated when a
    # dataset splits a report across several. Matched case-insensitively.
    REPORT_COLUMNS = (
        "report", "report_text", "radiology_report", "radiologist_report",
        "clinical_notes", "doctor_notes", "notes", "note", "text",
        "clinical_history", "history", "indication", "technique",
        "findings", "impression", "conclusion", "description",
    )

    def __init__(self, data_path: str, split: str = "train", *,
                 labeler: Any = None, extractor: Any = None,
                 report_col: str | Sequence[str] | None = None,
                 id_col: str = "StudyInstanceUID",
                 labels: Sequence[str] | None = None,
                 label_prefix: str = "", include_certainty: bool = True,
                 drop_empty_series: bool = True,
                 workers: int = 8, verbose: bool = True):
        """
        data_path         competition root (holds train.csv, train_series/, ...)
        split             'train' or 'test'; picks the csv names and image dir
        labeler           a ClinicalNoteLabeler; built with defaults if omitted
        extractor         a DICOMExtractor; built on data_path if omitted
        report_col        report column name(s). Auto-detected when None --
                          pass it explicitly once you know it.
        label_prefix      prefix for the label columns, e.g. 'y_'. Empty means
                          the label names themselves.
        include_certainty adds '<label>_certainty' columns (definite / probable
                          / negated / not_mentioned / ...). Keep them: they let
                          you weight a definite tear above a hedged one, or
                          drop hedged rows entirely, without re-running the
                          labeler.
        workers           threads for the header scan. IO-bound, so more than
                          cores is fine; drop to 1 to debug.
        """
        self.data_path = data_path
        self.split = split
        self.verbose = verbose
        self.workers = max(1, int(workers))
        self.drop_empty_series = drop_empty_series
        self.id_col = id_col
        self.label_prefix = label_prefix
        self.include_certainty = include_certainty

        self.study_csv = f"{split}.csv"
        self.series_csv = f"{split}_series.csv"
        self.series_dir = f"{split}_series"

        self.labeler = labeler if labeler is not None else ClinicalNoteLabeler()
        self.extractor = extractor if extractor is not None else DICOMExtractor(data_path)
        self.labels = list(labels if labels is not None else getattr(self.labeler, "labels", LABELS))
        self._report_col = report_col

        self.errors: list[dict] = []
        self._studies: pd.DataFrame | None = None
        self._series_meta: pd.DataFrame | None = None
        self._table: pd.DataFrame | None = None
        self._records: list[SeriesRecord] | None = None

    # -- small utilities --------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _read_csv(self, name: str) -> pd.DataFrame:
        path = os.path.join(self.data_path, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Set data_path/split to match your layout; "
                f"found: {sorted(os.listdir(self.data_path))[:20] if os.path.isdir(self.data_path) else 'no such dir'}"
            )
        return pd.read_csv(path)

    # -- the text side ----------------------------------------------------
    def resolve_report_columns(self, df: pd.DataFrame) -> list[str]:
        """Which column(s) hold the doctor's note.

        Auto-detection is a convenience for the first run, not a contract:
        competitions rename this column freely. It prefers a whole-report
        column, falls back to concatenating section columns (findings +
        impression), and raises with the actual column list rather than
        guessing wrong and labelling the entire corpus from nothing.
        """
        if self._report_col:
            cols = [self._report_col] if isinstance(self._report_col, str) else list(self._report_col)
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise KeyError(f"report column(s) {missing} not in {list(df.columns)}")
            return cols

        lower = {c.lower(): c for c in df.columns}
        whole = [lower[n] for n in self.REPORT_COLUMNS[:10] if n in lower]
        if whole:
            return whole[:1]
        sections = [lower[n] for n in self.REPORT_COLUMNS[10:] if n in lower]
        if sections:
            return sections

        # Last resort: the widest text column. Reports are long; ids are not.
        # `is_string_dtype` rather than `== object`: pandas >= 3 gives string
        # columns their own dtype and the object test silently finds nothing.
        text_cols = [c for c in df.columns
                     if c != self.id_col and (df[c].dtype == object
                                              or pd.api.types.is_string_dtype(df[c]))]
        if text_cols:
            widths = {c: df[c].astype(str).str.len().mean() for c in text_cols}
            best = max(widths, key=widths.get)
            if widths[best] > 40:
                warnings.warn(
                    f"No known report column; using {best!r} (mean length "
                    f"{widths[best]:.0f}). Pass report_col= to be explicit.")
                return [best]
        raise KeyError(
            f"No report column found in {list(df.columns)}. Pass report_col='<name>'.")

    def label_studies(self, df: pd.DataFrame | None = None,
                      relabel: bool = False) -> pd.DataFrame:
        """Run the labeler over every report. One row per study.

        Returns the original study csv plus the soft labels, their certainties
        and three diagnostics: `report_script`, `report_chars`, and
        `needs_review` (the labeler's own signal that it saw a script its
        vocabulary does not cover -- those rows are all-negative for the wrong
        reason and must not be trained on as negatives).
        """
        if self._studies is not None and df is None and not relabel:
            return self._studies                     # labelling is pure; do it once

        df = self._read_csv(self.study_csv) if df is None else df.copy()
        if self.id_col not in df.columns:
            raise KeyError(f"{self.id_col!r} not in {self.study_csv}: {list(df.columns)}")

        cols = self.resolve_report_columns(df)
        self._log(f"labelling {len(df)} reports from column(s) {cols}")
        reports = (df[cols].fillna("").astype(str)
                   .apply(lambda r: "\n".join(x for x in r if x.strip()), axis=1))
        df["report"] = reports

        rows = []
        for text in reports:
            res = self.labeler.extract(text)
            row: dict[str, Any] = {
                f"{self.label_prefix}{lab}": res[lab].value for lab in self.labels
            }
            if self.include_certainty:
                row.update({f"{lab}_certainty": res[lab].certainty for lab in self.labels})
            row["report_script"] = self.labeler.normalizer.detect_script(text)
            row["report_chars"] = len(text or "")
            row["needs_review"] = any(r.needs_review for r in res.values())
            rows.append(row)

        labelled = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)

        # A study csv that already ships ground-truth columns would be silently
        # overwritten by the labeler's estimates, which is how you end up
        # training on your own predictions. Keep both, renamed.
        dupes = [c for c in labelled.columns if list(labelled.columns).count(c) > 1]
        if dupes:
            warnings.warn(f"duplicate columns after labelling: {sorted(set(dupes))}")

        flagged = int(labelled["needs_review"].sum())
        if flagged:
            self._log(f"  {flagged} report(s) flagged needs_review "
                      f"(unsupported script) -- see builder.review_queue()")
        self._studies = labelled
        return labelled

    def review_queue(self, top: int = 20) -> pd.DataFrame:
        """The reports the labeler could not read, worst first."""
        df = self.study_table()
        bad = df[df["needs_review"]].copy()
        return bad[[self.id_col, "report_script", "report_chars", "report"]].head(top)

    # -- the pixel side ---------------------------------------------------
    def series_index(self) -> pd.DataFrame:
        """study -> series mapping, from the extractor."""
        mapping = self.extractor._getSeriesInstanceUID(self.series_csv)
        return pd.DataFrame(
            [(study, ser) for study, sers in mapping.items() for ser in sers],
            columns=["StudyInstanceUID", "SeriesInstanceUID"],
        )

    def scan(self, studies: Sequence[str] | None = None,
             limit_studies: int | None = None) -> pd.DataFrame:
        """Read every header and build the series metadata table.

        The expensive call. Threaded because it is IO-bound, and cached on the
        builder so `build()` is idempotent.
        """
        index = self.series_index()
        if studies is not None:
            index = index[index["StudyInstanceUID"].isin(set(studies))]
        if limit_studies:
            keep = index["StudyInstanceUID"].drop_duplicates().head(limit_studies)
            index = index[index["StudyInstanceUID"].isin(set(keep))]

        jobs = [(s, ser, os.path.join(self.data_path, self.series_dir, s, ser))
                for s, ser in zip(index["StudyInstanceUID"], index["SeriesInstanceUID"])]
        self._log(f"scanning {len(jobs)} series from "
                  f"{index['StudyInstanceUID'].nunique()} studies "
                  f"({self.workers} threads)")

        rows = []
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for i, row in enumerate(pool.map(lambda j: scan_series(*j), jobs), 1):
                rows.append(row)
                if self.verbose and i % 250 == 0:
                    self._log(f"  {i}/{len(jobs)} series")

        meta = pd.DataFrame(rows)
        self.errors = [r for r in rows if r.get("scan_error") or r.get("missing_files")]
        if self.errors:
            self._log(f"  {len(self.errors)} series with problems -- see builder.errors")
        self._series_meta = meta
        return meta

    # -- the join ---------------------------------------------------------
    def build(self, limit_studies: int | None = None,
              rescan: bool = False) -> pd.DataFrame:
        """One row per series: series metadata + study metadata + labels.

        Left join from the series side, so a study whose folder is missing
        drops out with a warning instead of appearing as a label with no
        pixels behind it.
        """
        studies = self.label_studies()
        meta = (self.scan(studies=studies[self.id_col].tolist(), limit_studies=limit_studies)
                if (self._series_meta is None or rescan or limit_studies) else self._series_meta)

        if self.drop_empty_series:
            n = len(meta)
            meta = meta[meta["n_slices"] > 0].copy()
            if n - len(meta):
                self._log(f"  dropped {n - len(meta)} empty/unreadable series "
                          f"(drop_empty_series=False to keep them)")

        table = meta.merge(studies, on=self.id_col, how="left", suffixes=("", "_study"))

        orphans = int(table["report"].isna().sum()) if "report" in table else 0
        if orphans:
            warnings.warn(f"{orphans} series have no matching row in {self.study_csv}")

        table["n_series_in_study"] = table.groupby(self.id_col)["SeriesInstanceUID"].transform("count")
        # Group by this, never by row, when you split train/val: every series of
        # a study shares one report, so a row-level split leaks the label.
        table["group"] = table[self.id_col]

        front = [c for c in [self.id_col, "SeriesInstanceUID", "n_slices", "plane",
                             "weighting", "series_description", "n_series_in_study"]
                 if c in table.columns]
        rest = [c for c in table.columns if c not in front and c != "paths"]
        self._table = table[front + rest + (["paths"] if "paths" in table else [])]
        self._records = None
        self._log(f"built {len(self._table)} series x {len(self._table.columns)} columns")
        return self._table

    # -- accessors --------------------------------------------------------
    def study_table(self) -> pd.DataFrame:
        """Study-level view: one row per study, labels plus series counts."""
        if self._table is None:
            return self.label_studies()
        agg = (self._table.groupby(self.id_col)
               .agg(n_series=("SeriesInstanceUID", "count"),
                    n_slices=("n_slices", "sum"),
                    planes=("plane", lambda s: sorted(set(s))),
                    weightings=("weighting", lambda s: sorted(set(s))))
               .reset_index())
        studies = self.label_studies()
        return studies.merge(agg, on=self.id_col, how="left")

    @property
    def table(self) -> pd.DataFrame:
        if self._table is None:
            self.build()
        return self._table

    @property
    def records(self) -> list[SeriesRecord]:
        """SeriesRecord per row, labels attached, pixels still on disk."""
        if self._records is None:
            df = self.table
            label_cols = [f"{self.label_prefix}{l}" for l in self.labels]
            meta_cols = [c for c in df.columns if c not in label_cols + ["paths"]]
            self._records = [
                SeriesRecord(
                    study_uid=r[self.id_col],
                    series_uid=r["SeriesInstanceUID"],
                    paths=tuple(r.get("paths") or ()),
                    meta={c: r[c] for c in meta_cols},
                    labels={l: r.get(f"{self.label_prefix}{l}") for l in self.labels},
                    report=r.get("report", "") or "",
                )
                for r in df.to_dict("records")
            ]
        return self._records

    def dataset(self, transform=None, load_pixels: bool = True, **volume_kwargs) -> KneeDataset:
        """Indexable dataset over the records. Extra kwargs go to `volume()`."""
        return KneeDataset(self.records, self.labels, transform, load_pixels, **volume_kwargs)

    # -- cache ------------------------------------------------------------
    def save(self, path: str) -> str:
        """Persist the built table so the next session skips the scan.

        `paths` is a tuple per row, which parquet handles and csv does not, so
        it is JSON-encoded on the way out and decoded on the way back in.
        """
        df = self.table.copy()
        df["paths"] = df["paths"].apply(lambda p: json.dumps(list(p or ())))
        if path.endswith(".parquet"):
            try:
                df.to_parquet(path, index=False)
            except ImportError:                      # no pyarrow/fastparquet here
                path = path[: -len(".parquet")] + ".csv"
                warnings.warn(f"no parquet engine installed; wrote {path} instead")
                df.to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)
        self._log(f"wrote {len(df)} rows to {path}")
        return path

    @classmethod
    def load(cls, path: str, data_path: str = "", **kwargs) -> "KneeDatasetBuilder":
        """Rebuild a builder from a cached table. No scan, no labelling."""
        df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
        df["paths"] = df["paths"].apply(
            lambda s: tuple(json.loads(s)) if isinstance(s, str) else tuple(s or ()))
        obj = cls(data_path or ".", verbose=False, **kwargs)
        obj._table = df
        return obj

    def __repr__(self) -> str:
        n = "unbuilt" if self._table is None else f"{len(self._table)} series"
        return f"KneeDatasetBuilder({self.data_path!r}, split={self.split!r}, {n})"


def build_knee_dataset(data_path: str, **kwargs) -> tuple[pd.DataFrame, KneeDataset]:
    """One-liner for the common case: the table and the dataset.

        df, ds = build_knee_dataset(DATA_PATH)
    """
    builder = KneeDatasetBuilder(data_path, **kwargs)
    return builder.build(), builder.dataset()
