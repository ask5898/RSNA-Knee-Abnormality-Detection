"""Join the images to the reports and hand back one DataFrame.

    df = build_dataset(DATA_PATH, max_studies=50)

One row per series:

    study     StudyInstanceUID
    series    SeriesInstanceUID
    image     (n_slices, H, W) float array -- the DICOM pixels, in anatomical order
    <12 labels> soft labels parsed from that study's report, plus `labels`,
                the same twelve as one vector in LABELS order

...and the metadata you need to actually use those three: which plane and
weighting the series is, how big a voxel is, which knee, and the study id to
group on when you split.

MEMORY. The `image` column holds decoded pixels, so the frame is as big as the
data it loaded: one knee series is roughly 30 x 512 x 512 x 4 bytes = 31 MB,
and a study has several. The whole training set will not fit in RAM this way.
Three levers, in the order worth reaching for:

    max_studies=50          build on a subset while you develop
    image_dtype=np.uint16   quarter the size, lose the rescale correction
    load_images=False       identical frame, `paths` instead of `image`

The last one is what to use at full scale: keep the frame, read the pixels per
batch from `paths` with `read_volume()`.
"""
import os
from concurrent.futures import ThreadPoolExecutor
from glob import glob
 
import cv2
import numpy as np
import pandas as pd
import pydicom

from ..labeling.rules import LABELS, ClinicalNoteLabeler
 
 
# =============================================================================
# Reading one series
# =============================================================================
def _tag(ds, *names, default=None):
    """First present, non-empty attribute. Vendors disagree on which tag holds what."""
    for n in names:
        v = getattr(ds, n, None)
        if v not in (None, ""):
            return v
    return default
 
 
def _f(v, default=None):
    """DICOM numbers arrive as DSfloat, IS, str, or a 1-element list."""
    if isinstance(v, (list, tuple)) or type(v).__name__ == "MultiValue":
        v = v[0] if len(v) else None
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
 
 
def _plane(ds):
    """'sagittal' | 'coronal' | 'axial' | 'oblique' | 'unknown'.
 
    Knee MRI is routinely prescribed oblique to the ligaments, so a normal that
    no axis clearly dominates is reported as oblique rather than rounded to the
    nearest plane and quietly mislabelled.
    """
    iop = _tag(ds, "ImageOrientationPatient")
    if iop is None or len(iop) != 6:
        return "unknown"
    iop = np.asarray(iop, float)
    n = np.abs(np.cross(iop[:3], iop[3:]))
    if not n.any():
        return "unknown"
    n = n / np.linalg.norm(n)
    axis = int(np.argmax(n))
    return ("sagittal", "coronal", "axial")[axis] if n[axis] >= 0.75 else "oblique"
 
 
def _weighting(description, ds):
    """T1 / T2 / PD / STIR / ... from the series name, falling back to timings.
 
    Knee protocols are named, not tagged: the weighting lives in
    SeriesDescription as free text a technologist typed. A guess, and treated
    as one -- use it to pick sequences, not as a feature.
    """
    t = (description or "").lower()
    for key, name in [("stir", "STIR"), ("t2*", "T2*"), ("pd", "PD"), ("proton", "PD"),
                      ("t1", "T1"), ("t2", "T2"), ("flair", "FLAIR"),
                      ("loc", "localizer"), ("scout", "localizer")]:
        if key in t:
            return name
    tr, te = _f(_tag(ds, "RepetitionTime")), _f(_tag(ds, "EchoTime"))
    if tr is not None and te is not None:
        if tr < 900 and te < 30:
            return "T1"
        if tr >= 2000:
            return "T2" if te >= 60 else "PD"
    return "unknown"
 
 
def order_slices(datasets, paths):
    """Paths sorted through-plane, the way the anatomy stacks.
 
    Position projected on the slice normal first, InstanceNumber second,
    filename last. Only the geometric sort is right for a series acquired
    interleaved; only the fallbacks work for a localizer with no position tags.
    Sorting by filename alone shuffles the stack on any scanner that does not
    zero-pad its exports.
    """
    iop = next((_tag(d, "ImageOrientationPatient") for d in datasets
                if _tag(d, "ImageOrientationPatient") is not None), None)
    if iop is not None and len(iop) == 6:
        iop = np.asarray(iop, float)
        normal = np.cross(iop[:3], iop[3:])
        pos = [_tag(d, "ImagePositionPatient") for d in datasets]
        if all(p is not None and len(p) == 3 for p in pos):
            proj = [float(np.dot(np.asarray(p, float), normal)) for p in pos]
            if len(set(proj)) > 1:
                return [p for _, p in sorted(zip(proj, paths))]
 
    nums = [_f(_tag(d, "InstanceNumber")) for d in datasets]
    if all(n is not None for n in nums) and len(set(nums)) > 1:
        return [p for _, p in sorted(zip(nums, paths))]
    return sorted(paths)
 
 
def read_volume(paths, dtype=np.float16, max_slices=None, size=None, normalize=False):
    """(n_slices, H, W) from ordered paths. Raises only if the slices differ in shape.
 
    max_slices keeps the central N: the knee joint sits mid-stack, and the
    outermost slices are mostly air.
    size=(H, W) resizes each slice as it is read: a 512x512 slice is 4x smaller
    at 256x256, and float16 halves it again.
    normalize scales the volume to 0..1 by its own 0.5th-99.5th percentiles
    (MR intensities have no fixed unit, so this is per volume, not per slice).
    """
    if max_slices and len(paths) > max_slices:
        start = (len(paths) - max_slices) // 2
        paths = paths[start: start + max_slices]
 
    slices = []
    for p in paths:
        ds = pydicom.dcmread(p)
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * _f(_tag(ds, "RescaleSlope"), 1.0) + _f(_tag(ds, "RescaleIntercept"), 0.0)
        if _tag(ds, "PhotometricInterpretation") == "MONOCHROME1":
            arr = arr.max() - arr          # MONOCHROME1 stores bright-is-low
        if size is not None:
            arr = cv2.resize(arr, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        slices.append(arr)
 
    shapes = {s.shape for s in slices}
    if len(shapes) > 1:
        raise ValueError(f"slices differ in shape: {sorted(shapes)}")
    vol = np.stack(slices)
    if normalize:
        lo, hi = np.percentile(vol, [0.5, 99.5])
        vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)
    return vol.astype(dtype)
 
 
def read_series(study, series, series_dir, load_images=False,
                image_dtype=np.float16, max_slices=None, image_size=None):
    """One series -> one row. Never raises: a broken series comes back flagged.
 
    A scan that dies on file 40,000 of 50,000 has cost you the other 49,999, so
    every failure lands in `error` and the run continues.
    """
    row = {"study": study, "series": series, "image": None, "paths": (),
           "n_slices": 0, "n_unreadable": 0, "error": ""}
 
    paths = sorted(glob(os.path.join(series_dir, "*.dcm")))
    if not paths:
        row["error"] = "no .dcm files"
        return row
 
    datasets, kept = [], []
    for p in paths:
        try:
            datasets.append(pydicom.dcmread(p, stop_before_pixels=True))
            kept.append(p)
        except Exception:
            row["n_unreadable"] += 1        # one bad file, not a bad series
    if not datasets:
        row["error"] = f"all {len(paths)} files unreadable"
        return row
 
    by_path = dict(zip(kept, datasets))
    ordered = order_slices(datasets, kept)
    head = by_path[ordered[0]]
 
    # Rows/Columns are in the header, so a series that cannot be stacked is
    # caught whether or not pixels get read -- otherwise load_images=False
    # hands back a row that blows up later, in the middle of training.
    shapes = {(int(_f(_tag(by_path[p], "Rows"), 0)),
               int(_f(_tag(by_path[p], "Columns"), 0))) for p in ordered}
    if len(shapes) > 1:
        row["error"] = f"slices differ in shape: {sorted(shapes)}"
 
    spacing = _tag(head, "PixelSpacing", default=[None, None])
    desc = str(_tag(head, "SeriesDescription", "ProtocolName", default=""))
    age = str(_tag(head, "PatientAge", default=""))
 
    row.update({
        "paths": tuple(ordered),
        "n_slices": len(ordered),
        "n_unreadable": row["n_unreadable"],
        "rows": int(_f(_tag(head, "Rows"), 0)) or None,
        "cols": int(_f(_tag(head, "Columns"), 0)) or None,
        # plane and weighting are how you choose which series to feed a model:
        # menisci are read on sagittal fat-sat, collaterals on coronal.
        "plane": _plane(head),
        "weighting": _weighting(desc, head),
        "fat_sat": any(k in desc.lower() for k in ("fs", "fat sat", "stir", "spair", "spir")),
        "series_description": desc,
        # geometry, so volumes can be resampled to a common physical size
        "pixel_spacing": _f(spacing[0]),
        "slice_thickness": _f(_tag(head, "SliceThickness")),
        "spacing_between_slices": _f(_tag(head, "SpacingBetweenSlices")),
        "laterality": str(_tag(head, "ImageLaterality", "Laterality", default="")),
        "patient_age": _f(age[:-1]) if age[-1:].upper() == "Y" else _f(age),
        "patient_sex": str(_tag(head, "PatientSex", default="")),
        "manufacturer": str(_tag(head, "Manufacturer", default="")),
    })
 
    if load_images and not row["error"]:
        try:
            row["image"] = read_volume(ordered, image_dtype, max_slices, image_size)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
    return row
 
 
# =============================================================================
# Reading the reports
# =============================================================================
REPORT_COLUMNS = ("report", "report_text", "radiology_report", "clinical_notes",
                  "doctor_notes", "notes", "text", "findings", "impression",
                  "conclusion")
 
 
def read_reports(data_path, split="train", report_col=None, labeler=None):
    """train.csv -> one row per study, with the 12 soft labels attached.
 
    The report column is auto-detected as a first-run convenience; pass
    report_col once you know the name. `needs_review` marks reports in a script
    the vocabulary cannot read -- those come out all-negative for the wrong
    reason, and training on them as negatives poisons the label set.
    """
    df = pd.read_csv(os.path.join(data_path, f"{split}.csv"))
    labeler = labeler or ClinicalNoteLabeler()
 
    if report_col:
        cols = [report_col] if isinstance(report_col, str) else list(report_col)
    else:
        lower = {c.lower(): c for c in df.columns}
        cols = [lower[n] for n in REPORT_COLUMNS if n in lower]
    if not cols:
        raise KeyError(f"No report column in {list(df.columns)}. Pass report_col='<name>'.")
 
    text = (df[cols].fillna("").astype(str)
            .apply(lambda r: "\n".join(x for x in r if x.strip()), axis=1))
 
    rows = []
    for t in text:
        res = labeler.extract(t)
        rows.append({
            **{lab: res[lab].value for lab in LABELS},
            "labels": np.array([res[lab].value for lab in LABELS], dtype=np.float32),
            "report": t,
            "report_script": labeler.normalizer.detect_script(t),
            "needs_review": any(r.needs_review for r in res.values()),
        })
 
    out = pd.concat([df[["StudyInstanceUID"]].reset_index(drop=True),
                     pd.DataFrame(rows)], axis=1)
    return out.rename(columns={"StudyInstanceUID": "study"})
 
 
# =============================================================================
# The join
# =============================================================================
def build_dataset(data_path, split="train", *, report_col=None, labeler=None,
                  load_images=False, image_dtype=np.float16, max_slices=None,
                  image_size=None, max_studies=None, workers=8, drop_failed=True,
                  verbose=True):
    """One DataFrame: study, series, paths, labels, and the metadata to use them.
 
    By default no pixels are loaded: each row keeps its slice `paths`, and
    SeriesDataset reads them batch by batch during training. The full training
    set does not fit in a Kaggle session's RAM, and loading it all is what
    killed the kernel.
 
    load_images      True also fills `image` with the pixels, after checking
                     they fit in free RAM. Only for subsets.
    max_studies      build on a subset
    max_slices       keep the central N slices of each series
    image_size       (H, W) to resize slices to when load_images=True
    drop_failed      leave out series that could not be read (they are listed
                     either way when verbose)
    """
    index = pd.read_csv(os.path.join(data_path, f"{split}_series.csv"))
    reports = read_reports(data_path, split, report_col, labeler)
 
    studies = list(dict.fromkeys(index["StudyInstanceUID"]))
    if max_studies:
        studies = studies[:max_studies]
        index = index[index["StudyInstanceUID"].isin(set(studies))]
 
    jobs = [(s, ser, os.path.join(data_path, f"{split}_series", s, ser))
            for s, ser in zip(index["StudyInstanceUID"], index["SeriesInstanceUID"])]
    if verbose:
        print(f"reading {len(jobs)} series from {len(studies)} studies"
              f"{' with pixels' if load_images else ' (metadata only)'} ...")
 
    # Headers only: cheap, and enough to know how much the pixels would cost.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda j: read_series(*j, load_images=False), jobs))
    series = pd.DataFrame(rows)
 
    if load_images:
        ok = series["error"] == ""
        depth = series.loc[ok, "n_slices"]
        if max_slices:
            depth = depth.clip(upper=max_slices)
        h, w = image_size if image_size else (series.loc[ok, "rows"], series.loc[ok, "cols"])
        need = float((depth * h * w).sum()) * np.dtype(image_dtype).itemsize
        try:
            import psutil
            free = psutil.virtual_memory().available
        except ImportError:
            free = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if verbose:
            print(f"  pixels need ~{need / 1e9:.1f} GB, {free / 1e9:.1f} GB of RAM free")
        if need > 0.6 * free:
            raise MemoryError(
                f"Loading pixels needs ~{need / 1e9:.1f} GB but only {free / 1e9:.1f} GB of RAM is free. "
                "Use load_images=False and SeriesDataset, or pass max_studies, max_slices or image_size.")
 
        def load(i):
            try:
                return read_volume(list(series.at[i, "paths"]), image_dtype, max_slices, image_size), ""
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"
 
        todo = series.index[ok]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            loaded = dict(zip(todo, pool.map(load, todo)))
        series["image"] = pd.Series([loaded.get(i, (None, ""))[0] for i in series.index],
                                    index=series.index, dtype=object)
        for i, (_, err) in loaded.items():
            if err:
                series.at[i, "error"] = err
 
    failed = series[series["error"] != ""]
    if verbose and len(failed):
        print(f"  {len(failed)} series had problems:")
        for _, r in failed.head(5).iterrows():
            print(f"    {r['series']}: {r['error']}")
    if drop_failed:
        series = series[(series["error"] == "") & (series["n_slices"] > 0)]
 
    # Left join from the series side: a label with no pixels behind it is not a
    # training row. Labels are per study and broadcast to each of its series.
    df = series.merge(reports, on="study", how="left")
    df["n_series_in_study"] = df.groupby("study")["series"].transform("count")
    # Group by this when you split. Every series of a study shares one report,
    # so a row-level split puts the same label on both sides and flatters CV.
    df["group"] = df["study"]
 
    front = ["study", "series", "image", "labels", *LABELS,
             "n_slices", "rows", "cols", "plane", "weighting", "fat_sat",
             "laterality", "pixel_spacing", "slice_thickness"]
    df = df[[c for c in front if c in df] + [c for c in df.columns if c not in front]]
 
    if verbose:
        mb = df.drop(columns=["image"]).memory_usage(deep=True).sum() / 1e6
        if load_images:
            mb += sum(im.nbytes for im in df["image"] if im is not None) / 1e6
        print(f"{len(df)} series x {len(df.columns)} columns, ~{mb:,.0f} MB in memory")
    return df.reset_index(drop=True)
 
 
# =============================================================================
# Reading pixels during training
# =============================================================================
class SeriesDataset:
    """One series per item, read from disk when a batch asks for it.
 
    Only the DataFrame (paths, labels) stays in memory, so the dataset can be
    any size. Each item is (volume, labels): volume is (depth, H, W) float32 in
    0..1, the same shape for every series so items batch together. depth slices
    are taken evenly across the stack, repeating slices when a series is short.
    Labels are NaN for studies without a report: mask them in the loss.
 
    Works with torch.utils.data.DataLoader as is:
        loader = DataLoader(SeriesDataset(df), batch_size=8, shuffle=True, num_workers=4)
    """
 
    def __init__(self, df, depth=16, size=(256, 256), label_col="labels"):
        self.paths = df["paths"].tolist()
        self.labels = df[label_col].tolist() if label_col in df else [None] * len(df)
        self.depth, self.size = depth, size
 
    def __len__(self):
        return len(self.paths)
 
    def __getitem__(self, i):
        paths = self.paths[i]
        idx = np.linspace(0, len(paths) - 1, self.depth).round().astype(int)
        label = self.labels[i]
        if label is None or np.ndim(label) == 0:     # no report for this study
            label = np.full(len(LABELS), np.nan, np.float32)
        try:
            vol = read_volume([paths[j] for j in idx], np.float32, size=self.size, normalize=True)
        except Exception as exc:                    # a file that fails to decode mid-training
            print(f"could not read {os.path.dirname(paths[0])}: {exc!r}")
            vol = np.zeros((self.depth, *self.size), np.float32)
            label = np.full(len(LABELS), np.nan, np.float32)
        return vol, np.asarray(label, np.float32)
