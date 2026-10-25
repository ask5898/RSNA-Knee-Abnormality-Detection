"""Build train_set.csv: one row per training series, with its DICOM paths, slice
count, plane, weighting, pixel spacing and the report labels (notebook cells 7-9).

    python scripts/build_train_set.py                      # all studies (reads every DICOM header)
    python scripts/build_train_set.py --max-studies 50     # a quick subset

Reading every header of the full training set takes a long time (an hour or
more on Kaggle); the result is reused by every training run.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rsna_knee import config as C  # noqa: E402
from rsna_knee.cli import add_common_args, apply_common_args  # noqa: E402
from rsna_knee.data.dicom import build_dataset  # noqa: E402
from rsna_knee.labeling.rules import LABELS  # noqa: E402
from rsna_knee.utils import display  # noqa: E402


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--max-studies", type=int, help="only the first N studies")
    p.add_argument("--out", help="where to write it (default: TRAIN_SET_CSV)")
    args = p.parse_args(argv)
    apply_common_args(args)
    out = args.out or C.TRAIN_SET_CSV

    df = build_dataset(C.DATA_PATH, max_studies=args.max_studies)     # drop max_studies for all of it
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out}: {len(df)} series from {df['study'].nunique()} studies")

    # One row, end to end.
    row = df.iloc[0]
    print(row["study"], row["series"])
    print("slices", row["n_slices"], "| first file", row["paths"][0])   # pixels stay on disk
    print("labels", dict(zip(LABELS, row["labels"].round(2))))
    print("series", row["plane"], row["weighting"], "fat_sat" if row["fat_sat"] else "")

    # Check the labels before training on them: a report in a script the
    # vocabulary cannot read comes out all-negative, which is indistinguishable
    # from a normal knee unless you look.
    print("\nflagged reports:", df["needs_review"].sum())
    display(df[df["needs_review"]][["study", "report_script", "report"]].drop_duplicates("study"))

    # Sagittal fat-sat is what the menisci are read on.
    sag = df[(df["plane"] == "sagittal") & df["fat_sat"]]
    print(len(sag), "sagittal fat-sat series")


if __name__ == "__main__":
    main()
