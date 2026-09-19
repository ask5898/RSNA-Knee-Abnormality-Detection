# -*- coding: utf-8 -*-
"""Command line entry point: build the index without opening a notebook.

    python -m rsna_knee build  --out knee_index.parquet
    python -m rsna_knee labels --out labels.csv        # reports only, no scan
    python -m rsna_knee check                          # what would break

`labels` exists because it is seconds rather than minutes: the report side has
no IO to speak of, so vocabulary work should not wait on a header scan.
"""
from __future__ import annotations

import argparse
import sys

from .builder import KneeDatasetBuilder
from .config import resolve_data_path


def _builder(args) -> KneeDatasetBuilder:
    return KneeDatasetBuilder(
        resolve_data_path(args.data_path),
        split=args.split,
        report_col=args.report_col,
        workers=args.workers,
        verbose=not args.quiet,
    )


def cmd_build(args) -> int:
    b = _builder(args)
    df = b.build(limit_studies=args.limit_studies)
    b.save(args.out)
    print(f"{len(df)} series, {df['StudyInstanceUID'].nunique()} studies, "
          f"{len(b.errors)} problem series")
    return 0


def cmd_labels(args) -> int:
    b = _builder(args)
    df = b.label_studies()
    df.to_csv(args.out, index=False) if args.out.endswith(".csv") else df.to_parquet(args.out)
    flagged = int(df["needs_review"].sum())
    print(f"{len(df)} reports -> {args.out}  ({flagged} need review)")
    print(b.labeler.coverage_report(zip(df["StudyInstanceUID"], df["report"])))
    return 0


def cmd_check(args) -> int:
    """Dry run: resolve the paths, label the reports, scan a couple of studies.

    Cheap enough to run before a long job, and it catches the failures that
    actually happen -- wrong data path, unexpected report column, an image
    directory that is not where the csv says it is.
    """
    b = _builder(args)
    df = b.build(limit_studies=args.limit_studies or 2)
    print(f"data path   {b.data_path}")
    print(f"reports     {len(b.label_studies())} rows, "
          f"column(s) {b.resolve_report_columns(b.label_studies())}")
    print(f"series      {len(df)} scanned, {len(b.errors)} with problems")
    print(f"planes      {sorted(set(df['plane']))}")
    print(f"weightings  {sorted(set(df['weighting']))}")
    for e in b.errors[:5]:
        print(f"  ! {e['SeriesInstanceUID']}: {e['scan_error'] or 'missing'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rsna_knee", description=__doc__.splitlines()[0])
    p.add_argument("--data-path", default=None, help="competition root (default: auto)")
    p.add_argument("--split", default="train", help="train | test")
    p.add_argument("--report-col", default=None, help="report column (default: auto-detect)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit-studies", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="scan, label, join, and cache the table")
    b.add_argument("--out", default="knee_index.parquet")
    b.set_defaults(func=cmd_build)

    l = sub.add_parser("labels", help="label the reports only (no DICOM scan)")
    l.add_argument("--out", default="labels.csv")
    l.set_defaults(func=cmd_labels)

    c = sub.add_parser("check", help="dry run on a couple of studies")
    c.set_defaults(func=cmd_check)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
