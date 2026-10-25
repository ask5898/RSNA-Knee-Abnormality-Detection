"""Download the competition data (and optionally your Kaggle datasets or models).

    python scripts/download_data.py                     # everything into data/rsna-knee-abnormality-detection
    python scripts/download_data.py --csv-only          # only the CSVs (reports, official labels, series lists)
    python scripts/download_data.py --list              # what the competition holds, with sizes
    python scripts/download_data.py --dataset ask5898/train-set --dest data/
    python scripts/download_data.py --model ask5898/knee-model-convnext/pytorch/default/1 --dest outputs/knee_model

Sign in first: KAGGLE_API_TOKEN, ~/.kaggle/access_token, ~/.kaggle/kaggle.json or
`kaggle auth login`. Accept the competition rules on Kaggle once before downloading.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rsna_knee import config as C  # noqa: E402
from rsna_knee.data import download as D  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", help=f"target folder (default: the competition data folder, {C.DATA_PATH})")
    what = p.add_mutually_exclusive_group()
    what.add_argument("--csv-only", action="store_true", help="only " + ", ".join(D.CSV_FILES))
    what.add_argument("--files", nargs="+", metavar="NAME", help="only these competition files")
    what.add_argument("--list", action="store_true", help="list the competition files and sizes, download nothing")
    what.add_argument("--dataset", metavar="OWNER/NAME", help="a Kaggle dataset instead of the competition")
    what.add_argument("--model", metavar="HANDLE", help="a Kaggle model version instead of the competition")
    p.add_argument("--max-files", type=int, default=50, help="with --list: stop after this many files (0 = all)")
    p.add_argument("--force", action="store_true", help="download again even if the files are there")
    args = p.parse_args(argv)

    if args.list:
        files = D.list_competition_files(max_files=args.max_files or None)
        for name, size in files:
            print(f"{size / 1e6:10.1f} MB  {name}")
        print(f"{len(files)} files listed, {sum(s for _, s in files) / 1e9:.2f} GB")
        return
    if args.dataset:
        dest = D.download_dataset(args.dataset, args.dest or "data", force=args.force)
    elif args.model:
        dest = D.download_model(args.model, args.dest or C.MODEL_DIR, force=args.force)
    else:
        files = list(D.CSV_FILES) if args.csv_only else args.files
        dest = D.download_competition(args.dest, files=files, force=args.force)
        print(D.summarize(dest))
    print(f"done: {dest}")


if __name__ == "__main__":
    main()
