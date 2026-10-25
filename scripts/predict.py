"""Write submission.csv from a trained model, with no Internet (notebook cell 14).

    python scripts/predict.py --model-dir outputs/knee_model --out submission.csv

Reads the test DICOMs from the competition data folder and the model from
MODEL_DIR. The settings the model was trained with (slices, crop, feature type)
come from its config.json, not from config.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from rsna_knee import config as C  # noqa: E402
from rsna_knee.cli import add_common_args, apply_common_args  # noqa: E402


def main(argv=None):
    p = add_common_args(argparse.ArgumentParser(description=__doc__,
                                                formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--out", help=f"submission file (default {C.SUBMISSION_PATH})")
    args = p.parse_args(argv)
    apply_common_args(args)
    if args.out:
        C.update(SUBMISSION_PATH=args.out)

    from rsna_knee.predict import predict
    predict()


if __name__ == "__main__":
    main()
