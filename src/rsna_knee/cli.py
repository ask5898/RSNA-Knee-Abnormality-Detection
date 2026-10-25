"""Command-line options shared by the scripts: where the data and model live, and
--set for any setting in config.py."""
from . import config as C


def add_common_args(parser):
    parser.add_argument("--data", help=f"competition data folder (default {C.DATA_PATH})")
    parser.add_argument("--train-set", help=f"train_set.csv (default {C.TRAIN_SET_CSV})")
    parser.add_argument("--model-dir", help=f"where the trained model is saved and loaded (default {C.MODEL_DIR})")
    parser.add_argument("--feature-cache", help=f"frozen path: slice-feature cache folder (default {C.FEATURE_CACHE})")
    parser.add_argument("--slice-cache", help=f"LoRA path: prepared-slice cache folder (default {C.SLICE_CACHE})")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="override any setting in config.py, e.g. --set FT_EPOCHS=5 --set cfg.image_size=224")
    return parser


def apply_common_args(args):
    paths = {"DATA_PATH": args.data, "TRAIN_SET_CSV": args.train_set, "MODEL_DIR": args.model_dir,
             "FEATURE_CACHE": args.feature_cache, "SLICE_CACHE": args.slice_cache}
    C.update(**{name: value for name, value in paths.items() if value})
    C.apply_overrides(args.set)
