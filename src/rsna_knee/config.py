"""Every path, switch and setting of the pipeline, in one place.

In the notebook these were globals shared by all cells: cell 0's run settings
and the constants at the top of cells 12, 13 and 14. The package reads them from
here when it runs, so changing one (`config.CROP_MM = 100` in Python, or
`--set CROP_MM=100` on the command line) changes every function that uses it,
as editing the notebook cell did. Values are the notebook's.

Paths default to the Kaggle locations inside a Kaggle notebook and to ./data
and ./outputs elsewhere; environment variables override both.
"""
import ast
import os
import types

from .training.common import TrainConfig

_MISSING = object()

ON_KAGGLE = os.path.exists("/kaggle/input")


def _path(env, kaggle, local):
    return os.environ.get(env, kaggle if ON_KAGGLE else local)


# =============================================================================
# Run settings (notebook cell 0)
# =============================================================================
DATA_PATH = _path("KNEE_DATA_PATH", "/kaggle/input/competitions/rsna-knee-abnormality-detection",
                  "data/rsna-knee-abnormality-detection")

# TRAIN = True   extract ConvNeXt features, train, and save the model to MODEL_DIR.
#                Needs Internet on once, to download the ConvNeXt weights.
# TRAIN = False  load the saved model from MODEL_DIR and only write submission.csv.
#                Works with Internet off: use this for the competition submission.
TRAIN = True
MODEL_DIR = _path("KNEE_MODEL_DIR", "/kaggle/working/knee_model", "outputs/knee_model")
# For the submission, attach the training run's output as an input and point
# MODEL_DIR at it, e.g. "/kaggle/input/<your-dataset>/knee_model".
TRAIN_SET_CSV = _path("KNEE_TRAIN_SET_CSV", "/kaggle/input/datasets/ask5898/train-set/train_set.csv",
                      "data/train_set.csv")
# Train features, reused if training is rerun. To skip the ~1 h extraction in a new
# session, attach a previous training run's output and point FEATURE_CACHE at its
# feature_cache folder, e.g. "/kaggle/input/<that-notebook>/feature_cache".
FEATURE_CACHE = _path("KNEE_FEATURE_CACHE", "/kaggle/working/feature_cache", "outputs/feature_cache")

# Which training runs when TRAIN = True:
#   FINETUNE = True   training/lora.py: ConvNeXt fine-tuned with LoRA adapters, end to end
#                     with the study head (one model; hours on a T4, mostly the first pass
#                     that prepares the slices)
#   FINETUNE = False  training/frozen.py: ConvNeXt frozen, features cached, 5 fold heads
#                     (minutes once the features exist)
# Either way the result is saved to MODEL_DIR in the same format, and predict.py submits it.
FINETUNE = True

# The hidden-test rerun of a submission never trains, whatever TRAIN says above.
if os.environ.get("KAGGLE_IS_COMPETITION_RERUN"):
    TRAIN = False

# =============================================================================
# Frozen-path training and model settings (notebook cell 12)
# =============================================================================
cfg = TrainConfig(
    model_name="convnext_tiny.fb_in22k_ft_in1k",   # 28 M parameters; convnext_small.fb_in22k_ft_in1k is ~2x slower
    n_slices=16, image_size=288,         # any size works; a multiple of 32 avoids edge effects. 224 faster, 384 sharper
    batch_size=8,                        # series per batch while extracting features
    num_workers=os.cpu_count() or 2,     # reading DICOMs is the slow part: use every CPU
    epochs=40, patience=8,               # head training; an epoch now takes seconds
    head_lr=1e-3, weight_decay=1e-4, dropout=0.1,
)
HEAD_BATCH = 32                          # studies per step when training the heads
SLICE_KEEP = 0.75                        # augmentation: each step sees a random 75% of the slices
SERIES_DROP = 0.2                        # augmentation: each series is hidden from the study with this chance
DROP_LOCALIZERS = True                   # scout images: no diagnostic value

# Which slices are embedded. The ends of a knee series are mostly empty or soft
# tissue, so only the middle CENTER_FRAC of each series is sampled.
CENTER_FRAC = 0.7
MAX_SERIES = 6                           # series per study the head looks at

# Each slice is cropped to the central CROP_MM x CROP_MM millimetres (using its
# PixelSpacing) before it is resized to image_size. The public baseline uses 130 mm.
# A slice whose field of view is smaller keeps all of it. None = no crop.
CROP_MM = 130

# What one slice is described by (the last ConvNeXt feature map, 768 channels):
#   "avg"      global average, layer-normed: what the model returns by default
#   "avg+max"  average and maximum, each layer-normed and joined (twice the size)
FEATURE_MODE = "avg+max"

# What the report-derived labels mean for training:
#   "soft"            use them as they are (present 1.0 ... not mentioned 0.2 ... absent 0.0)
#   "drop_uncertain"  present -> 1, absent or not mentioned -> 0, hedged ("possible ...") -> ignored.
#                     A hedged label is a guess in both directions; ignoring it removes that noise.
LABEL_MODE = "drop_uncertain"

# The report labeler (labeling/rules.py) had gaps that put wrong labels into train_set.csv: "Baker's cyst" and
# "medial meniscal tear" were never matched, and "ACL: intact" was read as a tear. It is
# fixed now. True recomputes the 12 labels of TRAIN_SET_CSV from the reports in train.csv
# with it (seconds, no DICOM reading), so train_set.csv does not have to be rebuilt.
RELABEL_FROM_REPORTS = True

# The official labels: radiologists read the images and marked a finding only past a
# severity threshold ("on the fence" counts as negative). Report labels mark any
# mention, so a small effusion is positive in the reports and may be negative here.
# These are the only training labels that mean what the test labels mean.
USE_GOLD = True
GOLD_WEIGHT = 8.0                        # a gold study counts this many times in the loss

# =============================================================================
# LoRA fine-tuning settings (notebook cell 13)
# =============================================================================
LORA_RANK = 8
LORA_ALPHA = 16                  # adapter output is scaled by alpha / rank
LORA_STAGES = (2, 3)             # ConvNeXt stages (0-3) that get adapters; 2 and 3 hold 12 of the 18 blocks
FT_SLICES = 12                   # slices per series; the submission uses the same number
FT_IMAGE = 224                   # pixels per side; the submission uses the same size
FT_EPOCHS = 8
FT_BATCH = 4                     # studies per step; lower it if the GPU runs out of memory
FT_LR_LORA = 2e-4
FT_LR_HEAD = 1e-3
FT_VAL_FOLD = 0
FT_TRAIN_ON_GOLD = False         # True: the official studies train too, weighted GOLD_WEIGHT, and no longer check
FT_TIME_LIMIT_H = 6.0            # no new epoch starts after this many hours; the best epoch is kept
SLICE_CACHE = "/tmp/knee_slice_cache"   # local disk, outside /kaggle/working: not part of the output

# =============================================================================
# Submission (notebook cell 14)
# =============================================================================
SUBMISSION_PATH = "submission.csv"
TEST_SPLIT = "test"              # test.csv, test_series.csv and test_series/ under DATA_PATH


def update(**values):
    """Change settings by name: config.update(CROP_MM=100, FT_EPOCHS=5)."""
    for name, value in values.items():
        current = globals().get(name, _MISSING)
        if (current is _MISSING or name.startswith("_") or callable(current)
                or isinstance(current, types.ModuleType)):
            raise KeyError(f"no setting called {name}")
        globals()[name] = value


def apply_overrides(pairs):
    """["CROP_MM=100", "cfg.image_size=224"] from the command line -> set them.

    Values are Python literals (numbers, strings, True/False/None, tuples);
    anything else is taken as a plain string."""
    for pair in pairs or []:
        name, _, raw = pair.partition("=")
        name = name.strip()
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        if name.startswith("cfg."):
            field = name[4:]
            if not hasattr(cfg, field):
                raise KeyError(f"TrainConfig has no field {field}")
            setattr(cfg, field, value)
        else:
            update(**{name: value})
