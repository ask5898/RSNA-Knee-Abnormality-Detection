"""Builds src/rsna_knee/ from notebooks/rsna-knee-abnormality.ipynb, one module per cell.

This is how the package was made from the notebook (a one-time migration; edit
the package directly from now on, since running this again overwrites the
generated modules). tests/check_equivalence.py confirms the two give identical
results.

Code is copied from the cells as it stands. The only edits, all listed here:
  * settings move to config.py; code reads them as C.NAME (scope-aware rewrite)
  * `if TRAIN ...:` blocks and the prediction block become functions
  * defaults that captured a setting at definition time resolve it at call time
  * explicit imports replace the notebook's shared namespace
  * cell 11 is split: shared utilities (common.py) vs the unused DINOv2 trainer (legacy/)
  * cell 12's pick_slices override patches common.pick_slices (the notebook did it by
    redefining the global that cell 11's dataset looks up)
"""
import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from rewrite import rewrite  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB = os.path.join(REPO, "notebooks", "rsna-knee-abnormality.ipynb")
OUT = sys.argv[1] if len(sys.argv) > 1 else REPO
PKG = os.path.join(OUT, "src", "rsna_knee")
cells = ["".join(c["source"]) for c in json.load(open(NB))["cells"]]

SETTINGS = {"DATA_PATH", "TRAIN", "MODEL_DIR", "TRAIN_SET_CSV", "FEATURE_CACHE", "FINETUNE",
            "cfg", "HEAD_BATCH", "SLICE_KEEP", "SERIES_DROP", "DROP_LOCALIZERS", "CENTER_FRAC", "MAX_SERIES",
            "CROP_MM", "FEATURE_MODE", "LABEL_MODE", "RELABEL_FROM_REPORTS", "USE_GOLD", "GOLD_WEIGHT",
            "LORA_RANK", "LORA_ALPHA", "LORA_STAGES", "FT_SLICES", "FT_IMAGE", "FT_EPOCHS", "FT_BATCH",
            "FT_LR_LORA", "FT_LR_HEAD", "FT_VAL_FOLD", "FT_TRAIN_ON_GOLD", "FT_TIME_LIMIT_H", "SLICE_CACHE",
            "SUBMISSION_PATH", "TEST_SPLIT"}


def rep(s, old, new, count=1):
    assert s.count(old) == count, ("expected", count, "match(es), found", s.count(old), old[:100])
    return s.replace(old, new)


def cut(s, start, end_before):
    """Remove s[start : end_before) and return (rest, removed)."""
    a = s.index(start)
    b = s.index(end_before, a)
    return s[:a] + s[b:], s[a:b]


def write(rel, text):
    while "\n\n\n\n" in text:                      # at most two blank lines, as PEP 8 has it
        text = text.replace("\n\n\n\n", "\n\n\n")
    path = os.path.join(OUT, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text.rstrip("\n") + "\n")
    print("wrote", rel)


def indent(block):
    return "\n".join(("    " + l) if l.strip() else "" for l in block.rstrip("\n").split("\n"))


# =============================================================================
# cell 3 -> labeling/rules.py (verbatim; stdlib only)
# =============================================================================
write("src/rsna_knee/labeling/rules.py", cells[3])

# =============================================================================
# cell 1 -> labeling/llm.py
# =============================================================================
c1 = cells[1]
c1 = rep(c1, """LABELS = pd.read_csv(os.path.join(DATA_PATH, 'train.csv')).columns[2:].to_list()
assert 'Report' not in LABELS and len(LABELS) == 12, LABELS
""", """def competition_labels(data_path=None):
    \"\"\"The 12 finding columns of train.csv, as the competition names them.

    In the notebook this was cell 1's global LABELS, read when the cell ran (and
    later replaced by cell 3's labeler names); the reader keeps its own copy.\"\"\"
    data_path = C.DATA_PATH if data_path is None else data_path
    labels = pd.read_csv(os.path.join(data_path, 'train.csv')).columns[2:].to_list()
    assert 'Report' not in labels and len(labels) == 12, labels
    return labels
""")
c1 = rep(c1, """        self.batch_size = batch_size    # reports per generate call; lower it if the GPU runs out of memory
        self.tokenizer""", """        self.batch_size = batch_size    # reports per generate call; lower it if the GPU runs out of memory
        self.labels = competition_labels()
        from transformers import AutoModelForCausalLM, AutoTokenizer   # optional dependency, only for this reader
        self.tokenizer""")
c1 = rep(c1, "report = pd.read_csv(os.path.join(DATA_PATH, str(self.report) + '.csv'))",
         "report = pd.read_csv(os.path.join(C.DATA_PATH, str(self.report) + '.csv'))")
c1 = rep(c1, """    def build_system_prompt(self, labels=LABELS) -> str:
        lines = [""", """    def build_system_prompt(self, labels=None) -> str:
        labels = self.labels if labels is None else labels
        lines = [""")
c1 = rep(c1, "            for label in LABELS:\n", "            for label in self.labels:\n")
c1 = rep(c1, "        return pd.DataFrame(rows, columns=[id_col] + LABELS)", "        return pd.DataFrame(rows, columns=[id_col] + self.labels)")
write("src/rsna_knee/labeling/llm.py", '''"""LLM report reader: Qwen3-8B reads each report into 12 labels (notebook cell 1).

Run it with scripts/label_reports_llm.py (notebook cell 2). Needs `transformers`
and a GPU. Its output (llm_labels.csv) is not used by the training pipeline,
which trains on the rule-based labels of rules.py.
"""
import json
import os
import re

import numpy as np
import pandas as pd
import torch

from .. import config as C

''' + c1)

# =============================================================================
# cell 6 -> data/dicom.py ; cell 10 -> data/splits.py (verbatim + imports)
# =============================================================================
c6 = cells[6]
a = c6.index("import os\n")
write("src/rsna_knee/data/dicom.py", c6[:a] + c6[a:].replace(
    "import pydicom\n", "import pydicom\n\nfrom ..labeling.rules import LABELS, ClinicalNoteLabeler\n", 1))

c10 = cells[10]
tree = ast.parse(c10)
last_import = max(n.end_lineno for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
lines = c10.split("\n")
lines.insert(last_import, "\nfrom ..labeling.rules import LABELS")
write("src/rsna_knee/data/splits.py", rep("\n".join(lines), "from dataclasses import dataclass\n", ""))

# =============================================================================
# cell 11 -> training/common.py + legacy/dinov2_end_to_end.py
# =============================================================================
LEGACY = {"load_backbone", "_fold_patch_embed", "_folded_conv", "DinoV2MultiLabel", "run_epoch",
          "make_loader", "train_fold", "cross_validate", "predict"}
c11 = cells[11]
t11 = ast.parse(c11)
L11 = c11.split("\n")
doc_node, body = t11.body[0], t11.body[1:]
docstring = "\n".join(L11[doc_node.lineno - 1:doc_node.end_lineno])
common_parts, legacy_parts = [], []
prev_end = doc_node.end_lineno
for node in body:
    start = prev_end                       # 0-based index of the first line after the previous node
    chunk = "\n".join(L11[start:node.end_lineno])
    name = getattr(node, "name", None)
    (legacy_parts if name in LEGACY else common_parts).append(chunk)
    prev_end = node.end_lineno
tail = "\n".join(L11[prev_end:])
common_src = "\n".join(common_parts) + tail
common_src = common_src.replace("from torch.utils.data import DataLoader, Dataset\n",
                                "from torch.utils.data import Dataset\n\n"
                                "from ..data.dicom import read_volume\n", 1)
for unused in ("import math\n", "import os\n", "import time\n", "import pandas as pd\n"):
    common_src = rep(common_src, unused, "")
write("src/rsna_knee/training/common.py", '''"""Shared training pieces (notebook cell 11): TrainConfig, the volume dataset,
slice sampling and normalisation, the attention pool, metrics and the loss.

Cell 11 also held the first, end-to-end DINOv2 trainer; nothing uses it any
more, and it lives on in rsna_knee/legacy/dinov2_end_to_end.py.
"""''' + "\n" + common_src)
write("src/rsna_knee/legacy/dinov2_end_to_end.py", docstring + '''
# The first training approach of the notebook (cell 11): DINOv2 fine-tuned end
# to end on full DICOM reads. Superseded by training/frozen.py and training/lora.py
# and not used by the pipeline; kept for reference.
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..data.splits import fold_frames
from ..labeling.rules import LABELS
from ..training.common import (AttentionPool, KneeVolumeDataset, TrainConfig, free_memory, masked_bce,
                               multilabel_metrics, seed_everything)

''' + "\n".join(legacy_parts))

# cell 0's DICOMExtractor (unused by the pipeline)
c0 = cells[0]
a = c0.index("class DICOMExtractor")
write("src/rsna_knee/legacy/dicom_extractor.py", '''"""The notebook's first DICOM reader (cell 0). Not used by the pipeline:
data/dicom.py replaced it. Kept for reference."""
import os
from collections import defaultdict
from glob import glob

import pandas as pd
import pydicom


''' + c0[a:])

# =============================================================================
# cell 12 -> training/frozen.py
# =============================================================================
c12 = cells[12]
c12, settings12 = cut(c12, "cfg = TrainConfig(", "def autocast(device):")
settings12 = rep(settings12, "# Cell 3's labeler had gaps", "# The report labeler (labeling/rules.py) had gaps")
settings12 = rep(settings12, "read as a tear. Cell 3\n# is fixed now.", "read as a tear. It is\n# fixed now.")
c12 = rep(c12, """_pick_slices_all = globals().get("_pick_slices_all", pick_slices)
""", """_pick_slices_all = getattr(_common, "_pick_slices_all", _common.pick_slices)
_common._pick_slices_all = _pick_slices_all
""")
c12 = rep(c12, """    return [lo + j for j in _pick_slices_all(keep, n_wanted, jitter, rng)]
""", """    return [lo + j for j in _pick_slices_all(keep, n_wanted, jitter, rng)]


_common.pick_slices = pick_slices     # what cell 11's dataset calls, as redefining it in the notebook did
""")
c12 = rep(c12, '''def load_gold(data_path=DATA_PATH):
    """The official labels in train.csv, renamed to LABELS and made 0/1.

    One row per study that has them; a missing cell stays NaN. Every other
    training study has only its report."""
''', '''def load_gold(data_path=None):
    """The official labels in train.csv, renamed to LABELS and made 0/1.

    One row per study that has them; a missing cell stays NaN. Every other
    training study has only its report."""
    data_path = DATA_PATH if data_path is None else data_path
''')
c12 = rep(c12, """# The frozen path: features once, then 5 fold heads. With FINETUNE = True (cell 0)
# the next cell trains instead, and this block is skipped.
if TRAIN and not FINETUNE:
""", '''def train_frozen():
    """The frozen path (cell 12's training block): features once, then 5 fold
    heads, saved to MODEL_DIR. scripts/train.py --method frozen runs it."""
''')
c12, n12, _ = rewrite(c12, SETTINGS)
c12 = rep(c12, "import contextlib\nimport json\nimport re\nimport time\n\nimport torch.nn.functional as F\n\nimport timm\n", "")
c12 = rep(c12, "\ndef autocast(device):", "\n\n\ndef autocast(device):")
c12 = rep(c12, "# With TRAIN = True this cell saves everything the submission needs to MODEL_DIR:",
          "# train_frozen() saves everything the submission needs to MODEL_DIR:")
c12 = rep(c12, "# is kept (once) so that running this cell again does not narrow the window twice.",
          "# is kept (once) so that importing this module again does not narrow the window twice.")
c12 = c12.replace("# Fast training: ConvNeXt features computed once, then only the heads train",
                  "# Fast training: ConvNeXt features computed once, then only the heads train (notebook cell 12)", 1)
c12 = c12.replace("# With TRAIN = False it only defines the functions; cell 14 loads the saved model.\n"
                  "# With FINETUNE = True (cell 0) its training block is skipped and cell 13 trains instead.",
                  "# train_frozen() runs this path; train_lora() in lora.py is the alternative, and\n"
                  "# predict.py loads what either saves.", 1)
head12 = '''"""Frozen-backbone training (notebook cell 12), and the pieces both training paths
and the submission share: slice selection, the backbone, features, the study
head, labels and the official labels."""
import contextlib
import json
import math
import os
import re
import time

import numpy as np
import pandas as pd
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import config as C
from ..data.dicom import read_reports
from ..data.splits import make_splits
from ..labeling.rules import LABELS
from ..utils import display
from . import common as _common
from .common import AttentionPool, KneeVolumeDataset, free_memory, multilabel_metrics, seed_everything

'''
write("src/rsna_knee/training/frozen.py", head12 + c12)

# =============================================================================
# cell 13 -> training/lora.py
# =============================================================================
c13 = cells[13]
c13, settings13 = cut(c13, "LORA_RANK = 8", "class LoRALinear(nn.Module):")
c13 = rep(c13, "def add_lora(backbone, stages=LORA_STAGES, rank=LORA_RANK, alpha=LORA_ALPHA):\n"
               '    """Wrap fc1 and fc2 of every block in `stages`. Returns the new trainable parameters."""\n',
               "def add_lora(backbone, stages=None, rank=None, alpha=None):\n"
               '    """Wrap fc1 and fc2 of every block in `stages`. Returns the new trainable parameters."""\n'
               "    stages = LORA_STAGES if stages is None else stages\n"
               "    rank = LORA_RANK if rank is None else rank\n"
               "    alpha = LORA_ALPHA if alpha is None else alpha\n")
c13 = rep(c13, "if TRAIN and FINETUNE:\n", '''def train_lora():
    """The LoRA path (cell 13's training block): one fine-tuned model, saved to
    MODEL_DIR at every new best epoch. scripts/train.py --method lora runs it."""
''')
c13, n13, _ = rewrite(c13, SETTINGS)
c13 = rep(c13, "import shutil\n\nclass LoRALinear(nn.Module):", "\n\nclass LoRALinear(nn.Module):")
c13 = c13.replace("# Fine-tuning: ConvNeXt-Tiny with LoRA adapters, trained end to end with the study head",
                  "# Fine-tuning: ConvNeXt-Tiny with LoRA adapters, trained end to end with the study head (cell 13)", 1)
c13 = c13.replace("# head_fold0.pt and config.json. Cell 14 loads them exactly like the frozen model:",
                  "# head_fold0.pt and config.json. predict.py loads them exactly like the frozen model:", 1)
c13 = c13.replace("#   -> StudyHead (cell 12) -> 12 logits -> loss", "#   -> StudyHead (frozen.py) -> 12 logits -> loss", 1)
c13 = c13.replace("# every other study trains, the 15% hold-out of cell 12 included.",
                  "# every other study trains, the 15% hold-out of the frozen path included.", 1)
head13 = '''"""LoRA fine-tuning of ConvNeXt-Tiny with the study head (notebook cell 13)."""
import json
import math
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .. import config as C
from ..labeling.rules import LABELS
from ..utils import display
from .common import GRAY_MEAN, GRAY_STD, TrainConfig, free_memory, multilabel_metrics, seed_everything
from .frozen import (PLANES, WEIGHTINGS, SafeVolumeDataset, StudyHead, autocast, build_studies, clean_labels,
                     create_backbone, embed, feature_dim, prepare_training_frame, weighted_bce)

'''
write("src/rsna_knee/training/lora.py", head13 + c13)

# =============================================================================
# cell 14 -> predict.py
# =============================================================================
c14 = cells[14]
c14 = rep(c14, '''SUBMISSION_PATH = "submission.csv"
TEST_SPLIT = "test"              # test.csv, test_series.csv and test_series/ under DATA_PATH
''', "")
c14 = rep(c14, '''def build_test_frame(labels, data_path=DATA_PATH, split=TEST_SPLIT, drop_localizers=True, workers=8):
    """One row per readable test series, shaped like the training frame.

    build_dataset can't be used here: it reads the reports, and test.csv has none.
    """
''', '''def build_test_frame(labels, data_path=None, split=None, drop_localizers=True, workers=8):
    """One row per readable test series, shaped like the training frame.

    build_dataset can't be used here: it reads the reports, and test.csv has none.
    """
    data_path = DATA_PATH if data_path is None else data_path
    split = TEST_SPLIT if split is None else split
''')
c14 = rep(c14, '''def submission_columns(data_path=DATA_PATH):
    """(id column, the 12 label columns) as the competition names them."""
''', '''def submission_columns(data_path=None):
    """(id column, the 12 label columns) as the competition names them."""
    data_path = DATA_PATH if data_path is None else data_path
''')
a = c14.index("# --- predict")
c14 = c14[:a] + '''def predict():
    """Load the model in MODEL_DIR, predict every test study, write SUBMISSION_PATH
    (cell 14's prediction code). Returns the submission DataFrame."""
''' + indent(c14[a:]) + "\n    return submission\n"
c14 = rep(c14, '"No saved model in {model_dir}. Run the notebook once with TRAIN = True, "',
          '"No saved model in {model_dir}. Train first (scripts/train.py), "')
c14 = rep(c14, '"or attach that run\'s output and set MODEL_DIR to its knee_model folder.")',
          '"or point MODEL_DIR at the knee_model folder of an earlier run.")')
c14, n14, _ = rewrite(c14, SETTINGS)       # sees `global ...` first: those assignments become C.X = ...
c14 = rep(c14, "    global CENTER_FRAC, MAX_SERIES, FEATURE_MODE, CROP_MM\n", "")
c14 = c14.replace("PLANES / WEIGHTINGS in cell 12 differ", "PLANES / WEIGHTINGS in training/frozen.py differ", 1)
head14 = '''"""Offline submission (notebook cell 14): load the saved model, predict the test
studies, write submission.csv. Needs no Internet: the backbone and heads come
from MODEL_DIR, the DICOMs from DATA_PATH. Nothing is trained."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch

from . import config as C
from .data.dicom import read_series
from .training.common import TrainConfig
from .training.frozen import (PLANES, WEIGHTINGS, StudyHead, build_studies, create_backbone, extract_features,
                              label_mapping, study_probs)
from .utils import display

'''
c14 = rep(c14, "from glob import glob            # same import as cell 0; `import glob` would break read_series\n", "")
c14 = rep(c14, c14[:c14.index("def load_trained")], "")
write("src/rsna_knee/predict.py", head14 + c14)

# =============================================================================
# config.py: cell 0's run settings + the settings blocks of cells 12, 13, 14
# =============================================================================
config = '''"""Every path, switch and setting of the pipeline, in one place.

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
''' + settings12.rstrip() + '''

# =============================================================================
# LoRA fine-tuning settings (notebook cell 13)
# =============================================================================
''' + settings13.rstrip() + '''

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
'''
write("src/rsna_knee/config.py", config)

write("src/rsna_knee/utils.py", '''"""Small helpers the notebook got from Jupyter."""


def display(obj):
    """Jupyter's display() inside a notebook; a plain print anywhere else."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            from IPython.display import display as _display
            return _display(obj)
    except ImportError:
        pass
    print(obj.to_string() if hasattr(obj, "to_string") else obj)
''')

for pkg in ["", "labeling", "data", "training", "legacy"]:
    path = os.path.join(PKG, pkg, "__init__.py")
    if not os.path.exists(path):
        write(os.path.relpath(path, OUT), {
            "": '"""RSNA Knee Abnormality Detection: report labels, ConvNeXt features or LoRA\nfine-tuning, a study-level head, and an offline Kaggle submission."""\n__version__ = "1.0.0"',
            "labeling": '"""Training labels from the radiology reports: rule-based (rules.py) and LLM (llm.py)."""',
            "data": '"""Getting the data (download.py), reading DICOMs and reports (dicom.py), splits (splits.py)."""',
            "training": '"""Training: shared pieces (common.py), frozen backbone (frozen.py), LoRA (lora.py)."""',
            "legacy": '"""Earlier code from the notebook that the pipeline no longer uses. Kept for reference."""',
        }[pkg])

print(f"rewritten references: frozen {n12}, lora {n13}, predict {n14}")
