"""Run the notebook's code cells in order, as Kaggle would, with test settings.

    python tests/_run_notebook.py NOTEBOOK COMP_DIR WORK_DIR --train --finetune

Runs inside WORK_DIR (the notebook writes train_set.csv and submission.csv to
the current folder)."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _offline  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("notebook")
p.add_argument("comp")
p.add_argument("work")
p.add_argument("--train", action="store_true")
p.add_argument("--finetune", action="store_true")
a = p.parse_args()
nb = json.load(open(a.notebook))
os.makedirs(a.work, exist_ok=True)
os.chdir(a.work)
_offline.patch()


def display(x):
    print(x.to_string() if hasattr(x, "to_string") else x)


overrides = {
    "DATA_PATH = '/kaggle/input/competitions/rsna-knee-abnormality-detection'": f"DATA_PATH = '{a.comp}'",
    "TRAIN = True": f"TRAIN = {a.train}",
    'MODEL_DIR = "/kaggle/working/knee_model"': f'MODEL_DIR = "{a.work}/knee_model"',
    'TRAIN_SET_CSV = "/kaggle/input/datasets/ask5898/train-set/train_set.csv"': f'TRAIN_SET_CSV = "{a.work}/train_set.csv"',
    'FEATURE_CACHE = "/kaggle/working/feature_cache"': f'FEATURE_CACHE = "{a.work}/feature_cache"',
    "BUILD_TRAIN_SET = False": f"BUILD_TRAIN_SET = {a.train}",
    "n_slices=16, image_size=288,": "n_slices=4, image_size=64,",
    "epochs=40, patience=8,": "epochs=6, patience=3,",
    "FT_SLICES = 12": "FT_SLICES = 4", "FT_IMAGE = 224": "FT_IMAGE = 64", "FT_EPOCHS = 8": "FT_EPOCHS = 2",
    "FT_BATCH = 4 ": "FT_BATCH = 2 ",
    'SLICE_CACHE = "/tmp/knee_slice_cache"': f'SLICE_CACHE = "{a.work}/slice_cache"',
    "FINETUNE = True": f"FINETUNE = {a.finetune}",
}
if a.train:
    overrides["CROP_MM = 130"] = "CROP_MM = 15"          # synthetic slices are only ~20 mm wide
namespace = {"display": display, "__name__": "__main__"}
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    for old, new in overrides.items():
        src = src.replace(old, new)
    exec(compile(src, f"cell{i}", "exec"), namespace)
print("notebook run finished")
