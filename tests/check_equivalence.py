"""The repo and the notebook produce the same results: same train_set.csv, same
models, same submission.csv, for both training paths.

    python tests/check_equivalence.py            # about 2 minutes on a CPU

Builds a small synthetic competition folder, then for the frozen path and the
LoRA path runs (1) the notebook, cell by cell, and (2) the package scripts, each
in its own process with the same settings, and compares what they write.
"""
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
NOTEBOOK = os.path.join(REPO, "notebooks", "rsna-knee-abnormality.ipynb")
sys.path.insert(0, HERE)
from make_synthetic_data import make  # noqa: E402

TEST_SETTINGS = ["cfg.n_slices=4", "cfg.image_size=64", "cfg.epochs=6", "cfg.patience=3",
                 "FT_SLICES=4", "FT_IMAGE=64", "FT_EPOCHS=2", "FT_BATCH=2"]


def run(args, cwd, log):
    with open(log, "w") as f:
        r = subprocess.run([sys.executable] + args, cwd=cwd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"failed: {' '.join(args)}\n--- last lines of {log}:\n" + "".join(open(log).readlines()[-25:]))


def script(name, *args, work, log):
    run([os.path.join(HERE, "_run_script.py"), os.path.join(REPO, "scripts", name), *args], work, log)


def package_run(comp, work, method):
    os.makedirs(work, exist_ok=True)
    common = ["--data", comp, "--train-set", f"{work}/train_set.csv", "--model-dir", f"{work}/knee_model",
              "--feature-cache", f"{work}/feature_cache", "--slice-cache", f"{work}/slice_cache"]
    sets = [x for s in TEST_SETTINGS for x in ("--set", s)]
    script("build_train_set.py", *common, work=work, log=f"{work}/build.log")
    script("train.py", "--method", method, *common, *sets, "--set", "CROP_MM=15", work=work, log=f"{work}/train.log")
    # a fresh process with the default settings: the model's config.json must bring back what it was trained with
    script("predict.py", *common, "--out", f"{work}/submission.csv", work=work, log=f"{work}/predict.log")


def notebook_run(comp, work, finetune):
    args = [os.path.join(HERE, "_run_notebook.py"), NOTEBOOK, comp, work, "--train"] + (["--finetune"] if finetune else [])
    os.makedirs(work, exist_ok=True)
    run(args, work, f"{work}/notebook.log")


def compare_csv(a, b, what, numeric_from=0):
    x, y = pd.read_csv(a), pd.read_csv(b)
    assert list(x.columns) == list(y.columns), (what, "columns differ")
    assert len(x) == len(y), (what, "row counts differ")
    num = x.columns[numeric_from:]
    num = [c for c in num if pd.api.types.is_numeric_dtype(x[c])]
    other = [c for c in x.columns if c not in num]
    assert x[other].astype(str).equals(y[other].astype(str)), (what, "text columns differ")
    diff = float(np.nanmax(np.abs(x[num].to_numpy(float) - y[num].to_numpy(float)))) if num else 0.0
    print(f"  {what:34s} {len(x):4d} rows, max difference {diff:.1e}")
    return diff


def main():
    root = tempfile.mkdtemp(prefix="knee_equiv_")
    comp = make(os.path.join(root, "comp"))
    worst = 0.0
    try:
        for method in ("frozen", "lora"):
            print(f"{method} path: running the notebook and the package ...")
            nb_dir, pkg_dir = os.path.join(root, f"notebook_{method}"), os.path.join(root, f"package_{method}")
            notebook_run(comp, nb_dir, finetune=method == "lora")
            package_run(comp, pkg_dir, method)
            worst = max(worst, compare_csv(f"{nb_dir}/train_set.csv", f"{pkg_dir}/train_set.csv", "train_set.csv"))
            worst = max(worst, compare_csv(f"{nb_dir}/submission.csv", f"{pkg_dir}/submission.csv",
                                           f"submission.csv ({method})"))
        assert worst < 1e-6, f"the package differs from the notebook by {worst}"
        print(f"OK: the package reproduces the notebook (largest difference {worst:.1e})")
    finally:
        if os.environ.get("KEEP_WORKDIR"):
            print("work files kept in", root)
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
