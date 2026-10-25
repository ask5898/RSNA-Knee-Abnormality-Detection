"""Offline submission (notebook cell 14): load the saved model, predict the test
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

def load_trained(model_dir):
    """Backbone, one head per fold, the saved settings, and a matching config."""
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No saved model in {model_dir}. Train first (scripts/train.py), "
            "or point MODEL_DIR at the knee_model folder of an earlier run.")
    with open(config_path) as f:
        meta = json.load(f)
    if meta.get("feature_mode") not in ("avg", "avg+max") or "crop_mm" not in meta:
        raise ValueError(f"{model_dir} holds a model from the DINOv2 version of this notebook "
                         f"({meta.get('model_name')}). Retrain with this version, or submit that model "
                         "with the DINOv2 notebook.")
    if meta.get("head") != "study":
        raise ValueError(f"{model_dir} was saved by an older version of this notebook (one head per series). "
                         "Retrain with this version, or use the old notebook for that model.")
    if meta["planes"] != PLANES or meta["weightings"] != WEIGHTINGS:
        raise ValueError("PLANES / WEIGHTINGS in training/frozen.py differ from the ones the model was trained with: "
                         f"{meta['planes']}, {meta['weightings']}")
    # The embeddings must be made the way the heads were trained on them
    C.CENTER_FRAC, C.MAX_SERIES, C.FEATURE_MODE = meta["center_frac"], meta["max_series"], meta["feature_mode"]
    C.CROP_MM = meta["crop_mm"]

    run_cfg = TrainConfig(model_name=meta["model_name"], image_size=meta["image_size"],
                          n_slices=meta["n_slices"], in_chans=meta["in_chans"],
                          batch_size=8, num_workers=os.cpu_count() or 2)
    backbone = create_backbone(run_cfg, weights=os.path.join(model_dir, "backbone.pt"))
    heads = []
    for k in meta["folds"]:
        head = StudyHead(meta["dim"], len(meta["labels"]), meta["dropout"])
        head.load_state_dict(torch.load(os.path.join(model_dir, f"head_fold{k}.pt"), map_location="cpu"))
        heads.append(head.eval().to(run_cfg.device))
    print(f"loaded {len(heads)} fold heads from {model_dir} (CV AUC {meta['cv_auc']:.3f}, "
          f"{meta['model_name']}, features '{C.FEATURE_MODE}', crop {C.CROP_MM} mm, "
          f"middle {C.CENTER_FRAC:.0%} of each series)")
    return backbone, heads, meta, run_cfg


def build_test_frame(labels, data_path=None, split=None, drop_localizers=True, workers=8):
    """One row per readable test series, shaped like the training frame.

    build_dataset can't be used here: it reads the reports, and test.csv has none.
    """
    data_path = C.DATA_PATH if data_path is None else data_path
    split = C.TEST_SPLIT if split is None else split
    index = pd.read_csv(os.path.join(data_path, f"{split}_series.csv"))
    jobs = [(s, ser, os.path.join(data_path, f"{split}_series", s, ser))
            for s, ser in zip(index["StudyInstanceUID"], index["SeriesInstanceUID"])]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda j: read_series(*j, load_images=False), jobs))
    frame = pd.DataFrame(rows)

    bad = (frame["error"] != "") | (frame["n_slices"] == 0)
    if bad.any():
        print(f"{bad.sum()} test series could not be read and are left out, "
              f"e.g. {frame.loc[bad, 'error'].iloc[0]}")
    frame = frame[~bad]
    if drop_localizers:          # the model was trained without them
        frame = frame[frame["weighting"] != "localizer"]
    frame = frame.reset_index(drop=True)
    for lab in labels:           # the Dataset reads label columns; test has none
        frame[lab] = np.nan
    print(f"{len(frame)} test series from {frame['study'].nunique()} studies")
    return frame


def submission_columns(data_path=None):
    """(id column, the 12 label columns) as the competition names them."""
    data_path = C.DATA_PATH if data_path is None else data_path
    sample = os.path.join(data_path, "sample_submission.csv")
    if os.path.exists(sample):
        cols = pd.read_csv(sample, nrows=0).columns.tolist()
        return cols[0], cols[1:]
    # train.csv is StudyInstanceUID, Report, then the 12 findings
    cols = pd.read_csv(os.path.join(data_path, "train.csv"), nrows=0).columns.tolist()
    return "StudyInstanceUID", cols[2:]


def predict():
    """Load the model in MODEL_DIR, predict every test study, write SUBMISSION_PATH
    (cell 14's prediction code). Returns the submission DataFrame."""
    # --- predict --------------------------------------------------------------------
    backbone, heads, meta, run_cfg = load_trained(C.MODEL_DIR)
    model_labels = meta["labels"]
    test_frame = build_test_frame(model_labels, drop_localizers=meta["drop_localizers"])

    print(f"embedding {len(test_frame)} test series ...")
    test_feats, test_ok = extract_features(test_frame, backbone, run_cfg)
    test_studies = build_studies(test_frame, test_ok)                  # the readable series of each study
    study_preds = study_probs(heads, test_feats, test_studies, run_cfg.device, model_labels)

    # --- submission.csv: one row per test study ------------------------------------
    id_col, sub_labels = submission_columns()
    mapping = label_mapping(model_labels, sub_labels)
    print("model label -> submission column")
    for m, s in mapping.items():
        print(f"  {m:20s} -> {s}")

    test_ids = pd.read_csv(os.path.join(C.DATA_PATH, f"{C.TEST_SPLIT}.csv"))["StudyInstanceUID"]
    submission = pd.DataFrame({id_col: test_ids.values})
    for m, s in mapping.items():
        submission[s] = submission[id_col].map(study_preds[m])

    # A study with no readable series has no prediction. AUC only ranks, so the
    # column mean is a neutral placeholder rather than a guess.
    cols = list(mapping.values())
    missing = submission[cols].isna().any(axis=1)
    if missing.any():
        print(f"{missing.sum()} test studies had no readable series; filled with the column mean")
        submission[cols] = submission[cols].fillna(submission[cols].mean()).fillna(0.5)

    assert len(submission) == len(test_ids) and not submission.isna().any().any()
    submission.to_csv(C.SUBMISSION_PATH, index=False)
    print(f"wrote {C.SUBMISSION_PATH}: {submission.shape[0]} studies x {submission.shape[1]} columns")
    display(submission.head())
    return submission
