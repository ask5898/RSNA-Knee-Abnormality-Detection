"""Test / train / validation splits, grouped by study.

    df = make_splits(df, n_folds=5, test_frac=0.15)
    df[["study", "series", "split", "fold"]]

THE ONE RULE. A study's series all share one report, so they all carry the
same twelve labels. Split rows and the same label lands on both sides of the
boundary: the model sees a sagittal slice of a knee in training and a coronal
slice of the *same* knee in validation, and your CV score measures memory, not
generalisation. So studies are split, never rows, and the row assignment is
joined back afterwards -- which makes leakage structurally impossible rather
than something to remember to check.

STRATIFICATION. Eight of the twelve findings are rare, and with a few hundred
studies a plain random split will hand some fold zero positives for a label,
which makes that label's AUC undefined and the fold means incomparable. Each
study is therefore keyed by the *rarest* finding it has, and that key is
balanced across the splits. It is a heuristic -- true multi-label
stratification needs iterative stratification (`iterstrat`) -- but it is the
one that protects the labels most at risk, which is what actually breaks.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from ..labeling.rules import LABELS


def study_strata(df, labels=None, threshold=0.5, study_col="study"):
    """One stratification key per study: the rarest finding it is positive for.

    Returns a frame of (study, stratum). "none" for a study with no positive
    finding, which is itself a class worth balancing -- normal knees are a
    large fraction of the set.
    """
    labels = list(labels if labels is not None else LABELS)
    studies = df.drop_duplicates(study_col).set_index(study_col)[labels]
    positive = studies > threshold

    # Rarest first, so the label most likely to vanish from a fold is the one
    # that decides the key.
    prevalence = positive.mean().sort_values()
    order = list(prevalence.index)

    def key(row):
        for lab in order:                 # rarest -> commonest
            if row[lab]:
                return lab
        return "none"

    return pd.DataFrame({
        study_col: studies.index,
        "stratum": positive.apply(key, axis=1).values,
    })


def _collapse_rare(strata, min_count):
    """Fold strata too small to split into one 'rare' bucket.

    StratifiedKFold warns (and stratifies badly) when a class has fewer members
    than folds. Merging the small ones keeps the split working and still
    separates those studies from the rest -- but the merged bucket can itself
    still be too small, so it is then folded into the largest class rather than
    left to trip the same warning.
    """
    strata = strata.copy()
    counts = strata["stratum"].value_counts()
    small = set(counts[counts < min_count].index)
    if not small:
        return strata

    strata["stratum"] = strata["stratum"].where(~strata["stratum"].isin(small), "rare")
    counts = strata["stratum"].value_counts()
    if counts.get("rare", 0) < min_count and len(counts) > 1:
        biggest = counts.drop(index="rare").idxmax()
        strata["stratum"] = strata["stratum"].replace("rare", biggest)
    return strata


def make_splits(df, n_folds=5, test_frac=0.15, seed=42, labels=None,
                threshold=0.5, drop_needs_review=True, study_col="study",
                verbose=True):
    """Add `split` ('train'/'test') and `fold` (0..n_folds-1, -1 for test).

    Validation is fold k of the training rows: it is not a third fixed set,
    because with a few hundred studies a fixed validation set is small enough
    that model selection on it is mostly noise. Cross-validation reuses every
    study for validation exactly once.

    drop_needs_review removes studies whose report the labeler could not read.
    Their labels are all-negative for the wrong reason, so training on them
    teaches the model that an abnormal knee is normal. Set False to keep them.
    """
    labels = list(labels if labels is not None else LABELS)
    out = df.copy()

    if drop_needs_review and "needs_review" in out.columns:
        flagged = out["needs_review"].fillna(False).astype(bool)
        if verbose and flagged.any():
            print(f"dropping {flagged.sum()} rows "
                  f"({out.loc[flagged, study_col].nunique()} studies) flagged needs_review")
        out = out[~flagged]

    strata = study_strata(out, labels, threshold, study_col)

    # --- hold out the test studies ---------------------------------------
    n_test_splits = max(2, int(round(1 / test_frac)))
    s = _collapse_rare(strata, n_test_splits)
    splitter = StratifiedKFold(n_splits=n_test_splits, shuffle=True, random_state=seed)
    rest_idx, test_idx = next(splitter.split(s, s["stratum"]))
    test_studies = set(s.iloc[test_idx][study_col])
    rest = s.iloc[rest_idx].reset_index(drop=True)

    # --- fold the remainder ----------------------------------------------
    rest = _collapse_rare(rest, n_folds)
    folder = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_of = {}
    for k, (_, val_idx) in enumerate(folder.split(rest, rest["stratum"])):
        for study in rest.iloc[val_idx][study_col]:
            fold_of[study] = k

    out["split"] = np.where(out[study_col].isin(test_studies), "test", "train")
    out["fold"] = out[study_col].map(fold_of).fillna(-1).astype(int)

    # Structural guarantee, asserted rather than assumed.
    overlap = set(out.loc[out["split"] == "test", study_col]) & \
              set(out.loc[out["split"] == "train", study_col])
    assert not overlap, f"study in both splits: {sorted(overlap)[:3]}"

    if verbose:
        print(f"test  {(out['split'] == 'test').sum():5d} series "
              f"({out.loc[out['split'] == 'test', study_col].nunique()} studies)")
        for k in range(n_folds):
            rows = out[(out["split"] == "train") & (out["fold"] == k)]
            print(f"fold {k} {len(rows):5d} series ({rows[study_col].nunique()} studies) "
                  f"-- validation when training fold {k}")
    return out


def fold_frames(df, fold):
    """(train, val) for one fold. Test stays out of both."""
    train_rows = df[(df["split"] == "train") & (df["fold"] != fold)]
    val_rows = df[(df["split"] == "train") & (df["fold"] == fold)]
    return train_rows.reset_index(drop=True), val_rows.reset_index(drop=True)
