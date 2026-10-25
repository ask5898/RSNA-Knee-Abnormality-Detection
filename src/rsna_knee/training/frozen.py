"""Frozen-backbone training (notebook cell 12), and the pieces both training paths
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

# ============================================================================
# Fast training: ConvNeXt features computed once, then only the heads train (notebook cell 12)
# ============================================================================
# The backbone is a ConvNeXt-Tiny pretrained on ImageNet-22k (timm), frozen, with
# its first convolution folded to one grey input channel. Because it is frozen,
# it returns the same slice embeddings every epoch.
# Reading every DICOM again each epoch is what made training take ~30 minutes
# per epoch. Here each series is read and embedded once, the embeddings are
# cached, and the heads train on them in seconds per epoch.
#
# What changed from the first version (CV/leaderboard AUC was ~0.6):
#   * The head works on a whole study, not on one series at a time. It pools the
#     slices of each series, adds an embedding of the series' plane and weighting
#     (sagittal PD, coronal T1, ...), and lets every finding attend to the series
#     that show it. Averaging all series equally diluted the signal.
#   * Only the middle of each series is sampled (CENTER_FRAC); the ends are air.
#   * Each slice is cropped to the central CROP_MM millimetres before resizing, so
#     every scanner shows the knee at the same physical scale. A CNN's filters are
#     tuned to a scale; a 160 mm and a 120 mm field of view resized to the same
#     pixels would show the same meniscus at different sizes.
#   * Each slice is described by the average and the maximum of the last feature
#     map (FEATURE_MODE): the average says what the slice looks like overall, the
#     maximum keeps a small bright spot (a tear, an oedema) that averaging dilutes.
#   * LABEL_MODE drops hedged report labels instead of training on 0.5-0.8 guesses.
#   * A per-finding table at the end shows where the model works and where not.
#   * The official labels in train.csv (58 studies, read from the images by
#     radiologists, the same kind of label the test set is scored on) are now used:
#     they replace the report labels for those studies, weigh GOLD_WEIGHT times
#     more in the loss, and the table reports AUC against them.
#
# train_frozen() saves everything the submission needs to MODEL_DIR:
#   backbone.pt       ConvNeXt weights (timm format), so nothing is downloaded later
#   head_fold{k}.pt   the trained study head of each fold
#   config.json       settings, label order and scores
# train_frozen() runs this path; train_lora() in lora.py is the alternative, and
# predict.py loads what either saves.


def autocast(device):
    return torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else contextlib.nullcontext()


# --- which slices -------------------------------------------------------------------
# cell 11's dataset calls pick_slices by name, so redefining it here changes what
# every dataset reads, in training and in the submission alike. The original
# is kept (once) so that importing this module again does not narrow the window twice.
_pick_slices_all = getattr(_common, "_pick_slices_all", _common.pick_slices)
_common._pick_slices_all = _pick_slices_all


def pick_slices(n_available, n_wanted, jitter=False, rng=None):
    """cell 11's evenly spaced pick, restricted to the middle CENTER_FRAC of the series."""
    keep = min(n_available, max(n_wanted, int(round(n_available * C.CENTER_FRAC))))
    lo = (n_available - keep) // 2
    return [lo + j for j in _pick_slices_all(keep, n_wanted, jitter, rng)]


_common.pick_slices = pick_slices     # what cell 11's dataset calls, as redefining it in the notebook did


# --- backbone and features ------------------------------------------------------------
def feature_dim(backbone):
    return backbone.num_features * (2 if C.FEATURE_MODE == "avg+max" else 1)


def create_backbone(cfg, weights=None):
    """Frozen ConvNeXt from timm. weights=None downloads the pretrained weights
    (Internet on); a path loads the copy saved in MODEL_DIR (works offline).
    in_chans=1: timm sums the pretrained RGB filters of the first convolution
    into one grey channel, which is equivalent for a grey image."""
    model = timm.create_model(cfg.model_name, pretrained=weights is None, num_classes=0,
                              in_chans=cfg.in_chans)
    if weights is not None:
        model.load_state_dict(torch.load(weights, map_location="cpu"))
    for p in model.parameters():
        p.requires_grad = False
    return model.eval()


def embed(backbone, x):
    """(N, 1, H, W) slices -> (N, feature_dim) embeddings."""
    fmap = backbone.forward_features(x)                       # (N, 768, H/32, W/32)
    avg = backbone.forward_head(fmap, pre_logits=True)        # global average + the head's LayerNorm
    if C.FEATURE_MODE != "avg+max":
        return avg
    peak = backbone.head.norm(fmap.amax(dim=(2, 3), keepdim=True)).flatten(1)
    return torch.cat([avg, peak], dim=-1)


def crop_mm(vol, spacing, mm=None):
    """The central mm x mm millimetres of every slice in vol (S, H, W).

    spacing is the in-plane PixelSpacing in mm. A missing spacing, or a field of
    view smaller than mm, leaves that axis whole."""
    mm = C.CROP_MM if mm is None else mm
    try:
        spacing = float(spacing)
    except (TypeError, ValueError):
        return vol
    if not mm or not np.isfinite(spacing) or spacing <= 0:
        return vol
    side = int(round(mm / spacing))
    H, W = vol.shape[-2:]
    h, w = min(side, H), min(side, W)
    top, left = (H - h) // 2, (W - w) // 2
    return vol[..., top:top + h, left:left + w]


class SafeVolumeDataset(KneeVolumeDataset):
    """KneeVolumeDataset that never raises: an unreadable series comes back as
    zeros with ok=False, so one bad file cannot stop a run or a submission.
    It also crops every slice to CROP_MM before cell 11 normalises and resizes it."""

    def _slices(self, row, idx):
        return crop_mm(super()._slices(row, idx), row.get("pixel_spacing"))

    def __getitem__(self, i):
        try:
            x, _ = super().__getitem__(i)
            return x, True
        except Exception as exc:
            print(f"unreadable series {self.rows.iloc[i]['series']}: {type(exc).__name__}: {exc}")
            return torch.zeros(self.n_slices, 1, self.image_size, self.image_size), False


@torch.no_grad()
def extract_features(frame, backbone, cfg):
    """Slice embeddings, (n_series, n_slices, feature_dim) float16, and an ok flag per series."""
    loader = DataLoader(SafeVolumeDataset(frame, LABELS, cfg.n_slices, cfg.image_size),
                        batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=cfg.device == "cuda")
    backbone = backbone.to(cfg.device)
    feats = np.zeros((len(frame), cfg.n_slices, feature_dim(backbone)), np.float16)
    ok = np.zeros(len(frame), bool)
    done, t0 = 0, time.time()
    for step, (x, good) in enumerate(loader):
        b, s = x.shape[:2]
        with autocast(cfg.device):
            f = embed(backbone, x.to(cfg.device, non_blocking=True).flatten(0, 1))
        feats[done:done + b] = f.reshape(b, s, -1).float().cpu().numpy()
        ok[done:done + b] = good.numpy()
        done += b
        if step % 100 == 0 or done == len(frame):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(frame)} series embedded, ~{(len(frame) - done) / rate / 60:.0f} min left")
    return feats, ok


# --- studies: which series belong together and what kind they are ------------------
PLANES = ["sagittal", "coronal", "axial"]            # oblique / unknown count as one more kind
WEIGHTINGS = ["PD", "T1", "T2", "STIR", "T2*"]       # anything else counts as one more kind
N_TYPES = 1 + (len(PLANES) + 1) * (len(WEIGHTINGS) + 1)   # 0 is padding


def series_type(plane, weighting):
    """Integer id (1..N_TYPES-1) for a series' plane x weighting."""
    p = PLANES.index(plane) if plane in PLANES else len(PLANES)
    w = WEIGHTINGS.index(weighting) if weighting in WEIGHTINGS else len(WEIGHTINGS)
    return 1 + p * (len(WEIGHTINGS) + 1) + w


def build_studies(frame, ok, max_series=None):
    """Group the readable series of `frame` by study.

    Returns {"ids": [study, ...], "sidx": (n_studies, K) row numbers into the
    feature array (-1 = padding), "types": (n_studies, K) series_type ids (0 = padding)}.
    A study with more than K series keeps one of each kind first, then the rest.
    """
    K = max_series or C.MAX_SERIES
    study = frame["study"].to_numpy()
    kind = np.array([series_type(p, w) for p, w in zip(frame["plane"], frame["weighting"])])
    rows_of = {}
    for i in np.flatnonzero(np.asarray(ok, bool)):
        rows_of.setdefault(study[i], []).append(i)
    ids, sidx, types = [], [], []
    for sid, rows in rows_of.items():
        seen, first, rest = set(), [], []
        for i in rows:
            (rest if kind[i] in seen else first).append(i)
            seen.add(kind[i])
        rows = (first + rest)[:K]
        ids.append(sid)
        sidx.append(rows + [-1] * (K - len(rows)))
        types.append([kind[i] for i in rows] + [0] * (K - len(rows)))
    return {"ids": ids, "sidx": np.array(sidx, np.int64).reshape(-1, K),
            "types": np.array(types, np.int64).reshape(-1, K)}


class StudyHead(nn.Module):
    """Slice embeddings of a study's series -> 12 logits.

    1. AttentionPool: the slices of each series -> one vector per series.
    2. + an embedding of the series' plane and weighting, so the head knows
       it is looking at a sagittal PD and not an axial T2.
    3. One learned query per finding attends over the study's series: ACL
       can lean on the sagittal PD, effusion on the axial, and a series that
       shows nothing relevant gets little weight. A plain mean could not do that.
    """

    def __init__(self, dim, n_labels, dropout=0.1):
        super().__init__()
        self.pool = AttentionPool(dim)
        self.type_emb = nn.Embedding(N_TYPES, dim)
        self.norm = nn.LayerNorm(dim)
        self.query = nn.Parameter(torch.randn(n_labels, dim) * dim ** -0.5)
        self.out_w = nn.Parameter(torch.randn(n_labels, dim) * dim ** -0.5)
        self.out_b = nn.Parameter(torch.zeros(n_labels))
        self.drop = nn.Dropout(dropout)

    def forward(self, feats, types, return_weights=False):    # feats (B, K, S, D), types (B, K)
        B, K, S, D = feats.shape
        pooled, slice_w = self.pool(feats.float().reshape(B * K, S, D))
        h = self.norm(pooled.reshape(B, K, D) + self.type_emb(types))
        score = torch.einsum("bkd,ld->blk", h, self.query) / D ** 0.5
        score = score.masked_fill((types == 0)[:, None, :], -1e4)
        w = torch.softmax(score, dim=-1)                       # (B, L, K): series weight per finding
        logits = (self.drop(torch.einsum("blk,bkd->bld", w, h)) * self.out_w).sum(-1) + self.out_b
        return (logits, w, slice_w.reshape(B, K, S)) if return_weights else logits


@torch.no_grad()
def study_probs(heads, feats, studies, device, labels, idx=None, batch=64):
    """Sigmoid probabilities averaged over the heads (one per fold):
    a DataFrame, one row per study in `studies` (or only those in `idx`).
    `feats` may be a numpy array or a tensor (e.g. already on the GPU)."""
    idx = np.arange(len(studies["ids"])) if idx is None else np.asarray(idx, int)
    for h in heads:
        h.eval()
    out = []
    for start in range(0, len(idx), batch):
        b = idx[start:start + batch]
        rows = np.clip(studies["sidx"][b], 0, None)            # padding points at row 0; its type is 0, so it is masked
        if torch.is_tensor(feats):
            x = feats[torch.from_numpy(rows).to(feats.device)].to(device)
        else:
            x = torch.from_numpy(np.asarray(feats[rows])).to(device)
        types = torch.from_numpy(studies["types"][b]).to(device)
        out.append(torch.stack([torch.sigmoid(h(x, types)) for h in heads]).mean(0).cpu().numpy())
    values = np.concatenate(out) if out else np.zeros((0, len(labels)), np.float32)
    return pd.DataFrame(values, index=[studies["ids"][i] for i in idx], columns=list(labels))


def clean_labels(values, mode):
    """Report-derived labels -> training targets (see LABEL_MODE)."""
    v = np.asarray(values, dtype=np.float32)
    if mode == "soft":
        return v
    out = np.full_like(v, np.nan)
    out[v >= 0.9] = 1.0          # present
    out[v <= 0.25] = 0.0         # absent, or not mentioned
    return out                   # hedged labels (0.5 to 0.8) and missing reports stay NaN


def relabel_from_reports(df):
    """Replace the 12 label columns of `df` by labels recomputed from the reports.

    Reports the labeler cannot read (a script outside its vocabulary) come out
    all-negative for the wrong reason, so those studies get no label (NaN)."""
    fresh = read_reports(C.DATA_PATH, "train").drop_duplicates("study").set_index("study")
    unreadable = fresh["needs_review"].to_numpy(bool)
    new = fresh[LABELS].copy()
    new.loc[unreadable, :] = np.nan
    old = df.drop_duplicates("study").set_index("study")[LABELS]
    out = df.drop(columns=[c for c in LABELS + ["labels"] if c in df.columns])
    out = out.merge(new, left_on="study", right_index=True, how="left")
    n_old = (old >= 0.9).sum()
    n_new = (new >= 0.9).sum()
    print(f"relabelled {len(new)} studies from their reports ({int(unreadable.sum())} unreadable ones left without labels)")
    print("  confident positives per finding, train_set.csv -> now:")
    for lab in LABELS:
        print(f"    {lab:20s} {int(n_old.get(lab, 0)):5d} -> {int(n_new[lab]):5d}")
    return out


def label_mapping(model_labels, sub_labels):
    """Model label -> competition column. By name where every name matches
    (ignoring case and punctuation), otherwise by position: the model uses
    cell 3's names ('bakers_cyst'), the competition its own ("Baker's"),
    in the same order."""
    assert len(model_labels) == len(sub_labels), (model_labels, sub_labels)
    key = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())
    by_name = {key(s): s for s in sub_labels}
    if all(key(m) in by_name for m in model_labels):
        return {m: by_name[key(m)] for m in model_labels}
    return dict(zip(model_labels, sub_labels))


def load_gold(data_path=None):
    """The official labels in train.csv, renamed to LABELS and made 0/1.

    One row per study that has them; a missing cell stays NaN. Every other
    training study has only its report."""
    data_path = C.DATA_PATH if data_path is None else data_path
    t = pd.read_csv(os.path.join(data_path, "train.csv"))
    cols = [c for c in t.columns if c not in ("StudyInstanceUID", "Report")]
    if len(cols) != len(LABELS):
        print(f"train.csv has {len(cols)} label columns, not {len(LABELS)}: official labels not used ({cols})")
        return pd.DataFrame(columns=LABELS, dtype=np.float32)
    mapping = label_mapping(LABELS, cols)
    g = t.set_index("StudyInstanceUID")[[mapping[lab] for lab in LABELS]].apply(pd.to_numeric, errors="coerce")
    g.columns = LABELS
    g = g[g.notna().any(axis=1)]
    values = np.unique(g.to_numpy()[~np.isnan(g.to_numpy())])
    top = float(values.max()) if len(values) else 1.0
    cut = 0.5 if top <= 1 else top / 2                    # 0/1 labels; a graded scale is cut at its midpoint
    print(f"{len(g)} training studies have official labels | values {values[:8].tolist()} | positive = value >= {cut}")
    print("  official positives per finding: " + ", ".join(f"{lab} {int((g[lab] >= cut).sum())}" for lab in LABELS))
    return (g >= cut).astype(np.float32).where(g.notna())


def weighted_bce(logits, y, w):
    """BCE over the known labels, each study weighted by w (gold studies count more)."""
    known = ~torch.isnan(y)
    if not known.any():
        return logits.sum() * 0.0
    loss = F.binary_cross_entropy_with_logits(logits.float(), torch.nan_to_num(y), reduction="none")
    wk = w[:, None] * known
    return (loss * wk).sum() / wk.sum()


def train_head_fold(feats, studies, Y, tab, fold, cfg, verbose=True, weights=None):
    """Train one fold's head on the cached features.

    feats    (n_series, S, D) slice embeddings
    studies  build_studies(...) of the training frame
    Y        (n_studies, 12) targets (clean_labels), NaN = unknown
    tab      one row per study, same order as `studies`, with its split and fold
    weights  (n_studies,) loss weight per study, default 1
    Returns (head, best_metrics, val_predictions).
    """
    seed_everything(cfg.seed + fold)
    dev = cfg.device
    known = ~np.isnan(Y).all(axis=1)
    in_train = (tab["split"] == "train").to_numpy()
    tr = np.where(in_train & (tab["fold"] != fold).to_numpy() & known)[0]
    va = np.where(in_train & (tab["fold"] == fold).to_numpy() & known)[0]

    X = torch.from_numpy(feats).to(dev)                                # float16, a few hundred MB
    rows = torch.from_numpy(np.clip(studies["sidx"], 0, None)).to(dev)
    types_all = torch.from_numpy(studies["types"]).to(dev)
    Yt = torch.from_numpy(Y).to(dev)
    Wt = torch.from_numpy(np.ones(len(Y), np.float32) if weights is None else weights.astype(np.float32)).to(dev)

    head = StudyHead(feats.shape[-1], len(LABELS), cfg.dropout).to(dev)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    steps = max(1, math.ceil(len(tr) / C.HEAD_BATCH)) * cfg.epochs
    warmup = max(1, int(steps * cfg.warmup_frac))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    n_keep = max(1, int(round(feats.shape[1] * C.SLICE_KEEP)))

    best, best_state, best_pred, stale = {"auc": -np.inf}, None, None, 0
    for epoch in range(cfg.epochs):
        head.train()
        perm = tr[np.random.permutation(len(tr))]
        total = 0.0
        for start in range(0, len(perm), C.HEAD_BATCH):
            idx = torch.from_numpy(perm[start:start + C.HEAD_BATCH]).to(dev)
            keep = torch.randperm(X.shape[1], device=dev)[:n_keep]          # a random 75% of the slices
            x = X[rows[idx]][:, :, keep]                                    # (B, K, n_keep, D)
            types = types_all[idx]
            if C.SERIES_DROP > 0:                                             # hide whole series, never all of them
                hidden = types.masked_fill(torch.rand(types.shape, device=dev) < C.SERIES_DROP, 0)
                types = torch.where((hidden == 0).all(dim=1, keepdim=True), types, hidden)
            loss = weighted_bce(head(x, types), Yt[idx], Wt[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            total += loss.item() * len(idx)

        val_pred = study_probs([head], X, studies, dev, LABELS, idx=va)
        metrics = multilabel_metrics(Y[va], val_pred.to_numpy(), LABELS)
        if verbose:
            print(f"  fold {fold} epoch {epoch:2d}  train loss {total / max(len(tr), 1):.4f}  "
                  f"val auc {metrics['auc']:.3f}  ap {metrics['ap']:.3f}")
        if best_state is None or (not np.isnan(metrics["auc"]) and metrics["auc"] > best["auc"]):
            best, best_pred, stale = metrics, val_pred, 0
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
            if stale >= cfg.patience:
                break

    head.load_state_dict(best_state)
    torch.save(best_state, os.path.join(C.MODEL_DIR, f"head_fold{fold}.pt"))
    del X, rows, types_all, Yt, Wt
    free_memory()
    return head.eval(), best, best_pred


def prepare_training_frame():
    """train_set.csv -> (df, gold): relabelled, localizers dropped, split into a
    hold-out and 5 folds, every official study in the folds. Both training paths
    (frozen heads below, LoRA fine-tuning in the next cell) start from this."""
    df = pd.read_csv(C.TRAIN_SET_CSV)
    if C.RELABEL_FROM_REPORTS:
        df = relabel_from_reports(df)
    if C.DROP_LOCALIZERS:
        df = df[df["weighting"] != "localizer"]
    gold = load_gold() if C.USE_GOLD else pd.DataFrame(columns=LABELS, dtype=np.float32)
    is_gold = df["study"].isin(gold.index)
    if "needs_review" in df.columns:                 # their labels do not come from the report: keep them
        df.loc[is_gold, "needs_review"] = False
    df = make_splits(df.reset_index(drop=True), n_folds=5, test_frac=0.15, seed=42).reset_index(drop=True)
    # Every gold study goes into the training folds, spread evenly, so each one is
    # trained on by four fold models and predicted out-of-fold by the fifth.
    is_gold = df["study"].isin(gold.index)
    if is_gold.any():
        n_folds = int(df.loc[df["split"] == "train", "fold"].max()) + 1
        fold_of = {s: i % n_folds for i, s in enumerate(dict.fromkeys(df.loc[is_gold, "study"]))}
        df.loc[is_gold, "split"] = "train"
        df.loc[is_gold, "fold"] = df.loc[is_gold, "study"].map(fold_of).astype(int)
        print(f"{len(fold_of)} gold studies spread over {n_folds} training folds")
    return df, gold


def train_frozen():
    """The frozen path (cell 12's training block): features once, then 5 fold
    heads, saved to MODEL_DIR. scripts/train.py --method frozen runs it."""
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    df, gold = prepare_training_frame()

    # ConvNeXt weights: downloaded once here (Internet on), saved for the offline submission
    backbone = create_backbone(C.cfg)
    torch.save(backbone.state_dict(), os.path.join(C.MODEL_DIR, "backbone.pt"))

    # Slice embeddings for every series, cached so rerunning this cell skips the reading.
    # The name carries everything that changes the embeddings, so a stale cache is never reused.
    feat_path = os.path.join(C.FEATURE_CACHE, f"train_features_{C.cfg.model_name}_{C.cfg.n_slices}x{C.cfg.image_size}"
                                            f"_{C.FEATURE_MODE}_c{int(round(C.CENTER_FRAC * 100))}_crop{C.CROP_MM or 0}.npy")
    index_path = feat_path.replace(".npy", "_index.csv")
    feats = None
    if os.path.exists(feat_path) and os.path.exists(index_path):
        cache_index = pd.read_csv(index_path)
        where = pd.Series(np.arange(len(cache_index)), index=cache_index["series"].astype(str))
        want = df["series"].astype(str)
        if where.index.is_unique and want.isin(where.index).all():
            order = where.loc[want].to_numpy()
            feats = np.ascontiguousarray(np.load(feat_path, mmap_mode="r")[order])
            df["ok"] = cache_index["ok"].astype(bool).to_numpy()[order]
            print(f"loaded cached features {feats.shape} from {feat_path}")
        else:
            print(f"the cache in {C.FEATURE_CACHE} does not cover these series: extracting again")
    if feats is None:
        print(f"extracting ConvNeXt features for {len(df)} series (one pass over the DICOMs) ...")
        feats, ok = extract_features(df, backbone, C.cfg)
        df["ok"] = ok
        try:
            os.makedirs(os.path.dirname(feat_path), exist_ok=True)
            np.save(feat_path, feats)
        except OSError:                  # FEATURE_CACHE is an attached, read-only input: save here instead
            feat_path = os.path.join(os.getcwd(), "feature_cache", os.path.basename(feat_path))
            index_path = feat_path.replace(".npy", "_index.csv")
            os.makedirs(os.path.dirname(feat_path), exist_ok=True)
            np.save(feat_path, feats)
        df[["study", "series", "ok"]].to_csv(index_path, index=False)
        print(f"saved the feature cache to {feat_path}")
    print(f"{(~df['ok']).sum()} series could not be read and are left out")
    backbone.to("cpu")
    free_memory()

    # One row per study: its series, its split/fold and its targets
    studies = build_studies(df, df["ok"])
    tab = df.drop_duplicates("study").set_index("study").loc[studies["ids"]]
    Y = clean_labels(tab[LABELS].to_numpy(np.float32), C.LABEL_MODE)
    W = np.ones(len(Y), np.float32)
    gold_rows = np.flatnonzero(tab.index.isin(gold.index))
    if len(gold_rows):                                   # official labels replace the report labels
        G = gold.loc[tab.index[gold_rows]].to_numpy(np.float32)
        Y[gold_rows] = np.where(np.isnan(G), Y[gold_rows], G)
        W[gold_rows] = C.GOLD_WEIGHT
    Y_df = pd.DataFrame(Y, index=tab.index, columns=LABELS)
    print(f"{len(studies['ids'])} studies | label mode '{C.LABEL_MODE}' | {len(gold_rows)} with official labels "
          f"(weight {C.GOLD_WEIGHT:g}) | {int(np.isnan(Y).all(axis=1).sum())} without usable labels")

    # Cross-validation: one head per fold
    folds = sorted(int(f) for f in tab.loc[tab["split"] == "train", "fold"].unique())
    heads, rows, oofs = [], [], []
    for fold in folds:
        print(f"\n=== fold {fold} ===")
        head, best, val_pred = train_head_fold(feats, studies, Y, tab, fold, C.cfg, weights=W)
        heads.append(head)
        oofs.append(val_pred)
        rows.append({"fold": fold, "val_auc": best["auc"], "val_ap": best["ap"],
                     "skipped_labels": len(best["skipped"])})
    summary = pd.DataFrame(rows)
    print(f"\nCV AUC {summary['val_auc'].mean():.3f} +/- {summary['val_auc'].std():.3f}")
    display(summary)

    # The held-out test split, once, with all fold heads averaged
    hold = np.where((tab["split"] == "test").to_numpy())[0]
    hold_pred = study_probs(heads, feats, studies, C.cfg.device, LABELS, idx=hold)
    hold_metrics = multilabel_metrics(Y[hold], hold_pred.to_numpy(), LABELS)
    print(f"hold-out test AUC {hold_metrics['auc']:.3f}  AP {hold_metrics['ap']:.3f}  "
          f"({len(hold_pred)} studies)")

    # Per finding: where the model is actually working. A finding near 0.5 has learned nothing.
    oof = pd.concat(oofs)
    oof_metrics = multilabel_metrics(Y_df.loc[oof.index].to_numpy(np.float32), oof.to_numpy(), LABELS)
    diagnosis = pd.DataFrame({
        "positives": [int((Y[:, i] == 1).sum()) for i in range(len(LABELS))],
        "oof_auc": [oof_metrics["per_label_auc"].get(lab, np.nan) for lab in LABELS],
        "holdout_auc": [hold_metrics["per_label_auc"].get(lab, np.nan) for lab in LABELS],
    }, index=LABELS)

    # Against the official labels: the only numbers here that measure what the leaderboard measures.
    #   report_vs_gold  how well the report labels agree with them: a ceiling for training on reports
    #   model_vs_gold   out-of-fold predictions of the image model: the closest estimate of the leaderboard
    # With ~58 studies, one finding's AUC moves by ~0.1 by chance; read the means, and the pattern.
    gold_auc = {"report": float("nan"), "model": float("nan")}
    if len(gold_rows):
        g_ids = [s for s in tab.index[gold_rows] if s in oof.index]
        G = gold.loc[g_ids].to_numpy(np.float32)
        rep_m = multilabel_metrics(G, tab.loc[g_ids, LABELS].to_numpy(np.float32), LABELS)
        mod_m = multilabel_metrics(G, oof.loc[g_ids].to_numpy(), LABELS)
        diagnosis["gold_positives"] = [int(np.nansum(G[:, i])) for i in range(len(LABELS))]
        diagnosis["report_vs_gold"] = [rep_m["per_label_auc"].get(lab, np.nan) for lab in LABELS]
        diagnosis["model_vs_gold"] = [mod_m["per_label_auc"].get(lab, np.nan) for lab in LABELS]
        gold_auc = {"report": rep_m["auc"], "model": mod_m["auc"]}
        print(f"\non the {len(g_ids)} studies with official labels:")
        print(f"  report labels vs official   mean AUC {rep_m['auc']:.3f}   (ceiling for learning from reports)")
        print(f"  image model (out of fold)   mean AUC {mod_m['auc']:.3f}   (closest estimate of the leaderboard)")
        print(f"  image model vs report labels mean AUC {oof_metrics['auc']:.3f}   (what CV measures)")
    display(diagnosis.round(3))
    weak = diagnosis.index[diagnosis["oof_auc"] < 0.6].tolist()
    if weak:
        print("below 0.6 out of fold (little more than chance):", ", ".join(weak))

    with open(os.path.join(C.MODEL_DIR, "config.json"), "w") as f:
        json.dump({
            "head": "study", "model_name": C.cfg.model_name, "image_size": C.cfg.image_size,
            "n_slices": C.cfg.n_slices, "in_chans": C.cfg.in_chans, "dim": int(feats.shape[-1]),
            "dropout": C.cfg.dropout, "feature_mode": C.FEATURE_MODE, "center_frac": C.CENTER_FRAC,
            "crop_mm": C.CROP_MM,
            "max_series": C.MAX_SERIES, "planes": PLANES, "weightings": WEIGHTINGS,
            "label_mode": C.LABEL_MODE, "relabelled": C.RELABEL_FROM_REPORTS, "labels": list(LABELS), "folds": folds,
            "drop_localizers": C.DROP_LOCALIZERS,
            "cv_auc": float(summary["val_auc"].mean()), "holdout_auc": float(hold_metrics["auc"]),
            "oof_auc_per_finding": {k: float(v) for k, v in oof_metrics["per_label_auc"].items()},
            "gold_studies": int(len(gold_rows)), "gold_weight": C.GOLD_WEIGHT,
            "gold_report_auc": float(gold_auc["report"]), "gold_model_oof_auc": float(gold_auc["model"]),
        }, f, indent=2)
    print(f"\nsaved model to {C.MODEL_DIR}: {sorted(os.listdir(C.MODEL_DIR))}")
