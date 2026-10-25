"""LoRA fine-tuning of ConvNeXt-Tiny with the study head (notebook cell 13)."""
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

# ============================================================================
# Fine-tuning: ConvNeXt-Tiny with LoRA adapters, trained end to end with the study head (cell 13)
# ============================================================================
# Cell 12 keeps ConvNeXt frozen and trains only the head on cached features, so
# the backbone sees knee MRI exactly as an ImageNet model would. This cell lets the
# backbone adapt, cheaply: small low-rank adapters (LoRA) are added to the two
# linear layers of every block in the last LORA_STAGES ConvNeXt stages, and only
# those adapters and the study head train. The pretrained weights stay frozen, so
# the model cannot drift far from what it already knows, and it fits on one T4.
#
#   one study: up to MAX_SERIES series x FT_SLICES slices
#     -> ConvNeXt + LoRA -> average + max of the last map (1,536 per slice)
#     -> StudyHead (cell 12) -> 12 logits -> loss
#   gradients reach the head AND the adapters.
#
# Speed: reading DICOMs is the slow part, so every slice the model will see is
# prepared once, exactly as the submission prepares it (middle 70%, 130 mm crop,
# scale, resize), and kept as uint8 in one file on local disk (SLICE_CACHE).
# Each epoch reads that file, not the DICOMs.
#
# One model, not five: five fine-tuned backbones would mean five passes over the
# test DICOMs at submission. Fold FT_VAL_FOLD validates and picks the best epoch;
# every other study trains, the 15% hold-out of the frozen path included. The 58 official
# studies stay out of training (FT_TRAIN_ON_GOLD = False) and are scored every
# epoch: that is the closest estimate of the leaderboard.
#
# At every new best epoch the adapters are merged into the ConvNeXt weights
# (W + scale * B @ A) and saved as an ordinary backbone.pt, with the head as
# head_fold0.pt and config.json. predict.py loads them exactly like the frozen model:
# nothing in the submission changes, and nothing about LoRA is needed offline.
# If the session stops early, MODEL_DIR already holds the best epoch so far.


class LoRALinear(nn.Module):
    """y = W x + b + scale * B (A x).

    W and b are the pretrained layer's own parameters, frozen and under their
    original names, so the merged weights load into a plain ConvNeXt. B starts
    at zero: before training the block computes exactly what it did before."""

    def __init__(self, base, rank, alpha):
        super().__init__()
        self.weight, self.bias = base.weight, base.bias
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale

    def merged_weight(self):
        return self.weight + (self.lora_B @ self.lora_A) * self.scale


def add_lora(backbone, stages=None, rank=None, alpha=None):
    """Wrap fc1 and fc2 of every block in `stages`. Returns the new trainable parameters."""
    stages = C.LORA_STAGES if stages is None else stages
    rank = C.LORA_RANK if rank is None else rank
    alpha = C.LORA_ALPHA if alpha is None else alpha
    params = []
    for si in stages:
        for block in backbone.stages[si].blocks:
            for name in ("fc1", "fc2"):
                lora = LoRALinear(getattr(block.mlp, name), rank, alpha)
                setattr(block.mlp, name, lora)
                params += [lora.lora_A, lora.lora_B]
    return params


def merged_state_dict(backbone):
    """The backbone's weights with every adapter folded in, under the original
    timm names: torch.load + load_state_dict into a plain ConvNeXt works."""
    sd = {k: v.detach().float().cpu().clone() for k, v in backbone.state_dict().items()
          if not k.endswith(("lora_A", "lora_B"))}
    for name, mod in backbone.named_modules():
        if isinstance(mod, LoRALinear):
            sd[f"{name}.weight"] = mod.merged_weight().detach().float().cpu()
    return sd


def _free_bytes(path):
    while path and not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path or "/").free


def build_slice_cache(frame, rows):
    """uint8 (len(rows), S, FT_IMAGE, FT_IMAGE) slices of the given frame rows,
    prepared exactly as the submission prepares them, and an ok flag per row.

    S = FT_SLICES, or fewer if the disk is too small for that. A cache already on
    disk for the same series is reused."""
    os.makedirs(C.SLICE_CACHE, exist_ok=True)
    series = frame["series"].astype(str).to_numpy()[rows]
    candidates = list(dict.fromkeys([C.FT_SLICES, max(4, C.FT_SLICES * 3 // 4), max(4, C.FT_SLICES // 2)]))
    name = lambda s: os.path.join(C.SLICE_CACHE, f"slices_{s}x{C.FT_IMAGE}_crop{C.CROP_MM or 0}"
                                               f"_c{int(round(C.CENTER_FRAC * 100))}.npy")
    for s in candidates:                                   # reuse a finished cache
        path, index_path = name(s), name(s).replace(".npy", "_index.csv")
        if os.path.exists(path) and os.path.exists(index_path):
            index = pd.read_csv(index_path)
            if index["series"].astype(str).tolist() == list(series):
                print(f"reusing the slice cache {path}")
                return np.load(path, mmap_mode="r"), index["ok"].astype(bool).to_numpy()

    free = _free_bytes(C.SLICE_CACHE)
    for s in candidates:
        need = len(rows) * s * C.FT_IMAGE * C.FT_IMAGE
        if need < 0.85 * free:
            break
    else:
        raise RuntimeError(f"The slice cache needs {need / 1e9:.1f} GB at {s} slices, but {C.SLICE_CACHE} has "
                           f"{free / 1e9:.1f} GB free. Lower FT_IMAGE or FT_SLICES, or point SLICE_CACHE elsewhere.")
    if s != C.FT_SLICES:
        print(f"not enough disk for {C.FT_SLICES} slices per series: using {s}")
    path, index_path = name(s), name(s).replace(".npy", "_index.csv")
    print(f"preparing {len(rows)} series x {s} slices at {C.FT_IMAGE} px "
          f"({need / 1e9:.1f} GB of {free / 1e9:.1f} GB free) into {path} ...")

    cache = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(rows), s, C.FT_IMAGE, C.FT_IMAGE))
    loader = DataLoader(SafeVolumeDataset(frame.iloc[rows], LABELS, s, C.FT_IMAGE),
                        batch_size=C.cfg.batch_size, shuffle=False, num_workers=C.cfg.num_workers)
    ok = np.zeros(len(rows), bool)
    done, t0 = 0, time.time()
    for step, (x, good) in enumerate(loader):
        b = x.shape[0]
        # back from the dataset's standardised values to 0-1, then 0-255
        cache[done:done + b] = (x[:, :, 0] * GRAY_STD + GRAY_MEAN).clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        ok[done:done + b] = good.numpy()
        done += b
        if step % 100 == 0 or done == len(rows):
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(rows)} series prepared, ~{(len(rows) - done) / rate / 60:.0f} min left")
    cache.flush()
    del cache
    pd.DataFrame({"series": series, "ok": ok}).to_csv(index_path, index=False)
    return np.load(path, mmap_mode="r"), ok


def study_forward(backbone, head, cache, crow, types, dev, n_keep=None, train=False):
    """Logits (B, 12) for B studies. crow (B, K) = cache rows, -1 for an empty slot;
    types (B, K) = series types, 0 for an empty or hidden slot. Only real series
    go through the backbone."""
    B, K = crow.shape
    valid = types > 0
    S = cache.shape[1]
    keep = np.sort(np.random.choice(S, n_keep, replace=False)) if train and n_keep and n_keep < S else np.arange(S)
    x = torch.from_numpy(np.ascontiguousarray(np.asarray(cache[crow[valid]])[:, keep])).to(dev)
    x = x.float().div_(255)                                           # (N, S', H, W) in 0-1
    if train:
        n = x.shape[0]
        # scanners differ in brightness and in where the knee sits; no flips (left and right matter)
        x = (x * torch.empty(n, 1, 1, 1, device=dev).uniform_(0.9, 1.1)
             + torch.empty(n, 1, 1, 1, device=dev).uniform_(-0.05, 0.05)).clamp_(0, 1)
        m = max(1, C.FT_IMAGE // 28)
        x = torch.roll(x, shifts=(int(np.random.randint(-m, m + 1)), int(np.random.randint(-m, m + 1))), dims=(-2, -1))
    x = (x - GRAY_MEAN) / GRAY_STD
    N, S2, H, W = x.shape
    with autocast(dev):
        f = embed(backbone, x.reshape(N * S2, 1, H, W))
    f = f.float().reshape(N, S2, -1)
    feats = torch.zeros(B, K, S2, f.shape[-1], device=dev)
    feats[torch.from_numpy(valid).to(dev)] = f
    return head(feats, torch.from_numpy(types).to(dev))


@torch.no_grad()
def ft_predict(backbone, head, cache, crow, types, ids, idx, dev, batch=8):
    """Sigmoid probabilities for the studies in idx: a DataFrame indexed by study."""
    backbone.eval()
    head.eval()
    out = []
    for start in range(0, len(idx), batch):
        b = idx[start:start + batch]
        out.append(torch.sigmoid(study_forward(backbone, head, cache, crow[b], types[b], dev)).cpu().numpy())
    values = np.concatenate(out) if out else np.zeros((0, len(LABELS)), np.float32)
    return pd.DataFrame(values, index=[ids[i] for i in idx], columns=LABELS)


def write_ft_config(model_name, n_slices, dim, val_auc, gold_auc, epoch, epochs_run, per_finding):
    with open(os.path.join(C.MODEL_DIR, "config.json"), "w") as f:
        json.dump({
            "head": "study", "model_name": model_name, "image_size": C.FT_IMAGE, "n_slices": int(n_slices),
            "in_chans": C.cfg.in_chans, "dim": int(dim), "dropout": C.cfg.dropout, "feature_mode": C.FEATURE_MODE,
            "center_frac": C.CENTER_FRAC, "crop_mm": C.CROP_MM, "max_series": C.MAX_SERIES,
            "planes": PLANES, "weightings": WEIGHTINGS, "label_mode": C.LABEL_MODE,
            "relabelled": C.RELABEL_FROM_REPORTS, "labels": list(LABELS), "folds": [0],
            "drop_localizers": C.DROP_LOCALIZERS,
            "cv_auc": float(val_auc), "gold_model_auc": float(gold_auc),
            "finetune": {"method": "lora", "rank": C.LORA_RANK, "alpha": C.LORA_ALPHA, "stages": list(C.LORA_STAGES),
                         "best_epoch": int(epoch), "epochs_run": int(epochs_run),
                         "train_on_gold": C.FT_TRAIN_ON_GOLD, "val_fold": C.FT_VAL_FOLD},
            "per_finding": per_finding,
        }, f, indent=2)


def train_lora():
    """The LoRA path (cell 13's training block): one fine-tuned model, saved to
    MODEL_DIR at every new best epoch. scripts/train.py --method lora runs it."""
    os.makedirs(C.MODEL_DIR, exist_ok=True)
    seed_everything(C.cfg.seed)
    dev = C.cfg.device
    df, gold = prepare_training_frame()

    # --- the slices: one pass over the DICOMs, then a file on local disk ---------------
    # Cache only the series the head will use: the same choice (one of each plane x
    # weighting first, at most MAX_SERIES) that the submission makes.
    first_choice = build_studies(df, np.ones(len(df), bool))
    rows = np.unique(first_choice["sidx"][first_choice["sidx"] >= 0])
    cache, ok_cached = build_slice_cache(df, rows)
    pos = np.full(len(df), -1, np.int64)
    pos[rows] = np.arange(len(rows))
    ok_rows = np.zeros(len(df), bool)
    ok_rows[rows] = ok_cached
    print(f"{(~ok_cached).sum()} of {len(rows)} cached series could not be read and are left out")

    studies = build_studies(df, ok_rows)
    crow_all = np.where(studies["sidx"] >= 0, pos[np.clip(studies["sidx"], 0, None)], -1)
    types_all = studies["types"]
    ids = studies["ids"]
    tab = df.drop_duplicates("study").set_index("study").loc[ids]

    # --- targets ----------------------------------------------------------------------------
    Y = clean_labels(tab[LABELS].to_numpy(np.float32), C.LABEL_MODE)
    W = np.ones(len(Y), np.float32)
    is_gold = tab.index.isin(gold.index)
    gold_rows = np.flatnonzero(is_gold)
    G = gold.loc[tab.index[gold_rows]].to_numpy(np.float32) if len(gold_rows) else np.zeros((0, len(LABELS)), np.float32)
    if C.FT_TRAIN_ON_GOLD and len(gold_rows):
        Y[gold_rows] = np.where(np.isnan(G), Y[gold_rows], G)
        W[gold_rows] = C.GOLD_WEIGHT
    known = ~np.isnan(Y).all(axis=1)
    in_val = ((tab["split"] == "train") & (tab["fold"] == C.FT_VAL_FOLD)).to_numpy() & ~is_gold
    va = np.flatnonzero(in_val & known)
    tr = np.flatnonzero(~in_val & known & (C.FT_TRAIN_ON_GOLD | ~is_gold))
    print(f"{len(ids)} studies: {len(tr)} train, {len(va)} validate (fold {C.FT_VAL_FOLD}), "
          f"{len(gold_rows)} official {'(trained on)' if C.FT_TRAIN_ON_GOLD else '(check only)'}")

    # --- model: pretrained ConvNeXt (downloaded here, Internet on) + adapters + head --------
    ft_cfg = TrainConfig(model_name=C.cfg.model_name, image_size=C.FT_IMAGE, n_slices=cache.shape[1],
                         in_chans=C.cfg.in_chans, batch_size=C.cfg.batch_size, num_workers=C.cfg.num_workers)
    backbone = create_backbone(ft_cfg)
    lora_params = add_lora(backbone)
    backbone = backbone.to(dev)
    head = StudyHead(feature_dim(backbone), len(LABELS), C.cfg.dropout).to(dev)
    trainable = lora_params + list(head.parameters())
    n_frozen = sum(p.numel() for p in backbone.parameters() if not p.requires_grad)
    print(f"trainable: {sum(p.numel() for p in lora_params):,} adapter + {sum(p.numel() for p in head.parameters()):,} "
          f"head parameters; {n_frozen:,} ConvNeXt parameters stay frozen")

    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": C.FT_LR_LORA, "weight_decay": 0.0},
        {"params": list(head.parameters()), "lr": C.FT_LR_HEAD, "weight_decay": C.cfg.weight_decay}])
    steps = max(1, math.ceil(len(tr) / C.FT_BATCH)) * C.FT_EPOCHS
    warmup = max(1, int(steps * C.cfg.warmup_frac))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    Yt = torch.from_numpy(Y).to(dev)
    Wt = torch.from_numpy(W).to(dev)
    n_keep = max(1, int(round(cache.shape[1] * C.SLICE_KEEP)))

    best = {"auc": -np.inf, "epoch": -1}
    best_state, history = None, []
    t_start = time.time()
    for epoch in range(C.FT_EPOCHS):
        if time.time() - t_start > C.FT_TIME_LIMIT_H * 3600:
            print(f"time limit of {C.FT_TIME_LIMIT_H} h reached: no further epochs")
            break
        backbone.train()
        head.train()
        perm = tr[np.random.permutation(len(tr))]
        total, t0 = 0.0, time.time()
        n_steps = math.ceil(len(perm) / C.FT_BATCH)
        for step in range(n_steps):
            b = perm[step * C.FT_BATCH:(step + 1) * C.FT_BATCH]
            types = types_all[b].copy()
            if C.SERIES_DROP > 0:                                        # hide whole series, never all of them
                hide = (np.random.rand(*types.shape) < C.SERIES_DROP) & (types > 0)
                hide[(np.where(hide, 0, types) == 0).all(axis=1)] = False
                types[hide] = 0
            logits = study_forward(backbone, head, cache, crow_all[b], types, dev, n_keep=n_keep, train=True)
            loss = weighted_bce(logits, Yt[b], Wt[b])
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, C.cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item() * len(b)
            if step == 19 or (step + 1) % 200 == 0:
                per_step = (time.time() - t0) / (step + 1)
                print(f"  epoch {epoch} step {step + 1}/{n_steps}: {per_step:.2f} s/step, "
                      f"~{per_step * (n_steps - step - 1) / 60:.0f} min left in this epoch")

        val_pred = ft_predict(backbone, head, cache, crow_all, types_all, ids, va, dev)
        val_m = multilabel_metrics(Y[va], val_pred.to_numpy(), LABELS)
        if len(gold_rows):
            gold_pred = ft_predict(backbone, head, cache, crow_all, types_all, ids, gold_rows, dev)
            gold_m = multilabel_metrics(G, gold_pred.to_numpy(), LABELS)
        else:
            gold_m = {"auc": float("nan"), "per_label_auc": {}}
        history.append({"epoch": epoch, "train_loss": total / max(len(tr), 1), "val_auc": val_m["auc"],
                        "official_auc": gold_m["auc"], "minutes": (time.time() - t0) / 60})
        print(f"epoch {epoch}: train loss {history[-1]['train_loss']:.4f}  val AUC {val_m['auc']:.3f}  "
              f"official-label AUC {gold_m['auc']:.3f}  ({history[-1]['minutes']:.0f} min)")

        if best_state is None or (not np.isnan(val_m["auc"]) and val_m["auc"] > best["auc"]):
            best = {"auc": val_m["auc"], "epoch": epoch, "gold": gold_m["auc"],
                    "val_per": val_m["per_label_auc"], "gold_per": gold_m["per_label_auc"]}
            best_state = {"lora": [p.detach().clone() for p in lora_params],
                          "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}}
            # save now, so a session that stops later still leaves a usable model
            torch.save(merged_state_dict(backbone), os.path.join(C.MODEL_DIR, "backbone.pt"))
            torch.save(best_state["head"], os.path.join(C.MODEL_DIR, "head_fold0.pt"))
            write_ft_config(C.cfg.model_name, cache.shape[1], feature_dim(backbone), val_m["auc"], gold_m["auc"],
                            epoch, epoch + 1,
                            {lab: {"val_auc": val_m["per_label_auc"].get(lab), "official_auc": gold_m["per_label_auc"].get(lab)}
                             for lab in LABELS})
            print(f"  saved epoch {epoch} to {C.MODEL_DIR}")

    display(pd.DataFrame(history).round(3))
    print(f"best epoch {best['epoch']}: val AUC {best['auc']:.3f}, official-label AUC {best['gold']:.3f}")

    # --- check: the saved, merged backbone computes what the adapted one did ---------------
    with torch.no_grad():
        for p, v in zip(lora_params, best_state["lora"]):
            p.copy_(v)
    merged = create_backbone(ft_cfg, weights=os.path.join(C.MODEL_DIR, "backbone.pt")).to(dev)
    probe_rows = crow_all[va[:2] if len(va) else tr[:2]]
    probe = torch.from_numpy(np.asarray(cache[probe_rows[probe_rows >= 0][:2]])[:, :2]).to(dev).float().div(255)
    probe = ((probe - GRAY_MEAN) / GRAY_STD).reshape(-1, 1, C.FT_IMAGE, C.FT_IMAGE)
    backbone.eval()
    with torch.no_grad():
        diff = (embed(backbone, probe) - embed(merged, probe)).abs().max().item()
    print(f"merged backbone vs adapters: largest difference {diff:.2e}")
    assert diff < 1e-3, "the merged backbone does not match the adapted one"

    diagnosis = pd.DataFrame({
        "val_auc": [best["val_per"].get(lab, np.nan) for lab in LABELS],
        "official_auc": [best["gold_per"].get(lab, np.nan) for lab in LABELS],
        "official_positives": [int(np.nansum(G[:, i])) if len(G) else 0 for i in range(len(LABELS))],
    }, index=LABELS)
    display(diagnosis.round(3))
    del backbone, merged, head, optimizer
    free_memory()
    print(f"\nsaved model to {C.MODEL_DIR}: {sorted(os.listdir(C.MODEL_DIR))}")
