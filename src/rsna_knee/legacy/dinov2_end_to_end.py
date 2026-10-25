"""
HOW A 2D MODEL EATS A 3D SERIES.
DINOv2 is a 2D ViT: it has no notion of a stack. A series is therefore fed as
N slices, each embedded independently, and the N embeddings are pooled into one
series vector before the head. The pooling is learned attention rather than a
mean, because a finding lives on a handful of slices -- a meniscal tear is two
or three of forty -- and a mean over forty slices divides that signal by forty.
Attention lets the model put its weight where the tear is, and the weights are
readable afterwards, which is the cheapest interpretability you will get.
 
WHAT IS NOT AUGMENTED, AND WHY.
No horizontal flips. On a coronal knee a flip swaps the medial and lateral
compartments, so `medial_OA` and `lateral_OA` become each other's labels; on a
sagittal knee it swaps anterior for posterior, which does the same to the
cruciates. The usual reflex augmentation silently corrupts four of the twelve
labels here. Intensity and mild crop jitter are safe and are used instead.
 
THE BACKBONE IS FROZEN BY DEFAULT.
A few hundred studies against 21M parameters fine-tunes straight into
memorisation. Frozen DINOv2 features plus a trained head is the honest
baseline, and it is what you should beat before unfreezing anything. When the
backbone is frozen the slice embeddings never change, so precomputing them once
turns each epoch from minutes into seconds -- worth doing as soon as you are
past the first run.
"""
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

 
 
# =============================================================================
# Model
# =============================================================================
def load_backbone(model_name, image_size, pretrained=True, in_chans=1):
    """DINOv2 via timm, then torch.hub, then a local checkpoint.
 
    Three routes because the weights come from the internet and Kaggle
    notebooks often have none: with the internet off, attach the DINOv2 weights
    as a dataset and point $DINOV2_CHECKPOINT at the .pth.
    """
    import timm
 
    try:
        # in_chans=1 makes timm sum the pretrained patch-embedding weights
        # across RGB, which is exactly right for a gray image and removes the
        # three-fold copy from every tensor in the pipeline.
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                                  img_size=image_size, in_chans=in_chans)
        return model, model.num_features
    except Exception as exc:
        print(f"timm could not fetch {model_name}: {type(exc).__name__}: {exc}")
 
    local = os.environ.get("DINOV2_CHECKPOINT")
    if local and os.path.exists(local):
        model = timm.create_model(model_name, pretrained=False, num_classes=0,
                                  img_size=image_size, in_chans=in_chans)
        state = torch.load(local, map_location="cpu")
        state = _fold_patch_embed(state.get("model", state), in_chans)
        missing, unexpected = model.load_state_dict(
            state.get("model", state), strict=False)
        print(f"loaded {local} (missing {len(missing)}, unexpected {len(unexpected)})")
        return model, model.num_features
 
    try:
        hub = {"vit_small_patch14_dinov2.lvd142m": "dinov2_vits14",
               "vit_base_patch14_dinov2.lvd142m": "dinov2_vitb14"}.get(model_name, "dinov2_vits14")
        model = torch.hub.load("facebookresearch/dinov2", hub)
        if in_chans == 1:
            model.load_state_dict(_fold_patch_embed(model.state_dict(), 1), strict=False)
            model.patch_embed.proj = _folded_conv(model.patch_embed.proj)
        return model, model.embed_dim
    except Exception as exc:
        raise RuntimeError(
            f"Could not load DINOv2 weights ({type(exc).__name__}: {exc}). "
            "With the internet off, attach the weights as a Kaggle dataset and "
            "set $DINOV2_CHECKPOINT, or pass pretrained=False to train from "
            "scratch (which will not work well on this much data)."
        ) from exc
 
 
def _fold_patch_embed(state, in_chans):
    """Sum a 3-channel patch-embedding kernel down to `in_chans`.
 
    A gray image repeated three times and convolved with three kernels gives
    exactly the same result as the image convolved with their sum, so this
    loses nothing and saves carrying the copies.
    """
    if in_chans != 1:
        return state
    out = dict(state)
    for k, v in state.items():
        if k.endswith("patch_embed.proj.weight") and v.ndim == 4 and v.shape[1] == 3:
            out[k] = v.sum(dim=1, keepdim=True)
    return out
 
 
def _folded_conv(conv):
    """The same conv with one input channel and summed weights."""
    folded = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride,
                       conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        folded.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            folded.bias.copy_(conv.bias)
    return folded
 
 
class DinoV2MultiLabel(nn.Module):
    """DINOv2 per slice -> attention pool over slices -> 12 logits."""
 
    def __init__(self, n_labels, cfg: TrainConfig):
        super().__init__()
        self.backbone, dim = load_backbone(cfg.model_name, cfg.image_size,
                                           cfg.pretrained, cfg.in_chans)
 
        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if cfg.unfreeze_last_n:
                blocks = getattr(self.backbone, "blocks", [])
                for blk in list(blocks)[-cfg.unfreeze_last_n:]:
                    for p in blk.parameters():
                        p.requires_grad = True
 
        self.pool = AttentionPool(dim)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(cfg.dropout),
                                  nn.Linear(dim, n_labels))
        self.frozen = cfg.freeze_backbone and not cfg.unfreeze_last_n
 
    def forward(self, x, return_weights=False):  # x: (B, S, 3, H, W)
        b, s = x.shape[:2]
        flat = x.flatten(0, 1)
        # No grad through a fully frozen backbone: it halves memory and lets
        # the batch be twice the size.
        with torch.set_grad_enabled(self.training and not self.frozen):
            feats = self.backbone(flat)
        feats = feats.reshape(b, s, -1).float()
        pooled, weights = self.pool(feats)
        logits = self.head(pooled)
        return (logits, weights) if return_weights else logits
 
 
def run_epoch(model, loader, loss_fn, cfg, optimizer=None, scheduler=None, scaler=None):
    """One pass. Optimiser given -> train; omitted -> evaluate."""
    train = optimizer is not None
    model.train(train)
    device = cfg.device
    use_amp = cfg.amp and device == "cuda"
 
    total, n, preds, targets = 0.0, 0, [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
 
        with torch.set_grad_enabled(train):
            with torch.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = loss_fn(logits, y)
 
            if train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
 
        total += float(loss) * len(y)
        n += len(y)
        preds.append(torch.sigmoid(logits.detach().float()).cpu().numpy())
        targets.append(y.detach().cpu().numpy())
 
    return total / max(n, 1), np.concatenate(preds), np.concatenate(targets)
 
 
def make_loader(df, labels, cfg, train):
    ds = KneeVolumeDataset(df, labels, cfg.n_slices, cfg.image_size, train, cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train,
                      num_workers=cfg.num_workers, pin_memory=(cfg.device == "cuda"),
                      drop_last=False)
 
 
def train_fold(df, fold, labels, cfg: TrainConfig, verbose=True):
    """Train one fold. Returns (checkpoint_path, history, best_metrics, oof).
 
    The trained weights come back as a path rather than a module: they are
    written every time validation improves anyway, and keeping five folds'
    worth of live model is how a 16 GB GPU runs out on the fourth fold.
    """
    seed_everything(cfg.seed + fold)
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.out_dir, f"fold{fold}.pt")
 
    train_df, val_df = fold_frames(df, fold)
    train_loader = make_loader(train_df, labels, cfg, train=True)
    val_loader = make_loader(val_df, labels, cfg, train=False)
 
    model = DinoV2MultiLabel(len(labels), cfg).to(cfg.device)
 
    # Two learning rates: a head from scratch wants a large one, a pretrained
    # backbone wants a small one or it forgets what it knows in one epoch.
    head_params = [p for n_, p in model.named_parameters()
                   if p.requires_grad and not n_.startswith("backbone")]
    back_params = [p for n_, p in model.named_parameters()
                   if p.requires_grad and n_.startswith("backbone")]
    groups = [{"params": head_params, "lr": cfg.head_lr}]
    if back_params:
        groups.append({"params": back_params, "lr": cfg.backbone_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
 
    steps = max(1, len(train_loader)) * cfg.epochs
    warmup = max(1, int(steps * cfg.warmup_frac))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))
 
    loss_fn = masked_bce                  # soft targets are fine: BCE takes [0,1]
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and cfg.device == "cuda")
 
    best = {"auc": -np.inf}
    best_state, best_preds, history, stale = None, None, [], 0
 
    for epoch in range(cfg.epochs):
        t0 = time.time()
        train_loss, _, _ = run_epoch(model, train_loader, loss_fn, cfg,
                                     optimizer, scheduler, scaler)
        val_loss, val_prob, val_true = run_epoch(model, val_loader, loss_fn, cfg)
        metrics = multilabel_metrics(val_true, val_prob, labels)
 
        history.append({"fold": fold, "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_auc": metrics["auc"],
                        "val_ap": metrics["ap"], "seconds": time.time() - t0})
        if verbose:
            print(f"  fold {fold} epoch {epoch}  train {train_loss:.4f}  "
                  f"val {val_loss:.4f}  auc {metrics['auc']:.3f}  "
                  f"ap {metrics['ap']:.3f}  ({time.time() - t0:.0f}s)")
 
        # `best_state is None` first: a fold whose validation set has no label
        # with both classes scores NaN, and without this it would never save a
        # checkpoint and would fall over building the out-of-fold frame.
        score = metrics["auc"]
        improved = best_state is None or (not np.isnan(score) and score > best["auc"])
        if improved:
            best, stale = metrics, 0
            best_preds = val_prob
            # .cpu() on the way out: a deepcopy of a CUDA state_dict puts a
            # second full set of weights on the GPU, next to the live model.
            best_state = {k: v.detach().to("cpu", copy=True)
                          for k, v in model.state_dict().items()}
            torch.save(best_state, ckpt_path)
        else:
            stale += 1
            if stale >= cfg.patience:
                if verbose:
                    print(f"  fold {fold}: no gain in {cfg.patience} epochs, stopping")
                break
 
    oof = val_df[["study", "series"]].copy()
    for i, lab in enumerate(labels):
        oof[f"pred_{lab}"] = best_preds[:, i]
        oof[f"true_{lab}"] = val_df[lab].values
    oof["fold"] = fold
 
    # Hand back the checkpoint path, not the model: five live fold models plus
    # their optimiser state is most of a GPU, and every one of them is already
    # on disk.
    del model
    free_memory()
    return ckpt_path, pd.DataFrame(history), best, oof
 
 
def cross_validate(df, labels=None, cfg=None, folds=None, verbose=True):
    """Train every fold. Returns (checkpoints, oof, history, summary).
 
    `checkpoints` maps fold -> path on disk. Pass it straight to predict(),
    which loads them one at a time.
 
    The out-of-fold frame is the useful artefact: every training study has a
    prediction from a model that never saw it, so thresholds and ensembling can
    be tuned on it without touching the test split.
    """
    labels = list(labels if labels is not None else LABELS)
    cfg = cfg or TrainConfig()
    folds = folds if folds is not None else sorted(
        df.loc[df["split"] == "train", "fold"].unique())
 
    models, oofs, histories, rows = {}, [], [], []
    for fold in folds:
        if verbose:
            print(f"\n=== fold {fold} ===")
        ckpt, history, best, oof = train_fold(df, fold, labels, cfg, verbose)
        models[fold] = ckpt                     # a path on disk, not a module
        oofs.append(oof)
        histories.append(history)
        rows.append({"fold": fold, "val_auc": best["auc"], "val_ap": best["ap"],
                     "skipped_labels": len(best["skipped"])})
 
    oof = pd.concat(oofs, ignore_index=True)
    summary = pd.DataFrame(rows)
    if verbose and len(summary):
        print(f"\nCV AUC {summary['val_auc'].mean():.3f} "
              f"+/- {summary['val_auc'].std():.3f}")
    return models, oof, pd.concat(histories, ignore_index=True), summary
 
 
@torch.no_grad()
def predict(models, df, labels, cfg, verbose=True):
    """Average the fold models over a held-out frame. Returns predictions + metrics.
 
    Touch the test split once, at the end. Every look at it that changes a
    decision turns it into a second validation set.
    """
    loader = make_loader(df, labels, cfg, train=False)
 
    # One fold on the device at a time, averaged as we go: holding five
    # models to average at the end costs five times the weights for no reason.
    mean_prob, n_models = None, 0
    for fold, model in models.items():
        if isinstance(model, str):              # a checkpoint path
            module = DinoV2MultiLabel(len(labels), cfg)
            module.load_state_dict(torch.load(model, map_location="cpu"))
        else:
            module = model
        module.eval().to(cfg.device)
 
        probs = []
        for x, _ in loader:
            logits = module(x.to(cfg.device, non_blocking=True))
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        probs = np.concatenate(probs)
        mean_prob = probs if mean_prob is None else mean_prob + probs
        n_models += 1
 
        module.to("cpu")
        del module
        free_memory()
    mean_prob /= max(n_models, 1)
    y_true = df[labels].to_numpy(dtype=np.float32)
    metrics = multilabel_metrics(y_true, mean_prob, labels)
 
    out = df[["study", "series"]].copy()
    for i, lab in enumerate(labels):
        out[f"pred_{lab}"] = mean_prob[:, i]
    if verbose:
        print(f"test AUC {metrics['auc']:.3f}  AP {metrics['ap']:.3f}  "
              f"({n_models} models, {len(df)} series)")
    return out, metrics
