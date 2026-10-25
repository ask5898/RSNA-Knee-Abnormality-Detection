"""Shared training pieces (notebook cell 11): TrainConfig, the volume dataset,
slice sampling and normalisation, the attention pool, metrics and the loss.

Cell 11 also held the first, end-to-end DINOv2 trainer; nothing uses it any
more, and it lives on in rsna_knee/legacy/dinov2_end_to_end.py.
"""
import ast
import gc
from dataclasses import dataclass, field
 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import Dataset

from ..data.dicom import read_volume
 
 
# =============================================================================
# Config
# =============================================================================
@dataclass
class TrainConfig:
    # data
    n_slices: int = 16              # slices sampled per series
    image_size: int = 224           # must be a multiple of 14 for a /14 ViT
    batch_size: int = 4             # volumes, so n_slices x this images per step
    num_workers: int = 2
 
    # model
    model_name: str = "vit_small_patch14_dinov2.lvd142m"
    pretrained: bool = True
    in_chans: int = 1               # gray in, no RGB replication
    freeze_backbone: bool = True
    unfreeze_last_n: int = 0        # last N transformer blocks, if not frozen
    dropout: float = 0.1
 
    # optimisation
    epochs: int = 8
    head_lr: float = 1e-3
    backbone_lr: float = 1e-5       # only used when something is unfrozen
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    patience: int = 3               # early stop after N epochs with no gain
    grad_clip: float = 1.0
 
    seed: int = 42
    amp: bool = True                # ignored on CPU
    out_dir: str = "checkpoints"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
 
 
def free_memory():
    """Return what was just dropped, instead of waiting for the collector.
 
    Worth calling between folds: Python frees the tensors when it gets round
    to it, and the next fold allocates before that happens.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
 
def gpu_report(prefix=""):
    """Current GPU allocation, when there is a GPU."""
    if not torch.cuda.is_available():
        return
    used = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"{prefix}GPU {used:.2f} GB in use, {peak:.2f} GB peak, {total:.0f} GB total")
 
 
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
 
# =============================================================================
# Dataset -- image and label, nothing else
# =============================================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
# One channel, not three. Replicating grayscale to RGB triples every tensor
# from the worker through pinned memory and across the bus, to carry the same
# number three times. The backbone's patch embedding is folded to one input
# channel instead (see load_backbone), which is what timm does for in_chans=1
# and is equivalent for a gray image. These are the ImageNet statistics
# averaged over the three channels.
GRAY_MEAN, GRAY_STD = 0.449, 0.226
 
 
def pick_slices(n_available, n_wanted, jitter=False, rng=None):
    """Evenly spaced slice indices, optionally jittered within each interval.
 
    Even spacing rather than the middle N: a knee series covers the whole joint
    and a finding can sit anywhere in it. Jitter makes the sampled set differ
    between epochs, which is augmentation the anatomy cannot object to.
    """
    if n_available <= n_wanted:
        idx = list(range(n_available))
        return idx + [n_available - 1] * (n_wanted - n_available)   # pad by repeat
 
    edges = np.linspace(0, n_available, n_wanted + 1)
    if jitter and rng is not None:
        return [int(min(n_available - 1, rng.uniform(edges[i], edges[i + 1])))
                for i in range(n_wanted)]
    return [int((edges[i] + edges[i + 1]) / 2) for i in range(n_wanted)]
 
 
def normalize_volume(vol):
    """Per-volume robust scaling to [0, 1].
 
    Per volume because MR intensities have no physical units -- they shift with
    coil, scanner and sequence, so a global constant is meaningless. Robust
    (1st/99th percentile) because a single bright artefact otherwise compresses
    the entire knee into the bottom of the range.
    """
    lo, hi = np.percentile(vol, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(vol.min()), float(vol.max())
    if hi <= lo:
        return np.zeros_like(vol, dtype=np.float32)
    return np.clip((vol - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
 
 
class KneeVolumeDataset(Dataset):
    """Yields (slices, labels) and nothing else.
 
    slices  float tensor (n_slices, 3, size, size), ImageNet-normalised
    labels  float tensor (12,) -- soft targets in [0, 1]
 
    Grayscale is replicated across three channels: DINOv2 was trained on RGB
    and its patch embedding expects three, and replication keeps the
    pretrained filters in the regime they were fitted on.
    """
 
    def __init__(self, df, labels, n_slices=16, image_size=224, train=False, seed=0):
        self.rows = df.reset_index(drop=True).copy()
        # read back from a CSV, each tuple of paths is one string: "('a.dcm', 'b.dcm', ...)"
        if len(self.rows) and isinstance(self.rows["paths"].iloc[0], str):
            self.rows["paths"] = self.rows["paths"].apply(ast.literal_eval)
        self.labels = list(labels)
        self.n_slices = n_slices
        self.image_size = image_size
        self.train = train
        self.seed = seed
 
    def __len__(self):
        return len(self.rows)
 
    def _slices(self, row, idx):
        """Only the sampled slices, decoded only if they are not already here.
 
        The order matters: pick the indices, then read. Reading the whole
        series and subscripting it decodes every slice to throw most away --
        on a 200-slice series that is twelve times the IO and twelve times the
        peak memory, per sample, every epoch.
        """
        img = row.get("image")
        if img is not None and not (isinstance(img, float) and np.isnan(img)):
            return np.asarray(img[idx], dtype=np.float32)
        paths = row["paths"]
        return read_volume([paths[j] for j in idx], np.float32)   # only the sampled slices
 
    def __getitem__(self, i):
        row = self.rows.iloc[i]
        rng = np.random.default_rng(self.seed + i if not self.train else None)
 
        img = row.get("image")              # .get: the column may have been dropped before saving
        have_pixels = img is not None and not (isinstance(img, float) and np.isnan(img))
        n_available = len(img) if have_pixels else int(row["n_slices"] or len(row["paths"]))
        idx = pick_slices(n_available, self.n_slices, jitter=self.train, rng=rng)
 
        # Scaled on the sampled slices rather than the whole series: the whole
        # series is what we are avoiding reading. The percentiles are close
        # enough on a dozen slices spanning the joint.
        vol = normalize_volume(self._slices(row, idx))
        x = torch.from_numpy(vol)                            # (S, H, W)
 
        if self.train:
            # Intensity jitter: scanners vary more than this between sites.
            x = torch.clamp(x * rng.uniform(0.9, 1.1) + rng.uniform(-0.05, 0.05), 0, 1)
 
        x = x.unsqueeze(1)                                   # (S, 1, H, W)
        x = F.interpolate(x, size=(self.image_size, self.image_size),
                          mode="bilinear", align_corners=False)
        x = (x - GRAY_MEAN) / GRAY_STD
 
        y = torch.tensor([float(row[l]) for l in self.labels], dtype=torch.float32)
        return x, y
 
 
class AttentionPool(nn.Module):
    """Pool slice embeddings into one series embedding, with learned weights.
 
    Returns the weights as well: `w[i]` is how much slice i mattered, which
    plots straight onto the stack and tells you whether the model is looking
    at the joint or at the edge of the field of view.
    """
 
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128),
                                   nn.Tanh(), nn.Linear(128, 1))
 
    def forward(self, x):                        # x: (B, S, D)
        w = torch.softmax(self.score(x), dim=1)  # (B, S, 1)
        return (x * w).sum(dim=1), w.squeeze(-1)
 
 
# =============================================================================
# Metrics
# =============================================================================
def multilabel_metrics(y_true, y_prob, labels, threshold=0.5):
    """Macro AUC and macro average precision over the labels that have both classes.
 
    Labels with no positive (or no negative) in a fold have no defined AUC.
    They are skipped and counted rather than scored as 0.5, which would drag a
    fold mean around for a reason that has nothing to do with the model.
    """
    aucs, aps, skipped = {}, {}, []
    for i, lab in enumerate(labels):
        known = ~np.isnan(y_true[:, i])         # NaN = no report for that study: not a negative
        col = (y_true[known, i] > threshold).astype(int)
        if len(col) == 0 or col.min() == col.max():
            skipped.append(lab)
            continue
        aucs[lab] = roc_auc_score(col, y_prob[known, i])
        aps[lab] = average_precision_score(col, y_prob[known, i])
    return {
        "auc": float(np.mean(list(aucs.values()))) if aucs else float("nan"),
        "ap": float(np.mean(list(aps.values()))) if aps else float("nan"),
        "per_label_auc": aucs,
        "skipped": skipped,
    }
 
 
# =============================================================================
# Train / evaluate
# =============================================================================
def masked_bce(logits, y):
    """BCE over the known labels only. NaN targets (studies without a report)
    are skipped; left in, one of them turns the whole loss into NaN."""
    known = ~torch.isnan(y)
    if not known.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits.float()[known], y[known]) 
