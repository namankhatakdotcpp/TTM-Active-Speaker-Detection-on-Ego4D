"""
Train BiLSTM + cross-modal attention fusion model for TTM detection.

Architecture:
  I3D video embeds   [B, T, 512] ─┐
                                   ├→ CrossModalAttention → BiLSTM → Bahdanau → Linear
  ResNet-18 audio embeds [B, T, 512] ─┘

Reads:
  fusion/data/video_embeds/{clip_id}.npy
  fusion/data/audio_embeds/{clip_id}.npy
  pipeline/data/clips_index_resplit.csv   (full 636k clips, proper 80/20 track split)
    -- OR --
  pipeline/data/balanced_clips_70k.csv    (35k TTM + 35k non-TTM, faster iteration)

Writes:
  fusion/checkpoints/best_model.pth
  fusion/results/metrics.json

Usage:
  # Full dataset (recommended for best F1):
  python train_fusion.py

  # Balanced 70k subset (faster, good for debugging):
  python train_fusion.py --clips-csv /path/to/balanced_clips_70k.csv
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_recall_curve,
)
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE     = os.path.dirname(os.path.abspath(__file__))
GROUP42  = os.path.dirname(HERE)
PIPELINE = os.path.join(GROUP42, "..", "ego4d_data", "v2", "full_scale", "pipeline")

sys.path.insert(0, HERE)
from fusion_model   import TTMFusionModel
from fusion_dataset import FusionDataset, collate_fn

# Priority order for default CSV:
#   1. clips_index_resplit.csv  — full 636k clips, proper 80/20 track-level split
#   2. balanced_clips_70k.csv   — 70k balanced subset (fallback)
_FULL_RESPLIT = os.path.join(PIPELINE, "data", "clips_index_resplit.csv")
_BALANCED_70K = os.path.join(PIPELINE, "data", "balanced_clips_70k.csv")
DEFAULT_CLIPS_CSV = _FULL_RESPLIT if os.path.exists(_FULL_RESPLIT) else _BALANCED_70K

VID_EMBED_DIR = os.path.join(HERE, "data", "video_embeds")
AUD_EMBED_DIR = os.path.join(HERE, "data", "audio_embeds")
CKPT_DIR      = os.path.join(HERE, "checkpoints")
RESULTS       = os.path.join(HERE, "results", "metrics.json")

# ── hyperparameters ───────────────────────────────────────────────────────────
DEFAULT_EPOCHS       = 50
DEFAULT_BATCH_SIZE   = 256
DEFAULT_LR           = 3e-4
DEFAULT_WEIGHT_DECAY = 5e-4  # stronger L2 — reduces overfitting on small val sets
DEFAULT_DROPOUT      = 0.5    # higher dropout for same reason
DEFAULT_NUM_WORKERS  = 4
SAVE_EVERY           = 5
DEFAULT_T_SEQ        = 16
DEFAULT_STRIDE       = 4      # 75% overlap → more train windows per track
DEFAULT_VAL_STRIDE   = 16     # non-overlapping val windows → unbiased metrics
EARLY_STOPPING_PATIENCE = 10  # more patience; full-dataset epochs are slow

FIXED_THRESHOLD = 0.5         # used throughout training; tuned once post-training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}")


# ── helpers ───────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples. gamma=0 → plain cross-entropy."""

    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def batch_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == labels).float().mean().item()


def make_weighted_sampler(dataset) -> WeightedRandomSampler:
    """Per-window inverse-frequency sampler → each batch is ~50:50."""
    labels       = np.array([s["label"] for s in dataset.samples])
    class_counts = np.bincount(labels)
    class_weight = 1.0 / class_counts.astype(np.float64)
    sample_weight = torch.tensor(class_weight[labels], dtype=torch.float)
    return WeightedRandomSampler(
        weights=sample_weight,
        num_samples=len(dataset),
        replacement=True,
    )


def verify_splits(train_ds, val_ds):
    train_keys = {(s["video_uid"], s["person_id"]) for s in train_ds.samples}
    val_keys   = {(s["video_uid"], s["person_id"]) for s in val_ds.samples}
    overlap    = train_keys & val_keys

    print("\n" + "=" * 54)
    print("SPLIT VERIFICATION")
    print("=" * 54)
    print(f"  Train : {len(train_ds.samples):,} windows | {len(train_keys):,} tracks")
    print(f"  Val   : {len(val_ds.samples):,} windows  | {len(val_keys):,} tracks")

    if overlap:
        print(f"\n  *** LEAKAGE: {len(overlap)} tracks in BOTH splits! ***")
    else:
        print("\n  OK — zero track overlap between train and val.")

    for name, ds in [("Train", train_ds), ("Val", val_ds)]:
        lbls = [s["label"] for s in ds.samples]
        n, pos = len(lbls), sum(lbls)
        print(f"  {name:5s}  neg={n-pos:,} ({100*(n-pos)/n:.1f}%)  "
              f"pos={pos:,} ({100*pos/n:.1f}%)")
    print("=" * 54 + "\n")


def find_optimal_threshold(probs: np.ndarray, labels: np.ndarray):
    """Scan precision-recall curve for threshold that maximises F1."""
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    denom = precision[:-1] + recall[:-1]
    f1s   = np.where(denom > 0, 2 * precision[:-1] * recall[:-1] / denom, 0.0)
    best  = int(np.argmax(f1s))
    return float(thresholds[best]), float(f1s[best])


def print_val_metrics(probs: np.ndarray, labels: np.ndarray, threshold: float, epoch):
    preds = (probs >= threshold).astype(int)
    f1    = float(f1_score(labels, preds, zero_division=0))
    ap    = float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan")
    cm    = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(labels, preds, target_names=["non-TTM", "TTM"], zero_division=0)
    prec_ttm = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_ttm  = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    W = 54
    print(f"\n  {'─'*W}")
    print(f"  Epoch {epoch}  |  threshold = {threshold:.2f}")
    print(f"  {'─'*W}")
    print(f"  {'':20s}  Predicted non-TTM  TTM")
    print(f"  Actual non-TTM     {tn:12d}  {fp:4d}")
    print(f"         TTM         {fn:12d}  {tp:4d}")
    print(f"  {'─'*W}")
    print(f"  TTM  precision={prec_ttm:.4f}  recall={rec_ttm:.4f}  F1={f1:.4f}  AP={ap:.4f}")
    print(f"  {'─'*W}")
    print(report)
    return f1


# ── training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, scaler, is_train, noise_scale=0.0):
    model.train(is_train)
    total_loss = total_acc = 0.0
    n = 0
    t0 = time.time()
    all_probs  = []
    all_labels = []

    if is_train:
        optimizer.zero_grad()

    for step, (v_feats, a_feats, labels, _) in enumerate(loader):
        v_feats = v_feats.to(device, non_blocking=True)
        a_feats = a_feats.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        lam      = 1.0
        labels_b = labels

        if is_train and noise_scale > 0.0:
            # Random modality dropout (5% chance each).
            r = np.random.random()
            if r < 0.05:
                v_feats = torch.zeros_like(v_feats)
            elif r < 0.10:
                a_feats = torch.zeros_like(a_feats)

            # Temporal masking — zero out 8% of time steps.
            tmask = torch.rand(v_feats.shape[0], v_feats.shape[1], device=device) < 0.08
            v_feats = v_feats.masked_fill(tmask.unsqueeze(-1), 0.0)
            a_feats = a_feats.masked_fill(tmask.unsqueeze(-1), 0.0)

            # Gaussian noise.
            v_feats = v_feats + torch.randn_like(v_feats) * noise_scale
            a_feats = a_feats + torch.randn_like(a_feats) * noise_scale

            # MixUp — 25% probability, Beta(0.4, 0.4) for softer mixing.
            if np.random.random() < 0.25:
                lam      = float(np.random.beta(0.4, 0.4))
                idx      = torch.randperm(v_feats.size(0), device=device)
                v_feats  = lam * v_feats + (1 - lam) * v_feats[idx]
                a_feats  = lam * a_feats + (1 - lam) * a_feats[idx]
                labels_b = labels[idx]

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits, _ = model(v_feats, a_feats)
            loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels_b)

        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            probs = F.softmax(logits.detach(), dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        total_loss += loss.item()
        total_acc  += batch_accuracy(logits.detach(), labels)
        n += 1

        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = (len(loader) - step - 1) / max((step + 1) / elapsed, 1e-9)
            phase = "train" if is_train else "val"
            print(f"\r  [{phase}] {step+1}/{len(loader)} | "
                  f"loss {total_loss/n:.4f} | acc {total_acc/n:.4f} | "
                  f"ETA {eta/60:.1f}min   ", end="", flush=True)

    print()
    return (
        total_loss / n,
        total_acc  / n,
        np.array(all_probs,  dtype=np.float32),
        np.array(all_labels, dtype=np.int32),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--clips-csv",         default=DEFAULT_CLIPS_CSV)
    p.add_argument("--epochs",            type=int,   default=DEFAULT_EPOCHS)
    p.add_argument("--batch-size",        type=int,   default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr",                type=float, default=DEFAULT_LR)
    p.add_argument("--weight-decay",      type=float, default=DEFAULT_WEIGHT_DECAY)
    p.add_argument("--dropout",           type=float, default=DEFAULT_DROPOUT)
    p.add_argument("--num-workers",       type=int,   default=DEFAULT_NUM_WORKERS)
    p.add_argument("--t-seq",             type=int,   default=DEFAULT_T_SEQ)
    p.add_argument("--stride",            type=int,   default=DEFAULT_STRIDE)
    p.add_argument("--val-stride",        type=int,   default=None)
    p.add_argument("--min-clips",         type=int,   default=None)
    p.add_argument("--fusion-type",       default="cross_attn",
                   choices=["concat", "cross_attn"])
    p.add_argument("--cross-attn-layers", type=int,   default=2)
    p.add_argument("--no-sampler",        action="store_true",
                   help="Disable WeightedRandomSampler (only for pre-balanced datasets).")
    return p.parse_args()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    early_stop_counter = 0
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

    val_stride = args.val_stride if args.val_stride is not None else DEFAULT_VAL_STRIDE

    # ── load & split dataset ──────────────────────────────────────────────────
    print(f"\nLoading dataset from: {args.clips_csv}")
    df = pd.read_csv(args.clips_csv)

    if "split" in df.columns:
        train_df = df[df["split"] == "train"].reset_index(drop=True)
        val_df   = df[df["split"] == "val"].reset_index(drop=True)
        split_source = "pre-computed split column"
    else:
        track_info = (
            df.groupby(["video_uid", "person_id"])["label"]
            .apply(lambda x: int(x.sum() >= len(x) / 2.0))
            .reset_index()
            .rename(columns={"label": "track_label"})
        )
        track_labels = track_info["track_label"].tolist()
        n_tracks = len(track_info)
        train_idx, val_idx = train_test_split(
            range(n_tracks), test_size=0.20, stratify=track_labels, random_state=42,
        )
        train_tracks = track_info.iloc[list(train_idx)][["video_uid", "person_id"]]
        val_tracks   = track_info.iloc[list(val_idx)][["video_uid", "person_id"]]
        train_df = df.merge(train_tracks, on=["video_uid", "person_id"]).reset_index(drop=True)
        val_df   = df.merge(val_tracks,   on=["video_uid", "person_id"]).reset_index(drop=True)
        split_source = "sklearn 80/20 track-level split"

    def _clip_stats(d):
        neg = int((d["label"] == 0).sum())
        pos = int((d["label"] == 1).sum())
        tracks = d.groupby(["video_uid", "person_id"]).ngroups
        return neg, pos, tracks

    tr_neg, tr_pos, tr_tracks = _clip_stats(train_df)
    va_neg, va_pos, va_tracks = _clip_stats(val_df)
    total_clips = len(df)

    print(f"\n  Total clips : {total_clips:,}")
    print(f"  Train clips : {len(train_df):,} | tracks={tr_tracks:,} | "
          f"neg={tr_neg:,} ({100*tr_neg/len(train_df):.1f}%) | "
          f"pos={tr_pos:,} ({100*tr_pos/len(train_df):.1f}%)")
    print(f"  Val   clips : {len(val_df):,} | tracks={va_tracks:,} | "
          f"neg={va_neg:,} ({100*va_neg/len(val_df):.1f}%) | "
          f"pos={va_pos:,} ({100*va_pos/len(val_df):.1f}%)")

    train_ds = FusionDataset(
        train_df, "train", VID_EMBED_DIR, AUD_EMBED_DIR,
        t_seq=args.t_seq, stride=args.stride, min_clips=args.min_clips,
    )
    val_ds = FusionDataset(
        val_df, "val", VID_EMBED_DIR, AUD_EMBED_DIR,
        t_seq=args.t_seq, stride=val_stride, min_clips=args.min_clips,
    )
    verify_splits(train_ds, val_ds)

    # ── data loader + loss ────────────────────────────────────────────────────
    # WeightedRandomSampler balances window-level distribution to ~50:50.
    # With balanced batches, CrossEntropyLoss is better calibrated than
    # FocalLoss(gamma=2): gamma=2 additionally focuses on hard examples, which
    # with already-balanced batches inflates false positives and hurts precision.
    if not args.no_sampler:
        sampler = make_weighted_sampler(train_ds)
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_fn, drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        sampler_status = "ON  (window 50:50, CrossEntropyLoss + label_smoothing=0.1)"
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True,
            collate_fn=collate_fn, drop_last=(len(train_ds) >= args.batch_size),
            persistent_workers=(args.num_workers > 0),
        )
        n_pos_w = sum(s["label"] for s in train_ds.samples)
        n_neg_w = len(train_ds.samples) - n_pos_w
        total_w = len(train_ds.samples)
        weights = torch.tensor(
            [total_w / (2.0 * n_neg_w), total_w / (2.0 * n_pos_w)],
            dtype=torch.float, device=device,
        )
        criterion = FocalLoss(gamma=1.5, weight=weights, label_smoothing=0.05)
        sampler_status = f"OFF (FocalLoss gamma=1.5, weights neg={weights[0]:.3f} pos={weights[1]:.3f})"

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=(args.num_workers > 0),
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = TTMFusionModel(
        video_dim=512, audio_dim=512,
        proj_dim=256, lstm_hidden=256, lstm_layers=2,
        num_classes=2, dropout=args.dropout,
        fusion_type=args.fusion_type,
        cross_attn_layers=args.cross_attn_layers,
        use_no_audio_token=True,
    ).to(device)

    p = model.count_parameters()
    print(f"\n  Model params — total: {p['total']/1e6:.2f}M  trainable: {p['trainable']/1e6:.2f}M")

    # ── optimizer + scheduler ─────────────────────────────────────────────────
    # ReduceLROnPlateau: halves LR only when val AP stagnates for 3 epochs.
    # Unlike CosineAnnealingLR, it does NOT decay blindly every epoch.
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max",   # maximise val AP
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    scaler = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_ap  = 0.0
    best_val_f1  = 0.0
    best_val_acc = 0.0
    history      = []

    # ── startup banner ────────────────────────────────────────────────────────
    tr_win_neg = sum(1 for s in train_ds.samples if s["label"] == 0)
    tr_win_pos = sum(1 for s in train_ds.samples if s["label"] == 1)
    va_win_neg = sum(1 for s in val_ds.samples   if s["label"] == 0)
    va_win_pos = sum(1 for s in val_ds.samples   if s["label"] == 1)

    W = 66
    print(f"\n  {'='*W}")
    print(f"  TRAINING CONFIGURATION")
    print(f"  {'='*W}")
    print(f"  CSV          : {os.path.basename(args.clips_csv)}")
    print(f"  Split source : {split_source}")
    print(f"  Train clips  : {len(train_df):,} | tracks={tr_tracks:,}")
    print(f"  Val   clips  : {len(val_df):,}  | tracks={va_tracks:,}")
    print(f"  Train windows: {len(train_ds):,}  neg={tr_win_neg:,} pos={tr_win_pos:,}")
    print(f"  Val   windows: {len(val_ds):,}   neg={va_win_neg:,} pos={va_win_pos:,}")
    print(f"  {'─'*W}")
    print(f"  Epochs       : {args.epochs}  (early stop patience={EARLY_STOPPING_PATIENCE})")
    print(f"  Batch size   : {args.batch_size}")
    print(f"  LR           : {args.lr}  (ReduceLROnPlateau ×0.5 after 3 stagnant epochs)")
    print(f"  Weight decay : {args.weight_decay}")
    print(f"  Dropout      : {args.dropout}")
    print(f"  Threshold    : {FIXED_THRESHOLD} (fixed during training; tuned once post-training)")
    print(f"  Fusion       : {args.fusion_type}  cross_attn_layers={args.cross_attn_layers}")
    print(f"  Sampler      : {sampler_status}")
    print(f"  Early stop   : patience={EARLY_STOPPING_PATIENCE}  metric=val AP")
    print(f"  Device       : {device}")
    print(f"  {'='*W}\n")

    # ── training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch}/{args.epochs}]  lr={current_lr:.2e}")

        train_loss, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            is_train=True, noise_scale=0.02,
        )
        val_loss, val_acc, val_probs, val_labels = run_epoch(
            model, val_loader, criterion, optimizer, scaler, is_train=False,
        )

        val_f1 = float(f1_score(val_labels, (val_probs >= FIXED_THRESHOLD).astype(int), zero_division=0))
        val_ap = (
            float(average_precision_score(val_labels, val_probs))
            if len(np.unique(val_labels)) > 1 else float("nan")
        )

        # Scheduler steps on val AP — only reduces LR when AP stagnates.
        if not np.isnan(val_ap):
            scheduler.step(val_ap)

        new_lr = optimizer.param_groups[0]["lr"]
        lr_tag = f"  *** LR reduced → {new_lr:.2e}" if new_lr < current_lr else ""

        elapsed = time.time() - t0
        record = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    5),
            "val_f1":     round(val_f1,     5),
            "val_ap":     round(val_ap,     5) if not np.isnan(val_ap) else None,
            "lr":         new_lr,
        }
        history.append(record)

        print(f"  train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} "
              f"f1={val_f1:.4f} AP={val_ap:.4f} | "
              f"time={elapsed/60:.1f}min{lr_tag}")

        print_val_metrics(val_probs, val_labels, threshold=FIXED_THRESHOLD, epoch=epoch)

        # Early stopping on val AP (threshold-agnostic, most reliable signal).
        if not np.isnan(val_ap) and val_ap > best_val_ap:
            best_val_ap  = val_ap
            best_val_f1  = val_f1
            best_val_acc = val_acc
            early_stop_counter = 0
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_acc":          val_acc,
                    "val_f1":           val_f1,
                    "val_ap":           val_ap,
                    "val_loss":         val_loss,
                },
                os.path.join(CKPT_DIR, "best_model.pth"),
            )
            print(f"  >> Best model saved  AP={val_ap:.4f}  F1={val_f1:.4f}  acc={val_acc:.4f}")
        else:
            early_stop_counter += 1
            print(f"  EarlyStopping counter: {early_stop_counter}/{EARLY_STOPPING_PATIENCE}")
            if early_stop_counter >= EARLY_STOPPING_PATIENCE:
                print("\nEarly stopping triggered.")
                break

        if epoch % SAVE_EVERY == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth"))

        with open(RESULTS, "w") as f:
            json.dump(
                {
                    "best_val_ap":  best_val_ap,
                    "best_val_f1":  best_val_f1,
                    "best_val_acc": best_val_acc,
                    "history":      history,
                    "config": {
                        "epochs":            args.epochs,
                        "clips_csv":         args.clips_csv,
                        "batch_size":        args.batch_size,
                        "lr":                args.lr,
                        "weight_decay":      args.weight_decay,
                        "dropout":           args.dropout,
                        "num_workers":       args.num_workers,
                        "t_seq":             args.t_seq,
                        "stride":            args.stride,
                        "val_stride":        val_stride,
                        "fusion_type":       args.fusion_type,
                        "cross_attn_layers": args.cross_attn_layers,
                        "weighted_sampler":  not args.no_sampler,
                        "min_clips":         train_ds.min_clips,
                        "fixed_threshold":   FIXED_THRESHOLD,
                    },
                },
                f, indent=2,
            )

    # ── post-training: single threshold tune on best checkpoint ───────────────
    print("\n" + "=" * 54)
    print("POST-TRAINING THRESHOLD TUNING  (done once, on best model)")
    print("=" * 54)

    best_ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")
    ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, val_probs, val_labels = run_epoch(
        model, val_loader, criterion, optimizer, scaler, is_train=False,
    )

    best_thresh, f1_tuned = find_optimal_threshold(val_probs, val_labels)
    f1_default = float(f1_score(val_labels, (val_probs >= FIXED_THRESHOLD).astype(int), zero_division=0))
    ap_final   = (
        float(average_precision_score(val_labels, val_probs))
        if len(np.unique(val_labels)) > 1 else float("nan")
    )

    print(f"\n  AUC-PR  : {ap_final:.4f}  (random baseline: {val_labels.mean():.4f})")
    print(f"\n  Threshold={FIXED_THRESHOLD:.2f}  F1={f1_default:.4f}")
    print(f"  Threshold={best_thresh:.3f}  F1={f1_tuned:.4f}  ← optimal (used for inference)")
    print(f"  Gain from threshold tuning: {f1_tuned - f1_default:+.4f}\n")

    print("── Metrics at optimal threshold ──")
    print_val_metrics(val_probs, val_labels, threshold=best_thresh, epoch="final")

    with open(RESULTS) as f:
        saved = json.load(f)
    saved["optimal_threshold"] = round(best_thresh, 4)
    saved["final_ap"]          = round(ap_final, 5) if not np.isnan(ap_final) else None
    with open(RESULTS, "w") as f:
        json.dump(saved, f, indent=2)

    print(f"\nDone. Best val AP={best_val_ap:.4f}  F1@{FIXED_THRESHOLD}={best_val_f1:.4f}  "
          f"F1@optimal={f1_tuned:.4f}  optimal_threshold={best_thresh:.3f}")
    print(f"Results saved to: {RESULTS}")


if __name__ == "__main__":
    main()
