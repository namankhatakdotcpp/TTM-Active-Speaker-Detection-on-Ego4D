"""
Evaluate the fusion TTM model on the FULL dataset (clips_index.csv).

This script performs full-dataset evaluation by:
  1. Loading clips_index.csv (all ~636k clips)
  2. Ignoring any "split" column — treating entire dataset as test set
  3. Creating windows WITHOUT class balancing or resampling
  4. Running inference on all windows
  5. Aggregating predictions at TRACK level (video_uid, person_id)
  6. Computing metrics if labels are available

Reads:
  fusion/checkpoints/best_model.pth
  pipeline/data/clips_index.csv (full dataset)
  fusion/data/video_embeds/  +  fusion/data/audio_embeds/

Writes:
  fusion/results/full_predictions_track_level.json
  fusion/results/full_metrics_track_level.json (if labels available)
  fusion/results/full_diagnostics.json

Usage:
  python evaluate_full.py
  python evaluate_full.py --threshold 0.5
  python evaluate_full.py --threshold best  (finds optimal threshold on labels)
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

HERE     = os.path.dirname(os.path.abspath(__file__))
GROUP42  = os.path.dirname(HERE)
PIPELINE = os.path.join(GROUP42, "..", "ego4d_data", "v2", "full_scale", "pipeline")

sys.path.insert(0, HERE)
from fusion_model   import TTMFusionModel
from fusion_dataset import collate_fn, T_SEQ, STRIDE, VID_DIM, AUD_DIM

DEFAULT_CLIPS_CSV = os.path.join(PIPELINE, "data", "clips_index.csv")
VID_EMBED_DIR = os.path.join(HERE, "data", "video_embeds")
AUD_EMBED_DIR = os.path.join(HERE, "data", "audio_embeds")
CKPT          = os.path.join(HERE, "checkpoints", "best_model.pth")
RESULTS_DIR   = os.path.join(HERE, "results")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────────────────────────────────────────────────────────
# Full Dataset (No Split, No Resampling)
# ────────────────────────────────────────────────────────────────────────────

class FullDatasetNoSplit(Dataset):
    """
    Load all clips from CSV and create windows WITHOUT class balancing.
    
    No split filtering — all clips are treated as test/evaluation set.
    No resampling — original class distribution is preserved.
    
    Args:
        clips_source    : CSV file path (str) OR a pre-filtered pandas DataFrame.
        video_embed_dir : directory of video embedding .npy files
        audio_embed_dir : directory of audio embedding .npy files
        t_seq           : clips per window (sequence length)
        stride          : step between window start positions within a track.
                          For full evaluation: use t_seq for non-overlapping windows
                          to avoid redundancy. Can use smaller stride for finer aggregation.
        min_clips       : minimum real (non-padded) clips a window must contain.
    """

    def __init__(
        self,
        clips_source,
        video_embed_dir: str,
        audio_embed_dir: str,
        t_seq: int = T_SEQ,
        stride: int = None,
        min_clips: int = None,
    ):
        self.video_dir = video_embed_dir
        self.audio_dir = audio_embed_dir
        self.t_seq     = t_seq
        self.stride    = stride if stride is not None else t_seq  # non-overlapping by default
        self.min_clips = max(1, t_seq // 4) if min_clips is None else min_clips

        # Group all clips by (video_uid, person_id) — NO split filtering
        groups = defaultdict(list)

        if isinstance(clips_source, str):
            df = pd.read_csv(clips_source)
        else:
            df = clips_source

        # Convert to list of dicts for consistent handling
        for row in df.to_dict("records"):
            key = (str(row["video_uid"]), str(row["person_id"]))
            groups[key].append(row)

        self.samples   = []
        label_counts   = Counter()
        total_windows  = 0
        skipped        = 0
        pad_fracs      = []

        for (video_uid, person_id), clips in groups.items():
            clips_sorted = sorted(clips, key=lambda r: int(r["clip_id"]))
            n = len(clips_sorted)

            for start in range(0, n, self.stride):
                chunk = clips_sorted[start : start + self.t_seq]
                total_windows += 1

                if len(chunk) < self.min_clips:
                    skipped += 1
                    continue

                clip_ids = [int(r["clip_id"]) for r in chunk]
                
                # Extract label if present
                label = None
                if "label" in chunk[0]:
                    try:
                        pos = sum(int(r["label"]) for r in chunk)
                        label = int(pos >= len(chunk) / 2.0)
                        label_counts[label] += 1
                    except (ValueError, TypeError):
                        label = None

                pad_fracs.append(1.0 - len(chunk) / self.t_seq)

                self.samples.append({
                    "utterance_id": (
                        f"{video_uid}:person{person_id}:"
                        f"{clip_ids[0]}-{clip_ids[-1]}"
                    ),
                    "video_uid": video_uid,
                    "person_id": person_id,
                    "clip_ids":  clip_ids,
                    "label":     label,
                })

        kept = len(self.samples)
        avg_windows_per_track = kept / max(len(groups), 1)
        avg_pad = float(np.mean(pad_fracs)) if pad_fracs else 0.0

        print(f"\n[FullDatasetNoSplit] FULL DATASET (NO SPLIT FILTERING)")
        print(f"  tracks        : {len(groups):,}")
        print(f"  windows total : {total_windows:,}  "
              f"(kept={kept:,}  skipped={skipped:,} below min_clips={self.min_clips})")
        if label_counts:
            print(f"  neg / pos     : {label_counts[0]:,} / {label_counts[1]:,}  "
                  f"(ratio {label_counts[0]/(label_counts[1] or 1):.2f}:1)")
        else:
            print(f"  labels        : NOT PRESENT in dataset")
        print(f"  avg windows/track : {avg_windows_per_track:.1f}")
        print(f"  avg padding/window: {100*avg_pad:.1f}%  "
              f"(t_seq={t_seq}, stride={self.stride}, min_clips={self.min_clips})")

        self.num_tracks = len(groups)
        self.num_windows = kept
        self.label_counts = label_counts

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        clip_ids = sample["clip_ids"]
        label    = sample["label"]

        video_seq = []
        audio_seq = []

        for cid in clip_ids[: self.t_seq]:
            v_path = os.path.join(self.video_dir, f"{cid}.npy")
            a_path = os.path.join(self.audio_dir, f"{cid}.npy")

            v = (np.load(v_path).astype(np.float32)
                 if os.path.exists(v_path) else np.zeros(VID_DIM, dtype=np.float32))
            a = (np.load(a_path).astype(np.float32)
                 if os.path.exists(a_path) else np.zeros(AUD_DIM, dtype=np.float32))

            video_seq.append(v.flatten()[:VID_DIM])
            audio_seq.append(a.flatten()[:AUD_DIM])

        # Pad to fixed window length
        while len(video_seq) < self.t_seq:
            video_seq.append(np.zeros(VID_DIM, dtype=np.float32))
            audio_seq.append(np.zeros(AUD_DIM, dtype=np.float32))

        video_feats = torch.tensor(np.stack(video_seq), dtype=torch.float32)
        audio_feats = torch.tensor(np.stack(audio_seq), dtype=torch.float32)

        label_tensor = (torch.tensor(label, dtype=torch.long)
                       if label is not None else torch.tensor(-1, dtype=torch.long))

        return video_feats, audio_feats, label_tensor, sample["utterance_id"]


def collate_fn_full(batch):
    """Custom collate for full dataset."""
    video  = torch.stack([b[0] for b in batch])
    audio  = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    uids   = [b[3] for b in batch]
    return video, audio, labels, uids


# ────────────────────────────────────────────────────────────────────────────
# Threshold Selection
# ────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(y_true, y_pred_proba):
    """Find threshold that maximises F1 for the positive (TTM) class."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.71, 0.01):
        f1 = f1_score(y_true, (y_pred_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), best_f1


# ────────────────────────────────────────────────────────────────────────────
# Track-Level Aggregation
# ────────────────────────────────────────────────────────────────────────────

def aggregate_to_track_level(
    window_predictions: List[Dict],
) -> Dict[str, List[Dict]]:
    """
    Aggregate window-level predictions to track level.
    
    Args:
        window_predictions: List of dicts with keys:
            - video_uid, person_id
            - probability (window-level prediction)
            - label (if available)
    
    Returns:
        tracks: Dict mapping (video_uid, person_id) → list of predictions per track
    """
    tracks = defaultdict(list)
    
    for pred in window_predictions:
        key = (pred["video_uid"], pred["person_id"])
        tracks[key].append(pred)
    
    track_aggregated = {}
    
    for (video_uid, person_id), window_preds in tracks.items():
        # Average probabilities across windows
        probs = np.array([p["probability"] for p in window_preds])
        avg_prob = float(np.mean(probs))
        
        # Majority voting for label (if available)
        labels = [p["label"] for p in window_preds if p["label"] is not None]
        if labels:
            majority_label = 1 if np.mean(labels) >= 0.5 else 0
        else:
            majority_label = None
        
        track_key = f"{video_uid}:person{person_id}"
        track_aggregated[track_key] = {
            "video_uid": video_uid,
            "person_id": person_id,
            "track_id": track_key,
            "avg_probability": avg_prob,
            "num_windows": len(window_preds),
            "true_label": majority_label,
            "window_probabilities": probs.tolist(),
        }
    
    return track_aggregated


# ────────────────────────────────────────────────────────────────────────────
# Metrics Computation
# ────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5):
    """Compute classification metrics at track level."""
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report, accuracy_score,
    )

    y_pred = (y_pred_proba >= threshold).astype(int)
    labels = [0, 1]
    present = sorted(set(y_true.tolist()))

    results = {
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "class_counts": dict(Counter(y_true.tolist())),
        "pred_pos_rate": float(y_pred.mean()),
    }

    auc = None
    if len(present) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
    results["auc_roc"] = auc

    report = classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=["not_ttm", "ttm"],
        digits=4, zero_division=0
    )
    results["report"] = report

    return results


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate fusion model on full dataset with track-level aggregation."
    )
    parser.add_argument(
        "--clips-csv",
        default=DEFAULT_CLIPS_CSV,
        help="Path to full clips CSV (default: clips_index.csv)",
    )
    parser.add_argument(
        "--checkpoint",
        default=CKPT,
        help="Path to saved model checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--t-seq",
        type=int,
        default=T_SEQ,
        help="Sequence length (window size) in clips",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Stride for window creation (default: t_seq, for non-overlapping windows)",
    )
    parser.add_argument(
        "--threshold",
        default="0.5",
        help="Classification threshold (default: 0.5, or 'best' to find optimal on labels)",
    )
    parser.add_argument(
        "--output-dir",
        default=RESULTS_DIR,
        help="Directory to save results",
    )
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ────────────────────────────────────────────────────────────────────────
    # Load Model
    # ────────────────────────────────────────────────────────────────────────
    print(f"\nLoading model from: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        sys.exit(1)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Infer config from checkpoint metadata if available
    # Default to "cross_attn" with 1 layer (matching train_fusion.py defaults)
    config = ckpt.get("config", {})
    fusion_type = config.get("fusion_type", "cross_attn")
    cross_attn_layers = config.get("cross_attn_layers", 1)

    model = TTMFusionModel(
        video_dim=512, audio_dim=512,
        proj_dim=256, lstm_hidden=256, lstm_layers=2,
        num_classes=2, dropout=0.4,
        fusion_type=fusion_type,
        cross_attn_layers=cross_attn_layers,
        use_no_audio_token=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Model loaded successfully")
    print(f"  fusion_type: {fusion_type}")
    print(f"  cross_attn_layers: {cross_attn_layers}")

    # ────────────────────────────────────────────────────────────────────────
    # Load Full Dataset
    # ────────────────────────────────────────────────────────────────────────
    print(f"\nLoading full dataset from: {args.clips_csv}")
    if not os.path.exists(args.clips_csv):
        print(f"ERROR: Clips CSV not found at {args.clips_csv}")
        sys.exit(1)

    stride = args.stride if args.stride is not None else args.t_seq
    ds = FullDatasetNoSplit(
        args.clips_csv,
        VID_EMBED_DIR,
        AUD_EMBED_DIR,
        t_seq=args.t_seq,
        stride=stride,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn_full,
    )

    # ────────────────────────────────────────────────────────────────────────
    # Run Inference
    # ────────────────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(ds):,} windows...")
    
    window_predictions = []
    all_labels_with_preds = []

    with torch.no_grad():
        for batch_idx, (v_feats, a_feats, labels, uids) in enumerate(loader):
            v_feats = v_feats.to(device)
            a_feats = a_feats.to(device)

            logits, attn = model(v_feats, a_feats)
            proba = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            for i, uid in enumerate(uids):
                # Parse utterance_id to extract video_uid, person_id
                parts = uid.split(":")
                video_uid = parts[0]
                person_id = parts[1].replace("person", "")
                
                label_val = labels[i].item() if labels[i].item() >= 0 else None

                window_predictions.append({
                    "utterance_id": uid,
                    "video_uid": video_uid,
                    "person_id": person_id,
                    "probability": float(proba[i]),
                    "label": label_val,
                })

                if label_val is not None:
                    all_labels_with_preds.append((label_val, proba[i]))

            if (batch_idx + 1) % max(1, len(loader) // 10) == 0:
                print(f"  [{batch_idx + 1:,} / {len(loader):,}] batches processed")

    print(f"Inference complete: {len(window_predictions):,} window predictions")

    # ────────────────────────────────────────────────────────────────────────
    # Determine Threshold
    # ────────────────────────────────────────────────────────────────────────
    threshold = 0.5
    
    if args.threshold == "best" and all_labels_with_preds:
        print(f"\nFinding optimal threshold...")
        y_true_for_thresh = np.array([x[0] for x in all_labels_with_preds])
        y_pred_for_thresh = np.array([x[1] for x in all_labels_with_preds])
        threshold, best_f1 = find_optimal_threshold(y_true_for_thresh, y_pred_for_thresh)
        print(f"  Optimal threshold: {threshold:.4f} (F1: {best_f1:.4f})")
    elif args.threshold != "best":
        try:
            threshold = float(args.threshold)
            print(f"Using threshold: {threshold:.4f}")
        except ValueError:
            print(f"WARNING: Invalid threshold '{args.threshold}', using default 0.5")
            threshold = 0.5

    # ────────────────────────────────────────────────────────────────────────
    # Aggregate to Track Level
    # ────────────────────────────────────────────────────────────────────────
    print(f"\nAggregating to track level...")
    track_predictions = aggregate_to_track_level(window_predictions)
    
    track_list = list(track_predictions.values())
    track_proba = np.array([t["avg_probability"] for t in track_list])
    track_labels = np.array([t["true_label"] for t in track_list if t["true_label"] is not None])

    print(f"  Total tracks: {len(track_list):,}")
    
    # Apply threshold to create final predictions
    for track in track_list:
        track["predicted_label"] = int(track["avg_probability"] >= threshold)
    
    # Compute label distribution
    pred_labels = np.array([t["predicted_label"] for t in track_list])
    pred_dist = Counter(pred_labels)
    
    print(f"  Predicted label distribution: {dict(pred_dist)}")
    if len(track_labels) > 0:
        true_dist = Counter(track_labels)
        print(f"  True label distribution     : {dict(true_dist)}")

    # ────────────────────────────────────────────────────────────────────────
    # Compute Metrics (if labels available)
    # ────────────────────────────────────────────────────────────────────────
    metrics = None
    metrics_best = None
    
    if len(track_labels) > 0:
        print(f"\nComputing metrics at track level...")
        metrics = compute_metrics(track_labels, track_proba, threshold=threshold)
        
        # If labels were available and we didn't use 'best' threshold, also find best
        if args.threshold != "best" and len(track_labels) > 1:
            best_t, best_f1 = find_optimal_threshold(track_labels, track_proba)
            metrics_best = compute_metrics(track_labels, track_proba, threshold=best_t)
            print(f"\n  Best threshold on tracks: {best_t:.4f} (F1: {best_f1:.4f})")
    else:
        print(f"\nNo labels available in dataset — skipping metrics computation")

    # ────────────────────────────────────────────────────────────────────────
    # Print Summary
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"FULL DATASET EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Dataset        : {args.clips_csv}")
    print(f"Checkpoint     : {args.checkpoint}")
    print(f"Tracks         : {len(track_list):,}")
    print(f"Windows        : {len(window_predictions):,}")
    print(f"Avg windows/track : {len(window_predictions) / max(len(track_list), 1):.2f}")
    print(f"t_seq          : {args.t_seq}")
    print(f"stride         : {stride}")
    print(f"Threshold      : {threshold:.4f}")
    
    if metrics:
        auc_text = "n/a (single class)" if metrics["auc_roc"] is None else f"{metrics['auc_roc']:.4f}"
        print(f"\nMetrics (Threshold={threshold:.4f}):")
        print(f"  Accuracy   : {metrics['accuracy']:.4f}")
        print(f"  F1 (TTM)   : {metrics['f1']:.4f}")
        print(f"  Precision  : {metrics['precision']:.4f}")
        print(f"  Recall     : {metrics['recall']:.4f}")
        print(f"  AUC-ROC    : {auc_text}")
        print(f"  Pred pos % : {100*metrics['pred_pos_rate']:.2f}%")
        print(f"\n{metrics['report']}")
        
        if metrics_best:
            print(f"\nMetrics (Best Threshold={metrics_best['threshold']:.4f}):")
            print(f"  Accuracy   : {metrics_best['accuracy']:.4f}")
            print(f"  F1 (TTM)   : {metrics_best['f1']:.4f}")
            print(f"  Precision  : {metrics_best['precision']:.4f}")
            print(f"  Recall     : {metrics_best['recall']:.4f}")
            print(f"  AUC-ROC    : {auc_text}")
            print(f"\n{metrics_best['report']}")
    
    print(f"{'='*70}\n")

    # ────────────────────────────────────────────────────────────────────────
    # Save Results
    # ────────────────────────────────────────────────────────────────────────
    print(f"Saving results to: {args.output_dir}")

    # Track-level predictions
    out_predictions = os.path.join(args.output_dir, "full_predictions_track_level.json")
    with open(out_predictions, "w") as f:
        json.dump(
            {
                "metadata": {
                    "dataset": args.clips_csv,
                    "checkpoint": args.checkpoint,
                    "threshold": threshold,
                    "num_tracks": len(track_list),
                    "num_windows": len(window_predictions),
                    "t_seq": args.t_seq,
                    "stride": stride,
                },
                "tracks": track_list,
            },
            f,
            indent=2,
        )
    print(f"  ✓ Track predictions: {out_predictions}")

    # Metrics
    if metrics:
        out_metrics = os.path.join(args.output_dir, "full_metrics_track_level.json")
        metrics_to_save = metrics.copy()
        if metrics_best:
            metrics_to_save["metrics_best_threshold"] = metrics_best
        
        with open(out_metrics, "w") as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"  ✓ Metrics: {out_metrics}")

    # Diagnostics
    diagnostics = {
        "num_tracks": len(track_list),
        "num_windows": len(window_predictions),
        "avg_windows_per_track": float(len(window_predictions) / max(len(track_list), 1)),
        "window_proba_stats": {
            "mean": float(np.mean(track_proba)),
            "std": float(np.std(track_proba)),
            "min": float(np.min(track_proba)),
            "max": float(np.max(track_proba)),
            "median": float(np.median(track_proba)),
        },
        "predicted_label_dist": dict(Counter([t["predicted_label"] for t in track_list])),
        "true_label_dist": dict(Counter(track_labels.tolist())) if len(track_labels) > 0 else None,
        "dataset_stats": {
            "num_tracks_in_dataset": ds.num_tracks,
            "num_windows_in_dataset": ds.num_windows,
            "label_counts_in_dataset": dict(ds.label_counts) if ds.label_counts else None,
        }
    }
    
    out_diag = os.path.join(args.output_dir, "full_diagnostics.json")
    with open(out_diag, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"  ✓ Diagnostics: {out_diag}\n")


if __name__ == "__main__":
    main()
