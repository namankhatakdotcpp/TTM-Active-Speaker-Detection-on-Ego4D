"""
Evaluate the fusion TTM model: AUC-ROC, F1, precision, recall.

Reads:
  fusion/checkpoints/best_model.pth
  pipeline/data/clips_index.csv
  fusion/data/video_embeds/  +  fusion/data/audio_embeds/

Writes:
  fusion/results/eval_metrics.json
  fusion/results/confusion_matrix.txt

Usage:
  python evaluate_fusion.py
  python evaluate_fusion.py --split val
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE     = os.path.dirname(os.path.abspath(__file__))
GROUP42  = os.path.dirname(HERE)
PIPELINE = os.path.join(GROUP42, "..", "ego4d_data", "v2", "full_scale", "pipeline")

sys.path.insert(0, HERE)
from fusion_model   import TTMFusionModel
from fusion_dataset import FusionDataset, collate_fn

DEFAULT_CLIPS_CSV = os.path.join(PIPELINE, "data", "balanced_clips_70k.csv")
VID_EMBED_DIR = os.path.join(HERE, "data", "video_embeds")
AUD_EMBED_DIR = os.path.join(HERE, "data", "audio_embeds")
CKPT          = os.path.join(HERE, "checkpoints", "best_model.pth")
RESULTS_DIR   = os.path.join(HERE, "results")
TRAIN_RESULTS  = os.path.join(HERE, "results", "metrics.json")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_best_threshold(y_true, y_pred_proba):
    """Find threshold that maximises F1 for the positive (TTM) class."""
    from sklearn.metrics import f1_score
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.20, 0.71, 0.01):
        f1 = f1_score(y_true, (y_pred_proba >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def compute_metrics(y_true, y_pred_proba, threshold: float = 0.5):
    """Compute AUC-ROC, F1, precision, recall from arrays."""
    from sklearn.metrics import (
        roc_auc_score, f1_score, precision_score, recall_score,
        confusion_matrix, classification_report,
    )

    y_pred = (y_pred_proba >= threshold).astype(int)
    labels = [0, 1]
    present = sorted(set(y_true.tolist()))

    auc = None
    if len(present) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
    f1     = f1_score(y_true, y_pred, zero_division=0)
    prec   = precision_score(y_true, y_pred, zero_division=0)
    rec    = recall_score(y_true, y_pred, zero_division=0)
    cm     = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred,
                                   labels=labels,
                                   target_names=["not_ttm", "ttm"],
                                   digits=4, zero_division=0)

    return {
        "auc_roc": auc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "confusion_matrix": cm.tolist(),
        "report": report,
        "class_counts": dict(Counter(y_true.tolist())),
        "pred_pos_rate": float(y_pred.mean()),
    }


def load_training_config():
    if not os.path.exists(TRAIN_RESULTS):
        return {}
    with open(TRAIN_RESULTS) as f:
        data = json.load(f)
    return data.get("config", {})


def infer_arch_from_checkpoint(sd: dict) -> dict:
    """
    Derive TTMFusionModel constructor kwargs from a checkpoint state_dict.
    Works for any checkpoint regardless of which training run produced it.
    """
    # proj_dim: video_proj.0.weight is [proj_dim, video_dim]
    proj_dim = sd["video_proj.0.weight"].shape[0]

    # lstm_hidden: bilstm.weight_ih_l0 is [4*hidden, input_size]
    lstm_hidden = sd["bilstm.weight_ih_l0"].shape[0] // 4

    # lstm_layers: weight_ih_l{i} exists for each layer (forward direction)
    lstm_layers = sum(1 for k in sd if k.startswith("bilstm.weight_ih_l") and "_reverse" not in k)

    # fusion_type: cross_attn layers present only in cross_attn mode
    fusion_type = "cross_attn" if any("cross_attn" in k for k in sd) else "concat"

    # cross_attn_layers: count CrossModalAttentionBlock layers
    cross_attn_layers = len({k.split(".")[2] for k in sd if k.startswith("cross_attn.layers.")}) if fusion_type == "cross_attn" else 1

    # use_no_audio_token: key is present only when the token was enabled
    use_no_audio_token = "no_audio_token" in sd

    return dict(
        video_dim=512,
        audio_dim=512,
        proj_dim=proj_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        num_classes=2,
        dropout=0.0,           # dropout doesn't affect inference
        fusion_type=fusion_type,
        cross_attn_layers=cross_attn_layers,
        use_no_audio_token=use_no_audio_token,
    )


def main():
    import pandas as pd
    train_cfg = load_training_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "all"],
                        help="'all' loads every row in --clips-csv regardless of split column.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--clips-csv", default=train_cfg.get("clips_csv", DEFAULT_CLIPS_CSV),
                        help="CSV to evaluate on.")
    parser.add_argument("--exclude-train-csv", default=None,
                        help="Exclude clips whose (video_uid, person_id) track appears in the "
                             "train split of this CSV. Use when --clips-csv is the full dataset "
                             "but you want only truly unseen tracks.")
    parser.add_argument("--t-seq", type=int, default=train_cfg.get("t_seq", 16))
    parser.add_argument("--stride", type=int, default=1,
                        help="Window stride for evaluation. stride=1 = maximum overlapping windows "
                             "= maximum support. Default: 1 (dense sampling).")
    parser.add_argument("--checkpoint", default=CKPT)
    parser.add_argument("--balanced-eval", action="store_true",
                        help="Undersample non-TTM windows to match TTM count before "
                             "computing metrics (equal support, honest precision/recall).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\nLoading model from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd   = ckpt["model_state_dict"]

    arch = infer_arch_from_checkpoint(sd)
    print(f"  Detected architecture:")
    print(f"    proj_dim={arch['proj_dim']}  lstm_hidden={arch['lstm_hidden']}  "
          f"lstm_layers={arch['lstm_layers']}")
    print(f"    fusion_type={arch['fusion_type']}  cross_attn_layers={arch['cross_attn_layers']}")
    print(f"    use_no_audio_token={arch['use_no_audio_token']}")

    model = TTMFusionModel(**arch).to(device)
    model.load_state_dict(sd)
    model.eval()

    stride = args.stride

    # ── build the evaluation DataFrame ───────────────────────────────────────
    print(f"\nBuilding eval dataset from: {args.clips_csv}  (split='{args.split}')")
    df = pd.read_csv(args.clips_csv)

    if args.split == "all":
        eval_df = df.copy()
    elif "split" in df.columns:
        eval_df = df[df["split"] == args.split].reset_index(drop=True)
    else:
        raise ValueError(f"--split '{args.split}' requested but CSV has no 'split' column. "
                         "Use --split all to evaluate on every row.")

    # Optionally remove any tracks that were used for TRAINING in another CSV.
    # This lets you test on the full clips_index.csv while guaranteeing that
    # every clip belongs to a track the model has never seen during training.
    if args.exclude_train_csv:
        train_ref = pd.read_csv(args.exclude_train_csv)
        if "split" in train_ref.columns:
            train_ref = train_ref[train_ref["split"] == "train"]
        train_tracks = set(zip(train_ref["video_uid"], train_ref["person_id"]))
        before = len(eval_df)
        eval_df = eval_df[
            ~eval_df.apply(lambda r: (r["video_uid"], r["person_id"]) in train_tracks, axis=1)
        ].reset_index(drop=True)
        removed = before - len(eval_df)
        print(f"  Excluded {removed:,} clips from {len(train_tracks):,} training tracks "
              f"(--exclude-train-csv).")

    n_ttm  = int((eval_df["label"] == 1).sum())
    n_nttm = int((eval_df["label"] == 0).sum())
    n_tracks = eval_df.groupby(["video_uid", "person_id"]).ngroups
    print(f"  Eval clips : {len(eval_df):,}  tracks={n_tracks}  "
          f"TTM={n_ttm:,} ({100*n_ttm/max(len(eval_df),1):.2f}%)  "
          f"non-TTM={n_nttm:,}")

    ds = FusionDataset(
        eval_df, args.split if args.split != "all" else "val",
        VID_EMBED_DIR, AUD_EMBED_DIR,
        t_seq=args.t_seq, stride=stride,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    all_labels = []
    all_proba  = []

    with torch.no_grad():
        for v_feats, a_feats, labels, _ in loader:
            v_feats = v_feats.to(device)
            a_feats = a_feats.to(device)

            logits, _ = model(v_feats, a_feats)
            proba = F.softmax(logits, dim=1)[:, 1].cpu().numpy()

            all_labels.extend(labels.numpy())
            all_proba.extend(proba)

    y_true  = np.array(all_labels)
    y_proba = np.array(all_proba)

    # ── Optional: balance the evaluation set ──────────────────────────────────
    # Undersample the majority (non-TTM) windows so support is equal.
    # This removes the threshold-calibration bias caused by extreme class ratios
    # and makes precision/recall/F1 directly comparable across experiments.
    if args.balanced_eval:
        rng = np.random.default_rng(args.seed)
        ttm_idx  = np.where(y_true == 1)[0]
        nttm_idx = np.where(y_true == 0)[0]
        n_minority = len(ttm_idx)
        if len(nttm_idx) > n_minority:
            keep_nttm = rng.choice(nttm_idx, size=n_minority, replace=False)
            keep_idx  = np.concatenate([ttm_idx, keep_nttm])
            keep_idx.sort()
            y_true  = y_true[keep_idx]
            y_proba = y_proba[keep_idx]
            print(f"\n[balanced-eval] Undersampled non-TTM {len(nttm_idx):,} → {n_minority:,} "
                  f"to match TTM count. Total windows: {len(y_true):,} (50:50)")
        else:
            print(f"\n[balanced-eval] Already balanced — no undersampling needed.")

    print("\nComputing metrics...")
    best_t = find_best_threshold(y_true, y_proba)
    metrics      = compute_metrics(y_true, y_proba, threshold=0.5)
    metrics_best = compute_metrics(y_true, y_proba, threshold=best_t)

    print(f"\n{'='*50}")
    print(f"Split         : {args.split}")
    print(f"Balanced eval : {'YES — equal support' if args.balanced_eval else 'NO — natural distribution'}")
    print(f"Windows       : {len(y_true):,}")
    print(f"CSV           : {args.clips_csv}")
    print(f"t_seq         : {args.t_seq}")
    print(f"stride        : {stride}")
    print(f"Classes       : {metrics['class_counts']}")
    auc_text = "n/a (single class present)" if metrics["auc_roc"] is None else f"{metrics['auc_roc']:.4f}"
    print(f"AUC-ROC       : {auc_text}")
    print(f"\n--- Threshold = 0.50 (default) ---")
    print(f"F1 (TTM)   : {metrics['f1']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(metrics["report"])
    print(f"--- Threshold = {best_t:.2f} (best F1) ---")
    print(f"F1 (TTM)   : {metrics_best['f1']:.4f}")
    print(f"Precision  : {metrics_best['precision']:.4f}")
    print(f"Recall     : {metrics_best['recall']:.4f}")
    print(metrics_best["report"])
    print(f"{'='*50}")

    tag = "full_test" if args.split == "all" else args.split
    out_json = os.path.join(RESULTS_DIR, f"eval_metrics_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "clips_csv": args.clips_csv,
                "exclude_train_csv": args.exclude_train_csv,
                "split": args.split,
                "t_seq": args.t_seq,
                "stride": stride,
                **{k: v for k, v in metrics.items() if k != "report"},
            },
            f,
            indent=2,
        )

    out_txt = os.path.join(RESULTS_DIR, f"confusion_matrix_{tag}.txt")
    with open(out_txt, "w") as f:
        f.write(f"Split: {args.split}\n")
        f.write(metrics["report"])

    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
