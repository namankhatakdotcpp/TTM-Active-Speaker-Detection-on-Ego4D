"""
create_balanced_split.py

Re-splits clips_index.csv at the (video_uid, person_id) TRACK level.

Problem with the original split
--------------------------------
The original split column has 1,278 train tracks / 43 val tracks (~3% val).
43 val tracks is far too few — metrics have very high variance epoch-to-epoch.

What this script does
---------------------
Reads the full clips_index.csv (636k clips, 1321 tracks) and assigns a new
80/20 split at the TRACK level so the same person never appears in both
train and val. Stratification is applied on the majority-vote track label
so the TTM:non-TTM track ratio is preserved.

Result
------
  Train : ~1057 tracks  (~509k clips)  ~94.4% non-TTM / ~5.6% TTM
  Val   :  ~264 tracks  (~127k clips)  ~94.4% non-TTM / ~5.6% TTM
  Leakage: zero — no (video_uid, person_id) pair in both splits

Output
------
  pipeline/data/clips_index_resplit.csv   (drop-in replacement for clips_index.csv)

Usage
-----
  python create_balanced_split.py
  python train_fusion.py --clips-csv <path>/clips_index_resplit.csv
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd

HERE    = Path(__file__).parent
GROUP42 = HERE.parent

CLIPS_CSV = (
    GROUP42.parent / "ego4d_data" / "v2" / "full_scale" / "pipeline" / "data" / "clips_index.csv"
)
OUT_CSV = CLIPS_CSV.parent / "clips_index_resplit.csv"


def majority_label(labels):
    return int(sum(labels) >= len(labels) / 2.0)


def stratified_track_split(pos_tracks, neg_tracks, val_frac, seed):
    """Split TTM-majority and non-TTM-majority tracks independently for class balance."""
    rng = random.Random(seed)
    pos = list(pos_tracks)
    neg = list(neg_tracks)
    rng.shuffle(pos)
    rng.shuffle(neg)

    n_val_pos = max(1, round(len(pos) * val_frac)) if pos else 0
    n_val_neg = max(1, round(len(neg) * val_frac)) if neg else 0

    val_keys   = set(pos[:n_val_pos])  | set(neg[:n_val_neg])
    train_keys = set(pos[n_val_pos:]) | set(neg[n_val_neg:])
    return train_keys, val_keys


def print_stats(df, label="SPLIT"):
    W = 64
    print(f"\n  {'─'*W}")
    print(f"  {label:^{W}}")
    print(f"  {'─'*W}")
    total_tracks = df.groupby(['video_uid', 'person_id']).ngroups
    for sp in ['train', 'val']:
        s = df[df.split == sp]
        n_tracks = s.groupby(['video_uid', 'person_id']).ngroups
        ttm     = int(s.label.sum())
        non_ttm = int((s.label == 0).sum())
        n       = len(s)
        print(f"  {sp.upper():5s}  clips={n:>8,}  tracks={n_tracks:>5,}  "
              f"TTM={ttm:>6,} ({100*ttm/n:.1f}%)  "
              f"non-TTM={non_ttm:>7,} ({100*non_ttm/n:.1f}%)")
    train_t = set(map(tuple, df[df.split=='train'][['video_uid','person_id']].values))
    val_t   = set(map(tuple, df[df.split=='val'][['video_uid','person_id']].values))
    overlap = train_t & val_t
    print(f"  {'─'*W}")
    print(f"  Total tracks : {total_tracks:,}  |  "
          f"Leakage: {'OK — zero overlap' if not overlap else f'ERROR — {len(overlap)} tracks overlap!'}")
    print(f"  {'─'*W}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val-frac", type=float, default=0.20,
                   help="Fraction of tracks for val (default 0.20)")
    p.add_argument("--seed",     type=int,   default=42)
    p.add_argument("--out",      default=str(OUT_CSV))
    args = p.parse_args()

    W = 64
    print(f"\n{'='*W}")
    print(f"  FULL DATASET TRACK-LEVEL RESPLIT")
    print(f"{'='*W}")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"\nReading: {CLIPS_CSV}")
    if not CLIPS_CSV.exists():
        raise FileNotFoundError(f"Not found: {CLIPS_CSV}")
    df = pd.read_csv(CLIPS_CSV)
    print(f"  {len(df):,} clips  |  "
          f"TTM={int(df.label.sum()):,}  non-TTM={int((df.label==0).sum()):,}  |  "
          f"tracks={df.groupby(['video_uid','person_id']).ngroups:,}")

    # Show original split for comparison
    if 'split' in df.columns:
        print(f"\n  Original split (for comparison):")
        print_stats(df, "ORIGINAL SPLIT")

    # ── Track-level majority-vote labels ──────────────────────────────────────
    track_labels = (
        df.groupby(['video_uid', 'person_id'])['label']
        .apply(majority_label)
        .reset_index()
        .rename(columns={'label': 'track_label'})
    )
    pos_tracks = set(
        map(tuple, track_labels[track_labels.track_label == 1][['video_uid','person_id']].values)
    )
    neg_tracks = set(
        map(tuple, track_labels[track_labels.track_label == 0][['video_uid','person_id']].values)
    )
    print(f"  Track-level majority labels:")
    print(f"    TTM-majority tracks     : {len(pos_tracks):,}")
    print(f"    non-TTM-majority tracks : {len(neg_tracks):,}")

    # ── Stratified 80/20 track split ─────────────────────────────────────────
    print(f"\n  Splitting {100*(1-args.val_frac):.0f}% train / "
          f"{100*args.val_frac:.0f}% val at track level  (seed={args.seed})")
    train_keys, val_keys = stratified_track_split(pos_tracks, neg_tracks, args.val_frac, args.seed)

    # ── Assign new split column ───────────────────────────────────────────────
    track_to_split = {k: 'val' for k in val_keys}
    track_to_split.update({k: 'train' for k in train_keys})
    df['split'] = df.apply(
        lambda r: track_to_split.get((r['video_uid'], r['person_id']), 'train'),
        axis=1
    )

    # ── Verify ────────────────────────────────────────────────────────────────
    print_stats(df, "NEW SPLIT")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  Saved: {out}")

    print(f"\n{'='*W}")
    print(f"  NEXT STEPS")
    print(f"{'='*W}")
    print(f"\n  # Train on full 636k clips with weighted sampler:")
    print(f"  python train_fusion.py \\")
    print(f"      --clips-csv {out}\n")
    print(f"  # Evaluate on full dataset after training:")
    print(f"  python evaluate_full.py\n")


if __name__ == "__main__":
    main()
