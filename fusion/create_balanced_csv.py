"""
create_balanced_csv.py

Creates a balanced, track-level-split CSV for fusion model training.

Sampling strategy (intelligent, NOT blind random)
-------------------------------------------------
1. Start with ALL TTM clips (35,207) from 462 TTM-containing tracks.
2. Sample non-TTM clips PROPORTIONALLY from TTM-containing tracks first:
     Each track's non-TTM quota  = round(track_ttm / total_ttm * target_non_ttm)
   This preserves within-track temporal context — the model sees both
   TTM and non-TTM clips from the same person in the same sequence.
3. If a track's non-TTM clips are fewer than its quota, take all of them
   and carry the shortfall to a fallback pool.
4. Fill any shortfall by sampling from pure non-TTM tracks, distributed
   proportionally to each track's clip count.

Why this matters
-----------------
Blind random sampling of non-TTM clips ignores track structure.
By preferring non-TTM from TTM-containing tracks, the model learns
the contrast within the same person's video — a much stronger signal
than comparing clips from completely different people.

Train/Val split
---------------
- Split is at the TRACK level (no person leaks between train and val).
- Stratified: TTM-containing tracks and pure non-TTM tracks split 80/20
  independently to preserve the proportion of each type in both splits.
- No temporal leakage possible.

Output columns (FusionDataset-compatible)
------------------------------------------
  clip_id, video_uid, person_id, label, split

Usage
------
  python create_balanced_csv.py                        # default 70k
  python create_balanced_csv.py --target-size 60000    # 60k
  python create_balanced_csv.py --val-frac 0.20        # default 20% val
  python create_balanced_csv.py --seed 42
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

HERE    = Path(__file__).parent
GROUP42 = HERE.parent

CLIPS_CSV = (
    GROUP42.parent / "ego4d_data" / "v2" / "full_scale" / "pipeline" / "data" / "clips_index.csv"
)
OUT_DIR = CLIPS_CSV.parent


def section(title, width=66):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def divider(width=66):
    print(f"  {'─'*width}")


def majority_label(labels):
    return int(sum(labels) >= len(labels) / 2.0)


def stratified_track_split(track_df, val_frac, seed):
    """
    Split tracks 80/20 independently for TTM-containing and pure non-TTM tracks.
    Returns a Series mapping (video_uid, person_id) → 'train' or 'val'.
    """
    rng = random.Random(seed)

    ttm_tracks  = track_df[track_df.has_ttm].index.tolist()
    pure_tracks = track_df[~track_df.has_ttm].index.tolist()

    rng.shuffle(ttm_tracks)
    rng.shuffle(pure_tracks)

    n_val_ttm  = max(1, round(len(ttm_tracks)  * val_frac)) if ttm_tracks  else 0
    n_val_pure = max(1, round(len(pure_tracks) * val_frac)) if pure_tracks else 0

    val_keys = set(ttm_tracks[:n_val_ttm]) | set(pure_tracks[:n_val_pure])

    return val_keys


def sample_non_ttm_intelligent(df, ttm_clips, target_non_ttm, seed):
    """
    Sample non-TTM clips proportionally from TTM-containing tracks first,
    then fill any shortfall from pure non-TTM tracks.

    Returns a DataFrame of sampled non-TTM clips.
    """
    rng = np.random.default_rng(seed)

    # ── TTM-containing tracks ─────────────────────────────────────────────────
    ttm_track_keys = set(
        zip(ttm_clips['video_uid'], ttm_clips['person_id'])
    )
    ttm_count_per_track = (
        ttm_clips.groupby(['video_uid', 'person_id']).size().rename('ttm_count')
    )
    total_ttm = len(ttm_clips)

    # Non-TTM clips that belong to TTM-containing tracks
    mask_in_ttm_tracks = df.apply(
        lambda r: (r['video_uid'], r['person_id']) in ttm_track_keys, axis=1
    )
    non_ttm_in_ttm_tracks = df[(df.label == 0) & mask_in_ttm_tracks]

    sampled_parts = []
    shortfall = 0

    for (vid, pid), ttm_cnt in ttm_count_per_track.items():
        # Proportional quota: tracks with more TTM contribute more non-TTM context
        quota = max(1, round(ttm_cnt / total_ttm * target_non_ttm))

        track_non_ttm = non_ttm_in_ttm_tracks[
            (non_ttm_in_ttm_tracks.video_uid == vid) &
            (non_ttm_in_ttm_tracks.person_id  == pid)
        ]

        if len(track_non_ttm) == 0:
            shortfall += quota
            continue

        if len(track_non_ttm) >= quota:
            idx = rng.choice(len(track_non_ttm), size=quota, replace=False)
            sampled_parts.append(track_non_ttm.iloc[idx])
        else:
            # Take all available; carry deficit to shortfall
            sampled_parts.append(track_non_ttm)
            shortfall += quota - len(track_non_ttm)

    from_ttm_tracks = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    already_selected = set(from_ttm_tracks['clip_id'].tolist()) if len(from_ttm_tracks) else set()

    total_from_ttm = len(from_ttm_tracks)

    # ── Fill shortfall from pure non-TTM tracks ───────────────────────────────
    # Any remaining budget (shortfall + rounding correction)
    still_needed = target_non_ttm - total_from_ttm

    from_pure_tracks = pd.DataFrame()
    if still_needed > 0:
        pure_non_ttm = df[
            (df.label == 0) &
            ~mask_in_ttm_tracks &
            ~df['clip_id'].isin(already_selected)
        ]

        if len(pure_non_ttm) == 0:
            print(f"  WARNING: No pure non-TTM clips available for shortfall ({still_needed:,})")
        else:
            # Sample proportionally by track size for diversity
            track_sizes = pure_non_ttm.groupby(['video_uid','person_id']).size()
            weights = (track_sizes / track_sizes.sum()).values

            selected_tracks = rng.choice(
                len(track_sizes),
                size=min(still_needed, len(pure_non_ttm)),
                replace=True,
                p=weights,
            )
            # Collect one clip per selection from the chosen track
            track_keys = track_sizes.index.tolist()
            per_track_pool = {
                k: pure_non_ttm[
                    (pure_non_ttm.video_uid == k[0]) & (pure_non_ttm.person_id == k[1])
                ] for k in track_keys
            }
            fill_rows = []
            track_pick_count = {}
            for ti in selected_tracks:
                k = track_keys[ti]
                track_pick_count[k] = track_pick_count.get(k, 0) + 1

            for k, cnt in track_pick_count.items():
                pool = per_track_pool[k]
                n_take = min(cnt, len(pool))
                idx = rng.choice(len(pool), size=n_take, replace=False)
                fill_rows.append(pool.iloc[idx])

            if fill_rows:
                from_pure_tracks = pd.concat(fill_rows, ignore_index=True)

    # ── Combine ───────────────────────────────────────────────────────────────
    parts = [p for p in [from_ttm_tracks, from_pure_tracks] if len(p) > 0]
    sampled_non_ttm = pd.concat(parts, ignore_index=True).drop_duplicates(subset='clip_id')

    print(f"\n  non-TTM sampling breakdown:")
    print(f"    From TTM-containing tracks : {total_from_ttm:,}")
    print(f"    From pure non-TTM tracks   : {len(from_pure_tracks):,}")
    print(f"    Total non-TTM sampled      : {len(sampled_non_ttm):,}  "
          f"(target was {target_non_ttm:,})")

    return sampled_non_ttm


def print_split_stats(df):
    W = 66
    divider()
    total_tracks = df.groupby(['video_uid','person_id']).ngroups
    for sp in ['train', 'val']:
        s = df[df.split == sp]
        if len(s) == 0:
            continue
        n_tracks = s.groupby(['video_uid','person_id']).ngroups
        ttm   = int(s.label.sum())
        n_non = int((s.label == 0).sum())
        n     = len(s)
        print(f"\n  {sp.upper()}")
        print(f"    clips   : {n:>8,}  ({100*n/len(df):.1f}% of balanced set)")
        print(f"    tracks  : {n_tracks:>8,}  ({100*n_tracks/total_tracks:.1f}% of tracks)")
        print(f"    TTM     : {ttm:>8,}  ({100*ttm/n:.2f}%)")
        print(f"    non-TTM : {n_non:>8,}  ({100*n_non/n:.2f}%)")

    train_t = set(map(tuple, df[df.split=='train'][['video_uid','person_id']].values))
    val_t   = set(map(tuple, df[df.split=='val'][['video_uid','person_id']].values))
    overlap = train_t & val_t
    divider()
    print(f"\n  Total balanced clips : {len(df):,}")
    print(f"  Total tracks        : {total_tracks:,}")
    print(f"  Leakage check       : {'OK — zero leakage' if not overlap else f'ERROR {len(overlap)} overlap!'}")
    divider()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv",         default=str(CLIPS_CSV))
    p.add_argument("--out-dir",     default=str(OUT_DIR))
    p.add_argument("--target-size", type=int, default=None,
                   help="Total balanced size (default: 2 × all TTM clips ≈ 70k). "
                        "Must be even. Half TTM, half non-TTM.")
    p.add_argument("--val-frac",    type=float, default=0.20)
    p.add_argument("--seed",        type=int,   default=42)
    args = p.parse_args()

    csv_path = Path(args.csv)
    out_dir  = Path(args.out_dir)

    # ── Load ──────────────────────────────────────────────────────────────────
    section("LOADING DATASET")
    print(f"\n  {csv_path}")
    df = pd.read_csv(csv_path)
    df['clip_id']   = df['clip_id'].astype(int)
    df['person_id'] = df['person_id'].astype(int)
    df['label']     = df['label'].astype(int)

    all_ttm_clips = df[df.label == 1].copy()
    all_non_clips = df[df.label == 0].copy()

    n_ttm = len(all_ttm_clips)
    n_non = len(all_non_clips)
    n_total = len(df)
    max_balanced = n_ttm * 2

    print(f"\n  Total clips    : {n_total:,}")
    print(f"  TTM clips      : {n_ttm:,}  ({100*n_ttm/n_total:.2f}%)")
    print(f"  non-TTM clips  : {n_non:,}  ({100*n_non/n_total:.2f}%)")
    print(f"  Max balanced   : {max_balanced:,}  (2 × all TTM)")

    # ── Determine target ──────────────────────────────────────────────────────
    if args.target_size is not None:
        target_total = args.target_size
        if target_total % 2 != 0:
            target_total -= 1
        target_per_class = target_total // 2
        if target_per_class > n_ttm:
            raise ValueError(
                f"--target-size {target_total} requires {target_per_class:,} TTM clips "
                f"but only {n_ttm:,} exist. "
                f"Max balanced size without oversampling: {max_balanced:,}."
            )
    else:
        target_per_class = n_ttm
        target_total     = n_ttm * 2

    print(f"\n  Target balanced size  : {target_total:,}")
    print(f"  Per class (TTM / non-TTM) : {target_per_class:,} each")

    # ── Step 1: Select TTM clips ───────────────────────────────────────────────
    section("STEP 1 — SELECT TTM CLIPS")
    rng_pd = np.random.default_rng(args.seed)

    if target_per_class < n_ttm:
        # Undersample TTM proportionally across tracks (maintain track distribution)
        ttm_track_counts = all_ttm_clips.groupby(['video_uid','person_id']).size()
        weights = (ttm_track_counts / ttm_track_counts.sum()).values

        selected_ttm_parts = []
        for (vid, pid), cnt in ttm_track_counts.items():
            n_take = max(1, round(cnt / n_ttm * target_per_class))
            track_ttm = all_ttm_clips[
                (all_ttm_clips.video_uid == vid) & (all_ttm_clips.person_id == pid)
            ]
            n_take = min(n_take, len(track_ttm))
            idx = rng_pd.choice(len(track_ttm), size=n_take, replace=False)
            selected_ttm_parts.append(track_ttm.iloc[idx])

        selected_ttm = pd.concat(selected_ttm_parts, ignore_index=True)
        # Trim/pad to exact target
        if len(selected_ttm) > target_per_class:
            selected_ttm = selected_ttm.sample(n=target_per_class, random_state=args.seed)
        print(f"\n  Selected TTM clips : {len(selected_ttm):,}  (undersampled from {n_ttm:,})")
    else:
        selected_ttm = all_ttm_clips.copy()
        print(f"\n  Selected TTM clips : {len(selected_ttm):,}  (ALL TTM clips)")

    # ── Step 2: Sample non-TTM clips (intelligent) ────────────────────────────
    section("STEP 2 — SAMPLE NON-TTM CLIPS (INTELLIGENT)")
    selected_non = sample_non_ttm_intelligent(df, selected_ttm, target_per_class, args.seed)

    # ── Step 3: Combine ───────────────────────────────────────────────────────
    section("STEP 3 — COMBINE & VERIFY BALANCE")
    balanced = pd.concat(
        [selected_ttm[['clip_id','video_uid','person_id','label']],
         selected_non[['clip_id','video_uid','person_id','label']]],
        ignore_index=True
    )
    balanced = balanced.drop_duplicates(subset='clip_id').reset_index(drop=True)

    n_bal_ttm = int((balanced.label == 1).sum())
    n_bal_non = int((balanced.label == 0).sum())
    print(f"\n  Combined balanced clips : {len(balanced):,}")
    print(f"    TTM     : {n_bal_ttm:,}  ({100*n_bal_ttm/len(balanced):.2f}%)")
    print(f"    non-TTM : {n_bal_non:,}  ({100*n_bal_non/len(balanced):.2f}%)")
    print(f"    Tracks  : {balanced.groupby(['video_uid','person_id']).ngroups:,}")

    # ── Step 4: Track-level 80/20 split ───────────────────────────────────────
    section("STEP 4 — TRACK-LEVEL STRATIFIED 80/20 SPLIT")

    track_summary = (
        balanced.groupby(['video_uid','person_id'])['label']
        .max()
        .rename('has_ttm')
        .reset_index()
    )
    track_summary['has_ttm'] = track_summary['has_ttm'].astype(bool)
    track_summary = track_summary.set_index(['video_uid','person_id'])

    val_keys = stratified_track_split(track_summary, args.val_frac, args.seed)

    balanced['split'] = balanced.apply(
        lambda r: 'val' if (r['video_uid'], r['person_id']) in val_keys else 'train',
        axis=1
    )

    print_split_stats(balanced)

    # ── Step 5: Save ──────────────────────────────────────────────────────────
    section("STEP 5 — SAVE")
    total_k = round(len(balanced) / 1000)
    out_name = f"balanced_clips_{total_k}k.csv"
    out_path = out_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ['clip_id', 'video_uid', 'person_id', 'label', 'split']
    balanced[cols].to_csv(out_path, index=False)
    print(f"\n  Saved : {out_path}")
    print(f"  Size  : {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # ── Final instructions ─────────────────────────────────────────────────────
    W = 66
    print(f"\n{'='*W}")
    print(f"  NEXT — TRAIN WITH BALANCED CSV")
    print(f"{'='*W}")
    print(f"""
  # Option 1: foreground (watch live)
  CUDA_VISIBLE_DEVICES=7 python train_fusion.py \\
      --clips-csv {out_path} \\
      --no-sampler

  # Option 2: background with logging (recommended for clusters)
  CUDA_VISIBLE_DEVICES=7 nohup python train_fusion.py \\
      --clips-csv {out_path} \\
      --no-sampler \\
      > output.log 2>&1 &
  echo "PID: $!"

  # Monitor
  tail -f output.log

  # Evaluate on full 636k clips after training
  python evaluate_full.py
""")


if __name__ == "__main__":
    main()
