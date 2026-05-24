"""
analyze_distribution.py

Comprehensive analysis of clips_index.csv dataset distribution.
Prints dataset stats, class imbalance, track structure, and
recommends optimal balanced dataset size.

Usage:
  python analyze_distribution.py
  python analyze_distribution.py --csv /path/to/clips_index.csv
"""

import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

HERE    = Path(__file__).parent
GROUP42 = HERE.parent

DEFAULT_CSV = (
    GROUP42.parent / "ego4d_data" / "v2" / "full_scale" / "pipeline" / "data" / "clips_index.csv"
)


def section(title, width=66):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def divider(width=66):
    print(f"  {'─'*width}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(DEFAULT_CSV))
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # ── Load ──────────────────────────────────────────────────────────────────
    section("LOADING DATASET")
    print(f"\n  Path : {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  Rows : {len(df):,}")
    print(f"  Cols : {df.columns.tolist()}")

    required = {'clip_id', 'video_uid', 'person_id', 'label'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── 1. CLIP-LEVEL DISTRIBUTION ────────────────────────────────────────────
    section("1. CLIP-LEVEL DISTRIBUTION")
    N = len(df)
    n_ttm = int((df.label == 1).sum())
    n_non = int((df.label == 0).sum())
    ratio = n_non / n_ttm

    print(f"\n  Total clips  : {N:>10,}")
    print(f"  TTM   (pos=1): {n_ttm:>10,}  ({100*n_ttm/N:.2f}%)")
    print(f"  non-TTM(neg=0): {n_non:>10,}  ({100*n_non/N:.2f}%)")
    print(f"  Imbalance ratio (neg:pos) : {ratio:.1f} : 1")

    bar_len = 40
    ttm_bar = int(bar_len * n_ttm / N)
    print(f"\n  Class bar  [{'█'*ttm_bar}{'░'*(bar_len-ttm_bar)}]  TTM={100*n_ttm/N:.1f}%")

    # ── 2. TRACK-LEVEL ANALYSIS ───────────────────────────────────────────────
    section("2. TRACK-LEVEL ANALYSIS  (video_uid, person_id)")
    track_stats = (
        df.groupby(['video_uid', 'person_id'])
        .agg(
            total_clips=('clip_id', 'count'),
            ttm_clips=('label', 'sum'),
        )
        .reset_index()
    )
    track_stats['non_ttm_clips'] = track_stats['total_clips'] - track_stats['ttm_clips']
    track_stats['ttm_frac'] = track_stats['ttm_clips'] / track_stats['total_clips']
    track_stats['has_ttm'] = track_stats['ttm_clips'] > 0

    n_tracks      = len(track_stats)
    n_ttm_tracks  = int(track_stats['has_ttm'].sum())
    n_pure_tracks = n_tracks - n_ttm_tracks

    print(f"\n  Total tracks              : {n_tracks:,}")
    print(f"  Tracks with ≥1 TTM clip   : {n_ttm_tracks:,}  ({100*n_ttm_tracks/n_tracks:.1f}%)")
    print(f"  Tracks with ONLY non-TTM  : {n_pure_tracks:,}  ({100*n_pure_tracks/n_tracks:.1f}%)")

    divider()
    print(f"  Clips per track (all tracks):")
    cpt = track_stats['total_clips']
    print(f"    min={cpt.min():,}  max={cpt.max():,}  mean={cpt.mean():.1f}  "
          f"median={cpt.median():.0f}  p95={cpt.quantile(.95):.0f}")

    ttm_tracks = track_stats[track_stats.has_ttm]
    ttm_cpt = ttm_tracks['ttm_clips']
    print(f"\n  TTM clips per TTM-containing track:")
    print(f"    min={ttm_cpt.min():,}  max={ttm_cpt.max():,}  mean={ttm_cpt.mean():.1f}  "
          f"median={ttm_cpt.median():.0f}")

    non_ttm_in_ttm_tracks = int(ttm_tracks['non_ttm_clips'].sum())
    print(f"\n  Non-TTM clips in TTM-containing tracks : {non_ttm_in_ttm_tracks:,}")
    print(f"  Non-TTM clips in pure non-TTM tracks   : {n_non - non_ttm_in_ttm_tracks:,}")

    # ── 3. SPLIT DISTRIBUTION (if exists) ────────────────────────────────────
    if 'split' in df.columns:
        section("3. EXISTING SPLIT DISTRIBUTION")
        for sp in ['train', 'val']:
            s = df[df.split == sp]
            n_s = len(s)
            if n_s == 0:
                continue
            sp_tracks = s.groupby(['video_uid', 'person_id']).ngroups
            sp_ttm    = int((s.label == 1).sum())
            sp_non    = int((s.label == 0).sum())
            print(f"\n  {sp.upper()}")
            print(f"    clips   : {n_s:>8,}  ({100*n_s/N:.1f}% of total)")
            print(f"    tracks  : {sp_tracks:>8,}")
            print(f"    TTM     : {sp_ttm:>8,}  ({100*sp_ttm/n_s:.2f}%)")
            print(f"    non-TTM : {sp_non:>8,}  ({100*sp_non/n_s:.2f}%)")
        divider()
        train_tr = set(map(tuple, df[df.split=='train'][['video_uid','person_id']].values))
        val_tr   = set(map(tuple, df[df.split=='val'][['video_uid','person_id']].values))
        overlap  = train_tr & val_tr
        status   = "OK — zero leakage" if not overlap else f"ERROR — {len(overlap)} tracks leak!"
        print(f"\n  Track leakage : {status}")

    # ── 4. RECOMMENDATION ─────────────────────────────────────────────────────
    section("4. BALANCED DATASET RECOMMENDATION")

    max_balanced   = n_ttm * 2
    rec_60k        = 60_000
    rec_70k        = min(max_balanced, 70_000)
    can_do_70k     = max_balanced >= 70_000

    print(f"""
  Max balanced size (use ALL TTM, equal non-TTM) : {max_balanced:,}
  ─────────────────────────────────────────────────────
  Option A — 60k : 30,000 TTM + 30,000 non-TTM
    Pro  : Smaller, faster training
    Con  : Discards {n_ttm - 30000:,} TTM clips

  Option B — 70k : {n_ttm:,} TTM + {n_ttm:,} non-TTM  ← RECOMMENDED
    Pro  : Uses ALL available TTM data — maximum positive signal
    Con  : Slightly larger (still fast given pre-extracted embeddings)

  Option C — 80k : Requires oversampling TTM ({80000//2 - n_ttm:,} synthetic clips)
    Pro  : More data
    Con  : Oversampled clips repeat patterns → overfitting risk. NOT recommended.
    """)

    print(f"  → Run: python create_balanced_csv.py            (default: 70k)")
    print(f"  → Run: python create_balanced_csv.py --target-size 60000  (60k)")

    # ── 5. TTM CLIP DISTRIBUTION ACROSS TRACKS ───────────────────────────────
    section("5. TOP 10 TTM-HEAVIEST TRACKS")
    top = ttm_tracks.nlargest(10, 'ttm_clips')[
        ['video_uid', 'person_id', 'total_clips', 'ttm_clips', 'non_ttm_clips', 'ttm_frac']
    ]
    print()
    print(f"  {'video_uid':>36}  {'pid':>4}  {'total':>7}  {'TTM':>6}  {'non-TTM':>8}  {'TTM%':>6}")
    print(f"  {'─'*36}  {'─'*4}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*6}")
    for _, row in top.iterrows():
        print(f"  {row.video_uid:>36}  {int(row.person_id):>4}  "
              f"{int(row.total_clips):>7,}  {int(row.ttm_clips):>6,}  "
              f"{int(row.non_ttm_clips):>8,}  {100*row.ttm_frac:>5.1f}%")

    print(f"\n  Analysis complete.\n")


if __name__ == "__main__":
    main()
