"""
Extract mel-spectrograms for only the clips in balanced_clips_50k.csv
that are not yet saved. Much faster than full extraction — touches only
the videos that contain missing clips.

Usage:
  python extract_balanced_audio.py
  python extract_balanced_audio.py --workers 32
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
import torchaudio.transforms as T

HERE         = os.path.dirname(os.path.abspath(__file__))
BALANCED_CSV = os.path.join(HERE, "..", "..", "ego4d_data", "v2", "full_scale",
                            "pipeline", "data", "balanced_clips_50k.csv")
CLIPS_CSV    = os.path.join(HERE, "..", "..", "ego4d_data", "v2", "full_scale",
                            "pipeline", "data", "clips_index.csv")
VIDEO_DIR    = "/usershome/cs671_user13/ego4d_data/v2/full_scale"
OUT_DIR      = os.path.join(HERE, "data", "audio_clips")
OUT_CSV      = os.path.join(HERE, "data", "audio_index.csv")

VIDEO_FPS  = 30
TARGET_SR  = 16000
N_FFT      = 400
HOP_LENGTH = 160
N_MELS     = 64
F_MIN      = 60.0
F_MAX      = 7600.0
TOP_DB     = 80.0


def make_mel_transform():
    return torch.nn.Sequential(
        T.MelSpectrogram(sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
                         n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX),
        T.AmplitudeToDB(top_db=TOP_DB),
    )


def _process_video(args):
    video_uid, clip_rows = args
    video_path = os.path.join(VIDEO_DIR, f"{video_uid}.mp4")
    if not os.path.exists(video_path):
        return []

    try:
        cmd = ["ffmpeg", "-v", "quiet", "-i", video_path,
               "-ac", "1", "-ar", str(TARGET_SR), "-f", "s16le", "-"]
        raw   = subprocess.check_output(cmd, timeout=300)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        return []

    total_samples = len(audio)
    mel_transform = make_mel_transform()
    results = []

    for row in clip_rows:
        out_path = os.path.join(OUT_DIR, f"{row['clip_id']}.npy")
        if os.path.exists(out_path):
            results.append({"clip_id": row["clip_id"], "npy_path": os.path.relpath(out_path, os.path.dirname(OUT_CSV)),
                            "label": row["label"], "split": row["split"], "video_uid": video_uid})
            continue

        frames  = [int(os.path.splitext(os.path.basename(p))[0])
                   for p in row["frame_paths"].split("|")]
        t_start = frames[0] / VIDEO_FPS
        t_end   = (frames[-1] + 1) / VIDEO_FPS
        s_start = max(0, min(int(round(t_start * TARGET_SR)), total_samples))
        s_end   = max(s_start + 1, min(int(round(t_end * TARGET_SR)), total_samples))

        segment = audio[s_start:s_end]
        if len(segment) < N_FFT:
            continue
        segment = torch.from_numpy(segment).unsqueeze(0)
        spec    = mel_transform(segment)
        np.save(out_path, spec.numpy().astype(np.float16))

        results.append({"clip_id": row["clip_id"], "npy_path": os.path.relpath(out_path, os.path.dirname(OUT_CSV)),
                        "label": row["label"], "split": row["split"], "video_uid": video_uid})
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    # Load balanced clip IDs
    balanced_ids = set()
    with open(BALANCED_CSV) as f:
        for row in csv.DictReader(f):
            balanced_ids.add(int(row["clip_id"]))

    # Check which are missing
    missing_ids = {cid for cid in balanced_ids
                   if not os.path.exists(os.path.join(OUT_DIR, f"{cid}.npy"))}
    print(f"Balanced clips: {len(balanced_ids):,}  Missing: {len(missing_ids):,}")

    if not missing_ids:
        print("All balanced clips already extracted!")
        return

    # Load clip metadata for missing clips only, group by video
    video_clips = defaultdict(list)
    with open(CLIPS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["clip_id"]) in missing_ids:
                video_clips[row["video_uid"]].append(row)

    video_items  = list(video_clips.items())
    total_videos = len(video_items)
    total_clips  = sum(len(v) for _, v in video_items)
    print(f"Videos to process: {total_videos}  |  Clips: {total_clips:,}  |  Workers: {args.workers}")

    t0     = time.time()
    done_v = 0
    new_rows = []

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_process_video, video_items):
            new_rows.extend(result)
            done_v += 1
            elapsed = time.time() - t0
            rate    = done_v / elapsed
            eta     = (total_videos - done_v) / rate if rate > 0 else 0
            print(f"\r  [{done_v}/{total_videos} videos] "
                  f"{len(new_rows):,} spectrograms  ETA: {eta/60:.0f}min   ",
                  end="", flush=True)

    print(f"\n\nDone in {(time.time()-t0)/60:.1f} min  |  New spectrograms: {len(new_rows):,}")

    # Merge with existing audio_index.csv
    existing = []
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="") as f:
            existing = list(csv.DictReader(f))

    existing_ids = {int(r["clip_id"]) for r in existing}
    merged = existing + [r for r in new_rows if int(r["clip_id"]) not in existing_ids]

    fields = ["clip_id", "npy_path", "label", "split", "video_uid"]
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged)

    print(f"Audio index updated: {len(merged):,} total entries → {OUT_CSV}")


if __name__ == "__main__":
    main()
