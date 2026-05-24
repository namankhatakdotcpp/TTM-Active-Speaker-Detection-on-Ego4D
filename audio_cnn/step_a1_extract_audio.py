# """
# Step A1: Extract per-clip Mel-spectrograms aligned to video clips.

# Strategy (fast — ~20–40 min for all 636K clips):
#   Load FULL audio of each video once with ffmpeg (153 videos).
#   Then slice all clip windows from the in-memory numpy array.
#   This replaces 636,406 ffmpeg calls with just 153.

# Temporal alignment (zero lag):
#   clip window = [first_frame / 30 , (last_frame+1) / 30]  seconds.
#   Converts to sample indices: start_sample = round(t_start * TARGET_SR).

# Reads:  pipeline/data/clips_index.csv
# Writes: data/audio_clips/{clip_id}.npy   [1, N_MELS=64, T≈54]  float16
#         data/audio_index.csv

# Usage:
#     python step_a1_extract_audio.py                  # all clips, 8 video workers
#     python step_a1_extract_audio.py --split val      # val only
#     python step_a1_extract_audio.py --workers 16     # more parallel videos
# """

# import argparse
# import csv
# import os
# import subprocess
# import sys
# import time
# from collections import defaultdict
# from multiprocessing import Pool

# import numpy as np
# import torch
# import torchaudio.transforms as T

# # ── paths ─────────────────────────────────────────────────────────────────────
# HERE      = os.path.dirname(os.path.abspath(__file__))
# CLIPS_CSV = os.path.join(HERE, "..", "..", "ego4d_data", "v2", "full_scale",
#                          "pipeline", "data", "clips_index.csv")
# VIDEO_DIR = "/usershome/cs671_user13/ego4d_data/v2/full_scale"
# OUT_DIR   = os.path.join(HERE, "data", "audio_clips")
# OUT_CSV   = os.path.join(HERE, "data", "audio_index.csv")

# # ── audio / spectrogram config ────────────────────────────────────────────────
# VIDEO_FPS  = 30
# TARGET_SR  = 16000
# N_FFT      = 400
# HOP_LENGTH = 160
# N_MELS     = 64
# F_MIN      = 60.0
# F_MAX      = 7600.0
# TOP_DB     = 80.0


# def make_mel_transform():
#     return torch.nn.Sequential(
#         T.MelSpectrogram(
#             sample_rate=TARGET_SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
#             n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX,
#         ),
#         T.AmplitudeToDB(top_db=TOP_DB),
#     )


# # ── per-video worker ──────────────────────────────────────────────────────────

# def _process_video(args):
#     """
#     Load full audio of one video, slice all its clips, save spectrograms.
#     Returns list of metadata dicts.
#     """
#     video_uid, clip_rows = args

#     video_path = os.path.join(VIDEO_DIR, f"{video_uid}.mp4")
#     #if not os.path.exists(video_path):
#         #return []

#     # ── 1. load full audio via ffmpeg → PCM float32 ───────────────────────────
#     try:
#         cmd = [
#             "ffmpeg", "-v", "quiet",
#             "-i", video_path,
#             "-ac", "1", "-ar", str(TARGET_SR),
#             "-f", "s16le", "-",
#         ]
#         raw   = subprocess.check_output(cmd, timeout=300)
#         audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
#     except Exception:
#         return []

#     total_samples = len(audio)
#     mel_transform = make_mel_transform()
#     results       = []

#     # ── 2. slice each clip window and compute spectrogram ─────────────────────
#     for row in clip_rows:
#         # Skip already-saved clips (resume support)
#         out_path_check = os.path.join(OUT_DIR, f"{row['clip_id']}.npy")
#         if os.path.exists(out_path_check):
#             results.append({
#                 "clip_id":   row["clip_id"],
#                 "npy_path":  os.path.relpath(out_path_check, os.path.dirname(OUT_CSV)),
#                 "label":     row["label"],
#                 "split":     row["split"],
#                 "video_uid": video_uid,
#             })
#             continue
#         frames   = [int(os.path.splitext(os.path.basename(p))[0])
#                     for p in row["frame_paths"].split("|")]
#         t_start  = frames[0]  / VIDEO_FPS
#         t_end    = (frames[-1] + 1) / VIDEO_FPS

#         s_start = int(round(t_start * TARGET_SR))
#         s_end   = int(round(t_end   * TARGET_SR))
#         s_start = max(0, min(s_start, total_samples))
#         s_end   = max(s_start + 1, min(s_end, total_samples))

#         segment = audio[s_start:s_end]

# # FIX: never drop clips — pad instead
#         if len(segment) < N_FFT:
#             pad_len = N_FFT - len(segment)
#         if len(segment) == 0:
#             segment = np.zeros(N_FFT, dtype=np.float32)
#         else:
#             segment = np.pad(segment, (0, pad_len), mode="constant")
#         segment  = torch.from_numpy(segment).unsqueeze(0)  # [1, T]
#         spec     = mel_transform(segment)   # [1, N_MELS, T_frames]

#         out_path = os.path.join(OUT_DIR, f"{row['clip_id']}.npy")
#         np.save(out_path, spec.numpy().astype(np.float16))

#         results.append({
#             "clip_id":   row["clip_id"],
#             "npy_path":  os.path.relpath(out_path, os.path.dirname(OUT_CSV)),
#             "label":     row["label"],
#             "split":     row["split"],
#             "video_uid": video_uid,
#         })

#     return results


# # ── main ──────────────────────────────────────────────────────────────────────

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--split",    choices=["train", "val", "all"], default="all")
#     p.add_argument("--workers",  type=int, default=8,
#                    help="Parallel video workers (each loads one full video audio)")
#     p.add_argument("--max-videos", type=int, default=None)
#     return p.parse_args()


# def main():
#     args = parse_args()
#     os.makedirs(OUT_DIR, exist_ok=True)

#     # Group clips by video
#     print(f"Reading: {CLIPS_CSV}")
#     video_clips = defaultdict(list)
#     with open(CLIPS_CSV, newline="") as f:
#         for row in csv.DictReader(f):
#             if args.split != "all" and row["split"] != args.split:
#                 continue
#             video_clips[row["video_uid"]].append(row)

#     video_items = list(video_clips.items())
#     if args.max_videos:
#         video_items = video_items[:args.max_videos]

#     total_videos = len(video_items)
#     total_clips  = sum(len(v) for _, v in video_items)
#     print(f"Videos: {total_videos}  |  Clips: {total_clips:,}  |  Workers: {args.workers}")
#     print("Strategy: load full audio per video (1 ffmpeg call/video)")

#     t0     = time.time()
#     done_v = 0
#     rows   = []

#     with Pool(processes=args.workers) as pool:
#         for result in pool.imap_unordered(_process_video, video_items):
#             rows.extend(result)
#             done_v += 1
#             elapsed = time.time() - t0
#             rate    = done_v / elapsed
#             eta     = (total_videos - done_v) / rate if rate > 0 else 0
#             print(f"\r  [{done_v}/{total_videos} videos] "
#                   f"{len(rows):,} spectrograms  "
#                   f"ETA: {eta/60:.0f}min   ", end="", flush=True)

#     print(f"\n\nDone in {(time.time()-t0)/60:.1f} min")
#     print(f"Spectrograms saved: {len(rows):,}")

#     fields = ["clip_id", "npy_path", "label", "split", "video_uid"]
#     with open(OUT_CSV, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fields)
#         writer.writeheader()
#         writer.writerows(rows)

#     print(f"Audio index: {OUT_CSV}")


# if __name__ == "__main__":
#     main()


"""
Step A1: Extract per-clip Mel-spectrograms aligned to video clips.
"""

import argparse
import csv
import os
import subprocess
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import torch
import torchaudio.transforms as T


# ── paths ─────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
CLIPS_CSV = os.path.join(HERE, "..", "..", "ego4d_data", "v2", "full_scale",
                         "pipeline", "data", "clips_index.csv")
VIDEO_DIR = "/usershome/cs671_user13/ego4d_data/v2/full_scale"
OUT_DIR   = os.path.join(HERE, "data", "audio_clips")
OUT_CSV   = os.path.join(HERE, "data", "audio_index.csv")


# ── config ─────────────────────────────────────────────────────────────
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
        T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            f_min=F_MIN,
            f_max=F_MAX,
        ),
        T.AmplitudeToDB(top_db=TOP_DB),
    )


# ── per-video worker ───────────────────────────────────────────────────
def _process_video(args):
    video_uid, clip_rows = args

    video_path = os.path.join(VIDEO_DIR, f"{video_uid}.mp4")

    if not os.path.exists(video_path):
        print(f"❌ Missing video: {video_uid}")
        return []

    # Load full audio
    try:
        cmd = [
            "ffmpeg", "-v", "quiet",
            "-i", video_path,
            "-ac", "1", "-ar", str(TARGET_SR),
            "-f", "s16le", "-"
        ]
        raw = subprocess.check_output(cmd, timeout=300)
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception:
        print(f"❌ FFMPEG FAILED: {video_uid}")
        return []

    total_samples = len(audio)
    mel_transform = make_mel_transform()
    results = []

    for row in clip_rows:

        frames = [int(os.path.splitext(os.path.basename(p))[0])
                  for p in row["frame_paths"].split("|")]

        t_start = frames[0] / VIDEO_FPS
        t_end   = (frames[-1] + 1) / VIDEO_FPS

        s_start = int(round(t_start * TARGET_SR))
        s_end   = int(round(t_end * TARGET_SR))

        s_start = max(0, min(s_start, total_samples))
        s_end   = max(s_start + 1, min(s_end, total_samples))

        # Safe segment extraction
        if s_end <= s_start:
            segment = np.zeros(N_FFT, dtype=np.float32)
        else:
            segment = audio[s_start:s_end]

        # 🔥 FIX: NEVER DROP CLIPS
        if len(segment) < N_FFT:
            pad_len = N_FFT - len(segment)
            if len(segment) == 0:
                segment = np.zeros(N_FFT, dtype=np.float32)
            else:
                segment = np.pad(segment, (0, pad_len), mode="constant")

        segment = torch.from_numpy(segment).unsqueeze(0)

        spec = mel_transform(segment)

        out_path = os.path.join(OUT_DIR, f"{row['clip_id']}.npy")
        np.save(out_path, spec.numpy().astype(np.float16))

        results.append({
            "clip_id":   row["clip_id"],
            "npy_path":  os.path.relpath(out_path, os.path.dirname(OUT_CSV)),
            "label":     row["label"],
            "split":     row["split"],
            "video_uid": video_uid,
        })

    return results


# ── main ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val", "all"], default="all")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-videos", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Reading: {CLIPS_CSV}")

    video_clips = defaultdict(list)
    with open(CLIPS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if args.split != "all" and row["split"] != args.split:
                continue
            video_clips[row["video_uid"]].append(row)

    video_items = list(video_clips.items())

    if args.max_videos:
        video_items = video_items[:args.max_videos]

    total_videos = len(video_items)
    total_clips  = sum(len(v) for _, v in video_items)

    print(f"Videos: {total_videos} | Clips: {total_clips:,} | Workers: {args.workers}")

    t0 = time.time()
    rows = []
    done_v = 0

    with Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(_process_video, video_items):
            rows.extend(result)
            done_v += 1

            elapsed = time.time() - t0
            rate = done_v / elapsed if elapsed > 0 else 0
            eta  = (total_videos - done_v) / rate if rate > 0 else 0

            print(f"\r[{done_v}/{total_videos}] "
                  f"{len(rows):,} clips | ETA {eta/60:.1f} min",
                  end="", flush=True)

    print(f"\n\nDone in {(time.time()-t0)/60:.1f} min")
    print(f"Total spectrograms: {len(rows):,}")

    fields = ["clip_id", "npy_path", "label", "split", "video_uid"]

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved index: {OUT_CSV}")


if __name__ == "__main__":
    main()

