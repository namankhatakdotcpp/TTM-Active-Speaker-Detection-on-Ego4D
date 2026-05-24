"""
Dataset for fusion model training.

Reads clips grouped by utterance from clips_index.csv.
For each utterance (sequence of consecutive clips from the same video),
loads precomputed video and audio embeddings and stacks them into
temporal sequences of length T.

Pre-extracted embeddings are expected at:
  video: group_42/fusion/data/video_embeds/{clip_id}.npy   [512,]
  audio: group_42/fusion/data/audio_embeds/{clip_id}.npy   [512,]

If embeddings are not yet extracted, run extract_embeddings.py first.

Returns:
  video_feats : FloatTensor [T, 512]
  audio_feats : FloatTensor [T, 512]
  label       : LongTensor scalar  (0 or 1)
  utterance_id: str
"""

import csv
import os
from collections import Counter, defaultdict
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

T_SEQ    = 16   # clips per window
STRIDE   = 4    # training stride  (4 = 75% overlap → 3× more windows than stride=16)
VID_DIM  = 512
AUD_DIM  = 512


class FusionDataset(Dataset):
    """
    Args:
        clips_source    : CSV file path (str) OR a pre-filtered pandas DataFrame.
                          When a str path is given, rows are filtered by the
                          ``split`` column.  When a DataFrame is given, all rows
                          are used as-is (caller is responsible for the split).
        split           : "train" or "val" — used for display and, when
                          clips_source is a file path, for filtering.
        video_embed_dir : directory of video embedding .npy files
        audio_embed_dir : directory of audio embedding .npy files
        t_seq           : clips per window (sequence length fed to BiLSTM)
        stride          : step between window start positions within a track.
                          Use a small value (e.g. 4) for training to maximise
                          window count; use t_seq for non-overlapping val windows.
        min_clips       : minimum real (non-padded) clips a window must contain
                          to be kept.  Windows below this threshold are mostly
                          zeros — they add noise rather than signal.
                          Default: t_seq // 4  (window must be ≥25% real data).
                          Pass 1 to disable the filter entirely.
    """

    def __init__(
        self,
        clips_source: Union[str, pd.DataFrame],
        split: str,
        video_embed_dir: str,
        audio_embed_dir: str,
        t_seq: int = T_SEQ,
        stride: int = STRIDE,
        min_clips: int = None,
    ):
        self.video_dir = video_embed_dir
        self.audio_dir = audio_embed_dir
        self.t_seq     = t_seq
        self.stride    = stride
        self.min_clips = max(1, t_seq // 4) if min_clips is None else min_clips

        # Group clips by tracked person within a video.
        groups = defaultdict(list)

        if isinstance(clips_source, str):
            with open(clips_source, newline="") as f:
                for row in csv.DictReader(f):
                    if row["split"] != split:
                        continue
                    key = (row["video_uid"], row["person_id"])
                    groups[key].append(row)
        else:
            for row in clips_source.to_dict("records"):
                key = (str(row["video_uid"]), str(row["person_id"]))
                groups[key].append(row)

        self.samples   = []
        label_counts   = Counter()
        total_windows  = 0   # before min_clips filter
        skipped        = 0   # windows dropped for being too short
        pad_fracs      = []  # padding fraction per kept window (for diagnostics)
        audio_covered  = 0   # windows where ≥1 clip has an audio embedding

        for (video_uid, person_id), clips in groups.items():
            clips_sorted = sorted(clips, key=lambda r: int(r["clip_id"]))
            n = len(clips_sorted)

            for start in range(0, n, self.stride):
                chunk = clips_sorted[start : start + self.t_seq]
                total_windows += 1

                # Skip windows that are too short — they would be padded with
                # zeros for more than (1 - min_clips/t_seq) of their length,
                # contributing more noise than real gradient signal.
                if len(chunk) < self.min_clips:
                    skipped += 1
                    continue

                clip_ids = [int(r["clip_id"]) for r in chunk]
                pos      = sum(int(r["label"]) for r in chunk)
                label    = int(pos >= len(chunk) / 2.0)
                label_counts[label] += 1
                pad_fracs.append(1.0 - len(chunk) / self.t_seq)

                # Track audio coverage for diagnostics
                has_any_audio = any(
                    os.path.exists(os.path.join(audio_embed_dir, f"{cid}.npy"))
                    for cid in clip_ids
                )
                if has_any_audio:
                    audio_covered += 1

                self.samples.append({
                    "utterance_id": (
                        f"{video_uid}:person{person_id}:"
                        f"{clip_ids[0]}-{clip_ids[-1]}"
                    ),
                    "video_uid": video_uid,
                    "person_id": person_id,
                    "clip_ids":  clip_ids,
                    "label":     label,
                    "split":     split,
                })

        kept = len(self.samples)
        avg_windows_per_track = kept / max(len(groups), 1)
        avg_pad = float(np.mean(pad_fracs)) if pad_fracs else 0.0
        audio_pct = 100.0 * audio_covered / max(kept, 1)

        print(f"\n[FusionDataset] {split}")
        print(f"  tracks        : {len(groups):,}")
        print(f"  windows total : {total_windows:,}  "
              f"(kept={kept:,}  skipped={skipped:,} below min_clips={self.min_clips})")
        print(f"  neg / pos     : {label_counts[0]:,} / {label_counts[1]:,}  "
              f"(ratio {label_counts[0]/(label_counts[1] or 1):.2f}:1)")
        print(f"  avg windows/track : {avg_windows_per_track:.1f}")
        print(f"  avg padding/window: {100*avg_pad:.1f}%  "
              f"(t_seq={t_seq}, stride={stride}, min_clips={self.min_clips})")
        print(f"  audio coverage    : {audio_covered:,}/{kept:,} windows "
              f"({audio_pct:.1f}%) have ≥1 real audio embed")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        clip_ids = sample["clip_ids"]
        label    = torch.tensor(sample["label"], dtype=torch.long)

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

        # Pad to fixed window length so batches can be stacked.
        while len(video_seq) < self.t_seq:
            video_seq.append(np.zeros(VID_DIM, dtype=np.float32))
            audio_seq.append(np.zeros(AUD_DIM, dtype=np.float32))

        video_feats = torch.tensor(np.stack(video_seq), dtype=torch.float32)  # [T, 512]
        audio_feats = torch.tensor(np.stack(audio_seq), dtype=torch.float32)  # [T, 512]

        return video_feats, audio_feats, label, sample["utterance_id"]


def collate_fn(batch):
    """Custom collate — drops utterance_id string from tensor stack."""
    video  = torch.stack([b[0] for b in batch])
    audio  = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    uids   = [b[3] for b in batch]
    return video, audio, labels, uids
