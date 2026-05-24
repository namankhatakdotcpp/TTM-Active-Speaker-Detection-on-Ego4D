"""
PyTorch Dataset for audio Mel-spectrogram clips.

Reads audio_index.csv produced by step_a1_extract_audio.py.
Returns (spec_tensor, label) where:
  spec_tensor: Float tensor  [1, N_MELS, T_FIXED]   (1 channel, 64 mel bins, fixed width)
  label:       Long tensor scalar  (0 or 1)

Fixed width T_FIXED = 50 frames (matches a ~0.53s clip at 16kHz / hop=160).
Shorter clips are zero-padded; longer clips are centre-cropped.
"""

import csv
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

T_FIXED = 50     # fixed time dimension fed to ResNet-18 (adjust if needed)
N_MELS  = 64


class AudioClipDataset(Dataset):
    """
    Args:
        audio_csv:  Path to data/audio_index.csv
        split:      "train" or "val"
        data_root:  Root directory for relative npy_path values.
                    Defaults to directory containing audio_csv.
        augment:    If True, apply SpecAugment-style time/freq masking.
    """

    def __init__(
        self,
        audio_csv: str,
        split: str = "train",
        data_root: str | None = None,
        augment: bool | None = None,
    ):
        self.split     = split
        self.data_root = data_root or os.path.dirname(os.path.abspath(audio_csv))
        self.augment   = augment if augment is not None else (split == "train")

        self.clips = []
        with open(audio_csv, newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] == split:
                    self.clips.append({
                        "npy_path": row["npy_path"],
                        "label":    int(row["label"]),
                        "clip_id":  int(row["clip_id"]),
                    })

        print(f"[AudioClipDataset] {split}: {len(self.clips):,} clips loaded")

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int):
        clip  = self.clips[idx]
        label = torch.tensor(clip["label"], dtype=torch.long)

        abs_path = os.path.join(self.data_root, clip["npy_path"])
        try:
            spec = np.load(abs_path).astype(np.float32)   # [1, N_MELS, T]
        except Exception:
            spec = np.zeros((1, N_MELS, T_FIXED), dtype=np.float32)

        spec = torch.from_numpy(spec)   # [1, N_MELS, T]

        # Fix time dimension
        spec = self._fix_length(spec)   # [1, N_MELS, T_FIXED]

        # Normalize per-clip: zero mean, unit std
        mean = spec.mean()
        std  = spec.std().clamp(min=1e-6)
        spec = (spec - mean) / std

        # SpecAugment (train only): random time + frequency masking
        if self.augment:
            spec = self._spec_augment(spec)

        return spec, label

    # ── helpers ───────────────────────────────────────────────────────────────

    def _fix_length(self, spec: torch.Tensor) -> torch.Tensor:
        T = spec.shape[-1]
        if T == T_FIXED:
            return spec
        if T < T_FIXED:
            pad = T_FIXED - T
            spec = torch.nn.functional.pad(spec, (0, pad))
        else:
            # centre crop
            start = (T - T_FIXED) // 2
            spec  = spec[..., start:start + T_FIXED]
        return spec

    def _spec_augment(self, spec: torch.Tensor) -> torch.Tensor:
        """SpecAugment: mask up to 10 time steps and 8 freq bins."""
        spec = spec.clone()
        # Time mask
        t_mask = torch.randint(0, 10, (1,)).item()
        t0     = torch.randint(0, max(1, T_FIXED - t_mask), (1,)).item()
        spec[..., t0:t0 + t_mask] = 0.0
        # Freq mask
        f_mask = torch.randint(0, 8, (1,)).item()
        f0     = torch.randint(0, max(1, N_MELS - f_mask), (1,)).item()
        spec[:, f0:f0 + f_mask, :] = 0.0
        return spec

    def class_weights(self) -> torch.Tensor:
        n_pos = sum(c["label"] for c in self.clips)
        n_neg = len(self.clips) - n_pos
        total = len(self.clips)
        w_neg = total / (2.0 * n_neg) if n_neg > 0 else 1.0
        w_pos = total / (2.0 * n_pos) if n_pos > 0 else 1.0
        return torch.tensor([w_neg, w_pos], dtype=torch.float)
