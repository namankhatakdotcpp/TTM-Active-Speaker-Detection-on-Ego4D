"""
Extract audio CNN embeddings for ALL clips that have mel-spectrogram .npy files.

The original extract_embeddings.py only processes clips listed in audio_index.csv
(~38k clips). This script scans the full audio_clips directory (~370k files) and
extracts embeddings for every clip, regardless of whether it appears in any CSV.

Reads:
  group_42/audio_cnn/data/audio_clips/{clip_id}.npy   [1, 64, T] mel-spectrogram

Writes:
  fusion/data/audio_embeds/{clip_id}.npy   [512,] float32

Usage:
  python extract_audio_full.py
  python extract_audio_full.py --batch-size 512 --num-workers 8
  python extract_audio_full.py --skip-existing   (skip clips already done)
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE    = os.path.dirname(os.path.abspath(__file__))
GROUP42 = os.path.dirname(HERE)

sys.path.insert(0, GROUP42)

AUDIO_CLIPS_DIR = os.path.join(GROUP42, "audio_cnn", "data", "audio_clips")
AUD_OUT_DIR     = os.path.join(HERE, "data", "audio_embeds")

# Find the best available audio checkpoint
def _find_audio_ckpt():
    candidates = [
        os.path.join(GROUP42, "audio_cnn", "checkpoints", "best_model.pth"),
        os.path.join(GROUP42, "audio_cnn", "checkpoints_balanced", "best_model.pth"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"No audio checkpoint found. Tried: {candidates}")

AUDIO_CKPT = _find_audio_ckpt()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioClipsFullDataset(Dataset):
    """
    Loads all mel-spectrogram .npy files from audio_clips_dir.
    Pads/crops the time axis to a fixed width so batches can be stacked.
    """

    def __init__(self, audio_clips_dir: str, skip_existing: bool = False,
                 out_dir: str = AUD_OUT_DIR, target_time: int = 128):
        self.clips_dir   = audio_clips_dir
        self.out_dir     = out_dir
        self.target_time = target_time

        all_files = [f for f in os.listdir(audio_clips_dir) if f.endswith(".npy")]

        if skip_existing:
            all_files = [
                f for f in all_files
                if not os.path.exists(os.path.join(out_dir, f))
            ]
            print(f"  Skipping already-extracted clips — {len(all_files):,} remaining")

        # Sort by clip_id (numeric) for reproducibility
        all_files.sort(key=lambda f: int(os.path.splitext(f)[0]))
        self.files = all_files

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fname   = self.files[idx]
        clip_id = int(os.path.splitext(fname)[0])
        path    = os.path.join(self.clips_dir, fname)

        try:
            spec = np.load(path).astype(np.float32)   # [1, 64, T]
        except Exception:
            spec = np.zeros((1, 64, self.target_time), dtype=np.float32)

        # Pad / crop time axis to target_time
        T = spec.shape[-1]
        if T < self.target_time:
            pad = np.zeros((1, spec.shape[1], self.target_time - T), dtype=np.float32)
            spec = np.concatenate([spec, pad], axis=-1)
        elif T > self.target_time:
            spec = spec[..., : self.target_time]

        return torch.from_numpy(spec), clip_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size",    type=int, default=256)
    p.add_argument("--num-workers",   type=int, default=4)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip clips that already have an embedding file")
    p.add_argument("--target-time",   type=int, default=128,
                   help="Fixed time dimension for mel-spectrograms (pad/crop)")
    args = p.parse_args()

    os.makedirs(AUD_OUT_DIR, exist_ok=True)

    print(f"Device          : {device}")
    print(f"Audio clips dir : {AUDIO_CLIPS_DIR}")
    print(f"Output dir      : {AUD_OUT_DIR}")
    print(f"Checkpoint      : {AUDIO_CKPT}")

    # ── count raw files ───────────────────────────────────────────────────────
    all_raw = [f for f in os.listdir(AUDIO_CLIPS_DIR) if f.endswith(".npy")]
    already_done = len([f for f in all_raw
                        if os.path.exists(os.path.join(AUD_OUT_DIR, f))])
    print(f"\nRaw audio clips : {len(all_raw):,}")
    print(f"Already done    : {already_done:,}")
    print(f"Remaining       : {len(all_raw) - already_done:,}\n")

    # ── load model ────────────────────────────────────────────────────────────
    from audio_cnn.audio_model import AudioEncoder

    model = AudioEncoder(pretrained=False).to(device)
    ckpt  = torch.load(AUDIO_CKPT, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"AudioEncoder loaded (checkpoint epoch {ckpt.get('epoch', '?')})")

    # ── dataset & loader ─────────────────────────────────────────────────────
    ds = AudioClipsFullDataset(
        AUDIO_CLIPS_DIR,
        skip_existing=args.skip_existing,
        out_dir=AUD_OUT_DIR,
        target_time=args.target_time,
    )
    if len(ds) == 0:
        print("Nothing to extract — all clips already have embeddings.")
        return

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    print(f"\nExtracting embeddings for {len(ds):,} clips "
          f"(batch_size={args.batch_size})...")

    done  = 0
    t0    = time.time()

    with torch.no_grad():
        for specs, clip_ids in loader:
            specs  = specs.to(device, non_blocking=True)
            embeds = model.encode(specs).cpu().numpy().astype(np.float32)

            for embed, cid in zip(embeds, clip_ids.tolist()):
                out_path = os.path.join(AUD_OUT_DIR, f"{cid}.npy")
                np.save(out_path, embed)

            done += len(embeds)

            if done % 10000 < args.batch_size or done == len(ds):
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (len(ds) - done) / max(rate, 1)
                print(f"  {done:>8,} / {len(ds):,}  |  "
                      f"{rate:.0f} clips/s  |  ETA {eta/60:.1f} min",
                      flush=True)

    elapsed = time.time() - t0
    print(f"\nDone. Extracted {done:,} embeddings in {elapsed/60:.1f} min.")
    print(f"Output dir: {AUD_OUT_DIR}")

    # ── summary ───────────────────────────────────────────────────────────────
    total_now = len([f for f in os.listdir(AUD_OUT_DIR) if f.endswith(".npy")])
    print(f"Total audio embeddings now available: {total_now:,}")


if __name__ == "__main__":
    main()
