"""
Step F0: Extract per-clip embeddings from trained video and audio encoders.

Reads:
  video checkpoints: group_42/video_cnn/checkpoints/best_model.pth
  audio checkpoints: group_42/audio_cnn/checkpoints/best_model.pth
  video clips:       pipeline/data/clips_index.csv  +  frame .jpg files
  audio clips:       audio_cnn/data/audio_index.csv + .npy mel-spectrograms

Writes:
  fusion/data/video_embeds/{clip_id}.npy   [512,]  float32
  fusion/data/audio_embeds/{clip_id}.npy   [512,]  float32

Usage:
  python extract_embeddings.py
  python extract_embeddings.py --split val      # val only (smoke test)
  python extract_embeddings.py --batch-size 128
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

HERE      = os.path.dirname(os.path.abspath(__file__))
GROUP42   = os.path.dirname(HERE)
PIPELINE  = os.path.join(GROUP42, "..", "ego4d_data", "v2", "full_scale", "pipeline")

sys.path.insert(0, GROUP42)
sys.path.insert(0, PIPELINE)

CLIPS_CSV     = os.path.join(PIPELINE, "data", "clips_index.csv")
AUDIO_CSV     = os.path.join(GROUP42, "audio_cnn", "data", "audio_index.csv")
VID_OUT_DIR   = os.path.join(HERE, "data", "video_embeds")
AUD_OUT_DIR   = os.path.join(HERE, "data", "audio_embeds")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def first_existing(*paths):
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"None of these paths exist: {paths}")


VIDEO_CKPT = first_existing(
    os.path.join(GROUP42, "video_cnn", "checkpoints", "best_model.pth"),
    os.path.join(GROUP42, "video_cnn", "checkpoints_slowfast", "best_model.pth"),
)

AUDIO_CKPT = first_existing(
    os.path.join(GROUP42, "audio_cnn", "checkpoints", "best_model.pth"),
    os.path.join(GROUP42, "audio_cnn", "checkpoints_balanced", "best_model.pth"),
)


# ── video embedding extraction ────────────────────────────────────────────────

def extract_video_embeddings(split, batch_size, num_workers):
    from video_cnn.model import I3D_TTM
    from dataset import TTMClipDataset

    os.makedirs(VID_OUT_DIR, exist_ok=True)

    print(f"\nExtracting video embeddings ({split})...")
    print(f"  Using video checkpoint: {VIDEO_CKPT}")
    model = I3D_TTM(pretrained=False, freeze_backbone=False).to(device)
    ckpt  = torch.load(VIDEO_CKPT, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    # strip torch.compile prefix if present
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    ds     = TTMClipDataset(CLIPS_CSV, split=split)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers > 0))

    done = 0
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
            embeds = model.encode(frames).cpu().numpy().astype(np.float32)

            # map back to clip_ids via dataset order
            start = done
            for i, embed in enumerate(embeds):
                cid = ds.clips[start + i]["clip_id"]
                out = os.path.join(VID_OUT_DIR, f"{cid}.npy")
                if not os.path.exists(out):
                    np.save(out, embed)
            done += len(embeds)
            if done % 10000 < batch_size:
                print(f"  Video embeds: {done:,}/{len(ds):,}", flush=True)

    print(f"  Video embeddings saved: {done:,}")


# ── audio embedding extraction ────────────────────────────────────────────────

def extract_audio_embeddings(split, batch_size, num_workers):
    from audio_cnn.audio_model   import AudioEncoder
    from audio_cnn.audio_dataset import AudioClipDataset

    os.makedirs(AUD_OUT_DIR, exist_ok=True)

    print(f"\nExtracting audio embeddings ({split})...")
    print(f"  Using audio checkpoint: {AUDIO_CKPT}")
    model = AudioEncoder(pretrained=False).to(device)
    ckpt  = torch.load(AUDIO_CKPT, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    ds     = AudioClipDataset(AUDIO_CSV, split=split, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True,
                        persistent_workers=(num_workers > 0))

    done = 0
    with torch.no_grad():
        for specs, labels in loader:
            specs  = specs.to(device, non_blocking=True)
            embeds = model.encode(specs).cpu().numpy().astype(np.float32)

            start = done
            for i, embed in enumerate(embeds):
                cid = ds.clips[start + i]["clip_id"]
                out = os.path.join(AUD_OUT_DIR, f"{cid}.npy")
                if not os.path.exists(out):
                    np.save(out, embed)
            done += len(embeds)
            if done % 10000 < batch_size:
                print(f"  Audio embeds: {done:,}/{len(ds):,}", flush=True)

    print(f"  Audio embeddings saved: {done:,}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split",      choices=["train", "val", "all"], default="all")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-audio", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    splits = ["train", "val"] if args.split == "all" else [args.split]

    for split in splits:
        if not args.skip_video:
            extract_video_embeddings(split, args.batch_size, args.num_workers)
        if not args.skip_audio:
            extract_audio_embeddings(split, args.batch_size, args.num_workers)

    print("\nEmbedding extraction complete.")


if __name__ == "__main__":
    main()
