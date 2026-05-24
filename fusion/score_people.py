"""
Score every tracked person in a video with the fusion model.

Outputs a TTM probability per temporal window and aggregate scores per person.
This uses already extracted clip-level fusion embeddings.
"""

import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from fusion_model import TTMFusionModel

HERE = os.path.dirname(os.path.abspath(__file__))
GROUP42 = os.path.dirname(HERE)
PIPELINE = os.path.join(GROUP42, "..", "ego4d_data", "v2", "full_scale", "pipeline")

CLIPS_CSV = os.path.join(PIPELINE, "data", "clips_index.csv")
VID_EMBED_DIR = os.path.join(HERE, "data", "video_embeds")
AUD_EMBED_DIR = os.path.join(HERE, "data", "audio_embeds")
CKPT = os.path.join(HERE, "checkpoints", "best_model.pth")
RESULTS_DIR = os.path.join(HERE, "results")

VID_DIM = 512
AUD_DIM = 512


def load_embed(path, dim):
    if os.path.exists(path):
        return np.load(path).astype(np.float32).flatten()[:dim]
    return np.zeros(dim, dtype=np.float32)


def build_windows(rows, t_seq, stride):
    rows = sorted(rows, key=lambda r: int(r["clip_id"]))
    for start in range(0, len(rows), stride):
        chunk = rows[start:start + t_seq]
        if chunk:
            yield chunk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video-uid", required=True)
    p.add_argument("--split", choices=["train", "val"], default=None)
    p.add_argument("--t-seq", type=int, default=16)
    p.add_argument("--stride", type=int, default=16)
    p.add_argument("--checkpoint", default=CKPT)
    args = p.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TTMFusionModel().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    by_person = defaultdict(list)
    with open(CLIPS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row["video_uid"] != args.video_uid:
                continue
            if args.split and row["split"] != args.split:
                continue
            by_person[row["person_id"]].append(row)

    if not by_person:
        raise SystemExit(f"No clips found for video_uid={args.video_uid}")

    results = {
        "video_uid": args.video_uid,
        "checkpoint": args.checkpoint,
        "people": [],
    }

    with torch.no_grad():
        for person_id, rows in sorted(by_person.items(), key=lambda x: x[0]):
            window_scores = []
            for chunk in build_windows(rows, args.t_seq, args.stride):
                video_seq = []
                audio_seq = []
                for row in chunk:
                    cid = int(row["clip_id"])
                    video_seq.append(load_embed(os.path.join(VID_EMBED_DIR, f"{cid}.npy"), VID_DIM))
                    audio_seq.append(load_embed(os.path.join(AUD_EMBED_DIR, f"{cid}.npy"), AUD_DIM))
                while len(video_seq) < args.t_seq:
                    video_seq.append(np.zeros(VID_DIM, dtype=np.float32))
                    audio_seq.append(np.zeros(AUD_DIM, dtype=np.float32))

                v = torch.tensor(np.stack(video_seq), dtype=torch.float32, device=device).unsqueeze(0)
                a = torch.tensor(np.stack(audio_seq), dtype=torch.float32, device=device).unsqueeze(0)
                logits, _ = model(v, a)
                prob = float(F.softmax(logits, dim=1)[0, 1].item())
                window_scores.append({
                    "start_clip_id": int(chunk[0]["clip_id"]),
                    "end_clip_id": int(chunk[-1]["clip_id"]),
                    "num_clips": len(chunk),
                    "ttm_probability": prob,
                })

            probs = [w["ttm_probability"] for w in window_scores]
            results["people"].append({
                "person_id": person_id,
                "num_clips": len(rows),
                "num_windows": len(window_scores),
                "mean_ttm_probability": float(np.mean(probs)),
                "max_ttm_probability": float(np.max(probs)),
                "predicted_ttm": bool(np.max(probs) >= 0.5),
                "window_scores": window_scores,
            })

    out_path = os.path.join(RESULTS_DIR, f"person_scores_{args.video_uid}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
