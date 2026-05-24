"""
Train SlowFast-R50 TTM classifier on balanced 50K clips.

Architecture : SlowFast-R50, Kinetics-400 pretrained
Dataset      : balanced_clips_50k.csv (25K TTM + 25K non-TTM in train)
Loss         : CrossEntropyLoss (balanced dataset → equal weights)
Optimizer    : AdamW  lr=1e-4  wd=1e-4
Schedule     : CosineAnnealingLR
Extras       : AMP (FP16), warm-up (freeze backbone 2 epochs), cudnn.benchmark

Outputs:
    checkpoints_slowfast/best_model.pth
    checkpoints_slowfast/epoch_*.pth
    results_slowfast/metrics.json
    logs/training_slowfast.log
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

HERE     = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.join(HERE, "..", "..", "ego4d_data", "v2", "full_scale", "pipeline")

sys.path.insert(0, PIPELINE)
from dataset import TTMClipDataset

sys.path.insert(0, HERE)
from model_slowfast import SlowFast_TTM

CLIPS_CSV = os.path.join(PIPELINE, "data", "balanced_clips_50k.csv")
CKPT_DIR  = os.path.join(HERE, "checkpoints_slowfast")
RESULTS   = os.path.join(HERE, "results_slowfast", "metrics.json")

# ── hyperparameters ───────────────────────────────────────────────────────────
EPOCHS        = 20
BATCH_SIZE    = 8
GRAD_ACCUM    = 4       # effective batch = 32
LR            = 1e-4
WEIGHT_DECAY  = 1e-4
NUM_WORKERS   = 8
SAVE_EVERY    = 5
WARMUP_EPOCHS = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name(0)}")


# ── helpers ───────────────────────────────────────────────────────────────────

def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, scaler, grad_accum, is_train):
    model.train(is_train)
    total_loss = total_acc = 0.0
    n = 0
    t0 = time.time()

    if is_train:
        optimizer.zero_grad()

    for step, (frames, labels) in enumerate(loader):
        # frames: [B, T, C, H, W] → [B, C, T, H, W]
        frames = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(frames)
            loss   = criterion(logits, labels) / grad_accum

        if is_train:
            scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        total_acc  += accuracy(logits.detach(), labels)
        n += 1

        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta     = (len(loader) - step - 1) / ((step + 1) / elapsed)
            phase   = "train" if is_train else "val"
            print(f"\r  [{phase}] {step+1}/{len(loader)} | "
                  f"loss {total_loss/n:.4f} | acc {total_acc/n:.4f} | "
                  f"ETA {eta/60:.1f}min   ", end="", flush=True)

    print()
    return total_loss / n, total_acc / n


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

    print(f"\nLoading balanced datasets from: {CLIPS_CSV}")
    train_ds = TTMClipDataset(CLIPS_CSV, split="train")
    val_ds   = TTMClipDataset(CLIPS_CSV, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
        persistent_workers=True, prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=True, prefetch_factor=2,
    )

    # Equal class weights since dataset is balanced
    criterion = nn.CrossEntropyLoss()

    print("\nLoading SlowFast-R50 (Kinetics-400 pretrained)...")
    model = SlowFast_TTM(pretrained=True, dropout=0.5, freeze_backbone=True).to(device)
    p = model.count_parameters()
    print(f"Params — total: {p['total']/1e6:.1f}M  trainable: {p['trainable']/1e6:.1f}M")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    history      = []

    print(f"\nTraining: {EPOCHS} epochs, effective batch = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Warm-up: backbone frozen for first {WARMUP_EPOCHS} epochs\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        if epoch == WARMUP_EPOCHS + 1:
            print("  >> Unfreezing backbone")
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=LR / 5, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

        print(f"Epoch [{epoch}/{EPOCHS}]  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, scaler, GRAD_ACCUM, is_train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion,
                                          optimizer, scaler, GRAD_ACCUM, is_train=False)
        scheduler.step()

        elapsed = time.time() - t0
        record  = {"epoch": epoch,
                   "train_loss": round(train_loss, 5), "train_acc": round(train_acc, 5),
                   "val_loss":   round(val_loss,   5), "val_acc":   round(val_acc,   5),
                   "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)

        print(f"  train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"time={elapsed/60:.1f}min")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_acc": val_acc, "val_loss": val_loss},
                       os.path.join(CKPT_DIR, "best_model.pth"))
            print(f"  >> Best model saved (val_acc={val_acc:.4f})")

        if epoch % SAVE_EVERY == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict()},
                       os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth"))

        with open(RESULTS, "w") as f:
            json.dump({"best_val_acc": best_val_acc, "history": history}, f, indent=2)

    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    print(f"Checkpoints: {CKPT_DIR}")


if __name__ == "__main__":
    main()
