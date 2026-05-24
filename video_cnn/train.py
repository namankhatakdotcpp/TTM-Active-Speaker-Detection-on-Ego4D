"""
Training script for I3D-R50 TTM classifier.

Architecture : I3D-R50, Kinetics-400 pretrained, 2-class TTM head
Dataset      : clips_index.csv (636,406 clips, 16 frames each, 224×224)
Loss         : CrossEntropyLoss with inverse-frequency class weights
Optimizer    : AdamW  (lr=1e-4, wd=1e-4)
Schedule     : CosineAnnealingLR
Memory tricks: AMP (FP16) + gradient accumulation (effective batch = 32)

Usage (direct):
    python train.py

Usage (background, SSH-safe):
    nohup python train.py > logs/training.log 2>&1 &

Outputs:
    checkpoints/best_model.pth     ← best val accuracy
    checkpoints/epoch_{n}.pth      ← every SAVE_EVERY epochs
    results/metrics.json           ← full train/val history
    logs/training.log              ← stdout redirected by train_run.sh
"""

import csv
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# ── paths (resolve relative to this file) ─────────────────────────────────────
HERE       = os.path.dirname(os.path.abspath(__file__))
PIPELINE   = os.path.join(HERE, "..", "..", "ego4d_data", "v2", "full_scale", "pipeline")
CLIPS_CSV  = os.path.join(PIPELINE, "data", "clips_index.csv")
CKPT_DIR   = os.path.join(HERE, "checkpoints")
RESULTS    = os.path.join(HERE, "results", "metrics.json")

sys.path.insert(0, PIPELINE)
from dataset import TTMClipDataset

from model import I3D_TTM

# ── hyperparameters ───────────────────────────────────────────────────────────
EPOCHS          = 20
BATCH_SIZE      = 8          # reduced to avoid OOM on shared GPU
GRAD_ACCUM      = 4          # effective batch = 32
LR              = 1e-4
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 8
PIN_MEMORY      = True
PREFETCH        = 2
SAVE_EVERY      = 5
WARMUP_EPOCHS   = 2
RESUME_CKPT     = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "checkpoints", "best_model.pth")

# ── device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True   # auto-tune conv kernels for fixed input size
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ── helpers ───────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(
    model, loader, criterion, optimizer, scaler,
    grad_accum, is_train, epoch_idx
):
    model.train(is_train)
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    if is_train:
        optimizer.zero_grad()

    t0 = time.time()
    for step, (frames, labels) in enumerate(loader):
        # frames: (B, T, C, H, W)  →  model expects (B, C, T, H, W)
        frames = frames.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
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
        n_batches  += 1

        # Progress print every 50 steps
        if (step + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate    = (step + 1) / elapsed
            eta     = (len(loader) - step - 1) / rate
            phase   = "train" if is_train else "val"
            print(
                f"\r  [{phase}] step {step+1}/{len(loader)} | "
                f"loss {total_loss/n_batches:.4f} | "
                f"acc {total_acc/n_batches:.4f} | "
                f"ETA {eta/60:.1f}min   ",
                end="", flush=True
            )

    print()  # newline after \r
    return total_loss / n_batches, total_acc / n_batches


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)

    # ── datasets & loaders ────────────────────────────────────────────────────
    print(f"\nLoading datasets from: {CLIPS_CSV}")
    train_ds = TTMClipDataset(CLIPS_CSV, split="train")
    val_ds   = TTMClipDataset(CLIPS_CSV, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
        persistent_workers=True, prefetch_factor=PREFETCH,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=True, prefetch_factor=PREFETCH,
    )

    # ── class-weighted loss ───────────────────────────────────────────────────
    weights   = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Class weights — neg: {weights[0]:.3f}  pos: {weights[1]:.3f}")

    # ── model ─────────────────────────────────────────────────────────────────
    print("\nLoading I3D-R50 (Kinetics-400 pretrained)...")
    model = I3D_TTM(pretrained=True, dropout=0.5, freeze_backbone=True).to(device)
    # torch.compile gives ~20% speedup on PyTorch 2.x (skipped during warm-up head-only phase)
    params = model.count_parameters()
    print(f"Parameters — total: {params['total']/1e6:.1f}M  trainable: {params['trainable']/1e6:.1f}M")

    # ── optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    # ── resume from checkpoint ────────────────────────────────────────────────
    best_val_acc = 0.0
    history = []
    start_epoch = 1

    if os.path.exists(RESUME_CKPT):
        ckpt = torch.load(RESUME_CKPT, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        best_val_acc = ckpt.get("val_acc", 0.0)
        start_epoch  = ckpt.get("epoch", 1) + 1
        print(f"  >> Resumed from {RESUME_CKPT} (epoch {ckpt.get('epoch',1)}, val_acc={best_val_acc:.4f})")
        # If resuming past warm-up, unfreeze backbone immediately
        if start_epoch > WARMUP_EPOCHS + 1:
            model.unfreeze_backbone()
            optimizer = AdamW(model.parameters(), lr=LR / 5, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)
        # Load existing metrics history if available
        if os.path.exists(RESULTS):
            with open(RESULTS) as f:
                saved = json.load(f)
                history = saved.get("history", [])

    # ── training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training — epochs {start_epoch}–{EPOCHS}, effective batch = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"Warm-up: backbone frozen for first {WARMUP_EPOCHS} epochs\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_t0 = time.time()

        # Unfreeze backbone after warm-up
        if epoch == WARMUP_EPOCHS + 1:
            print("  >> Unfreezing backbone weights")
            model.unfreeze_backbone()
            # Rebuild optimizer with all params (no torch.compile — saves ~2GB VRAM on shared GPU)
            optimizer = AdamW(model.parameters(), lr=LR / 5, weight_decay=WEIGHT_DECAY)
            scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS - WARMUP_EPOCHS, eta_min=1e-6)

        print(f"Epoch [{epoch}/{EPOCHS}]  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler,
            GRAD_ACCUM, is_train=True, epoch_idx=epoch,
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer, scaler,
            GRAD_ACCUM, is_train=False, epoch_idx=epoch,
        )

        scheduler.step()
        epoch_time = time.time() - epoch_t0

        # Logging
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "train_acc":  round(train_acc,  5),
            "val_loss":   round(val_loss,   5),
            "val_acc":    round(val_acc,    5),
            "lr":         optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        print(
            f"  train loss={train_loss:.4f}  acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f}  acc={val_acc:.4f} | "
            f"time={epoch_time/60:.1f}min"
        )

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CKPT_DIR, "best_model.pth")
            torch.save({
                "epoch":     epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc":   val_acc,
                "val_loss":  val_loss,
            }, ckpt_path)
            print(f"  >> Best model saved (val_acc={val_acc:.4f})")

        # Periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth")
            torch.save({
                "epoch":     epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_acc":   val_acc,
            }, ckpt_path)
            print(f"  >> Checkpoint saved: epoch_{epoch:03d}.pth")

        # Write metrics after every epoch
        with open(RESULTS, "w") as f:
            json.dump({"best_val_acc": best_val_acc, "history": history}, f, indent=2)

    # ── final summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Training complete.")
    print(f"Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoints: {CKPT_DIR}")
    print(f"Metrics:     {RESULTS}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
