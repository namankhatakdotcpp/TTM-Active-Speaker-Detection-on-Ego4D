"""
Training script for ResNet-18 audio TTM classifier.

Architecture : ResNet-18 on log-Mel spectrogram, ImageNet pretrained (1-ch adapted)
Input        : [B, 1, 64, 50]  — log-Mel spectrogram per 16-frame clip
Loss         : CrossEntropyLoss with class weights
Optimizer    : AdamW  lr=1e-4
Schedule     : CosineAnnealingLR
Extras       : AMP (FP16), SpecAugment (in dataset), cudnn.benchmark

Outputs:
    checkpoints/best_model.pth
    checkpoints/epoch_*.pth
    results/metrics.json
    logs/training.log
"""

import json
import os
import sys
import time

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

HERE      = os.path.dirname(os.path.abspath(__file__))
AUDIO_CSV = os.path.join(HERE, "data", "audio_index.csv")
CKPT_DIR  = os.path.join(HERE, "checkpoints")
RESULTS   = os.path.join(HERE, "results", "metrics.json")

sys.path.insert(0, HERE)
from audio_dataset import AudioClipDataset
from audio_model   import AudioEncoder

# ── hyperparameters ───────────────────────────────────────────────────────────
EPOCHS       = 20
BATCH_SIZE   = 128        # spectrograms are tiny — large batch fits easily
LR           = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 16
SAVE_EVERY   = 5

# ── device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
print(f"Device: {device}")
if device.type == "cuda":
    print(f"GPU:    {torch.cuda.get_device_name(0)}")


# ── helpers ───────────────────────────────────────────────────────────────────

def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, scaler, is_train):
    model.train(is_train)
    total_loss = total_acc = 0.0
    n = 0
    t0 = time.time()

    if is_train:
        optimizer.zero_grad()

    for step, (spec, labels) in enumerate(loader):
        spec   = spec.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=(device.type == "cuda")):
            logits = model(spec)
            loss   = criterion(logits, labels)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        total_acc  += accuracy(logits.detach(), labels)
        n += 1

        if (step + 1) % 100 == 0:
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

    print(f"\nLoading audio dataset from: {AUDIO_CSV}")
    train_ds = AudioClipDataset(AUDIO_CSV, split="train", augment=True)
    val_ds   = AudioClipDataset(AUDIO_CSV, split="val",   augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=True, prefetch_factor=4)

    weights   = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"Class weights — neg: {weights[0]:.3f}  pos: {weights[1]:.3f}")

    print("\nLoading ResNet-18 audio encoder (ImageNet pretrained)...")
    model  = AudioEncoder(pretrained=True, dropout=0.3).to(device)
    params = model.count_parameters()
    print(f"Params — total: {params['total']/1e6:.1f}M  trainable: {params['trainable']/1e6:.1f}M")

    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scaler    = GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    history      = []

    print(f"\nTraining: {EPOCHS} epochs, batch={BATCH_SIZE}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f"Epoch [{epoch}/{EPOCHS}]  lr={optimizer.param_groups[0]['lr']:.2e}")

        train_loss, train_acc = run_epoch(model, train_loader, criterion,
                                          optimizer, scaler, is_train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion,
                                          optimizer, scaler, is_train=False)
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
