"""
Create balanced_clips_50k.csv:
  train: 25,000 TTM (label=1) + 25,000 non-TTM (label=0)  = 50,000
  val  : all 2,679 TTM        + 2,679 non-TTM              = 5,358

Reads:  pipeline/data/clips_index.csv
Writes: pipeline/data/balanced_clips_50k.csv
"""

import csv
import os
import random

random.seed(42)

HERE      = os.path.dirname(os.path.abspath(__file__))
IN_CSV    = os.path.join(HERE, "..", "ego4d_data", "v2", "full_scale",
                         "pipeline", "data", "clips_index.csv")
OUT_CSV   = os.path.join(HERE, "..", "ego4d_data", "v2", "full_scale",
                         "pipeline", "data", "balanced_clips_50k.csv")

# ── load all rows split by label ──────────────────────────────────────────────
train_pos, train_neg = [], []
val_pos,   val_neg   = [], []

with open(IN_CSV, newline="") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    for row in reader:
        lbl   = int(row["label"])
        split = row["split"]
        if split == "train":
            (train_pos if lbl == 1 else train_neg).append(row)
        else:
            (val_pos if lbl == 1 else val_neg).append(row)

print(f"Train  — pos: {len(train_pos):,}  neg: {len(train_neg):,}")
print(f"Val    — pos: {len(val_pos):,}  neg: {len(val_neg):,}")

# ── balanced sampling ─────────────────────────────────────────────────────────
N_TRAIN = 25_000
N_VAL   = len(val_pos)   # use all val positives, equal negatives

random.shuffle(train_pos)
random.shuffle(train_neg)
random.shuffle(val_neg)

selected_train = train_pos[:N_TRAIN] + train_neg[:N_TRAIN]
selected_val   = val_pos             + val_neg[:N_VAL]

random.shuffle(selected_train)
random.shuffle(selected_val)

print(f"\nBalanced train: {len(selected_train):,} clips (50/50)")
print(f"Balanced val  : {len(selected_val):,} clips (50/50)")
print(f"Total         : {len(selected_train)+len(selected_val):,}")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(selected_train + selected_val)

print(f"\nWritten: {OUT_CSV}")
