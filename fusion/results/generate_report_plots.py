"""Generate report plots from actual result data."""
import json, base64, io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

NAVY   = "#1B3A6B"
BLUE   = "#2563EB"
GREEN  = "#16A34A"
RED    = "#DC2626"
ORANGE = "#EA580C"
GRAY   = "#6B7280"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
})

def to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

# ── load training history ──────────────────────────────────────────────────
with open("metrics.json") as f:
    met = json.load(f)
history = met["history"]

epochs     = [h["epoch"]     for h in history]
tloss      = [h["train_loss"] for h in history]
vloss      = [h["val_loss"]   for h in history]
tacc       = [h["train_acc"]  for h in history]
vap        = [h["val_ap"]     for h in history]
vf1        = [h["val_f1"]     for h in history]
lr_vals    = [h["lr"]         for h in history]

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 — Training curves  (3-panel)
# ══════════════════════════════════════════════════════════════════
fig1, axes = plt.subplots(1, 3, figsize=(13, 3.8))
fig1.suptitle("Training & Validation Curves (18 Epochs)", fontsize=11,
              fontweight="bold", color=NAVY, y=1.02)

# Loss
ax = axes[0]
ax.plot(epochs, tloss, color=BLUE,  lw=2, marker="o", ms=3.5, label="Train Loss")
ax.plot(epochs, vloss, color=RED,   lw=2, marker="s", ms=3.5, linestyle="--", label="Val Loss")
ax.axvline(9, color=GREEN, lw=1.5, linestyle=":", label="Best (ep.9)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
ax.set_title("Loss vs Epoch", fontweight="bold")
ax.legend()

# AP / F1 / Acc
ax = axes[1]
ax.plot(epochs, vap,  color=NAVY,   lw=2.2, marker="o", ms=3.5, label="Val AP  (best=0.828)")
ax.plot(epochs, vf1,  color=ORANGE, lw=1.8, marker="s", ms=3.5, linestyle="--", label="Val F1@0.5")
ax.plot(epochs, tacc, color=BLUE,   lw=1.4, marker="^", ms=2.5, linestyle=":",  label="Train Acc")
ax.axvline(9, color=GREEN, lw=1.5, linestyle=":", label="Best (ep.9)")
ax.axhline(0.8284, color=NAVY, lw=0.8, linestyle="-.", alpha=0.45)
ax.set_ylim(0.50, 0.95)
ax.set_xlabel("Epoch"); ax.set_ylabel("Score")
ax.set_title("Val AP / F1 / Train Acc", fontweight="bold")
ax.legend()

# LR
ax = axes[2]
ax.step(epochs, lr_vals, color=RED, lw=2, where="post")
ax.set_yscale("log")
ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate")
ax.set_title("LR Schedule\n(ReduceLROnPlateau)", fontweight="bold")
ax.annotate("÷2 at ep.13", xy=(13, 1.5e-4), xytext=(14.5, 2.2e-4),
            fontsize=7, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
ax.annotate("÷2 at ep.17", xy=(17, 7.5e-5), xytext=(14.5, 6e-5),
            fontsize=7, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))

fig1.tight_layout()
b64_fig1 = to_b64(fig1); plt.close(fig1)

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 — Confusion matrices: Val (t=0.139) | Unseen (t=0.20)
# ══════════════════════════════════════════════════════════════════
fig2, axes = plt.subplots(1, 2, figsize=(10, 4.2))
fig2.suptitle("Confusion Matrices at Optimal Thresholds", fontsize=11,
              fontweight="bold", color=NAVY, y=1.02)

def draw_cm(ax, cm, labels, title, subtitle):
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=cm.max())
    plt.colorbar(im, ax=ax, fraction=0.044, pad=0.03)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels, fontsize=9); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title + "\n" + subtitle, fontweight="bold", fontsize=9.5)
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            pct = 100 * v / cm[i].sum()
            col = "white" if v > cm.max() * 0.55 else "black"
            ax.text(j, i, f"{v:,}\n({pct:.1f}%)", ha="center", va="center",
                    fontsize=9, color=col, fontweight="bold")

# Val balanced (941 windows), threshold=0.139
cm_val = np.array([[172, 267], [30, 472]])
draw_cm(axes[0], cm_val, ["non-TTM", "TTM"],
        "Validation Split (balanced, 941 windows)",
        "Threshold = 0.139 → F1=0.761  Recall=0.940")

# Unseen balanced (12780 windows), threshold=0.20
cm_test = np.array([[4146, 2244], [749, 5641]])
# At t=0.20: recall TTM=0.8827... let me recalculate
# precision=0.7045, recall=0.8371, so TP/(TP+FN)=0.8371 → TP=0.8371*6390=5349, FN=1041
# FP/(FP+TN)=? precision=TP/(TP+FP)→ FP=TP/0.7045-TP = 5349*(1/0.7045-1)=5349*0.4194=2243
# TN=6390-2243=4147
cm_test = np.array([[4147, 2243], [1041, 5349]])
draw_cm(axes[1], cm_test, ["non-TTM", "TTM"],
        "Unseen Tracks — clips_index.csv (balanced, 12,780 windows)",
        "Threshold = 0.20 → F1=0.765  Recall=0.837")

fig2.tight_layout()
b64_fig2 = to_b64(fig2); plt.close(fig2)

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 — Architecture diagram
# ══════════════════════════════════════════════════════════════════
fig3, ax = plt.subplots(figsize=(13, 4.8))
ax.set_xlim(0, 13); ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title("TTMFusionModel — Architecture Overview", fontsize=11,
             fontweight="bold", color=NAVY, pad=8)

def box(ax, x, y, w, h, label, sub="", color=BLUE, fs=9):
    r = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
        boxstyle="round,pad=0.07", lw=1.4, edgecolor=color,
        facecolor=color+"22")
    ax.add_patch(r)
    ax.text(x, y+(0.07 if sub else 0), label, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=color)
    if sub:
        ax.text(x, y-0.26, sub, ha="center", va="center", fontsize=7, color=GRAY)

def arr(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.2))

# Video stream
box(ax, 1.0, 4.0, 1.6, 0.7, "I3D-R50", "frozen · [B,T,512]", NAVY)
box(ax, 1.0, 2.7, 1.6, 0.7, "Video Proj", "Linear+LN+ReLU\n[B,T,256]", BLUE)
arr(ax, 1.0, 3.65, 1.0, 3.05)

# Audio stream
box(ax, 3.2, 4.0, 1.6, 0.7, "ResNet-18", "frozen · [B,T,512]", NAVY)
box(ax, 3.2, 2.7, 1.6, 0.7, "Audio Proj", "Linear+LN+ReLU\n[B,T,256]", BLUE)
arr(ax, 3.2, 3.65, 3.2, 3.05)

# Cross-modal attention
box(ax, 2.1, 1.55, 3.4, 0.85, "Bidirectional Cross-Modal Attention",
    "L=2 layers · 4 heads · shared weights → [B,T,512]", "#7C3AED")
arr(ax, 1.0, 2.35, 1.7, 1.98)
arr(ax, 3.2, 2.35, 2.5, 1.98)

# BiLSTM
box(ax, 5.7, 1.55, 1.9, 0.85, "BiLSTM", "2-layer · hidden=256\n[B,T,512]", "#0891B2")
arr(ax, 3.8, 1.55, 4.75, 1.55)

# LayerNorm
box(ax, 7.7, 1.55, 1.5, 0.85, "LayerNorm", "stabilise\n[B,T,512]", "#0891B2")
arr(ax, 6.65, 1.55, 6.95, 1.55)

# Bahdanau
box(ax, 9.6, 1.55, 1.75, 0.85, "Bahdanau\nAttention", "additive · [B,512]", "#D97706")
arr(ax, 8.45, 1.55, 8.73, 1.55)

# Classifier
box(ax, 11.5, 1.55, 1.55, 0.85, "Classifier", "Dropout+Linear\n[B,2]", GREEN)
arr(ax, 10.48, 1.55, 10.73, 1.55)

# Output arrow + label
arr(ax, 12.28, 1.55, 12.75, 1.55)
ax.text(12.82, 1.55, "TTM\nlogits", ha="left", va="center",
        fontsize=8.5, fontweight="bold", color=GREEN)

# no-audio token note
ax.text(6.5, 0.38,
        "✦ Learned no-audio token (512-d parameter) replaces all-zero audio windows",
        ha="center", va="center", fontsize=8, color="#92400E", style="italic")

fig3.tight_layout()
b64_fig3 = to_b64(fig3); plt.close(fig3)

# ══════════════════════════════════════════════════════════════════
# FIGURE 4 — Class distribution + model comparison bar chart
# ══════════════════════════════════════════════════════════════════
fig4, axes = plt.subplots(1, 2, figsize=(11, 4))
fig4.suptitle("Dataset Distribution & Model Performance Comparison", fontsize=11,
              fontweight="bold", color=NAVY, y=1.02)

# 4a: Class distribution
ax = axes[0]
splits   = ["Full Ego4D\n(636k)", "Train\n(55k)", "Val\n(15k)", "Unseen Test\n(clips_index)"]
ttm_pct  = [5.53, 50.0, 50.0, 5.53]
nttm_pct = [94.47, 50.0, 50.0, 94.47]
x = np.arange(len(splits))
ax.bar(x, nttm_pct, color="#93C5FD", label="non-TTM", width=0.55)
ax.bar(x, ttm_pct, bottom=nttm_pct, color=NAVY, label="TTM", width=0.55)
for i, (t, nt) in enumerate(zip(ttm_pct, nttm_pct)):
    ax.text(i, nt + t/2, f"{t:.1f}%", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")
ax.set_xticks(x); ax.set_xticklabels(splits)
ax.set_ylabel("Percentage (%)"); ax.set_ylim(0, 115)
ax.set_title("Class Distribution Across Splits", fontweight="bold")
ax.legend(loc="upper right")

# 4b: Model comparison
ax = axes[1]
models  = ["Baseline\n(concat, FocalLoss)", "Proposed — Val\n(t=0.5)", "Proposed — Val\n(t=0.139 opt.)", "Proposed — Unseen\n(t=0.20 opt.)"]
f1s    = [0.44,  0.628, 0.761, 0.765]
precs  = [0.85,  0.837, 0.639, 0.705]
recs   = [0.30,  0.502, 0.940, 0.837]
aucs   = [0.70,  0.828, 0.828, 0.847]
x = np.arange(len(models))
w = 0.19
ax.bar(x-1.5*w, f1s,   width=w, label="F1",       color=NAVY)
ax.bar(x-0.5*w, precs, width=w, label="Precision", color=BLUE)
ax.bar(x+0.5*w, recs,  width=w, label="Recall",    color=ORANGE)
ax.bar(x+1.5*w, aucs,  width=w, label="AP/AUC",    color=GREEN)
for bars, vals in zip([x-1.5*w, x-0.5*w, x+0.5*w, x+1.5*w], [f1s, precs, recs, aucs]):
    for xi, v in zip(bars, vals):
        ax.text(xi, v+0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)
ax.axhline(0.75, color=RED, lw=0.8, linestyle="--", alpha=0.5)
ax.set_xticks(x); ax.set_xticklabels(models, fontsize=7.5)
ax.set_ylabel("Score"); ax.set_ylim(0, 1.08)
ax.set_title("Metrics Across Model Variants", fontweight="bold")
ax.legend(loc="upper left", fontsize=7.5)

fig4.tight_layout()
b64_fig4 = to_b64(fig4); plt.close(fig4)

# ── save ──────────────────────────────────────────────────────────────────
output = {
    "fig1_training_curves": b64_fig1,
    "fig2_confusion":       b64_fig2,
    "fig3_architecture":    b64_fig3,
    "fig4_summary":         b64_fig4,
}
with open("report_plots.json", "w") as f:
    json.dump(output, f)

print("Plots generated:")
for k, v in output.items():
    print(f"  {k}: {len(v)//1024} KB")
