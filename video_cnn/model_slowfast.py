"""
SlowFast-R50 TTM binary classifier.

Input : two-pathway list from SlowFastDataset:
          slow [B, 3, 8,  224, 224]  — 8 frames (strided)
          fast [B, 3, 32, 224, 224]  — 32 frames (temporally upsampled)
Output: logits [B, 2]
Embed : [B, 2304]  — for fusion (before classification head)

Pre-trained weights: Kinetics-400 (loaded via torch.hub).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlowFast_TTM(nn.Module):
    """
    Args:
        pretrained  : load Kinetics-400 weights
        dropout     : dropout before TTM head
        freeze_backbone: freeze all layers except the classification head
    """

    EMBED_DIM = 2304   # SlowFast-R50 penultimate feature size

    def __init__(
        self,
        num_classes: int = 2,
        pretrained:  bool = True,
        dropout:     float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "slowfast_r50",
            pretrained=pretrained,
            verbose=False,
        )

        # Replace Kinetics-400 head (400 classes) with TTM binary head
        in_features = self.backbone.blocks[-1].proj.in_features  # 2304
        self.backbone.blocks[-1].proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    # ── pathway preparation ───────────────────────────────────────────────────

    @staticmethod
    def prepare_pathways(frames: torch.Tensor):
        """
        From [B, 3, 16, H, W] (16-frame clips) produce SlowFast input:
          slow: [B, 3, 8,  H, W]  — every-other frame
          fast: [B, 3, 32, H, W]  — temporal 2× upsample
        """
        slow = frames[:, :, ::2, :, :]                                   # [B,3,8,H,W]
        fast = F.interpolate(frames, size=(32, frames.shape[3], frames.shape[4]),
                             mode="nearest")                             # [B,3,32,H,W]
        return [slow, fast]

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: [B, 3, 16, H, W]  (C-first, 16 frames)
        Returns:
            logits: [B, 2]
        """
        pathways = self.prepare_pathways(frames)
        return self.backbone(pathways)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Returns 2304-d embedding (before head) for fusion.
        """
        pathways = self.prepare_pathways(frames)
        # Run all blocks except the final projection in the head
        x = pathways
        for i, block in enumerate(self.backbone.blocks):
            if i < len(self.backbone.blocks) - 1:
                x = block(x)
            else:
                # Head block: run pooling only, skip proj
                x = block.dropout(block.output_pool(x).flatten(1))
        return x

    # ── param management ──────────────────────────────────────────────────────

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        # keep head trainable
        for p in self.backbone.blocks[-1].proj.parameters():
            p.requires_grad = True

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def count_parameters(self) -> dict:
        total    = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


# ── sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = SlowFast_TTM(pretrained=False).to(device)
    x      = torch.randn(2, 3, 16, 224, 224, device=device)
    logits = model(x)
    print(f"Input  : {x.shape}")
    print(f"Logits : {logits.shape}")
    p = model.count_parameters()
    print(f"Params — total: {p['total']/1e6:.1f}M  trainable: {p['trainable']/1e6:.1f}M")
