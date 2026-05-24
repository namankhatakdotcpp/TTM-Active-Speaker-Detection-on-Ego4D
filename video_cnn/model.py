"""
I3D-R50 wrapper for TTM (Talking To Me) binary classification.

Backbone: i3d_r50 from facebookresearch/pytorchvideo (Kinetics-400 pretrained).
Head:     Replace the 400-class Kinetics head with a 2-class TTM head.

Input shape expected by this model:
    (B, C, T, H, W)  — standard 3D CNN layout
    C=3, T=16, H=224, W=224

The backbone is loaded from torch hub — weights are downloaded once and cached
at ~/.cache/torch/hub/checkpoints/.
"""

import torch
import torch.nn as nn


class I3D_TTM(nn.Module):
    """
    I3D-R50 fine-tuned for TTM binary classification.

    Args:
        num_classes:   Number of output classes (2 for TTM).
        pretrained:    Load Kinetics-400 pretrained weights via torch hub.
        dropout:       Dropout before the classification head.
        freeze_backbone: If True, freeze all backbone layers (only train head).
                         Useful for a quick warm-up phase.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ── load backbone ────────────────────────────────────────────────────
        self.backbone = torch.hub.load(
            "facebookresearch/pytorchvideo",
            "i3d_r50",
            pretrained=pretrained,
        )

        # ── replace classification head ──────────────────────────────────────
        # pytorchvideo wraps the head as model.blocks[-1]
        # The projection layer is at model.blocks[-1].proj
        in_features = self.backbone.blocks[-1].proj.in_features
        self.backbone.blocks[-1].proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        # ── optionally freeze backbone ───────────────────────────────────────
        if freeze_backbone:
            self.freeze_backbone()

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, H, W)  float tensor, ImageNet-normalised
        Returns:
            logits: (B, num_classes)
        """
        return self.backbone(x)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the penultimate I3D feature before the classification head.

        Args:
            x: (B, C, T, H, W)
        Returns:
            features: (B, F)
        """
        for i, block in enumerate(self.backbone.blocks):
            if i < len(self.backbone.blocks) - 1:
                x = block(x)
            else:
                # Mirror the head forward pass but skip the final projection.
                x = block.dropout(block.output_pool(x).flatten(1))
        return x

    # ── helpers ───────────────────────────────────────────────────────────────

    def freeze_backbone(self):
        """Freeze all backbone layers; keep the classification head trainable."""
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Then explicitly unfreeze the head we replaced
        for param in self.backbone.blocks[-1].proj.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """Unfreeze all layers (call after warm-up phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def count_parameters(self) -> dict:
        total   = sum(p.numel() for p in self.parameters())
        trained = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trained}


# ── quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = I3D_TTM(pretrained=False).to(device)   # pretrained=False for speed here
    x = torch.randn(2, 3, 16, 224, 224, device=device)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    params = model.count_parameters()
    print(f"Params — total: {params['total']/1e6:.1f}M  trainable: {params['trainable']/1e6:.1f}M")
